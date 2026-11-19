"""End-to-end processing of one job (one field / one sequence of frames).

Memory-bounded design (works for 60 Mpx sensors and long sequences):
  pass 1  calibrate + flatten every frame, detect stars, write to disk (memmap)
  reg.    star matching with a polynomial distortion model (registration.py)
  hot px  detector-fixed defects from the dithered frames
  pass 2  resample every frame onto the reference grid (memmap)
  search  tile by tile (search.py): template, noise, filter, shift-and-add, vetting
  ident.  N-body predictions of known objects, matching, forced measurements
"""
from __future__ import annotations

import dataclasses
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
from astropy.time import Time
from scipy import ndimage

from . import __version__
from . import calibration as cal
from . import search as SR
from . import throughput as TP
from . import transients as TR
from . import inject as INJ
from . import classifier as CL
from .confidence import annotate as annotate_confidence
from . import tracking as T
from .astrometry import find_astap, header_wcs, header_wcs_kind, pixel_rate_to_sky, solve_astap
from .fields import split_fields
from .fitsmeta import group_key, list_fits, load_image, parse_section, read_meta
from .gpu import get_backend
from .identify import match_candidates, predict_field
from .imaging import track_crops, write_candidate_images, write_motion_frames, write_motion_gif
from .instruments import InstrumentStore
from .jobs import Job, RunContext
from .masters import MasterLibrary
from .mpcdata import COMETS_NAME, MPCORB_NAME, MpcStore
from .defects import LineProfiler, chip_fixed, chip_fixed_weak, inpaint_pixels
from .registration import FrameRegistration, PolyTransform, antialias, detect_stars, register_from_stars, warp
from .settings import SearchSettings, Settings
from .tracking import _quiet


class PipelineError(RuntimeError):
    pass


def job_search_params(settings: Settings, job: Job) -> SearchSettings:
    known = SearchSettings.__dataclass_fields__
    over = {k: _coerce(getattr(settings.search, k), v) for k, v in (job.params or {}).items() if k in known}
    return replace(settings.search, **over)


def _coerce(default, v):
    """Job parameter (often a string from the CLI / GUI) to the type of the default."""
    if isinstance(default, bool):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "ano", "on")
    return type(default)(v)


def _crop_masters(m: cal.Masters, sl) -> cal.Masters:
    if sl is None:
        return m
    return cal.Masters(dark=None if m.dark is None else m.dark[sl], flat=None if m.flat is None else m.flat[sl],
                       notes=m.notes)


def _astap_wcs(settings, inst, ref_meta, sl, b, ctx, hint: bool):
    """Plate solution of the reference frame with ASTAP (cropped / binned like the frames) or None."""
    exe = find_astap(settings.astap.executable)
    if not exe:
        if not hint:
            ctx.warn("ASTAP nenalezen a snímky nemají WCS – bez RA/Dec a bez identifikace známých objektů.")
        else:
            ctx.log("ASTAP nenalezen – WCS z polohy montáže nelze nahradit řešením.")
        return None
    # scale of the file as written (hardware binning from the header), not the profile's native one
    hw_bin = max(1, int(ref_meta.ybin or 1)) / max(1, int(inst.binning or 1))
    fov_h = inst.pixel_scale_arcsec * hw_bin * ref_meta.height / 3600.0
    radius = min(settings.astap.search_radius_deg, 2.0) if hint else settings.astap.search_radius_deg
    res = solve_astap(ref_meta.path, exe, ref_meta.ra_deg, ref_meta.dec_deg, fov_h, radius,
                      settings.astap.database_dir, settings.astap.timeout_s, settings.astap.sip)
    ctx.log(f"{res.message} ({res.seconds:.1f} s, pole {fov_h:.2f}°)")
    w = res.wcs
    if w is not None and sl is not None:
        w = w.slice(sl)
    if w is not None and b > 1:
        w = w.slice((slice(None, None, b), slice(None, None, b)))
    return w


def _sky_coords(cands, wcs, mjd_ref: float) -> None:
    for c in cands:
        c.ra, c.dec, c.rate_arcsec_min, c.pa_deg = pixel_rate_to_sky(wcs, c.x, c.y, c.vx, c.vy)
        c.sky_matrix = _local_sky_matrix(wcs, c.x, c.y, c.ra, c.dec)
        for p in c.positions:
            sky = wcs.pixel_to_world(p["x"], p["y"])
            p["ra"], p["dec"] = float(sky.ra.deg), float(sky.dec.deg)
            p["mjd"] = float(mjd_ref + p["dt_min"] / 1440.0)


def _pointing_offset(cands, known, scale: float, span_min: float, max_arcsec: float,
                     min_snr: float = 15.0) -> tuple[float, float, list] | None:
    """Pixel offset (measured − predicted) of a mount-position WCS from bright known objects.

    Pairs: a vetted detection with SNR ≥ ``min_snr`` within ``max_arcsec`` of a
    predicted known object whose sky velocity vector agrees (same tolerance as
    the identification). Two or more pairs must agree within 3 px; a single
    pair is used only if it is the one possible pair for that object."""
    rate_tol = max(0.05, 2.0 * scale / max(span_min, 1.0))
    lim_px = max_arcsec / scale
    pairs = []
    for k in known:
        if k.x is None:
            continue
        opts = []
        for c in cands:
            if c.auto_status != "candidate" or c.snr < min_snr or c.injected is not None:
                continue
            dx, dy = c.x - k.x, c.y - k.y
            if math.hypot(dx, dy) > lim_px:
                continue
            if c.rate_arcsec_min is not None and c.pa_deg is not None:
                cpa, kpa = math.radians(c.pa_deg), math.radians(k.pa_deg)
                dr = math.hypot(c.rate_arcsec_min * math.sin(cpa) - k.rate_arcsec_min * math.sin(kpa),
                                c.rate_arcsec_min * math.cos(cpa) - k.rate_arcsec_min * math.cos(kpa))
                if dr > max(rate_tol, 0.35 * k.rate_arcsec_min):
                    continue
            opts.append((dx, dy, c, k))
        pairs.extend((*o, len(opts)) for o in opts)
    if not pairs:
        return None
    best = []
    for p in pairs:
        grp = [q for q in pairs if math.hypot(q[0] - p[0], q[1] - p[1]) <= 3.0]
        if len({id(q[3]) for q in grp}) > len({id(q[3]) for q in best}):
            best = grp
    objs = {id(q[3]) for q in best}
    if len(objs) == 1 and not (len(best) == 1 and best[0][4] == 1):
        return None                            # one object, several possible detections: ambiguous
    dx = float(np.median([q[0] for q in best]))
    dy = float(np.median([q[1] for q in best]))
    return dx, dy, [q[3].name for q in best]


CALIB_HINT = (" Vyberte master ručně, nebo zpracujte bez něj: tlačítko „Kalibrace…“ u úlohy "
              "(parametry úlohy calib_dark / calib_flat = cesta k souboru | none).")


def _calib_choice(job: Job, kind: str) -> str:
    v = str((job.params or {}).get(f"calib_{kind}", "auto") or "auto").strip()
    return "auto" if v.lower() in ("", "auto") else ("none" if v.lower() in ("none", "off", "false", "0") else v)


def _select_masters(job: Job, lib: MasterLibrary, ref_meta, inst, ctx: RunContext):
    """Dark / flat by the job's choice (auto | none | file path). Returns Selections or None (= not used)."""
    from .masters import manual_selection
    out = {}
    for kind in ("dark", "flat"):
        choice = _calib_choice(job, kind)
        if choice == "none":
            out[kind] = None
            continue
        s = manual_selection(kind, choice, ref_meta, inst) if choice != "auto" else (
            lib.select_dark(ref_meta) if kind == "dark" else lib.select_flat(ref_meta))
        ctx.log(f"{kind}: {s.path or '—'}  ← {s.reason}")
        for w in s.warnings:
            ctx.warn(w)
        if not s.ok:
            if choice != "auto":
                raise PipelineError(f"Master {kind}: " + s.reason + CALIB_HINT)
            if kind == "dark" or inst.require_flat:
                raise PipelineError(f"Master {kind}: " + s.reason + CALIB_HINT)
            ctx.warn("Bez master flatu – spoléhá se jen na umělý flat.")
            s = None
        out[kind] = s
    dsel, fsel = out["dark"], out["flat"]
    if dsel is None and _calib_choice(job, "dark") == "none":
        bias = lib.select_bias(ref_meta)
        if bias.ok:
            ctx.warn(f"Zpracování bez darku (zvoleno) – odečten jen master bias {Path(bias.path).name}; "
                     "horké pixely zachytí maska z ditheringu.")
            from .masters import Selection
            dsel = Selection("dark", bias.path, "Bez darku – použit master bias.")
        else:
            ctx.warn("Zpracování bez darku (zvoleno) – bez odečtu darku i biasu"
                     + ("; flat se tak dělí i offset kamery, vinětace se opraví jen umělým flatem." if fsel else ".")
                     + " Horké pixely zachytí maska z ditheringu.")
    if fsel is None and _calib_choice(job, "flat") == "none":
        ctx.warn("Zpracování bez master flatu (zvoleno) – spoléhá se jen na umělý flat.")
    return dsel, fsel


def run_job(job: Job, ctx: RunContext, settings: Settings) -> None:
    t_start = time.time()
    timings: dict[str, float] = {}
    sp = job_search_params(settings, job)
    _validate_search_settings(sp)
    store = ctx.store
    jdir = store.job_dir(job.id)
    work = jdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    def tick(name):
        timings[name] = round(time.time() - tick.t, 2)
        tick.t = time.time()
    tick.t = time.time()

    inst = InstrumentStore(settings.instruments_dir).get(job.instrument_id)
    ctx.log(f"Přístroj: {inst.name or inst.id}  ({inst.pixel_scale_arcsec:.3f}″/px nativně, "
            f"softwarový binning {inst.software_binning})")

    # ---------------------------------------------------------------- frames
    ctx.stage("Načítání snímků", 0.01)
    if job.files:
        paths = [Path(p) for p in job.files if Path(p).is_file()]
        if len(paths) < len(job.files):
            ctx.warn(f"{len(job.files) - len(paths)} snímků pole už neexistuje – vynechány.")
    else:
        paths = list_fits(job.lights_dir)
    if job.field:
        ctx.log("Zorné pole " + (f"{job.field.get('index')}/{job.field.get('total')}: " if job.field.get("total", 1) > 1
                                 else ": ") + str(job.field.get("text", "")))
    elif settings.field_split and len(paths) >= 5:
        # safety net (CLI/API callers, re-runs of old jobs): never mix fields of view in one stack
        fs = split_fields(paths, inst, settings.field_tolerance_fov)
        if fs.multiple:
            best = max(fs.fields, key=lambda f: f.n)
            ctx.warn(f"Adresář obsahuje {len(fs.fields)} zorných polí – zpracováno jen největší ({best.describe()}). "
                     f"Pro všechna pole založte úlohu přes GUI / CLI, které je rozdělí do samostatných úloh.")
            paths = [Path(p) for p in best.files]
            job.field = {"index": best.index, "total": len(fs.fields), "label": best.label,
                         "ra_deg": best.ra_deg, "dec_deg": best.dec_deg, "n": best.n, "text": best.describe()}
    if len(paths) < 2:
        raise PipelineError(f"V adresáři {job.lights_dir} je jen {len(paths)} FITS snímků (min. 2, synthetic tracking 5)"
                            + (" pro toto zorné pole." if job.field else "."))
    metas = [read_meta(p, inst) for p in paths]
    groups = Counter((m.exptime, m.filter, m.xbin, m.ybin, m.width, m.height) for m in metas)
    key, cnt = groups.most_common(1)[0]
    want = str((job.params or {}).get("frame_group") or "").strip()
    if len(groups) > 1:
        chosen = None
        if want:
            chosen = next((k for k in groups if group_key(next(m for m in metas if (
                m.exptime, m.filter, m.xbin, m.ybin, m.width, m.height) == k)) == want), None)
        if chosen is not None:
            key, cnt = chosen, groups[chosen]
            ctx.log(f"Smíšené snímky ({len(groups)} skupin) – podle volby použita skupina: "
                    f"{key[0]} s, filtr {key[1]}, bin {key[2]}x{key[3]} ({cnt} snímků).")
        else:
            ctx.warn(f"Smíšené snímky ({len(groups)} skupin) – použita největší skupina: "
                     f"{key[0]} s, filtr {key[1]}, bin {key[2]}x{key[3]} ({cnt} snímků)"
                     + (f"; zvolená skupina {want} v tomto poli není." if want else
                        ". Jinou skupinu lze zvolit při zakládání úlohy."))
    metas = [m for m in metas if (m.exptime, m.filter, m.xbin, m.ybin, m.width, m.height) == key]
    if any(m.mjd_mid is None for m in metas):
        raise PipelineError("Některé snímky nemají čitelný DATE-OBS – časy jsou pro tracking nezbytné.")
    metas.sort(key=lambda m: m.mjd_mid)
    n = len(metas)
    job.n_frames = n
    mjd = np.array([m.mjd_mid for m in metas])
    span_min = (mjd[-1] - mjd[0]) * 1440.0
    ref_i = n // 2
    ref_meta = metas[ref_i]
    ctx.log(f"{n} snímků, expozice {key[0]} s, filtr {key[1]}, rozsah {span_min:.1f} min")
    # observatory
    obs_code = (inst.observatory_code or settings.observatory_code).strip().upper()
    has_profile_site = inst.observer_lat_deg is not None and inst.observer_lon_deg is not None
    lat = inst.observer_lat_deg if inst.observer_lat_deg is not None else settings.observer_lat_deg
    lon = inst.observer_lon_deg if inst.observer_lon_deg is not None else settings.observer_lon_deg
    alt = inst.observer_alt_m if inst.observer_alt_m is not None else settings.observer_alt_m
    site_source = "profil přístroje" if has_profile_site else "staré společné nastavení"
    if not inst.observatory_code or not has_profile_site:
        ctx.warn("Profil přístroje nemá úplné vlastní údaje stanoviště – chybějící hodnoty jsou převzaty "
                 "ze starého společného nastavení. Doplňte je v Přístroje → Upravit.")
    if inst.site_from_header and ref_meta.site_lat is not None and ref_meta.site_lon is not None:
        lat, lon = ref_meta.site_lat, ref_meta.site_lon
        alt = ref_meta.site_alt if ref_meta.site_alt is not None else alt
        site_source = "FITS hlavička"
        ctx.log(f"Poloha observatoře z hlavičky: {lat:.4f}°, {lon:.4f}°, {alt:.0f} m")
    else:
        ctx.log(f"Stanoviště ({site_source}): MPC {obs_code or '—'}, {lat:.4f}°, {lon:.4f}°, {alt:.0f} m")
    # crop
    sl = None
    if inst.crop == "auto":
        sl = parse_section(ref_meta.datasec, (ref_meta.height, ref_meta.width))
    elif inst.crop not in ("", "none"):
        sl = parse_section(inst.crop, (ref_meta.height, ref_meta.width))
    if sl is not None:
        ctx.log(f"Ořez na užitečnou oblast čipu: řádky {sl[0].start}–{sl[0].stop}, sloupce {sl[1].start}–{sl[1].stop}")
    tick("frames")

    # --------------------------------------------------------------- masters
    ctx.stage("Výběr master snímků", 0.02)
    lib = MasterLibrary(inst, cache_file=settings.instruments_dir / f"{inst.id}.masters.json")
    lib.scan()
    dsel, fsel = _select_masters(job, lib, ref_meta, inst, ctx)
    masters = _crop_masters(cal.load_masters(dsel, fsel), sl)
    tick("masters")

    # ---------------------------------------------------- pass 1: calibration
    ctx.stage("Kalibrace, umělý flat, detekce hvězd", 0.03)
    b = max(1, int(inst.software_binning))
    shape0 = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) if sl is not None else (ref_meta.height, ref_meta.width)
    shape = (shape0[0] // b, shape0[1] // b) if b > 1 else shape0
    norm_mm = _track(ctx, np.lib.format.open_memmap(work / "norm.npy", mode="w+", dtype=np.float32, shape=(n,) + shape))
    stars: list = [None] * n
    wcs_list: list = [None] * n
    wcs_kind: list = [None] * n            # "solved" | "pointing" (mount position only) | None
    bstats: list = [None] * n
    workers = _workers(settings, per_task_mb=12 * shape[0] * shape[1] * 4 / 2**20)

    def calib_one(i):
        m = metas[i]
        raw, hdr = load_image(m.path)
        if sl is not None:
            raw = raw[sl]
        img, bad = cal.calibrate(raw, masters, inst.saturation_adu)
        del raw
        if getattr(inst, "cosmics", False):
            img, bad = cal.remove_cosmics(img, bad, inst.gain_e_adu, inst.read_noise_e)
        if b > 1:
            img = cal.software_bin(img, b)
            bad = cal.software_bin(bad.astype(np.float32), b) > 0
        nrm, bad, st = cal.flatten_and_normalise(img, bad, max(16, sp.background_mesh_px // b))
        nrm[bad] = np.nan
        if nrm.shape != shape:
            raise PipelineError(f"Snímek {Path(m.path).name} má jiný rozměr {nrm.shape} než ostatní {shape}")
        norm_mm[i] = nrm
        w_i = header_wcs(hdr)
        wcs_kind[i] = header_wcs_kind(hdr) if w_i is not None else None
        if w_i is not None and sl is not None:
            w_i = w_i.slice(sl)
        if w_i is not None and b > 1:
            w_i = w_i.slice((slice(None, None, b), slice(None, None, b)))
        stars[i], wcs_list[i], bstats[i] = detect_stars(nrm, sat_sigma=None), w_i, st

    _parallel(calib_one, range(n), workers, ctx, 0.03, 0.25)
    norm_mm.flush()
    ctx.log(f"Pozadí: medián {np.median([s['bkg_median'] for s in bstats]):.1f} ADU, "
            f"gradient (p-p) {np.median([s['bkg_ptp'] for s in bstats]):.1f} ADU, "
            f"šum {np.median([s['rms_median'] for s in bstats]):.2f} ADU; hvězd na snímek ~{int(np.median([len(s) for s in stars]))}")
    tick("calibration")

    # ------------------------------------------------------------ registration
    ctx.stage("Sesazení snímků podle hvězd", 0.29)
    h, w = shape
    regs: list[FrameRegistration]
    norm_path = work / "norm.npy"
    max_off = float(settings.field_tolerance_fov) * min(h, w)
    try:
        wl = wcs_list if all(x is not None for x in wcs_list) else None
        failed: list = []
        regs = register_from_stars(stars, shape, ref_i, inst.registration_order, wl, ctx.log,
                                   failed=failed, max_offset_px=max_off)
        if len(failed) >= (n - 1) / 2:
            # the reference may be the odd one out (e.g. a frame of another field):
            # try a few other references and keep the one most frames agree with
            best = (len(failed), ref_i, regs, failed)
            for cand in [f[0] for f in failed][:: max(1, len(failed) // 3)][:3]:
                if len(stars[cand]) < 30:
                    continue
                f2: list = []
                r2 = register_from_stars(stars, shape, cand, inst.registration_order, wl, None,
                                         failed=f2, max_offset_px=max_off)
                if len(f2) < best[0]:
                    best = (len(f2), cand, r2, f2)
            if best[1] != ref_i:
                ctx.log(f"Referenční snímek {ref_i} se shoduje s menšinou snímků – nová reference snímek {best[1]}.")
            _, ref_i, regs, failed = best
        if failed:
            keep = [i for i in range(n) if regs[i] is not None]
            names = ", ".join(Path(metas[i].path).name for i, _ in failed[:6]) + (" …" if len(failed) > 6 else "")
            ctx.warn(f"{len(failed)} z {n} snímků se nepodařilo sesadit podle hvězd s referenčním snímkem "
                     f"(jiné zorné pole, mraky, rozmazání?) – vyřazeny, aby se nesmíchala různá pole: {names}")
            if len(keep) < 2:
                raise PipelineError(f"Po vyřazení nesesazených snímků zbývá jen {len(keep)} (min. 2).")
            new_path = work / "norm_keep.npy"
            new_mm = _track(ctx, np.lib.format.open_memmap(new_path, mode="w+", dtype=np.float32,
                                                           shape=(len(keep),) + shape))
            for k, i in enumerate(keep):
                new_mm[k] = norm_mm[i]
            new_mm.flush()
            _release_memmap(norm_mm)
            norm_mm = None
            _safe_unlink(norm_path, ctx)
            norm_mm, norm_path = new_mm, new_path
            ref_i = keep.index(ref_i)
            metas = [metas[i] for i in keep]
            stars = [stars[i] for i in keep]
            wcs_list = [wcs_list[i] for i in keep]
            bstats = [bstats[i] for i in keep]
            regs = [regs[i] for i in keep]
            mjd = mjd[keep]
            n = len(keep)
            job.n_frames = n
            ref_meta = metas[ref_i]
            span_min = (mjd[-1] - mjd[0]) * 1440.0
            ctx.log(f"Pokračuje se s {n} snímky ({span_min:.1f} min).")
    except PipelineError:
        raise
    except RuntimeError as e:
        ctx.warn(f"Registrace podle hvězd selhala ({e}) – použita fázová korelace (jen posun).")
        regs = []
        for i in range(n):
            dy, dx = T.measure_offset(np.nan_to_num(norm_mm[ref_i]), np.nan_to_num(norm_mm[i]))
            tr = PolyTransform.fit(np.array([[0, 0], [w, 0], [0, h], [w, h]], float),
                                   np.array([[-dx, -dy], [w - dx, -dy], [-dx, h - dy], [w - dx, h - dy]], float),
                                   1, ((w - 1) / 2, (h - 1) / 2), max(w, h) / 2)
            inv = PolyTransform.fit(np.array([[-dx, -dy], [w - dx, -dy], [-dx, h - dy], [w - dx, h - dy]], float),
                                    np.array([[0, 0], [w, 0], [0, h], [w, h]], float),
                                    1, ((w - 1) / 2, (h - 1) / 2), max(w, h) / 2)
            regs.append(FrameRegistration(tr, inv, 0, 0.0, "phase", (-dx, -dy)))
    rms = [r.rms_px for i, r in enumerate(regs) if i != ref_i]
    shifts = [math.hypot(*r.offset) for r in regs]
    ctx.log(f"Sesazení: medián RMS {np.median(rms):.3f} px, max {max(rms):.3f} px; dithering až {max(shifts):.1f} px")
    if max(rms) > 0.3:
        ctx.warn(f"Horší sesazení některého snímku (RMS {max(rms):.2f} px) – odečet hvězd bude méně čistý.")
    tick("registration")

    # ------------------------------------------------------------ hot pixels
    ctx.stage("Horké pixely a defekty čipu", 0.31)
    dithered = max(shifts) >= 3.0
    lines = LineProfiler(shape, block=128)
    hot_total = 0
    hot_stars = 0
    # weak warm pixels: a single pixel 2–6 σ in the chip median gives a track SNR 7–25 when the
    # dither drifts slowly (11P, 28 frames, 11 px drift: 7 400 false detections); the level is
    # 6 × the noise of a median of n unit-noise values, never below 1.5 σ
    weak_thr = max(1.5, 6.0 * 1.2533 / math.sqrt(max(n, 1)))
    inpaint_hot = bool(getattr(sp, "inpaint_hot_pixels", True))
    weak_total = 0

    def _offsets(cy, cx, y0):
        offs = np.empty((n, len(cy), 2), np.float32)
        for i in range(n):
            fx, fy = regs[i].ref_to_frame(cx.astype(float), (cy + y0).astype(float))
            offs[i, :, 0] = np.asarray(fy) - (cy + y0)
            offs[i, :, 1] = np.asarray(fx) - cx
        return offs

    for y0 in range(0, h, 1024):
        band = np.array(norm_mm[:, y0:y0 + 1024, :])
        med = SR.temporal_stats(band)[0]
        lines.add(med, y0)
        hot = (med > 6.0) if dithered else np.zeros(med.shape, bool)
        if hot.any():
            # a small or clustered dithering leaves stars in the chip-coordinate median:
            # keep only pixels that are bright on the chip, not on the sky (161P: a star on
            # the comet's path was masked in every frame and its hole zeroed the comet)
            cy, cx = np.nonzero(hot)
            fixed = chip_fixed(band, med, np.stack([cy, cx], 1), _offsets(cy, cx, y0))
            hot_stars += int((~fixed).sum())
            hot[:] = False
            hot[cy[fixed], cx[fixed]] = True
        if dithered and weak_thr < 6.0:
            with np.errstate(invalid="ignore"):
                weak = (med > weak_thr) & (med <= 6.0)
            if weak.any():
                wy, wx = np.nonzero(weak)
                wf = chip_fixed_weak(band, med, np.stack([wy, wx], 1), _offsets(wy, wx, y0))
                hot[wy[wf], wx[wf]] = True
                weak_total += int(wf.sum())
        del band
        if hot.any():
            lab, nl = ndimage.label(hot)
            sizes = ndimage.sum(hot, lab, index=np.arange(1, nl + 1))
            small = np.isin(lab, np.nonzero(sizes <= 6)[0] + 1)
            # only very hot pixels get their neighbours too (charge spreading); a 1-px dilation of
            # every hot pixel made the masked area 5× larger (1.5 % of the chip in 88P/V)
            with np.errstate(invalid="ignore"):
                very = small & (np.nan_to_num(med, nan=0.0) > 20.0)
            if very.any():
                small |= ndimage.binary_dilation(very, iterations=1)
            sy, sx = np.nonzero(small)
            for i in range(n):
                view = norm_mm[i, y0:y0 + 1024, :]
                if inpaint_hot:
                    inpaint_pixels(view, sy, sx)
                else:
                    view[small] = np.nan
                del view               # a slice keeps the file mapped (WinError 32)
            hot_total += int(small.sum())
    if dithered:
        ctx.log(f"{'Nahrazeno mediánem sousedů' if inpaint_hot else 'Zamaskováno'} {hot_total} px pevně vázaných na čip (horké pixely, defekty)"
                + (f", z toho {weak_total} slabých teplých pixelů ({weak_thr:.1f}–6 σ v mediánu čipu)"
                   if weak_total else "")
                + (f"; {hot_stars} px nad prahem patří hvězdám (malý dithering) – ponechány." if hot_stars else "."))
    else:
        ctx.log("Dithering < 3 px – horké pixely se odečtou se statickým polem (doporučeno ditherovat).")
    # bad columns / rows: weak per pixel, but a long line that moves with the dither
    # (or with a drift of the pointing) is fitted by a straight track
    defects = lines.find() if getattr(sp, "mask_bad_columns", True) else None
    if defects is not None and (defects.columns or defects.column_blocks or defects.rows or defects.row_blocks):
        line_px = 0
        for y0 in range(0, h, 1024):
            for i in range(n):
                view = norm_mm[i, y0:y0 + 1024, :]
                k = defects.apply(view, y0)
                if i == 0:
                    line_px += k
                del view
        norm_mm.flush()
        ctx.log(f"Vadné sloupce/řádky čipu: {defects.describe()} ({100 * line_px / (h * w):.2f} % plochy) – zamaskováno "
                f"ve všech snímcích. Nejvýraznější: {defects.worst()}.")
    else:
        ctx.log("Vadné sloupce/řádky čipu: žádné nenalezeny.")
    norm_mm.flush()
    tick("hotpixels")

    # ---------------------------------------------------- pass 2: resampling
    ctx.stage("Převzorkování na referenční snímek", 0.33)
    reg_mm = _track(ctx, np.lib.format.open_memmap(work / "reg.npy", mode="w+", dtype=np.float32, shape=(n,) + shape))
    aa_sigma = 0.0
    aa_target = float(getattr(sp, "antialias_fwhm_px", 0.0) or 0.0)
    cy0, cx0 = shape[0] // 2, shape[1] // 2
    with _quiet():
        fw0 = T.estimate_fwhm(np.nan_to_num(np.array(norm_mm[ref_i, max(0, cy0 - 512):cy0 + 512,
                                                               max(0, cx0 - 512):cx0 + 512])))
    if aa_target > 0:
        aa_below = min(float(getattr(sp, "antialias_below_fwhm_px", 2.3) or 0.0), 0.95 * aa_target)
        if fw0 >= aa_below:
            ctx.log(f"FWHM hvězd {fw0:.2f} px ≥ {aa_below:.2f} px – snímky se před převzorkováním nevyhlazují "
                    f"(vyhlazení dělá ze šumu zrnka velikosti hvězdy a přidává falešné detekce).")
        else:
            aa_sigma = math.sqrt((aa_target / 2.3548) ** 2 - (fw0 / 2.3548) ** 2)
            ctx.log(f"Hvězdy jsou podvzorkované (FWHM {fw0:.2f} px): před převzorkováním se snímky vyhladí na "
                    f"FWHM ≈ {aa_target:.1f} px (σ {aa_sigma:.2f} px). Interpolace by jinak měnila tvar hvězd podle "
                    f"zlomku pixelu a po odečtu by zůstávala rezidua a tmavé kroužky.")

    mask_transients = bool(getattr(sp, "mask_transients", True))
    if not mask_transients:
        ctx.log("Maskování kosmiky a stop družic je vypnuté (nastavení mask_transients).")

    cosmic_hits: dict = {}

    def warp_one(i):
        f = np.array(norm_mm[i])
        if mask_transients:
            # candidates only – they are confirmed as single-frame events after registration
            g = TR.mask_cosmics(f, fw0, apply=False)
            if isinstance(g, np.ndarray) and g.any():
                cy, cx = np.nonzero(g)
                cosmic_hits[i] = _frame_to_grid(regs[i], cx.astype(float), cy.astype(float), i == ref_i)
        if aa_sigma > 0:
            f = antialias(f, aa_sigma)
        reg_mm[i] = f if i == ref_i else warp(f, regs[i].ref_to_frame, shape)

    _parallel(warp_one, range(n), _workers(settings, per_task_mb=8 * shape[0] * shape[1] * 4 / 2**20),
              ctx, 0.33, 0.1)
    reg_mm.flush()
    _release_memmap(norm_mm)
    norm_mm = None                         # release the mapping before deleting (Windows)
    _safe_unlink(norm_path, ctx)
    if mask_transients:
        if cosmic_hits:
            order = list(np.argsort(mjd))
            c_px, c_kept = TR.confirm_cosmics(reg_mm, order, cosmic_hits,
                                              grow=1 + int(math.ceil(2 * aa_sigma)))
            reg_mm.flush()
            ctx.log(f"Kosmické záření: zamaskováno {c_px} px v {len(cosmic_hits)} snímcích"
                    + (f"; {c_kept} ostrých špiček ponecháno – jsou i v sousedních snímcích (hvězdy, planetky "
                       f"v snímcích s lepším seeingem)" if c_kept else "") + ".")
        try:
            npx, tframes = TR.mask_trails(reg_mm, fw0)
            reg_mm.flush()
            if tframes:
                ctx.log(f"Stopy družic/letadel: zamaskováno {npx} px ve snímcích {', '.join(str(i) for i in tframes)} "
                        f"(jen v těch snímcích, jinde zůstávají data).")
        except Exception as e:                    # never stop a job because of this aid
            ctx.warn(f"Maskování stop družic selhalo ({e}).")
    tick("resample")

    # ------------------------------------------------------------ astrometry
    ctx.stage("Astrometrie referenčního snímku", 0.44)
    wcs_ref = wcs_list[ref_i]
    n_wcs = sum(x is not None for x in wcs_list)
    if 0 < n_wcs < n:
        ctx.log(f"{n - n_wcs} z {n} snímků nemá WCS v hlavičce – sesazení jen podle hvězd, "
                f"astrometrie z ostatních snímků.")
    wcs_src_kind = wcs_kind[ref_i]
    if wcs_ref is None and n_wcs:
        wcs_ref, src_j = _wcs_via_registration(wcs_list, regs, ref_i, shape)
        if wcs_ref is not None:
            wcs_src_kind = wcs_kind[src_j]
            ctx.log(f"Referenční snímek nemá WCS – převzat ze snímku {src_j} přes transformaci sesazení.")
    wcs_source = "header" if wcs_ref is not None else None
    if wcs_ref is not None:
        if wcs_list[ref_i] is not None:
            ctx.log("WCS referenčního snímku převzat z hlavičky.")
        if wcs_src_kind == "pointing":
            # CRVAL = mount position (INDI …): only as good as the pointing – solve if we can
            solved = _astap_wcs(settings, inst, ref_meta, sl, b, ctx, hint=True)
            if solved is not None:
                wcs_ref, wcs_source = solved, "astap"
                ctx.log("WCS v hlavičce byl jen poloha montáže – nahrazen řešením ASTAP.")
            else:
                wcs_source = "pointing"
                ctx.warn("WCS v hlavičce zapsal ovladač z polohy montáže (CRVAL = RA/Dec montáže), není to "
                         "astrometrické řešení snímku – RA/Dec mohou být posunuté o desítky ″ až ′. Nulový bod se "
                         "zkusí dokalibrovat podle jasných známých objektů; pro přesnou astrometrii nainstalujte "
                         "ASTAP s databází hvězd (Nastavení → ASTAP).")
    else:
        wcs_ref = _astap_wcs(settings, inst, ref_meta, sl, b, ctx, hint=False)
        if wcs_ref is not None:
            wcs_source = "astap"
    scale = _pixel_scale(wcs_ref) or (inst.pixel_scale_arcsec * b * max(1, int(ref_meta.xbin or 1))
                                      / max(1, int(inst.binning or 1)))
    tick("astrometry")

    # ------------------------------------------------------ throughput maps
    maps = None
    if getattr(sp, "throughput_maps", True):
        try:
            maps = TP.measure(reg_mm, stars[ref_i], cell=max(64, int(sp.throughput_cell_px) // b))
        except Exception as e:                       # diagnostics must never stop a job
            ctx.warn(f"Mapu propustnosti se nepodařilo změřit ({e}) – použije se škálování po dlaždicích.")
            maps = None
        if maps is not None:
            if maps.spread() < 0.08 and float(maps.per_frame_min().min()) > 0.8:
                ctx.log(f"Propustnost je po poli rovnoměrná (rozdíly do {100 * maps.spread():.0f} %) – "
                        f"stačí škálování po dlaždicích.")
                maps = None
            else:
                ctx.log(f"Mapa propustnosti z {maps.n_stars} hvězd (buňka {maps.cell} px): rozdíly v poli až "
                        f"{100 * maps.spread():.0f} % – šablona hvězdného pole se odečítá s mapou každého snímku.")
                for line in maps.describe():
                    ctx.log("  " + line)
                if float(maps.per_frame_min().min()) < sp.throughput_min:
                    ctx.warn(f"V části pole mají některé snímky v apertuře hvězd výrazně méně světla – zakrytí "
                             f"(horizont, kopule), oblačnost, nebo posunuté/rozmazané hvězdy (vítr, chvění montáže, "
                             f"horší sesazení). Místa s „propustností“ pod {100 * sp.throughput_min:.0f} % se v těchto "
                             f"snímcích do hledání nepočítají.")

    # twilight: frames taken with the Sun above −18° that are dimmed only by the brightening
    # sky keep a usable signal down to ~30 % throughput; with a short morning series those
    # minutes decide whether slow asteroids move enough (Coihueco 314, 10. 11.: 6 of 19 frames
    # dropped at 19–44 %, 23 min left, (266) Aline 14.7 mag moved 1 FWHM and was lost). Only
    # then the limit is lowered; a night series (Sun < −18° in every frame) is unchanged.
    if maps is not None:
        tw = _twilight_frames(mjd, lat, lon, alt)
        tmin_tw = float(getattr(sp, "twilight_throughput_min", 0.3))
        if tw is not None and tw.any() and tmin_tw < sp.throughput_min:
            # the frames that would be dropped must all be twilight frames (clouds at night: unchanged)
            low0 = (maps.grid.reshape(n, -1) < sp.throughput_min).mean(axis=1)
            dim = low0 >= getattr(sp, "throughput_drop_frac", 0.5)
            if (dim & tw).any() and not (dim & ~tw).any():
                sp = dataclasses.replace(sp, throughput_min=tmin_tw)
                ctx.log(f"Svítání/soumrak: {int(tw.sum())} snímků se Sluncem nad −18° – zeslabené snímky se "
                        f"ponechají do propustnosti {100 * tmin_tw:.0f} % (váha podle propustnosti), noční "
                        f"snímky beze změny.")

    # frames obstructed over most of the field (twilight, clouds at the end of a series) carry
    # (almost) no data: the search leaves those pixels out, but TrackNorm, the injected SNRs and
    # the vetting still counted them – Coihueco 312 (7. 11.): 6 of 19 frames at 19–42 %, every
    # detection got "zakryté místo", injections at nominal SNR 10–15 were found 7–13 of 25,
    # nothing passed. Such frames are dropped from the job.
    if maps is not None:
        low = (maps.grid.reshape(n, -1) < sp.throughput_min).mean(axis=1)
        dead = [i for i in range(n) if low[i] >= getattr(sp, "throughput_drop_frac", 0.5) and i != ref_i]
        if dead and n - len(dead) >= 5:
            keep = [i for i in range(n) if i not in dead]
            # "throughput" = star flux in a small aperture after registration: a frame whose stars are
            # displaced or smeared (wind, mount shake, tracking) loses it as much as under a cloud.
            # SMPH Aquitania (15. 11., clear sky): registration residuals 0.8–1.2 px vs 0.08 px.
            rms_all = [r.rms_px for r in regs if r is not None and r.rms_px > 0]
            rms_med = float(np.median(rms_all)) if rms_all else 0.0
            blur = [i for i in dead if regs[i] is not None and rms_med > 0
                    and regs[i].rms_px > max(0.3, 3 * rms_med)]
            dim = [i for i in dead if i not in blur]
            parts = []
            if dim:
                parts.append(f"{', '.join(str(i) for i in dim)}: hvězdy zeslabené na většině pole (oblačnost, "
                             f"mlha, rosa, zakrytí)")
            if blur:
                parts.append(f"{', '.join(str(i) for i in blur)}: hvězdy nepravidelně posunuté nebo rozmazané "
                             f"(sesazení RMS {min(regs[i].rms_px for i in blur):.2f}–"
                             f"{max(regs[i].rms_px for i in blur):.2f} px, běžně {rms_med:.2f} px – vítr, chvění "
                             f"montáže, vedení), proto v malé apertuře méně světla")
            ctx.warn(f"Snímky s „propustností“ (tok hvězd v apertuře proti typickému) pod "
                     f"{100 * sp.throughput_min:.0f} % na většině pole – vyřazeny ze hledání "
                     f"(zbývá {len(keep)} snímků). " + "; ".join(parts) + ".")
            new_mm = _track(ctx, np.lib.format.open_memmap(work / "reg_keep.npy", mode="w+", dtype=np.float32,
                                                           shape=(len(keep),) + shape))
            for k, i in enumerate(keep):
                new_mm[k] = reg_mm[i]
            new_mm.flush()
            _release_memmap(reg_mm)
            reg_mm = None
            _safe_unlink(work / "reg.npy", ctx)
            reg_mm = new_mm
            ref_i = keep.index(ref_i)
            metas = [metas[i] for i in keep]
            stars = [stars[i] for i in keep]
            wcs_list = [wcs_list[i] for i in keep]
            regs = [regs[i] for i in keep]
            mjd = mjd[keep]
            maps.grid = maps.grid[keep]
            n = len(keep)
            job.n_frames = n
            ref_meta = metas[ref_i]
            span_min = (mjd[-1] - mjd[0]) * 1440.0
            if float(maps.per_frame_min().min()) >= 0.8 and maps.spread() < 0.08:
                maps = None

    # ---------------------------------------------------------------- search
    ctx.stage("Synthetic tracking", 0.45)
    with _quiet():
        cy, cx = h // 2, w // 2
        cen = np.nanmedian(reg_mm[:, max(0, cy - 512):cy + 512, max(0, cx - 512):cx + 512], axis=0)
    fwhm = T.estimate_fwhm(np.nan_to_num(cen))
    dts = (mjd - mjd[ref_i]) * 1440.0
    # WCS refined on UCAC4 (header WCS of a wide field can be off by several px in the corners:
    # CTA-N 320/2, corner 2.8 px median / 7 px at the very corner → RA/Dec of candidates, known
    # objects and the DSS comparison off by ~1′)
    wcs_refined = None
    if wcs_ref is not None and getattr(settings, "ucac4_dir", "") and getattr(sp, "refine_wcs", True):
        try:
            from . import ucac4 as U4r
            from . import twilight as TWr
            from astropy.time import Time as _T
            u4r = U4r.find_root(settings.ucac4_dir)
            if u4r is not None:
                idx_r = sorted({max(0, ref_i - 1), ref_i, min(n - 1, ref_i + 1)})
                with _quiet():
                    img_r = np.nanmedian(np.asarray(reg_mm[idx_r], np.float32), axis=0)
                r0, r1, d0, d1, _ = TWr.field_box(wcs_ref, w, h)
                cat_r = U4r.query(u4r, r0, r1, d0, d1, U4r.band_for_filter(key[1]),
                                  float(_T(mjd[ref_i], format="mjd").jyear), mag_max=13.5)
                new_w, info_r = TWr.refine_wcs(img_r, wcs_ref, fwhm, cat_r)
                del img_r
                wcs_refined = info_r
                if new_w is not None:
                    wcs_ref = new_w
                    wcs_source = f"{wcs_source}+UCAC4"
                    ctx.log(f"WCS zpřesněn podle {info_r['pairs']} hvězd UCAC4: odchylka (medián) "
                            + ", ".join(f"{k} {info_r['header_px'].get(k, '–')} → {v} px" for k, v in info_r['refined_px'].items())
                            + ".")
                    if max(info_r["header_px"].values()) > 1.5:
                        ctx.warn(f"WCS v hlavičce byl v části pole o {max(info_r['header_px'].values()):.1f} px "
                                 f"(medián, nejhůř v rohu) mimo – použito zpřesněné WCS.")
                else:
                    ctx.log(f"WCS podle UCAC4 nezpřesněn ({info_r.get('note') or 'hlavička sedí stejně dobře'}"
                            + (f"; hlavička {info_r.get('header_px')}" if info_r.get('header_px') else "") + ").")
        except Exception as e:            # noqa: BLE001 – never stop a job for this
            ctx.warn(f"Zpřesnění WCS podle UCAC4 selhalo ({e}) – použito WCS z hlavičky.")
    # search mode: synthetic tracking (unchanged), the separate catalogue comparison for a few
    # twilight frames ("zdroj navíc", twilight.py), or both
    from . import twilight as TWL
    search_mode, mode_reason = TWL.choose_mode(getattr(sp, "search_mode", "auto"), n, span_min,
                                               _twilight_frames(mjd, lat, lon, alt),
                                               int(getattr(sp, "twilight_max_frames", 6)),
                                               float(getattr(sp, "twilight_max_span_min", 0.0)))
    ctx.log(f"Režim hledání: {dict(tracking='synthetic tracking', twilight='zdroj navíc (porovnání s UCAC4)', both='synthetic tracking + zdroj navíc')[search_mode]} – {mode_reason}.")
    tn = SR.TrackNorm(dts, fwhm)
    vmin_phys = tn.min_speed(sp.min_efficiency)
    # Lower speed limit: below ``min_efficiency`` the object moves less than
    # ~1 FWHM, the static template absorbs most of it and the track sum is
    # dominated by star-subtraction residuals.  TrackNorm divides by the tiny
    # white-noise sigma of such tracks, so residuals reach SNR 100–600 and
    # bright slow asteroids lock onto the slowest grid speed (305/pole 2:
    # Metis 0.35, Libussa 0.49 -> 0.11″/min, then rejected).  Objects slower
    # than the floor are still found at the floor speed – over the whole
    # sequence that is a sub-pixel difference.
    rmin_px = max(sp.rate_min_arcsec_min / scale, vmin_phys)
    rmax_px = sp.rate_max_arcsec_min / scale
    if vmin_phys * scale > sp.rate_min_arcsec_min:
        ctx.log(f"Nejnižší hledaná rychlost {vmin_phys * scale:.2f}″/min (nastaveno {sp.rate_min_arcsec_min:.2f}): "
                f"za {span_min:.0f} min se pomalejší objekt posune o méně než ~1 FWHM ({fwhm:.1f} px). "
                "Pomalejší objekty se najdou na této rychlosti (rozdíl pod 1 px); pro přesnější rychlost prodlužte sérii.")
    grid = T.make_velocity_grid(rmin_px, rmax_px, span_min, sp.step_px)
    if not len(grid) and search_mode != "twilight":
        raise PipelineError(f"Pro tuto krátkou sérii nevznikla žádná hledaná rychlost: minimum "
                            f"{rmin_px * scale:.2f}″/min je nad maximem {rmax_px * scale:.2f}″/min. "
                            "Prodlužte sérii nebo zvyšte maximální rychlost.")
    be = get_backend(settings.gpu.backend, settings.gpu.device, settings.gpu.memory_fraction)
    margin = int(math.ceil(rmax_px * span_min / 2 + 4 * fwhm + 16))
    tile = max(256, int(sp.tile_px))
    tiles = [(y0, x0) for y0 in range(0, h, tile) for x0 in range(0, w, tile)]
    ctx.log(f"FWHM {fwhm:.2f} px, měřítko {scale:.3f}″/px; mřížka {len(grid)} rychlostí "
            f"({rmin_px * scale:.2f}–{rmax_px * scale:.2f}″/min); {len(tiles)} dlaždic, přesah {margin} px; backend {be.name}")
    img_dir = jdir / "img"
    injections: list[dict] = []
    pre_known = None
    if int(getattr(sp, "inject_n", 0) or 0) > 0 and search_mode != "twilight":
        # realistic injections: the frame's own star profile and, for half of them, the motion
        # of the field's known objects (29P: Gaussians at random directions misled the ranking)
        psf = None
        try:
            psf = INJ.empirical_psf(np.array(reg_mm[ref_i]), stars[ref_i], fwhm)
        except Exception as e:          # noqa: BLE001 – fall back to a Gaussian
            ctx.log(f"Profil hvězd pro injekce se nepodařilo změřit ({e}) – injekce jako gaussián.")
        vel = []
        if wcs_ref is not None:
            try:
                pre_known = _predict_known(settings, sp, MpcStore(settings.catalogs_dir), wcs_ref, w, h,
                                           mjd[ref_i], lat, lon, alt, ctx)
                if pre_known:
                    vel = [_known_pixel_velocity(wcs_ref, k) for k in pre_known
                           if k.kind == "asteroid" and (k.mag is None or k.mag <= sp.forced_mag_limit + 1.5)]
            except Exception as e:      # noqa: BLE001 – identification later reports problems itself
                ctx.log(f"Rychlosti známých objektů pro injekce nejsou k dispozici ({e}).")
                pre_known = None
        injections = INJ.plan(int(sp.inject_n), (h, w), dts, fwhm, tn, rmin_px, rmax_px, sp.inject_snr_min,
                              sp.inject_snr_max, scale, seed=int(abs(mjd[ref_i]) * 1e4) % (2 ** 31),
                              border=int(4 * fwhm) + 24, velocities=vel if len(vel) >= 5 else None,
                              gain=INJ.psf_gain(psf, fwhm), psf=psf,
                              comet_fraction=float(getattr(sp, "inject_comet_fraction", 0.0) or 0.0))
        INJ.add_to_frames(reg_mm, injections, dts, fwhm, maps, psf=psf)
        reg_mm.flush()
        n_field = sum(o.get("velocity") == "field" for o in injections)
        ctx.log(f"Injekce: do snímků přidáno {len(injections)} syntetických objektů (nominální SNR "
                f"{sp.inject_snr_min:g}–{sp.inject_snr_max:g}, {rmin_px * scale:.2f}–{rmax_px * scale:.2f}″/min; "
                f"profil {'hvězd ze snímku' if psf is not None else 'gaussián'}; {n_field} s pohybem podle "
                f"{len(vel)} známých planetek pole; {sum(o.get('kind') == 'comet' for o in injections)} komet "
                f"s komou 4–120″). Nalezené se v seznamu kandidátů neukazují (filtr „injekce“).")

    tw_injections: list[dict] = []
    if search_mode == "twilight" and int(getattr(sp, "twilight_inject_n", 0) or 0) > 0:
        psf_s = None
        try:
            psf_s = INJ.empirical_psf(np.array(reg_mm[ref_i]), stars[ref_i], fwhm)
        except Exception:               # noqa: BLE001 – Gaussian then
            psf_s = None
        tw_injections = INJ.plan_static(int(sp.twilight_inject_n), (h, w), n, fwhm,
                                        float(sp.twilight_inject_snr_min), float(sp.twilight_inject_snr_max), scale,
                                        seed=int(abs(mjd[ref_i]) * 1e4) % (2 ** 31) + 7,
                                        border=int(40 + 3 * 12 + 4 * fwhm + 10), psf=psf_s,
                                        comet_fraction=float(getattr(sp, "twilight_inject_comet_fraction", 0.5)))
        INJ.add_to_frames(reg_mm, tw_injections, dts, fwhm, maps, psf=psf_s)
        reg_mm.flush()
        ctx.log(f"Injekce pro „zdroj navíc“: do snímků přidáno {len(tw_injections)} statických zdrojů – "
                f"{sum(o['kind'] == 'point' for o in tw_injections)} bodových (nova, kometa v outburstu), "
                f"{sum(o['kind'] == 'comet' for o in tw_injections)} komet s komou 10–120″; nominální SNR "
                f"{sp.twilight_inject_snr_min:g}–{sp.twilight_inject_snr_max:g}. Nalezené jsou pod filtrem „injekce“.")

    # median throughput of each frame over the field: tells a covered spot from field-wide dimming
    frame_tp_med = (np.median(maps.grid.reshape(maps.grid.shape[0], -1), axis=1)
                    if maps is not None else None)

    near_m = max(0.0, float(getattr(sp, "near_threshold_margin", 0.0)))

    def tile_one(ti):
        y0, x0 = tiles[ti]
        Y0, X0 = max(0, y0 - margin), max(0, x0 - margin)
        Y1, X1 = min(h, y0 + tile + margin), min(w, x0 + tile + margin)
        block = np.array(reg_mm[:, Y0:Y1, X0:X1])
        if np.isfinite(block).sum() < 0.2 * block.size:
            return []
        # the PSF of wide-field optics changes over the field: FWHM per tile
        med_t = SR.temporal_stats(block)[0]          # also the static sky of the "motion layer" preview
        fw_t = T.estimate_fwhm(np.nan_to_num(med_t))
        if not (0.6 * fwhm < fw_t < 1.8 * fwhm):
            fw_t = fwhm
        amp_t = maps.tile(Y0, Y1, X0, X1) if maps is not None else None
        pt = SR.prepare_tile(block, fw_t, sp.bright_star_mask_sigma, cap=sp.frame_cap_sigma,
                             amp=amp_t, amp_min=sp.throughput_min,
                             chip_shift=_chip_shift(regs, 0.5 * (X0 + X1), 0.5 * (Y0 + Y1), sp))
        # optimal sum: every frame weighted by its throughput in this tile (pt.f / f_raw are
        # multiplied, TrackNorm knows the weights); uniform throughput → weights ≡ 1
        w_t = _frame_weights(amp_t, sp)
        if w_t is not None:
            pt.f *= w_t[:, None, None]
            if pt.f_raw is not pt.f:                  # cap = 0 → the same array
                pt.f_raw *= w_t[:, None, None]
        tn_t = SR.TrackNorm(dts, fw_t, pt.noise_sigma, weights=w_t, amps=w_t)
        core = (slice(y0 - Y0, min(y0 + tile, h) - Y0), slice(x0 - X0, min(x0 + tile, w) - X0))
        edge = int(2 * fwhm) + 2
        size = int(max(48, min(96, 14 * fwhm)))
        HARD = {"nekonzistentní segmenty", "jednosnímkový artefakt", "reziduum hvězdy", "mimo pole v části snímků",
                "nekonzistentní snímky", "bez směru pohybu", "pohyb s driftem čipu"}

        def detect(dts, tn_t):
            """Search + peaks + velocity refinement + basic vetting for one time assignment."""
            scales = np.ones(len(grid)) if getattr(sp, "velocity_noise_norm", True) else None
            with gpu_lock:                  # one GPU search at a time, CPU preparation overlaps
                best, bidx = SR.search_tile(be, pt.f, dts, grid, tn_t, should_stop=ctx.should_stop,
                                            scales=scales, noise_mask=_sky_mask(pt, fw_t))
            # coarse grid loses up to ~1σ: collect peaks 1σ below the threshold, refine, then cut
            cc = SR.peaks(best, bidx, grid, sp.snr_threshold - 1.0, fw_t, core, sp.max_candidates)
            del best, bidx
            cc = [c for c in cc if edge <= c.x + X0 < w - edge and edge <= c.y + Y0 < h - edge]

            def scale_of(c):
                if scales is None:
                    return 1.0
                k = int(np.argmin((grid.vx - c.vx) ** 2 + (grid.vy - c.vy) ** 2))
                return float(scales[k])

            SR.refine(cc, pt.f, dts, grid.dv, tn_t, min_speed=rmin_px, max_speed=rmax_px * 1.1)
            for c in cc:
                c.noise_scale = round(scale_of(c), 3)
                c.snr /= c.noise_scale
            cc = [c for c in cc if c.snr >= sp.snr_threshold - near_m]
            # velocity refinement on the un-clipped frames: clipping flattens bright objects
            SR.refine(cc, pt.f_raw, dts, grid.dv / 2, tn_t, min_speed=rmin_px, max_speed=rmax_px * 1.1)
            SR.vet(cc, pt, dts, tn_t, sp.segments, sp.segment_z_max, sp.bright_star_mask_sigma, fw_t)
            for c in cc:
                # numbers in units of the empirical noise at this velocity (ratios inside vet unchanged)
                c.noise_scale = round(scale_of(c), 3)
                if c.noise_scale > 1.0:
                    c.snr /= c.noise_scale
                    if c.snr_raw is not None:
                        c.snr_raw /= c.noise_scale
                    if c.seg_snr:
                        c.seg_snr = [round(v / c.noise_scale, 2) for v in c.seg_snr]
            return cc

        cshift = _chip_shift(regs, 0.5 * (X0 + X1), 0.5 * (Y0 + Y1), sp, force=True)

        def vet_full(cc, dts, tn_t):
            """All automatic checks of the detections (no images, tile coordinates)."""
            for c in cc:
                c.fwhm = round(fw_t, 2)
                chip_ov = 0.0
                if cshift is not None:
                    c.chip_ratio, chip_ov = SR.chip_path(pt.f_raw, dts, c, cshift, fw_t)
                # a slow, drift-like dithering: a warm pixel / chip pattern moves over the sky like an
                # object. Where the best detector-fixed path stays within 1 FWHM of the linear track in
                # ≥ 60 % of the frames and carries ≥ 0.8 × its signal, the two cannot be told apart –
                # rejected (11P/Coihueco 8. 11.: R run 8 090 of 11 582 passing unknowns, none of 21 known
                # objects and 82 injections; V run tile 758 of 823). With a random dither the paths
                # coincide in a few frames only and the rule does not apply. Inherent: an object moving
                # exactly with the drift is lost – dither more (randomly) to avoid that.
                if (c.chip_ratio is not None and c.chip_ratio >= sp.drift_chip_ratio
                        and chip_ov >= sp.drift_chip_overlap):
                    c.flags.append("pohyb s driftem čipu")
                    c.auto_status = "auto_rejected"
                # "reziduum hvězdy" (slow track over a bright star) is decided by the mover test:
                # a real asteroid passing a star (Adelinda, 305/field_01) is clearly a mover
                if "reziduum hvězdy" in c.flags and not (HARD - {"reziduum hvězdy"}) & set(c.flags):
                    tsv0 = SR.static_vs_mover(block, dts, c, fw_t)
                    if tsv0 is not None:
                        c.mover_vs_static = round(tsv0, 3)
                    if tsv0 is not None and tsv0 >= sp.star_rescue_mover:
                        c.flags.remove("reziduum hvězdy")
                        c.auto_status = "candidate"
                    elif _star_beside_track(c, fw_t, sp):
                        c.flags.remove("reziduum hvězdy")
                        c.flags.append("hvězda vedle stopy")
                        c.auto_status = "candidate"
                # astrometry: PSF fit in sub-stacks along the track -> positions, rate, direction
                fitted = False
                if c.auto_status == "candidate" or c.snr >= sp.snr_threshold + 2:
                    pts = SR.measure_substacks(block, dts, c, sp.segments, fw_t, amp=amp_t)
                    mot = SR.fit_motion(pts)
                    if mot is not None and all(p[4] for p in pts):
                        x0f, y0f, vxf, vyf = mot
                        if math.hypot(x0f - c.x, y0f - c.y) < 2.0 and math.hypot(vxf, vyf) <= rmax_px * 1.3:
                            # the fitted track must carry the signal at least as well as the search one
                            # (a fit pulled away by a neighbour or a residual loses it)
                            s_old = SR.track_snr(pt.f_raw, dts, c.x, c.y, c.vx, c.vy, tn_t)
                            s_new = SR.track_snr(pt.f_raw, dts, x0f, y0f, vxf, vyf, tn_t)
                            if s_new >= 0.8 * s_old:
                                moved = (math.hypot(vxf - c.vx, vyf - c.vy) * float(dts[-1] - dts[0])
                                         > fw_t)
                                c.x, c.y, c.vx, c.vy = x0f, y0f, vxf, vyf
                                fitted = True
                                if moved:
                                    _revet(c, pt, dts, tn_t, sp, fw_t, HARD)
                            else:
                                c.flags.append("astrometrie nejistá")
                    c.positions = [{"dt_min": round(p[0], 4), "x": p[1] + X0, "y": p[2] + Y0, "flux": round(p[3], 1),
                                    "ok": p[4]} for p in pts]
                    _substack_rescue(c, HARD, dts, sp.segments, amp_t, fw_t)
                # the search only tries speeds ≥ rmin_px; a PSF fit that finds (almost) no motion
                # means a static source whose flux or shape changes (variable star, star
                # residual from seeing/obstruction) – the sum along a slow track picks it up
                if fitted and math.hypot(c.vx, c.vy) < sp.static_rate_factor * rmin_px:
                    c.flags.append("téměř statický zdroj")
                    c.auto_status = "auto_rejected"
                if amp_t is not None:
                    tv = [float(amp_t[i, yi, xi]) for i, t in enumerate(dts)
                          for xi, yi in [(int(round(c.x + c.vx * t)), int(round(c.y + c.vy * t)))]
                          if 0 <= yi < amp_t.shape[1] and 0 <= xi < amp_t.shape[2]]
                    if tv:
                        c.throughput_min = round(min(tv), 3)
                        # share of frames that are dim here – one bad frame (twilight, a cloud)
                        # must not mark the whole field
                        c.throughput_low_frac = round(float(np.mean(np.array(tv) < 0.6)), 3)
                        # "zakryté místo" means the spot is covered (dome, horizon, a cloud) – dimmer
                        # than the rest of the field in the same frame. Field-wide dimming (twilight,
                        # uniform haze) is already in the frame weights and TrackNorm: Coihueco 317
                        # (12. 11.) dawn, frames to 32 %, every detection SNR < 10 got the flag –
                        # 2 of 100 injections passed, (382) Dodona (χ² 0.72) rejected.
                        if frame_tp_med is not None and len(tv) == len(frame_tp_med):
                            tva = np.array(tv)
                            c.throughput_local_frac = round(float(np.mean(
                                (tva < 0.6) & (tva < sp.cover_local_ratio * frame_tp_med))), 3)
                        if float(np.mean(np.array(tv) < 0.7)) >= 0.25:
                            c.flags.append("snížená propustnost")
                # star residuals in dense fields (Milky Way): a slow track (≲ 1.2 FWHM over the whole
                # sequence) right on a star much brighter than the object itself. The star-free
                # astrometry cannot tell these apart (its template follows the track), so decide
                # from the static sky: measured on CTA-S1 fields 4/5 (2883 false / 26 real) this
                # removes ~60 % of the false candidates and none of the real ones.
                motion_fw = math.hypot(c.vx, c.vy) * float(dts[-1] - dts[0]) / max(fw_t, 0.5)
                fl = [p["flux"] for p in c.positions if p.get("ok") and p.get("flux")]
                own_peak = (float(np.median(fl)) / (2 * math.pi * (fw_t / 2.3548) ** 2)) if fl else None
                if own_peak is not None:
                    t_mid = 0.5 * float(dts[0] + dts[-1])
                    mxi, myi = int(round(c.x + c.vx * t_mid)), int(round(c.y + c.vy * t_mid))
                    rr = max(2, int(round(1.5 * fw_t)))
                    win = pt.static[max(0, myi - rr):myi + rr + 1, max(0, mxi - rr):mxi + rr + 1]
                    if win.size:
                        c.star_ratio = round(float(np.nanmax(win)) / max(own_peak, 0.05), 2)
                        _, star_offset = _static_peak(pt.static, mxi, myi, rr)
                        # A very slow bright asteroid contributes to the temporal
                        # median itself.  If the static maximum is centred on the
                        # candidate it may be the object, not a neighbouring star;
                        # let the mover-vs-static and independent checks decide.
                        external_star = star_offset is not None and star_offset >= max(1.0, 0.6 * fw_t)
                        if (motion_fw < sp.slow_star_motion_fwhm and c.star_ratio > sp.slow_star_ratio
                                and external_star):
                            # A strong detection present in every third of the series is a real
                            # slow object beside the star, not the star's residual: (372) Palma,
                            # CTA-S1 320/4 (SNR 14.4, thirds 5.7/2.2/6.3, median/mean 0.96) was
                            # lost. Over the last 40 jobs this kept 5 real (Palma + 4 injections)
                            # and no other detection rejected only for this reason (33 real lost).
                            seg = c.seg_snr or []
                            exp_s = c.snr / math.sqrt(len(seg)) if seg else 0.0
                            consistent = (c.snr >= 8.0 and seg and exp_s > 0 and min(seg) / exp_s >= 0.2
                                          and (c.median_ratio or 0.0) >= 0.6)
                            if consistent and c.auto_status == "candidate":
                                c.flags.append("pomalý u hvězdy?")
                            else:
                                c.flags.append("pomalý u hvězdy")
                                c.auto_status = "auto_rejected"
                # "reziduum hvězdy" where the static template peak is the slow object itself:
                # centred on the track and not much brighter than the object (a real star
                # residual sits on a star ~9× brighter, measured on CTA-S1 304). 305/pole 1:
                # Tatjana, Carmen, Olympia (ratio 1.6–2.5, offset < 0.5 px) were lost.
                # Also a neighbouring star of similar brightness (Tatjana: 2.5×, 3 px) is not
                # the case the rule was made for; there the mover-vs-static test decides.
                if ("reziduum hvězdy" in c.flags
                        and (c.star_ratio is None or c.star_ratio < sp.slow_star_ratio)
                        and not (HARD - {"reziduum hvězdy"}) & set(c.flags)
                        and "pomalý u hvězdy" not in c.flags and "astrometrie nejistá" not in c.flags):
                    t_mid = 0.5 * float(dts[0] + dts[-1])
                    _, off = _static_peak(pt.static, c.x + c.vx * t_mid, c.y + c.vy * t_mid,
                                          max(2, int(round(1.5 * fw_t))))
                    self_like = off is not None and off < max(1.0, 0.6 * fw_t)
                    tsv1 = SR.static_vs_mover(block, dts, c, fw_t)
                    if tsv1 is not None:
                        c.mover_vs_static = round(tsv1, 3)
                    if (tsv1 is not None and tsv1 >= sp.static_var_warn
                            and (c.star_ratio is not None or self_like)):
                        c.flags = [f for f in c.flags if f not in ("reziduum hvězdy", "u hvězdy")]
                        c.flags.append("objekt ve statické šabloně" if self_like else "hvězda podobné jasnosti u stopy")
                        c.auto_status = "candidate"
                if "astrometrie nejistá" in c.flags:
                    c.auto_status = "auto_rejected"        # fitted track loses the signal: not a coherent mover
                cover_frac = (c.throughput_local_frac if c.throughput_local_frac is not None
                              else c.throughput_low_frac)
                if cover_frac is not None and cover_frac >= 0.25 and c.snr < 10:
                    c.flags.append("zakryté místo")
                    c.auto_status = "auto_rejected"
                # width of the object vs. the stars: hot/warm pixels that follow a drift of the
                # pointing, cosmic remnants and resampling noise are 1–2 px wide, a real object at
                # least as wide as the stars. Forward-modelled, so slow objects partly lost in the
                # static template are not mistaken for sharp ones; for undersampled stars the two
                # models hardly differ and Δχ² stays small (simulation: 0 of 464 real objects at
                # FWHM 2.0–2.8 px flagged; 600 mm / IMX533 !Hunting/308: 57 of 77 false removed).
                if c.auto_status == "candidate" and sp.sharp_ratio > 0:
                    shp = SR.sharpness(block, dts, c, fw_t, amp=amp_t)
                    if shp is not None:
                        c.obj_fwhm, c.sharp_dchi2 = round(shp[0], 2), round(shp[1], 1)
                        if shp[0] < sp.sharp_ratio * fw_t:
                            # A bright track whose signal is NOT on the detector-fixed path cannot be
                            # a hot/warm pixel; at high SNR Δχ² grows huge for any small width bias of
                            # undersampled stars ((51) Nemausa, CTA-S1 315/5: SNR 53, width 0.49 × FWHM,
                            # chip_ratio 0.19). Over all jobs the only other "sharp" detection with
                            # SNR ≥ 15 was a warm pixel with chip_ratio 12.9.
                            bright_mover = (c.snr >= sp.sharp_keep_snr and c.chip_ratio is not None
                                            and c.chip_ratio < sp.sharp_keep_chip)
                            if shp[1] >= sp.sharp_dchi2 and not bright_mover:
                                c.flags.append("ostřejší než hvězdy")
                                c.auto_status = "auto_rejected"
                            elif shp[1] >= sp.sharp_warn_dchi2:
                                c.flags.append("užší než hvězdy?")
                if c.auto_status == "candidate":
                    tsv = SR.static_vs_mover(block, dts, c, fw_t)
                    if tsv is not None:
                        c.mover_vs_static = round(tsv, 3)
                        if tsv < sp.static_var_reject:
                            c.flags.append("proměnná hvězda")
                            c.auto_status = "auto_rejected"
                        elif tsv < sp.static_var_warn:
                            c.flags.append("spíš proměnná hvězda")
                if c.auto_status == "candidate":
                    chk = SR.independent_check(block, dts, c, fw_t)
                    if chk is not None:
                        c.snr_check, c.frames_with_signal = round(chk[0], 2), round(chk[1], 2)
                        # the median template also removes part of a slow mover: compare with what a
                        # real object of this speed keeps (efficiency 0.3 at 0.7 FWHM … 0.85 at 4.5 FWHM)
                        eff = max(tn_t.efficiency(math.hypot(c.vx, c.vy)), 0.2)
                        if chk[0] / eff < sp.check_snr_min or chk[0] < sp.check_snr_frac * c.snr * eff:
                            c.flags.append("nepotvrzeno nezávislým měřením")
                            c.auto_status = "auto_rejected"
                        elif eff >= 0.6 and chk[1] < sp.check_frames_min:
                            c.flags.append("signál jen v části snímků")
                            c.auto_status = "auto_rejected"

        cc = detect(dts, tn_t)
        # empirical noise level of this tile: the same search and checks on the frames with
        # their times shuffled (a mover is scrambled, noise and star residuals stay)
        thr_t, ninfo = sp.snr_threshold, None
        if getattr(sp, "null_permutations", 0) > 0 and cc:
            area_mpx = (core[0].stop - core[0].start) * (core[1].stop - core[1].start) / 1e6
            thr_t, ninfo = SR.noise_threshold(detect, vet_full, dts, fw_t, pt.noise_sigma, sp, area_mpx,
                                              seed=1000 + ti, weights=w_t)
            tile_noise[ti] = ninfo
        # "druhá šance": a band just below the threshold gets the full vetting; what passes every
        # check is kept (listed apart from the candidates at the end), the rest is dropped.
        # Zeuxo/Petrina (322, CTA-N 15. 11.): SNR 6.0 ± 0.1 on the track, lost by hundredths
        lo_t = thr_t - near_m if thr_t <= sp.snr_threshold + 1e-6 else thr_t
        below = [c for c in cc if c.snr < lo_t]
        near = [c for c in cc if lo_t <= c.snr < thr_t]
        cc = [c for c in cc if c.snr >= thr_t]
        vet_full(cc + near, dts, tn_t)
        near = [c for c in near if c.auto_status == "candidate"]
        n_near_t = len(near)
        cc += near
        for c in below:                   # numeric record kept, no checks and no previews (speed)
            c.fwhm = round(fw_t, 2)
            c.flags.append("pod prahem šumu dlaždice")
            c.auto_status = "auto_rejected"
            c.preview_skipped = f"SNR pod prahem šumu této dlaždice ({thr_t:.2f})"
            if c.static_star_x is not None:
                c.static_star_x += X0
                c.static_star_y += Y0
            c.x += X0
            c.y += Y0
        resid_t = (block - med_t[None]).astype(np.float32) if cc else None
        for k, c in enumerate(cc):
            # previews for every detection (also the rejected ones – the user must
            # be able to check every automatic decision)
            # the same per-frame clipped frames the search sums (±frame_cap_sigma): the map shows what
            # the detection was made of – an unclipped satellite trail looks far stronger than its
            # real share (one frame ≤ cap/√n of the SNR)
            snr_map = (track_crops(pt.f, dts, c.x, c.y, c.vx, c.vy, size).sum(axis=0)
                       / tn_t.sigma(math.hypot(c.vx, c.vy)))
            c.images = write_candidate_images(img_dir, f"t{ti:03d}_{k:04d}",
                                              track_crops(block, dts, c.x, c.y, c.vx, c.vy, size),
                                              sp.segments, diff_crops=snr_map, fwhm=fw_t,
                                              motion_crops=track_crops(resid_t, dts, c.x, c.y,
                                                                       c.vx, c.vy, size))
            # the GUI animates the frame strip; the GIF is only a fallback (~40 % of the
            # preview time per detection – dense fields have 10 000+ detections)
            try:
                c.images["frames"] = write_motion_frames(img_dir, f"t{ti:03d}_{k:04d}", block, dts,
                                                         c.x, c.y, c.vx, c.vy, size, amp=amp_t)
            except Exception:      # noqa: BLE001 – fall back to the GIF
                c.images["motion"] = write_motion_gif(img_dir, f"t{ti:03d}_{k:04d}", block, dts,
                                                      c.x, c.y, c.vx, c.vy, size, fwhm=fw_t)
            if c.static_star_x is not None:
                c.static_star_x += X0
                c.static_star_y += Y0
            c.x += X0
            c.y += Y0
        note = (f" · práh šumu {thr_t:.2f} ({len(below)} pod prahem, odhad falešných nad {sp.snr_threshold:g}: {ninfo['expected_at_base']:.1f})" if ninfo else "")
        ctx.log(f"  dlaždice {ti + 1}/{len(tiles)}: {len(cc) + len(below) - n_near_t} detekcí, "
                f"{sum(c.auto_status == 'candidate' for c in cc) - n_near_t} prošlo"
                f"{f' (+{n_near_t} těsně pod prahem)' if n_near_t else ''}{note}{_mem_note()}")
        return cc + below

    import threading
    gpu_lock = threading.Lock() if be.is_gpu else _NoLock()
    # prepare_tile copies + the shuffled-time search and vetting of the noise estimate
    per_tile_mb = (14 if maps is not None else 13) * n * (tile + 2 * margin) ** 2 * 4 / 2**20
    tile_noise: list = [None] * len(tiles)
    n_tile_workers = _workers(settings, per_task_mb=per_tile_mb)
    ctx.log(f"Dlaždice zpracovává {n_tile_workers} vláken (odhad {per_tile_mb / 1024:.1f} GB na dlaždici){_mem_note()}.")
    if search_mode == "twilight":
        results = []
        ctx.log("Synthetic tracking se v režimu „zdroj navíc“ nespouští.")
    else:
        results = _parallel(tile_one, range(len(tiles)), n_tile_workers, ctx, 0.45, 0.4)
    cands: list[T.Candidate] = [c for r in results for c in (r or [])]
    for i, c in enumerate(cands):
        c.id = i
    cands = _dedupe_keep_images(cands, fwhm, span_min)
    n_near = sum(1 for c in cands if c.snr < sp.snr_threshold and c.auto_status == "candidate")
    ctx.log(f"Celkem {len(cands) - n_near} detekcí nad SNR {sp.snr_threshold}; "
            f"{sum(c.auto_status == 'candidate' for c in cands) - n_near} prošlo automatickým prověřením"
            + (f"; navíc {n_near} těsně pod prahem (SNR {sp.snr_threshold - near_m:g}–{sp.snr_threshold:g}) "
               f"prošlo všemi testy" if n_near else ""))
    tick("search")
    known_rejected: list = []
    wcs_shift = None
    inj_summary = None
    if injections:
        INJ.match(cands, injections, dts, fwhm)
        n_sharp = _sharp_rescue(cands, sp)
        if n_sharp:
            INJ.match(cands, injections, dts, fwhm)
            ctx.log(f"Šířka objektů kalibrovaná injekcemi (profil hvězd z pole): {n_sharp} detekcí "
                    f"zamítnutých jen jako „ostřejší než hvězdy“ je stejně široké jako injekce – vráceno "
                    f"s varováním „užší než hvězdy?“ (podvzorkované hvězdy, FWHM {fwhm:.1f} px).")
        inj_summary, lines = INJ.summarise(injections, sp.snr_threshold)
        for line in lines:
            ctx.log(line)

    # ------------------------------------------------------------ candidates
    if wcs_ref is not None:
        _sky_coords(cands, wcs_ref, mjd[ref_i])
    else:
        for c in cands:
            c.rate_arcsec_min = math.hypot(c.vx, c.vy) * scale
            c.pa_deg = (math.degrees(math.atan2(-c.vx, -c.vy))) % 360

    # -------------------------------------------------------------- identify
    ctx.stage("Identifikace známých objektů", 0.88)
    known_list = []
    mstore = MpcStore(settings.catalogs_dir)
    catalog_versions = {kind: mstore.status(kind).to_dict() for kind in (MPCORB_NAME, COMETS_NAME)}
    if wcs_ref is not None:
        if pre_known is not None:
            known_list = pre_known
        else:
            known_list = _predict_known(settings, sp, mstore, wcs_ref, w, h, mjd[ref_i], lat, lon, alt, ctx)
        if known_list is None:
            ctx.warn("MPCORB ani CometEls nejsou stažené – identifikace přeskočena.")
            known_list = []
        else:
            if wcs_source == "pointing":
                # mount-position WCS: measure its zero point on bright known objects, shift it
                # (CRPIX) and recompute everything that depends on it
                off = _pointing_offset(cands, known_list, scale, span_min, sp.pointing_wcs_search_arcsec)
                if off is not None:
                    odx, ody, names = off
                    wcs_ref = wcs_ref.deepcopy()
                    wcs_ref.wcs.crpix = wcs_ref.wcs.crpix + np.array([odx, ody])
                    wcs_ref.wcs.set()
                    wcs_source = "pointing+known"
                    _sky_coords(cands, wcs_ref, mjd[ref_i])
                    for k in known_list:
                        k.x, k.y = (float(v) for v in wcs_ref.world_to_pixel_values(k.ra, k.dec))
                    known_list = [k for k in known_list if 0 <= k.x < w and 0 <= k.y < h]
                    wcs_shift = {"dx_px": round(odx, 2), "dy_px": round(ody, 2),
                                 "arcsec": round(math.hypot(odx, ody) * scale, 1), "objects": names}
                    ctx.warn(f"WCS z polohy montáže posunut o ({odx:+.1f}, {ody:+.1f}) px = "
                             f"{math.hypot(odx, ody) * scale:.0f}″ podle známých objektů ({', '.join(names)}). "
                             "Natočení ani měřítko se nekalibrují – astrometrie není vhodná pro hlášení MPC.")
                else:
                    ctx.warn(f"WCS z polohy montáže se nepodařilo dokalibrovat (žádný jasný známý objekt "
                             f"jednoznačně spárovaný do {sp.pointing_wcs_search_arcsec:.0f}″) – identifikace "
                             "může známé objekty minout.")
            # wide-field: the tolerance must cover ~2.5 px of astrometric error
            # A faint/slow mover has only a ~2 px trail over this sequence and
            # the three-substack astrometric fit can move either endpoint by
            # about one pixel.  Use the resulting 2 px velocity uncertainty;
            # the full vector is still compared, so an opposite direction is
            # never accepted merely because the speeds are equal.
            # Slow objects (≲ 1 FWHM over the series) partly sit in their own static template and
            # the sub-stack fit overestimates their rate (~2×: (1558) Jarnefelt 0.38 vs 0.18″/min,
            # (2218) Wotho 0.50 vs 0.27″/min, Coihueco 8. 11., both right on the predicted position);
            # the velocity tolerance is therefore at least 1.2 FWHM over the series.
            match_candidates(cands, known_list, max(sp.identify_tolerance_arcsec, 2.5 * scale),
                             rate_tol_abs=max(0.05, max(2.0, 1.2 * fwhm) * scale / max(span_min, 1.0)),
                             comet_tol_arcsec=sp.comet_tolerance_arcsec)
            n_bad = _unmatch_implausible(cands, known_list)
            if n_bad:
                ctx.log(f"Identifikace: {n_bad} shod se známým objektem zrušeno – detekce je o víc než "
                        f"2,5 mag jasnější, než odpovídá předpovědi a ostatním identifikovaným objektům "
                        f"(jiný zdroj na tom místě, např. reziduum hvězdy).")
            # forced measurement on the predicted track of every known object
            todo = [k for k in known_list if k.mag is None or k.mag <= sp.forced_mag_limit]

            size_k = int(max(48, min(96, 14 * fwhm)))

            def forced_one(j):
                k = todo[j]
                vk = _known_pixel_velocity(wcs_ref, k)
                k.forced_snr, k.forced_dxy, k.forced_note, k.forced_noise, k.saturated = _forced(
                    reg_mm, k.x, k.y, vk, dts, fwhm, tn, sp, maps=maps, regs=regs)
                if k.forced_noise is not None:
                    k.forced_noise = round(float(k.forced_noise), 3)
                # frames along the predicted track for the "Známé objekty" preview (stacks and the
                # animation are built in the GUI on request; this strip is all that is stored)
                try:
                    k.images = {"frames": write_motion_frames(img_dir, f"k{j:04d}", reg_mm, dts,
                                                              k.x, k.y, vk[0], vk[1], size_k)}
                except Exception:          # noqa: BLE001 – a viewer aid, never fail the job
                    k.images = None
            _parallel(forced_one, range(len(todo)), _workers(settings, per_task_mb=200), None, 0, 0)
            ctx.log(f"Známých objektů v poli: {len(known_list)}; spárováno s detekcemi: "
                    f"{sum(1 for k in known_list if k.matched_candidate is not None)}")
            # safety net: known objects are the built-in test of the vetting rules – a rule that
            # rejects a known asteroid would reject a new comet just the same
            lost = [c for c in cands if c.known and c.auto_status != "candidate" and c.injected is None]
            if lost:
                by_flag = Counter(f for c in lost for f in c.flags)
                ctx.warn(f"Prověřování zamítlo {len(lost)} identifikovaných známých objektů: "
                         + ", ".join(f"{c.known['name']} (SNR {c.snr:.1f}: {', '.join(c.flags) or '–'})" for c in lost[:8])
                         + ("…" if len(lost) > 8 else "")
                         + f". Stejně by dopadly i neznámé objekty – zkontrolujte zamítnuté kandidáty "
                           f"(nejčastější důvod: {by_flag.most_common(1)[0][0]}).")
            known_rejected = [{"name": c.known["name"], "snr": round(c.snr, 1), "flags": c.flags} for c in lost]
    cat_zp = None
    u4root = None
    cat_stars: dict = {}
    if wcs_ref is not None and getattr(settings, "ucac4_dir", ""):
        from . import ucac4 as U4
        u4root = U4.find_root(settings.ucac4_dir)
        if u4root is None:
            ctx.warn(f"UCAC4: ve složce {settings.ucac4_dir} nejsou zónové soubory u4b/z001… – fotometrie "
                     "z katalogu přeskočena.")
        else:
            try:
                cat_zp = _zp_ucac4(u4root, reg_mm, wcs_ref, float(mjd[ref_i]), fwhm, key[1], stars_out=cat_stars)
                if cat_zp is None or len(cat_zp[0]) < 10:
                    ctx.log("UCAC4: v poli je málo izolovaných nepřesycených hvězd s APASS – nulový bod "
                            "z katalogu nelze určit.")
                    cat_zp = None
            except Exception as e:          # noqa: BLE001 – photometry must never stop a job
                ctx.warn(f"UCAC4: čtení katalogu selhalo ({e}) – fotometrie z katalogu přeskočena.")
                cat_zp = None
    photometry = (_photometry(cands, known_list, fwhm, span_min, catalog=cat_zp)
                  if (known_list or cat_zp) else None)
    if photometry is not None and photometry["source"] == "ucac4":
        ctx.log(f"Fotometrie (přibližná): nulový bod z {photometry['n']} hvězd UCAC4 (APASS {photometry['band']}, "
                f"filtr snímků {key[1]}), rozptyl ±{photometry['scatter']:.2f} mag"
                + (f"; kontrola na {photometry['known_n']} známých planetkách: změřeno − předpověď (V) = "
                   f"{photometry['known_offset']:+.2f} mag"
                   + ("" if photometry["band"] == "V" else f" (obsahuje barvu planetek V − {photometry['band']})")
                   if photometry.get("known_offset") is not None else "")
                + ". Pomalé objekty (< 1 FWHM za sérii) vycházejí slabší.")
    elif photometry is not None:
        ctx.log(f"Fotometrie (přibližná): nulový bod z {photometry['n']} identifikovaných planetek "
                f"(předpověď MPC, V), rozptyl ±{photometry['scatter']:.2f} mag – jasnost všech detekcí "
                f"s PSF fitem v pásmu snímků (filtr {key[1]}) převedena na škálu V. Pomalé objekty "
                f"(< 1 FWHM za sérii) vycházejí slabší (část světla ve statické šabloně)."
                + ("" if u4root else " Přesnější: nastavte složku UCAC4 v Data → Nastavení."))
    if cat_stars:
        n_free = _catalog_no_star_rescue(cands, cat_stars, fwhm, sp, span_min)
        if n_free:
            ctx.log(f"Katalog UCAC4: {n_free} detekcí zamítnutých jen kvůli „hvězdě“ u stopy leží na místě, kde "
                    f"UCAC4 žádnou hvězdu nemá, ač by ji při té jasnosti obsahoval – vráceno jako „bez hvězdy "
                    f"v UCAC4“ (statické maximum je nejspíš objekt sám, i u pomalého pohybu).")
        probe: list = []
        n_cat = _catalog_stars_reject(cands, cat_stars, fwhm, span_min, probe=probe)
        if probe:
            ctx.warn(f"Pravidlo „hvězda z katalogu UCAC4“ by zamítlo i {len(probe)} skutečných objektů "
                     f"(ponechány): {', '.join(probe[:8])}{'…' if len(probe) > 8 else ''} – stejně by mohlo "
                     f"zamítnout i neznámou planetku těsně u hvězdy.")
        if n_cat:
            ctx.log(f"Katalog UCAC4: {n_cat} detekcí leží přímo na hvězdě katalogu (pomalé: hvězda jasná "
                    f"jako detekce nebo jasnější; rychlejší: stopa přes hvězdu se znaky rezidua) – zamítnuto jako „hvězda z katalogu UCAC4“ (reziduum statické "
                    f"hvězdy, ne planetka).")
    n_slow = _slow_star_second_look(cands)
    if n_slow:
        ctx.log(f"„Pomalý u hvězdy“: {n_slow} silných detekcí viditelných ve všech třetinách série "
                f"vráceno s varováním „pomalý u hvězdy?“ (pravidlo zamítalo i skutečné pomalé planetky u hvězd).")
    n_cross = _crossing_reject(cands, dts, fwhm)
    if n_cross:
        ctx.log(f"Křížení stop: {n_cross} detekcí bere světlo jen z průchodu silnějšího pohybujícího se "
                f"objektu přes svou stopu (jeden snímek, nejsilnější segment právě tam) – zamítnuto.")
    n_sat = _satellite_reject(cands, reg_mm, dts, fwhm, sp)
    if n_sat:
        ctx.log(f"Stopy družic: {n_sat} detekcí přes práh jen díky slabé stopě družice v jednom snímku "
                f"(pod maskou přechodných jevů) – bez toho snímku pod prahem, zamítnuto.")
    tick("identify")
    inj_rank = None
    if injections and getattr(sp, "inject_rank_keep", 0) > 0:
        from . import rank as RK
        inj_rank = RK.calibrate(cands, fwhm, keep=sp.inject_rank_keep,
                                soft=getattr(sp, "inject_rank_keep_soft", 0.97))
        if inj_rank is None:
            ctx.log("Kalibrace injekcemi: málo injekcí nebo kandidátů – přeskočena.")
        else:
            ctx.log(f"Kalibrace injekcemi (ponechat {sp.inject_rank_keep:.0%} injekcí, AUC {inj_rank['cv_auc']}): "
                    f"z {inj_rank['others']} neznámých kandidátů odloženo {inj_rank['set_aside']} "
                    f"(z toho hraničních {inj_rank.get('borderline', 0)}), "
                    f"zbývá {inj_rank['remaining']}; příznaky "
                    + ("všechny." if inj_rank.get("feature_set") == "full" else
                       f"bez šířky objektu a dráhy čipu (s nimi by pod hranicí byly známé: "
                       f"{', '.join(inj_rank.get('full_set_known_below', [])[:5])})."))
            if inj_rank["known_below_cut"]:
                ctx.warn(f"Kalibrace injekcemi by odložila {len(inj_rank['known_below_cut'])} z "
                         f"{inj_rank['known_scored']} identifikovaných známých objektů "
                         f"({', '.join(inj_rank['known_below_cut'][:8])}) – známé se neodkládají, ale stejně "
                         "by dopadly i skutečné neznámé." + (" Hranice proto snížena pod nejslabší z nich." if inj_rank.get("cut_lowered_for_known") else ""))

    # ------------------------------------------------- twilight: source in addition to UCAC4
    tw_dicts, tw_stats = [], None
    if search_mode in ("twilight", "both"):
        ctx.stage("Zdroj navíc – porovnání s UCAC4", 0.93)
        if wcs_ref is None:
            ctx.warn("Zdroj navíc: snímky nemají WCS – porovnání s katalogem nelze udělat.")
        elif u4root is None:
            ctx.warn("Zdroj navíc: chybí katalog UCAC4 (Data → Nastavení → složka UCAC4) – porovnání přeskočeno.")
        else:
            try:
                astap_exe = find_astap(settings.astap.executable) if hasattr(settings, "astap") else None
                astap_dir = Path(astap_exe).parent if astap_exe else None
                if settings.astap.database_dir:
                    astap_dir = Path(settings.astap.database_dir) if (Path(settings.astap.database_dir) / "deep_sky.csv").exists() else astap_dir
                from . import ucac4 as U4b
                tw_dicts, tw_stats = TWL.pipeline_run(
                    reg_mm, dts, wcs_ref, fwhm, u4root, float(mjd[ref_i]), (lat, lon, alt), settings.catalogs_dir,
                    astap_dir, known_list, injections or tw_injections, regs, img_dir, band=U4b.band_for_filter(key[1]), log=ctx.log,
                    dss_online=bool(getattr(settings.online, "sky_images", False))
                               and bool(getattr(settings.online, "twilight_dss_check", True)),
                    pgc_online=bool(getattr(settings.online, "sky_images", False))
                               and bool(getattr(settings.online, "twilight_pgc", True)),
                    sn_inject_n=40 if int(getattr(sp, "twilight_inject_n", 0) or 0) > 0 and search_mode == "twilight" else 0)
                ctx.log(f"Zdroj navíc: {tw_stats.get('detections')} zdrojů v mediánu, {tw_stats.get('matched')} spárováno "
                        f"s UCAC4 (astrometrie {tw_stats.get('astrometry_rms_px')} px, mez ~{tw_stats.get('limit_mag_detected')} mag); "
                        f"kandidátů {sum(1 for d in tw_dicts if d['auto_status'] == 'candidate' and d.get('injected') is None)}, "
                        f"zamítnuto {sum(1 for d in tw_dicts if d['auto_status'] != 'candidate')}.")
            except Exception as e:        # noqa: BLE001 – the side search must never break a job
                import traceback
                ctx.warn(f"Zdroj navíc selhal: {e}")
                ctx.log(traceback.format_exc())
                tw_dicts = []

    if tw_injections:
        inj_summary, lines = INJ.summarise_static(tw_injections, tw_dicts)
        for line in lines:
            ctx.log(line)
        injections = tw_injections

    # --------------------------------------------------------------- outputs
    ctx.stage("Výstupy", 0.96)
    _near_second_chance(cands, injections, sp, ctx)
    cand_dicts = [c.to_dict() for c in cands]
    for d in cand_dicts:
        d["status"] = "unreviewed" if d["auto_status"] == "candidate" else "rejected"
        if d["auto_status"] == "candidate" and d.get("known") and getattr(sp, "auto_accept_known", True):
            d["status"] = "accepted"          # identified known object: nothing to review
            d["auto_accepted"] = True
        if d.get("injected") is not None:
            d["status"] = "injected"
    ml_ctx = {"min_rate": rmin_px * scale, "fwhm": fwhm, "n_frames": n, "span_min": span_min, "pixel_scale": scale}
    model = CL.load(settings.root, inst.id)
    if model is not None:
        CL.score(model, cand_dicts, ml_ctx)
        # rejected only because of a star nearby, but the classifier is confident: give it back
        STAR_ONLY = {"u hvězdy", "reziduum hvězdy", "pomalý u hvězdy", "snížená propustnost"}
        back = 0
        for d in cand_dicts:
            if (d["auto_status"] != "candidate" and d.get("injected") is None and set(d["flags"]) <= STAR_ONLY
                    and (d.get("ml_score") or 0) >= sp.ml_rescue):
                d["auto_status"] = "candidate"
                d["flags"].append("vráceno klasifikátorem")
                d["status"] = "accepted" if d.get("known") and getattr(sp, "auto_accept_known", True) else "unreviewed"
                back += 1
        if back:
            ctx.log(f"Klasifikátor vrátil {back} kandidátů zamítnutých jen kvůli blízké hvězdě (skóre ≥ {100 * sp.ml_rescue:.0f} %).")
            known_rejected = [{"name": d["known"]["name"], "snr": round(d["snr"], 1), "flags": d["flags"]}
                              for d in cand_dicts if d.get("known") and d["auto_status"] != "candidate"
                              and d.get("injected") is None]
        ctx.log(f"Klasifikátor ({'model přístroje ' + inst.id if model.get('instrument') else 'společný model'}, "
                f"naučen {model.get('trained')}, {model.get('n_pos')} skutečných / "
                f"{model.get('n_neg')} zamítnutých): skóre doplněno ke všem detekcím.")
    if tw_dicts and cand_dicts:
        # "both": an object the synthetic tracking already has is not repeated ((43) Ariadne
        # twice in CTA-N 318). Tracking positions are at the reference epoch, the median's at the
        # mean time of the frames: allow for the motion in between.
        t_mean = float(np.mean(dts))
        keep = []
        for d in tw_dicts:
            dup = None
            for c in cand_dicts:
                if c["auto_status"] != "candidate" and not c.get("known") and c.get("injected") is None:
                    continue
                cx, cy = c["x"] + c["vx"] * t_mean, c["y"] + c["vy"] * t_mean
                if math.hypot(cx - d["x"], cy - d["y"]) <= max(3.0, 2.0 * fwhm):
                    dup = c
                    break
            if dup is None:
                keep.append(d)
            else:
                dup.setdefault("flags", []).append("také jako zdroj navíc")
        if len(keep) < len(tw_dicts):
            ctx.log(f"Zdroj navíc: {len(tw_dicts) - len(keep)} objektů už našel synthetic tracking – neopakují se.")
        tw_dicts = keep
    if tw_dicts:
        base_id = max([d["id"] for d in cand_dicts], default=-1) + 1
        for k, d in enumerate(tw_dicts):
            d["id"] = base_id + k
            d["status"] = "unreviewed" if d["auto_status"] == "candidate" else "rejected"
            if d["auto_status"] == "candidate" and d.get("known") and getattr(sp, "auto_accept_known", True):
                d["status"] = "accepted"
                d["auto_accepted"] = True
            if d.get("injected") is not None:
                d["status"] = "injected"
        if tw_injections:                  # candidate ids are known only now
            ids = {d["injected"]: d["id"] for d in tw_dicts if d.get("injected") is not None}
            for o in tw_injections:
                if o.get("detected") is not None:
                    o["detected"] = ids.get(o["id"], o["detected"])
        cand_dicts += tw_dicts
    annotate_confidence(cand_dicts)
    store.write_json(job.id, "candidates.json", cand_dicts)
    if injections:
        store.write_json(job.id, "injections.json", injections)
    store.write_json(job.id, "known.json", [k.to_dict() for k in known_list])
    store.write_json(job.id, "frames.json", {"mjd": mjd.tolist(), "ref_index": ref_i,
                                             "paths": [m.path for m in metas],
                                             "registration": [r.summary() for r in regs]})
    _release_memmap(reg_mm)
    reg_mm = None
    import gc
    gc.collect()
    import os
    if os.environ.get("SYNTHTRACK_KEEP_WORK") or getattr(sp, "keep_work", False):   # diagnostics: keep the registered cube
        (work / "KEEP").write_text("keep_work\n", encoding="utf-8")                  # …also for JobStore.cleanup_work
    else:
        shutil.rmtree(work, ignore_errors=True)
    timings["total"] = round(time.time() - t_start, 2)
    job.summary = {
        "n_frames": n, "exptime": key[0], "filter": key[1], "span_min": round(span_min, 2),
        "fwhm_px": round(fwhm, 2), "pixel_scale": round(scale, 4), "grid_size": len(grid),
        "min_rate_arcsec_min": round(rmin_px * scale, 3),
        "max_rate_arcsec_min": round(rmax_px * scale, 3),
        "backend": be.name, "detections": len(cand_dicts),
        "noise_threshold": _noise_summary(tile_noise, sp),
        "injection_rank": inj_rank,
        "candidates": sum(1 for c in cand_dicts if c["auto_status"] == "candidate" and c.get("injected") is None),
        "known_in_field": len(known_list),
        "known_detected": sum(1 for k in known_list if k.matched_candidate is not None),
        "known_forced_above_threshold": sum(
            1 for k in known_list
            if k.forced_snr is not None and k.forced_snr >= sp.snr_threshold
            and k.forced_note not in {"jasná hvězda na stopě", "mimo pole", "mimo pole v části snímků"}),
        "unknown_candidates": sum(1 for c in cand_dicts if c["auto_status"] == "candidate" and not c.get("known")
                                  and c.get("injected") is None),
        "registration_rms_px": round(float(np.median(rms)), 3) if rms else None,
        "throughput": maps.to_summary() if maps is not None else None,
        "known_rejected": known_rejected,
        "known_saturated": [k.name for k in known_list if k.saturated],
        "photometry": photometry,
        "injection": inj_summary,
        "masters": {"dark": dsel.to_dict() if dsel else None, "flat": fsel.to_dict() if fsel else None},
        "wcs": wcs_ref is not None,
        "wcs_source": wcs_source,
        "wcs_refined": wcs_refined,
        "wcs_shift": wcs_shift,
        "site": [lat, lon, alt], "site_source": site_source, "observatory_code": obs_code,
        "field": job.field or None,
        "mjd_ref": float(mjd[ref_i]),
        "search_mode": search_mode, "search_mode_reason": mode_reason, "twilight": tw_stats,
        "timings_s": timings,
        "run": {"synthtrack_version": __version__, "search": asdict(sp), "instrument": inst.to_dict(),
                "catalogs": catalog_versions,
                "classifier": None if model is None else {k: model.get(k) for k in
                    ("kind", "version", "instrument", "trained", "n_pos", "n_neg")}},
    }
    ctx.log(f"Hotovo za {timings['total']:.1f} s")


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _validate_search_settings(sp: SearchSettings) -> None:
    """Fail early with a readable error instead of obscure numerical failures."""
    if sp.rate_min_arcsec_min < 0 or sp.rate_max_arcsec_min <= 0:
        raise PipelineError("Rozsah rychlostí musí být kladný.")
    if sp.rate_max_arcsec_min <= sp.rate_min_arcsec_min:
        raise PipelineError("Maximální hledaná rychlost musí být větší než minimální.")
    if sp.step_px <= 0:
        raise PipelineError("Krok mřížky rychlostí musí být větší než nula.")
    if sp.snr_threshold <= 0:
        raise PipelineError("Práh SNR musí být větší než nula.")
    if sp.segments < 1:
        raise PipelineError("Počet časových segmentů musí být alespoň 1.")
    if not 0 < sp.min_efficiency <= 1:
        raise PipelineError("Minimální účinnost musí být v intervalu (0, 1].")
    if sp.tile_px < 64 or sp.max_candidates < 1:
        raise PipelineError("Velikost dlaždice musí být alespoň 64 px a limit kandidátů alespoň 1.")


def _local_sky_matrix(wcs, x: float, y: float, ra: float, dec: float) -> list[list[float]]:
    """Local pixel -> tangent-plane matrix for exact north-up preview rendering."""
    sx = wcs.pixel_to_world(x + 1.0, y)
    sy = wcs.pixel_to_world(x, y + 1.0)
    cos_dec = math.cos(math.radians(dec))

    def east(r):
        return ((float(r.ra.deg) - ra + 180.0) % 360.0 - 180.0) * cos_dec

    return [[east(sx), east(sy)],
            [float(sx.dec.deg) - dec, float(sy.dec.deg) - dec]]


def _available_memory_mb() -> float | None:
    """Physically available RAM (Linux / Windows / macOS), None if unknown."""
    import os
    try:
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (ValueError, OSError):
        pass
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 2**20
    return None


def _workers(settings: Settings, per_task_mb: float) -> int:
    """Threads for per-frame / per-tile work: CPU cores, setting, and ≤ 60 % of free RAM."""
    import os
    n = max(1, int(getattr(settings, "cpu_workers", 0) or (os.cpu_count() or 1)))
    n = min(n, 16)
    mem = _available_memory_mb()
    if mem is not None and per_task_mb > 0:
        n = min(n, max(1, int(0.6 * mem / per_task_mb)))
    return n


def _parallel(fn, items, workers: int, ctx, p0: float, dp: float) -> list:
    """Run ``fn`` over ``items`` with a thread pool (NumPy/SciPy release the GIL).

    Results keep the input order; progress and cancellation are reported via ``ctx``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = list(items)
    out = [None] * len(items)
    if workers <= 1 or len(items) <= 1:
        for j, it in enumerate(items):
            out[j] = fn(it)
            if ctx is not None:
                ctx.progress(p0 + dp * (j + 1) / len(items))
                ctx.check()
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): j for j, it in enumerate(items)}
        done = 0
        for fu in as_completed(futs):
            out[futs[fu]] = fu.result()
            done += 1
            if ctx is not None:
                ctx.progress(p0 + dp * done / len(items))
                if ctx.should_stop():
                    for f2 in futs:
                        f2.cancel()
                    ctx.check()
    return out


def _predict_known(settings, sp, mstore, wcs_ref, w, h, mjd_ref, lat, lon, alt, ctx):
    """Known asteroids/comets predicted on the reference frame (pixel x, y set), None without catalogs."""
    mp, cm = mstore.load(MPCORB_NAME), mstore.load(COMETS_NAME)
    if mp is None and cm is None:
        return None
    cen_sky = wcs_ref.pixel_to_world((w - 1) / 2, (h - 1) / 2)
    radius = max(cen_sky.separation(wcs_ref.pixel_to_world(px, py)).deg
                 for px, py in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))) * 1.02
    t_ref = Time(mjd_ref, format="mjd", scale="utc")
    from .ephem import load_ephemeris
    eph = load_ephemeris(settings.ephemeris, [settings.catalogs_dir])
    t_id = time.time()
    known_list = predict_field(mp, cm, t_ref, cen_sky.ra.deg, cen_sky.dec.deg, radius, lat, lon, alt,
                               mag_limit=sp.forced_mag_limit + 1.5, eph=eph, log=ctx.log)
    ctx.log(f"Efemerida: {eph.name}; N-body propagace {time.time() - t_id:.1f} s")
    for k in known_list:
        kx, ky = wcs_ref.world_to_pixel_values(k.ra, k.dec)
        if 0 <= kx < w and 0 <= ky < h:
            k.x, k.y = float(kx), float(ky)
    return [k for k in known_list if k.x is not None]


def _frame_weights(amp_t, sp) -> np.ndarray | None:
    """Per-frame weights of one tile = median throughput there (None = all 1).

    Frames are in unit noise and the throughput map is measured from star fluxes on those
    frames, so A_i is the frame's relative signal-to-noise and w_i = A_i is the optimal
    weight. Frames under ``throughput_min`` over most of the field are dropped earlier."""
    if amp_t is None or not getattr(sp, "frame_weighting", True):
        return None
    w = np.array([float(np.median(a)) for a in amp_t], float)
    w = np.clip(np.nan_to_num(w, nan=1.0), 0.05, 1.5)
    if np.all(np.abs(w - 1.0) < 0.05):
        return None
    return w


def _mem_note() -> str:
    """Free RAM for the job log (a crash without a trace is often memory)."""
    m = _available_memory_mb()
    return f" · volná RAM {m / 1024:.1f} GB" if m is not None else ""


def _noise_summary(tile_noise, sp) -> dict | None:
    """Per-tile empirical thresholds for the summary (None when switched off)."""
    t = [x for x in (tile_noise or []) if x]
    if not t:
        return None
    thr = [x["threshold"] for x in t]
    return {"base": sp.snr_threshold, "min": min(thr), "median": float(np.median(thr)), "max": max(thr),
            "expected_false_base": round(sum(x["expected_at_base"] for x in t), 1),
            "expected_false": round(sum(x["expected_at_threshold"] for x in t), 1),
            "target_per_mpx": getattr(sp, "null_false_per_mpx", None), "tiles": t}


def _known_pixel_velocity(wcs, k) -> tuple[float, float]:
    """Pixel velocity (px/min) of a known object from its rate and position angle."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c0 = SkyCoord(k.ra * u.deg, k.dec * u.deg)
    c1 = c0.directional_offset_by(k.pa_deg * u.deg, (k.rate_arcsec_min * 10) * u.arcsec)
    x0, y0 = wcs.world_to_pixel(c0)
    x1, y1 = wcs.world_to_pixel(c1)
    return (float(x1 - x0) / 10.0, float(y1 - y0) / 10.0)


def _substack_rescue(c, hard, dts, segments: int, amp, fwhm: float) -> None:
    """Overrule the matched-filter consistency tests when the star-free sub-stacks agree.

    χ² and the segment test compare the frames with a matched-filter model that only
    approximates how much of a slow object's light stays in the static template; the
    deficit is largest in the middle of the series. The sub-stacks (local template without
    the object, PSF fit, ``measure_substacks``) do not have that problem. Their fluxes are
    corrected for the throughput of their frames (dawn, clouds) and must agree within 20 %.

    * "nekonzistentní snímky" → warning, for snr_raw ≥ 30 ((419) Aurelia 12.5 mag, 88P/
      Coihueco 9. 11.: χ² 4.5, fluxes 368/359/398);
    * "nekonzistentní segmenty" → warning, for SNR ≥ 10 and motion ≤ 2.5 FWHM over the series
      ((441) Bathilde 14.2 mag, Coihueco 314/1 10. 11., dawn, 13 frames, 1.3 FWHM: segments
      7.6/3.3/8.3, fluxes 42/32/26 at throughput 1.05/0.94/0.63 → 40/34/41)."""
    pos = c.positions or []
    fl = [p["flux"] for p in pos if p.get("ok") and p.get("flux")]
    if len(fl) < 3 or len(fl) < len(pos):
        return
    fl = np.array(fl, float)
    if amp is not None:
        n = len(dts)
        corr = []
        for seg in np.array_split(np.arange(n), max(1, segments)):
            tv = []
            for i in seg:
                xi, yi = int(round(c.x + c.vx * float(dts[i]))), int(round(c.y + c.vy * float(dts[i])))
                if 0 <= yi < amp.shape[1] and 0 <= xi < amp.shape[2]:
                    tv.append(float(amp[i, yi, xi]))
            corr.append(max(float(np.mean(tv)), 0.2) if tv else 1.0)
        if len(corr) == len(fl):
            fl = fl / np.array(corr)
    m = float(np.mean(fl))
    if m <= 0 or float(np.std(fl)) / m > 0.2:
        return
    motion = math.hypot(c.vx, c.vy) * float(np.ptp(dts)) / max(fwhm, 0.5)
    changed = False
    if "nekonzistentní snímky" in c.flags and (c.snr_raw or 0) >= 30:
        c.flags = [f for f in c.flags if f != "nekonzistentní snímky"]
        c.flags.append("χ² vysoké, podstacky souhlasí")
        changed = True
    if "nekonzistentní segmenty" in c.flags and c.snr >= 10 and motion <= 2.5:
        c.flags = [f for f in c.flags if f != "nekonzistentní segmenty"]
        c.flags.append("segmenty nesouhlasí, podstacky souhlasí")
        changed = True
    if changed and not set(c.flags) & hard:
        c.auto_status = "candidate"


VET_FLAGS = {"nekonzistentní segmenty", "jednosnímkový artefakt", "reziduum hvězdy", "u hvězdy",
             "mimo pole v části snímků", "nekonzistentní snímky", "bez směru pohybu"}


def _revet(c, pt, dts, tn_t, sp, fw_t, hard) -> None:
    """Repeat the basic vetting on the PSF-fitted motion when it differs from the search one.

    A bright slow object is partly in its own template; the GLRT then prefers a slower
    velocity (smaller σ) and the basic checks ran on a track that misses the object in
    part of the series. The sub-stack fit finds the true motion ((149) Medusa 12.4 mag,
    88P/Coihueco 9. 11.: search 0.13″/min, fit 0.61″/min, segments [56, 108, −0.6])."""
    import copy
    t = copy.copy(c)
    t.flags = []
    t.auto_status = "candidate"
    SR.vet([t], pt, dts, tn_t, sp.segments, sp.segment_z_max, sp.bright_star_mask_sigma, fw_t)
    keep = [f for f in c.flags if f not in VET_FLAGS]
    for k in ("snr", "snr_raw", "seg_snr", "median_ratio", "frames_used", "star_contaminated", "chi2",
              "snr_reverse", "star_ratio"):
        if hasattr(t, k):
            setattr(c, k, getattr(t, k))
    ns = c.noise_scale or 1.0
    if ns > 1.0:
        c.snr /= ns
        if c.snr_raw is not None:
            c.snr_raw /= ns
        if c.seg_snr:
            c.seg_snr = [round(v / ns, 2) for v in c.seg_snr]
    c.flags = keep + [f for f in t.flags if f not in keep]
    c.auto_status = "auto_rejected" if (set(c.flags) & hard or t.auto_status != "candidate") else "candidate"


def _frame_to_grid(reg, x: np.ndarray, y: np.ndarray, identity: bool = False):
    """Detector (frame) pixel → reference grid (rows, cols, int): inverse of ``ref_to_frame``
    by fixed-point iteration (the mapping is a small shift + distortion)."""
    if identity:
        return np.rint(y).astype(int), np.rint(x).astype(int)
    gx, gy = x.copy(), y.copy()
    for _ in range(4):
        fx, fy = reg.ref_to_frame(gx, gy)
        gx = gx - (np.asarray(fx) - x)
        gy = gy - (np.asarray(fy) - y)
    return np.rint(gy).astype(int), np.rint(gx).astype(int)


def _star_beside_track(c, fwhm: float, sp) -> bool:
    """The star that caused "reziduum hvězdy" lies beside the track, not under it.

    A residual of a badly subtracted star sits on the star (< 1 FWHM). With a short focal
    length almost every main-belt asteroid moves < 3 FWHM per series, so a star 1.6–2.2
    FWHM away flagged real objects ((1048) Feodosia, (488) Kreusa, (1304) Arosa, CTA-S1
    315/4). Kept only when the track is also consistent: χ² ≤ 1.3, median/mean ≥ 0.6
    and every third of the series ≥ 2 σ. Over 15 jobs: injections 46/144 back,
    unknowns 9/1372."""
    if c.static_star_x is None or c.static_star_y is None or not c.seg_snr:
        return False
    d = math.hypot(c.static_star_x - c.x, c.static_star_y - c.y) / max(fwhm, 0.5)
    return (d >= sp.star_beside_fwhm and c.chi2 <= sp.star_beside_chi2
            and c.median_ratio >= 0.6 and min(c.seg_snr) >= 2.0)


def _sharp_rescue(cands, sp, min_inj: int = 20) -> int:
    """Calibrate "ostřejší než hvězdy" on the injections of this run.

    Injections carry the field's empirical PSF and go through the same resampling and
    width measurement; with undersampled stars (short focal lengths, FWHM ≈ 2–2.7 px) real
    asteroids measured 0.55–0.65 × the star FWHM at SNR 7 ((1979) Sakharov, (3984) Chacos,
    88P/Coihueco 8. 11.) and were rejected by the fixed 0.65. The cut becomes
    max(0.45, min(sharp_ratio, 2 % quantile of the injections − 0.05)); detections rejected
    only for sharpness above it get the warning instead. Returns how many were restored."""
    rel = [c.obj_fwhm / c.fwhm for c in cands
           if c.injected is not None and c.obj_fwhm and c.fwhm]
    if len(rel) < min_inj:
        return 0
    # the narrowest injection, not the 2 % quantile: CTA-N 318/4 (786) Bredichina 0.57 × FWHM was
    # rejected with 79 injections at 0.575–0.73 (2 % quantile 0.642 → cut 0.592)
    cut = max(0.45, min(sp.sharp_ratio, float(np.quantile(rel, 0.02)) - 0.05, float(np.min(rel)) - 0.03))
    # the other warnings that do not reject on their own ((372) Palma, CTA-S1 320/4: 0.577 × FWHM with
    # the cut at 0.513, but "snížená propustnost" + "pomalý u hvězdy?" kept it out of the rescue)
    soft = {"ostřejší než hvězdy", "u hvězdy", "spíš proměnná hvězda", "užší než hvězdy?",
            "snížená propustnost", "pomalý u hvězdy?", "hvězda vedle stopy", "také jako zdroj navíc"}
    n = 0
    for c in cands:
        if ("ostřejší než hvězdy" in c.flags and set(c.flags) <= soft and c.obj_fwhm and c.fwhm
                and c.obj_fwhm / c.fwhm >= cut):
            c.flags = [f for f in c.flags if f != "ostřejší než hvězdy"]
            if "užší než hvězdy?" not in c.flags:
                c.flags.append("užší než hvězdy?")
            c.auto_status = "candidate"
            n += 1
    return n


STAR_FLAGS = {"u hvězdy", "reziduum hvězdy", "pomalý u hvězdy"}
STAR_SOFT = {"snížená propustnost", "hvězda vedle stopy"}




def _unmatch_implausible(cands, known_list, tol_mag: float = 2.5) -> int:
    """Drop identifications whose detection is far too bright for the predicted magnitude.
    For background-limited sums SNR ∝ flux, so pred_mag + 2.5·log10(SNR) is about the same for
    every correct match (the job's zero point). CTA-N 318/5: a star residual (SNR 7) 29″ from the
    19.6 mag (50647) 2000 EN88 was "identified" while 14.7 mag Penelope had SNR 5.5."""
    pts = [(c, c.known["pred_mag"] + 2.5 * math.log10(max(c.snr, 1.0))) for c in cands
           if c.known and c.known.get("kind") == "asteroid" and c.known.get("pred_mag") is not None
           and c.snr and c.snr > 0]
    good = [z for c, z in pts if c.auto_status == "candidate" and c.injected is None]
    if len(good) < 3:
        return 0
    z0 = float(np.median(good))
    n = 0
    for c, z in pts:
        if z > z0 + tol_mag:
            nm = c.known["name"]
            c.flags.append(f"u polohy {nm} ({c.known['pred_mag']:.1f} mag), ale o {z - z0:.1f} mag jasnější, "
                           f"než by byl – jiný zdroj")
            c.known = None
            for k in known_list:
                if k.name == nm and k.matched_candidate == c.id:
                    k.matched_candidate = None
            n += 1
    return n


def _TRAIL_NFAR(fwhm: float, half: int = 20) -> int:
    t = np.arange(-half, half + 1, dtype=float)
    return int((np.abs(t) > 2.0 * fwhm).sum())


def _trail_in_frame(sm: np.ndarray, px: float, py: float, fwhm: float, half: int = 20) -> tuple[float, float]:
    """Significance of a straight trail through (px, py) in a smoothed unit-noise residual frame:
    best angle, the light along ±half px outside the PSF core (the object itself excluded).
    Returns (z, angle_deg)."""
    from scipy import ndimage
    g = max(fwhm / 2.355, 0.5)
    norm = 1.0 / (2.0 * math.sqrt(math.pi) * g)          # σ of a pixel of the smoothed unit-noise map
    t = np.arange(-half, half + 1, dtype=float)
    far = np.abs(t) > 2.0 * fwhm
    nfar = int(far.sum())
    best = (0.0, 0.0)
    for ang in range(0, 180, 3):
        a = math.radians(ang)
        vals = ndimage.map_coordinates(sm, [py + t[far] * math.sin(a), px + t[far] * math.cos(a)], order=1,
                                       mode="nearest")
        # median, not mean: a star residual crossed by the line is a few samples, a trail is all of
        # them (mean: the null had a 1 % tail at z 54 from stars); samples correlated over ~2 px
        z = float(np.median(vals)) / norm * math.sqrt(nfar / 2.0) / 1.2533
        if z > best[0]:
            best = (z, float(ang))
    return best


def _satellite_reject(cands, reg_mm, dts, fwhm: float, sp, half: int = 40) -> int:
    """A faint satellite trail (below the transient mask) crossing the track in one frame adds up
    to frame_cap_sigma/√n to the SNR – enough to lift a 5σ noise peak over the threshold
    (CTA-N 318/1 #269: SNR 6.3, the last frame 8σ on a trail, z = 23 along it). The frame(s) with a
    trail through the position (z ≥ 16) are left out; below the threshold without them → rejected."""
    from scipy import ndimage
    n = reg_mm.shape[0]
    t = np.asarray(dts, float)
    cap = float(getattr(sp, "frame_cap_sigma", 4.0) or 4.0)
    thr = float(sp.snr_threshold)
    g = max(fwhm / 2.355, 0.5)
    norm = 1.0 / (2.0 * math.sqrt(math.pi) * g)
    nrej = 0
    for c in cands:
        if c.auto_status != "candidate" or c.known or c.injected is not None or c.snr >= 2 * thr:
            continue
        xi, yi = int(round(c.x)), int(round(c.y))
        h, w = reg_mm.shape[1:]
        Y0, Y1, X0, X1 = max(0, yi - half), min(h, yi + half), max(0, xi - half), min(w, xi + half)
        if Y1 - Y0 < 40 or X1 - X0 < 40:
            continue
        blk = np.array(reg_mm[:, Y0:Y1, X0:X1], np.float32)
        med = SR.temporal_stats(blk)[0]
        drop, hits = 0.0, []
        for i in range(n):
            r = blk[i] - med
            ok = np.isfinite(r)
            if ok.sum() < 0.5 * r.size:
                continue
            md = float(np.median(r[ok]))
            s = float(1.4826 * np.median(np.abs(r[ok] - md))) or 1.0
            sm = ndimage.gaussian_filter(np.where(ok, (r - md) / s, 0.0), g)
            px, py = c.x + c.vx * t[i] - X0, c.y + c.vy * t[i] - Y0
            if not (3 <= px < sm.shape[1] - 3 and 3 <= py < sm.shape[0] - 3):
                continue
            z, ang = _trail_in_frame(sm, px, py, fwhm)
            if z >= 16.0:              # null (random places, CTA-N 318/1): 99.9 % 12.7, max 13.6; trails 23–26
                pt = float(ndimage.map_coordinates(sm, [[py], [px]], order=1)[0]) / norm
                # only the trail's own light is taken away: its level at the position = the median
                # along the line (same smoothed map, same units); whatever is above it is the
                # object itself and stays in the sum (a real asteroid under a trail keeps its frame)
                trail = z * 1.2533 / math.sqrt(_TRAIL_NFAR(fwhm) / 2.0)
                own = max(0.0, pt - trail)
                drop += min(cap, max(pt, 0.0)) - min(cap, own)
                hits.append(i + 1)
        if not hits:
            continue
        snr_wo = (c.snr * math.sqrt(n) - drop) / math.sqrt(n)
        fr = ", ".join(str(k) for k in hits)
        # a margin: a borderline real object that the trail lifted a little keeps only a warning
        if snr_wo < thr - 0.3:
            c.flags.append(f"stopa družice přes polohu (snímek {fr}) – bez ní SNR {snr_wo:.1f}")
            c.auto_status = "auto_rejected"
            nrej += 1
        else:
            c.flags.append(f"stopa družice přes polohu (snímek {fr}), bez ní SNR {snr_wo:.1f}")
    return nrej


def _slow_star_second_look(cands) -> int:
    """After all vetting: a detection rejected only as "pomalý u hvězdy" (plus soft warnings) that is
    strong and present in every third of the series is kept with a warning. In-tile check alone
    missed (372) Palma (CTA-S1 320/4, SNR 14.4, thirds 5.7/2.2/6.3): it was rejected by another
    rule at that moment, which a later rescue lifted, leaving only "pomalý u hvězdy"."""
    soft = {"pomalý u hvězdy", "snížená propustnost", "užší než hvězdy?", "spíš proměnná hvězda",
            "také jako zdroj navíc"}
    n = 0
    for c in cands:
        if c.auto_status == "candidate" or "pomalý u hvězdy" not in c.flags or not set(c.flags) <= soft:
            continue
        seg = c.seg_snr or []
        exp_s = c.snr / math.sqrt(len(seg)) if seg else 0.0
        if (c.snr >= 8.0 and seg and exp_s > 0 and min(seg) / exp_s >= 0.2 and (c.median_ratio or 0.0) >= 0.6
                and (c.chi2 or 0.0) <= 2.5):
            c.flags = ["pomalý u hvězdy?" if f == "pomalý u hvězdy" else f for f in c.flags]
            c.auto_status = "candidate"
            n += 1
    return n

def _crossing_reject(cands, dts, fwhm: float) -> int:
    """A wrong velocity that crosses the path of a brighter moving object picks up its light in the
    frame(s) of the crossing: the sum is significant, but the cross sits on the object in one frame
    only (CTA-N 318: #296 P1, #187 P2, #197 P3 – each 0.2–0.9 px from a 2× stronger candidate in
    one frame, the last segment carrying the signal). Rejected when the crossing frame lies in the
    strongest segment and the two motions differ by more than 2 FWHM over the series."""
    t = np.asarray(dts, float)
    if len(t) < 3 or len(cands) < 2:
        return 0
    order = np.argsort(t)
    nseg = None
    strong = [b for b in cands if (b.auto_status == "candidate" or b.known or b.injected is not None)]
    span = float(np.ptp(t))
    n = 0
    for a in cands:
        if a.auto_status != "candidate" or a.known or not a.seg_snr:
            continue
        nseg = len(a.seg_snr)
        segs = np.array_split(order, nseg)
        for b in strong:
            if b is a or b.snr < 1.5 * a.snr:
                continue
            if math.hypot(a.vx - b.vx, a.vy - b.vy) * span <= 2 * fwhm:
                continue
            d = np.hypot(a.x + a.vx * t - (b.x + b.vx * t), a.y + a.vy * t - (b.y + b.vy * t))
            k = int(np.argmin(d))
            if d[k] > max(1.5 * fwhm, 2.0):
                continue
            s_k = next(i for i, sg in enumerate(segs) if k in sg)
            if int(np.argmax(a.seg_snr)) != s_k:
                continue
            a.flags.append(f"kříží stopu silnějšího objektu #{b.id} (snímek {k + 1}, {d[k]:.1f} px) – světlo z něj")
            a.auto_status = "auto_rejected"
            n += 1
            break
    return n

def _catalog_no_star_rescue(cands, cat: dict, fwhm: float, sp, span_min: float = 0.0) -> int:
    """Give back detections rejected only for a "star" at the track where UCAC4 has no star.

    The star tests look at the static (clipped-mean) sky; a slow object sits partly in that
    template itself, so its own light looks like a star next to the track. When the static
    peak is bright enough to be in UCAC4 (estimated from the detection's magnitude and the
    peak ratio, ≤ ``ucac4_complete_mag``) and the catalogue has no star within 1.5 FWHM of the
    track or of the peak, it is not a star – the object itself (or a variable / new source),
    worth a look even when slow. Fainter peaks cannot be judged from the catalogue: unchanged.
    Only detections whose flags are all star-related (or soft) are touched.

    Second case: a slow mover (≤ 3 FWHM over the series) rejected only for inconsistent
    segments, with no UCAC4 star near the track and consistent frames (χ² ≤ 2, median/mean
    ≥ 0.6). The segment expectation of slow tracks (own light in the template) is the least
    reliable part of the vetting, worst under changing throughput: Coihueco 317 dawn, 35 slow
    injections lost this way, no unknown detection with these flags at all."""
    from scipy.spatial import cKDTree
    if len(cat.get("x", [])) == 0:
        return 0
    lim = float(getattr(sp, "ucac4_complete_mag", 15.5))
    if lim <= 0:
        return 0
    tree = cKDTree(np.stack([cat["x"], cat["y"]], 1))
    rad = max(1.5 * fwhm, 2.0)
    n = 0
    for c in cands:
        fl = set(c.flags)
        if c.auto_status == "candidate":
            continue
        motion = math.hypot(c.vx, c.vy) * span_min / max(fwhm, 0.5)
        if ("nekonzistentní segmenty" in fl and fl <= {"nekonzistentní segmenty"} | STAR_SOFT
                and span_min > 0 and motion <= 3.0 and (c.chi2 is None or c.chi2 <= 2.0)
                and (c.median_ratio is None or c.median_ratio >= 0.6)
                and tree.query_ball_point([c.x, c.y], 40 * fwhm)
                and not tree.query_ball_point([c.x, c.y], rad + 0.5 * motion * fwhm)):
            c.flags = [f if f != "nekonzistentní segmenty" else "segmenty nesouhlasí (pomalý, bez hvězdy v UCAC4)"
                       for f in c.flags]
            c.auto_status = "candidate"
            n += 1
            continue
        if not (fl & STAR_FLAGS) or not fl <= STAR_FLAGS | STAR_SOFT:
            continue
        if c.mag is None or c.star_ratio is None or c.static_star_x is None:
            continue
        m_peak = c.mag - 2.5 * math.log10(max(c.star_ratio, 1e-3))
        if m_peak > lim:
            continue
        # the catalogue must cover the spot (gaps, field edge): some star within 40 FWHM
        if not tree.query_ball_point([c.x, c.y], 40 * fwhm):
            continue
        if tree.query_ball_point([c.x, c.y], rad) or tree.query_ball_point([c.static_star_x, c.static_star_y], rad):
            continue
        c.flags = [f for f in c.flags if f not in STAR_FLAGS]
        c.flags.append(f"bez hvězdy v UCAC4 (statické maximum ≈ {m_peak:.1f} mag)")
        c.auto_status = "candidate"
        n += 1
    return n


def _catalog_stars_reject(cands, cat: dict, fwhm: float, span_min: float, max_motion: float = 3.0,
                          mag_margin: float = -1.0, probe: list | None = None) -> int:
    """Reject slow detections that sit on a catalogue star at least as bright as themselves.

    A bright star whose image changes slightly during the series (seeing, registration of
    a part of the frames, a field corner) leaves a residual that the mover tests cannot
    always tell from a slow asteroid in its own template: 88P/Coihueco 11. 11., 9–12 mag
    stars at 0.28–0.32″/min passed with "spíš proměnná hvězda" and ML 0.8–0.99. A real
    asteroid that bright would be a known object; an unknown one on a star brighter than
    itself cannot be measured anyway. Fast movers (> ``max_motion`` FWHM over the series)
    and identified known objects are never touched. Returns how many were rejected.

    ``probe``: known objects and recovered injections are never rejected, but the ones the
    rule *would* reject are appended (name / "injekce #id") – the built-in test of the rule."""
    from scipy.spatial import cKDTree
    if len(cat.get("x", [])) == 0:
        return 0
    tree = cKDTree(np.stack([cat["x"], cat["y"]], 1))
    rad = max(1.5 * fwhm, 2.0)
    n = 0
    for c in cands:
        # also the ones already rejected for star-related reasons: the classifier rescue
        # ("vráceno klasifikátorem") returned 133 such star residuals in the next 88P run
        protected = bool(c.known) or c.injected is not None

        def _reject(m_star):
            nonlocal n
            if protected:
                if probe is not None:
                    probe.append(c.known["name"] if c.known else f"injekce #{c.injected}")
                return
            c.flags.append(f"hvězda z katalogu UCAC4 ({m_star:.1f} mag)")
            if c.auto_status == "candidate":
                n += 1
            c.auto_status = "auto_rejected"

        if math.hypot(c.vx, c.vy) * span_min / max(fwhm, 0.5) > max_motion:
            # faster tracks: a residual of a bright star the track runs across. Only with the
            # signs of a star residual (variable-star test or residual flag, static peak ≥ 20×
            # the object) and a catalogue star within 1 FWHM of the track. 317/Coihueco dawn:
            # 11 of the 12 detections the user rejected (stars 9–16 mag at 0.1–0.6 FWHM,
            # peak ratio 27–430); passing injections near stars: ≥ 1.5 FWHM, ratio ≤ 17.
            # A bright star (≤ 10.5 mag) leaves residuals further out: up to 2 FWHM when the
            # detection is flagged as a residual (317 Pole 2 #104, 9.4 mag at 1.8 FWHM, no own
            # flux measured → peak ratio unknown).
            fl = set(c.flags)
            # ≥ 20: the user-rejected residuals had 27–430; one real (injected) object crossing a
            # star within 1 FWHM had 11 with the variable-star sign (317/3 #125, 1 of 33 near stars)
            ratio_ok = (c.star_ratio >= 20) if c.star_ratio is not None else ("reziduum hvězdy" in fl)
            if ({"spíš proměnná hvězda", "reziduum hvězdy"} & fl) and ratio_ok:
                dt = [p.get("dt_min") for p in (c.positions or []) if p.get("dt_min") is not None]
                t0, t1 = (min(dt), max(dt)) if dt else (-0.5 * span_min, 0.5 * span_min)
                hits = set()
                for t in np.linspace(t0, t1, 15):
                    for k in tree.query_ball_point([c.x + c.vx * t, c.y + c.vy * t], 2.0 * fwhm):
                        d = math.hypot(cat["x"][k] - c.x - c.vx * t, cat["y"][k] - c.y - c.vy * t)
                        # the wider zone only with the residual flag itself: an injection with
                        # just the variable-star sign passed 1.9 FWHM from a 10-mag star (317/3)
                        if d <= fwhm or (d <= 2.0 * fwhm and cat["mag"][k] <= 10.5
                                         and "reziduum hvězdy" in fl):
                            hits.add(k)
                if hits:
                    _reject(float(np.min(cat["mag"][list(hits)])))
            continue
        nb = tree.query_ball_point([c.x, c.y], rad)
        if not nb:
            continue
        nb = np.asarray(nb)
        d = np.hypot(cat["x"][nb] - c.x, cat["y"][nb] - c.y)
        # centred on the star (≤ 1 FWHM): the star at most 0.5 mag fainter than the detection;
        # off-centre (1–1.5 FWHM, a residual of a star changing shape): the star ≥ 1 mag brighter.
        # A residual is always much fainter than its star (88P: 131 of 133 hold with −1 mag), a
        # faint real asteroid next to a faint star keeps its chance (with +1 mag everywhere every
        # UCAC4 star counts and ~12 % of the field would be closed to slow faint objects)
        base = c.mag if c.mag is not None else 15.0
        lim = np.where(d <= fwhm, base + 0.5, base + mag_margin)
        hit = cat["mag"][nb] <= lim
        if not hit.any():
            continue
        _reject(float(np.min(cat["mag"][nb][hit])))
    return n


def _twilight_frames(mjd, lat, lon, alt) -> np.ndarray | None:
    """True for frames taken with the Sun above −18° (astronomical twilight or day)."""
    try:
        from astropy.coordinates import AltAz, EarthLocation, get_sun
        from astropy.time import Time
        import astropy.units as u
        loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, (alt or 0.0) * u.m)
        t = Time(np.asarray(mjd, float), format="mjd", scale="utc")
        sun = get_sun(t).transform_to(AltAz(obstime=t, location=loc))
        return np.asarray(sun.alt.deg) > -18.0
    except Exception:          # noqa: BLE001 – without the Sun's position nothing is relaxed
        return None


def _sky_mask(pt, fwhm: float) -> np.ndarray:
    """Pixels away from stars (static sky < 1.5 σ, dilated by ~2 FWHM) for noise estimates."""
    st = np.nan_to_num(pt.static, nan=99.0)
    star = ndimage.binary_dilation(st > 1.5, iterations=max(1, int(round(2 * fwhm))))
    return ~star


def _drift_velocity(cshift: np.ndarray | None, dts: np.ndarray, fwhm: float) -> tuple[float, float] | None:
    """Sky velocity (px/min) of a detector-fixed point when the dither is a slow linear drift.

    ``cshift`` (n, 2) = detector − grid offsets (dy, dx) per frame. A detector-fixed point
    is at sky = detector − offset, so it moves with −d(offset)/dt. None when the offsets
    are not a line within 1 FWHM (random dithering: a warm pixel then jumps on the sky and
    ``chip_ratio`` alone tells it apart)."""
    if cshift is None or len(dts) < 4:
        return None
    t = np.asarray(dts, float)
    A = np.stack([np.ones_like(t), t], 1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(cshift, float), rcond=None)       # (2, 2): rows = const, slope
    res = np.asarray(cshift, float) - A @ coef
    if float(np.sqrt((res ** 2).sum(axis=1).mean())) > fwhm:
        return None
    return -float(coef[1, 1]), -float(coef[1, 0])


def _chip_shift(regs, cx: float, cy: float, sp, force: bool = False) -> np.ndarray | None:
    """(n, 2) detector − grid offsets (dy, dx) of every frame at a point (local translation)."""
    if not regs or not (force or getattr(sp, "chip_pattern", True)):
        return None
    out = np.zeros((len(regs), 2))
    for i, r in enumerate(regs):
        fx, fy = r.ref_to_frame(np.array([float(cx)]), np.array([float(cy)]))
        out[i] = (float(np.asarray(fy).ravel()[0]) - cy, float(np.asarray(fx).ravel()[0]) - cx)
    return out


def _forced(reg_mm, x, y, v, dts, fwhm, tn, sp, half=96, search_px=1.0, maps=None, regs=None):
    """SNR on the predicted track of a known object (best within ±search_px) and a note.

    The search box is small (predictions are good to < 1 px); a ±2 px box in a
    dense field found star residuals with SNR 30–50 for 18th-mag objects.
    The note says why the value may be unreliable."""
    n, h, w = reg_mm.shape
    xi, yi = int(round(x)), int(round(y))
    Y0, Y1, X0, X1 = max(0, yi - half), min(h, yi + half), max(0, xi - half), min(w, xi + half)
    block = np.array(reg_mm[:, Y0:Y1, X0:X1])
    if block.shape[1] < 32 or block.shape[2] < 32:
        return None, None, "mimo pole", None, None
    amp = maps.tile(Y0, Y1, X0, X1) if maps is not None else None
    pt = SR.prepare_tile(block, fwhm, sp.bright_star_mask_sigma, cap=0, amp=amp, amp_min=sp.throughput_min,
                         chip_shift=_chip_shift(regs, 0.5 * (X0 + X1), 0.5 * (Y0 + Y1), sp))
    sig = tn.sigma(math.hypot(*v))
    best, bxy = -1e9, (0.0, 0.0)
    for dy in np.arange(-search_px, search_px + 0.01, 0.5):
        for dx in np.arange(-search_px, search_px + 0.01, 0.5):
            acc, used = SR._sum_track(pt.f_raw, dts, x - X0 + dx, y - Y0 + dy, v[0], v[1], range(n), 0)
            if used < 0.7 * n:
                continue
            val = float(acc[0, 0]) / sig
            if val > best:
                best, bxy = val, (dx, dy)
    note = None
    if best <= -1e8:
        return None, None, "mimo pole v části snímků", None, None
    # empirical noise of exactly this statistic: the same sum at random positions of the block
    # with the same velocity. For slow tracks the model σ is ~25 % too small (integer shifts,
    # star residuals) and the best of the 5×5 offsets adds look-elsewhere – (3995) Sakaino,
    # (1773), (828) showed forced SNR 7–8 while the search map had 4.8 there (Coihueco 8. 11.)
    noise = _forced_noise(pt.f_raw, dts, v, sig, n, fwhm)
    if noise is not None:
        best = best / noise
    # what lies on the track: a bright star / masked core makes the value meaningless
    r = max(2, int(round(1.5 * fwhm)))
    hh, ww = pt.static.shape
    near, masked = 0.0, 0
    for i in (0, n // 2, n - 1):
        cx = int(round(x - X0 + bxy[0] + v[0] * dts[i]))
        cy = int(round(y - Y0 + bxy[1] + v[1] * dts[i]))
        y0, y1, x0, x1 = max(0, cy - r), min(hh, cy + r + 1), max(0, cx - r), min(ww, cx + r + 1)
        if y1 > y0 and x1 > x0:
            near = max(near, float(pt.static[y0:y1, x0:x1].max()))
            masked += int(pt.bright[y0:y1, x0:x1].any())
    mid_i = n // 2
    mcx = x - X0 + bxy[0] + v[0] * dts[mid_i]
    mcy = y - Y0 + bxy[1] + v[1] * dts[mid_i]
    _, peak_offset = _static_peak(pt.static, mcx, mcy, r)
    track_span = math.hypot(*v) * float(np.max(dts) - np.min(dts))
    self_like = (track_span < 1.5 * fwhm and peak_offset is not None
                 and peak_offset < max(1.0, 0.6 * fwhm))
    zeroed = all(float(np.abs(pt.f_raw[:, int(min(max(round(y - Y0 + bxy[1] + v[1] * dts[i]), 0), hh - 1)),
                                            int(min(max(round(x - X0 + bxy[0] + v[0] * dts[i]), 0), ww - 1))]).max()) == 0.0
                 for i in (0, n // 2, n - 1))
    if zeroed:
        holes = int((~np.isfinite(block[:, int(round(y - Y0)), int(round(x - X0))])).sum())
        note = f"data na stopě vynulována (maska; chybí ve {holes} z {n} snímků)"
        sat = _saturated_mover(block, x - X0, y - Y0, v, dts, fwhm)
        if sat is not None:
            sat["dx"] = round(sat["dx"], 2)
            sat["dy"] = round(sat["dy"], 2)
            note = (f"přesycený objekt na předpovězené dráze (díra v datech se pohybuje s ním ve "
                    f"{sat['frames']} z {n} snímků, odchylka {math.hypot(sat['dx'], sat['dy']):.1f} px)")
            return None, [sat["dx"], sat["dy"]], note, noise, sat
    elif masked >= 2 and self_like:
        note = "pomalý zdroj ve statické šabloně"
    elif masked >= 2:
        note = "jasná hvězda na stopě"
    elif near > 0.5 * sp.bright_star_mask_sigma and not _own_light(pt.static, block, x - X0 + bxy[0],
                                                                    y - Y0 + bxy[1], v, dts, fwhm, near):
        note = "hvězda u stopy"
    elif max(abs(bxy[0]), abs(bxy[1])) >= search_px - 1e-6:
        note = "maximum na okraji hledání"
    return round(best, 2), [round(bxy[0], 2), round(bxy[1], 2)], note, noise, None


def _own_light(static, block, x, y, v, dts, fwhm, near) -> bool:
    """Is the bright static sky at the track the object itself?

    A bright slow asteroid sits in its own clipped-mean template: (149) Medusa, (419) Aurelia
    12.5 mag (88P/Coihueco 9. 11.) got "hvězda u stopy" with no star anywhere near. It is the
    object when the brightest static pixel lies on the track (≤ 0.7 FWHM) and is not
    brighter than the object's own median peak in the frames."""
    n, hh, ww = block.shape
    t = np.linspace(float(np.min(dts)), float(np.max(dts)), 25)
    px, py = x + v[0] * t, y + v[1] * t
    r = max(2, int(round(1.5 * fwhm)))
    xi, yi = int(round(float(px.mean()))), int(round(float(py.mean())))
    span = int(math.ceil(math.hypot(*v) * float(np.ptp(dts)) / 2)) + r
    y0, y1, x0, x1 = max(0, yi - span), min(hh, yi + span + 1), max(0, xi - span), min(ww, xi + span + 1)
    if y1 <= y0 or x1 <= x0:
        return False
    w_ = np.where(np.isfinite(static[y0:y1, x0:x1]), static[y0:y1, x0:x1], -np.inf)
    j = np.unravel_index(int(np.argmax(w_)), w_.shape)
    sy, sx = y0 + j[0], x0 + j[1]
    if float(np.min(np.hypot(px - sx, py - sy))) > 0.7 * fwhm:
        return False
    peaks = []
    for i in range(n):
        cx, cy = int(round(x + v[0] * float(dts[i]))), int(round(y + v[1] * float(dts[i])))
        b = block[i, max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2]
        if b.size and np.isfinite(b).any():
            peaks.append(float(np.nanmax(b)))
    return bool(peaks) and float(w_[j]) <= 1.0 * float(np.median(peaks))


def _cand_flux(c) -> float | None:
    """Median PSF-fit flux of the sub-stacks (σ·px units of the registered frames)."""
    fl = [p["flux"] for p in (c.positions or []) if p.get("ok") and p.get("flux") and p["flux"] > 0]
    return float(np.median(fl)) if fl else None


def _clipped_zp(zs, min_n: int) -> tuple[float, float, int] | None:
    """Median zero point with MAD clipping: (zp, scatter, n used)."""
    if len(zs) < min_n:
        return None
    z = np.asarray(zs, float)
    for _ in range(3):
        med = float(np.median(z))
        mad = 1.4826 * float(np.median(np.abs(z - med)))
        keep = np.abs(z - med) <= max(3 * mad, 0.3 if len(z) < 30 else 0.15)
        if keep.sum() < min_n:
            break
        z = z[keep]
    zp = float(np.median(z))
    scatter = float(1.4826 * np.median(np.abs(z - zp))) if len(z) > 2 else 0.3
    return zp, scatter, int(len(z))


def _zp_known(cands, known_list, fwhm: float, span_min: float) -> list[float]:
    """V_pred + 2.5 log F of the identified asteroids (SNR ≥ 8, motion ≥ 0.8 FWHM, no comets)."""
    by_name = {k.name: k for k in known_list}
    zs = []
    for c in cands:
        if not c.known or c.injected is not None or c.known.get("kind") == "comet":
            continue
        k = by_name.get(c.known["name"])
        f = _cand_flux(c)
        motion = math.hypot(c.vx, c.vy) * span_min / max(fwhm, 0.5)
        if k is None or k.mag is None or f is None or motion < 0.8 or c.snr < 8:
            continue
        zs.append(k.mag + 2.5 * math.log10(f))
    return zs


def _zp_ucac4(root, reg_mm, wcs, mjd_ref: float, fwhm: float, filt, max_stars: int = 400, stars_out: dict | None = None,
              seed: int = 11) -> tuple[list[float], str] | None:
    """m_APASS + 2.5 log F of isolated, unsaturated UCAC4 stars in the static (median) sky.

    The stars are fitted exactly like the detections (Gaussian of the stars' FWHM + plane,
    ``search._fit_gauss``, total flux in σ·px of the registered frames), so the zero point
    applies directly to ``positions[].flux``. Band = APASS band closest to the filter."""
    from astropy.time import Time

    from . import ucac4 as U4
    n, h, w = reg_mm.shape
    band = U4.band_for_filter(filt)
    xs = np.r_[np.linspace(0, w - 1, 9), np.full(9, w - 1), np.linspace(0, w - 1, 9), np.zeros(9)]
    ys = np.r_[np.zeros(9), np.linspace(0, h - 1, 9), np.full(9, h - 1), np.linspace(0, h - 1, 9)]
    ra, dec = (np.asarray(a, float) for a in wcs.pixel_to_world_values(xs, ys))
    dmin, dmax = float(np.nanmin(dec)) - 0.05, float(np.nanmax(dec)) + 0.05
    ra = ra % 360.0
    if dmax > 89.0 or dmin < -89.0:
        r0, r1 = 0.0, 360.0
    elif float(np.ptp(ra)) > 180.0:                 # the field contains RA 0h
        r0, r1 = float(np.min(ra[ra > 180])) - 0.05, float(np.max(ra[ra < 180])) + 0.05
    else:
        r0, r1 = float(np.min(ra)) - 0.05, float(np.max(ra)) + 0.05
    epoch = float(Time(mjd_ref, format="mjd").jyear)
    cat = U4.query(root, r0 % 360.0, r1 % 360.0, dmin, dmax, band, epoch, mag_max=16.5)
    if len(cat["ra"]) < 10:
        return None
    px, py = (np.asarray(a, float) for a in wcs.world_to_pixel_values(cat["ra"], cat["dec"]))
    ax, ay = (np.asarray(a, float) for a in wcs.world_to_pixel_values(cat["ra_all"], cat["dec_all"]))
    if stars_out is not None:
        stars_out.update(x=ax, y=ay, mag=np.asarray(cat["mag_all"], float))
    R = 6
    inside = (px > R + 2) & (px < w - R - 3) & (py > R + 2) & (py < h - R - 3)
    idx = np.nonzero(inside)[0]
    if len(idx) < 10:
        return None
    # isolated: no other catalogue star within 3 FWHM brighter than (mag + 3)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([ax, ay], 1))
    iso = []
    for j in idx:
        nb = tree.query_ball_point([px[j], py[j]], 3.0 * fwhm)
        if all(math.hypot(ax[k] - px[j], ay[k] - py[j]) < 0.7 or cat["mag_all"][k] > cat["mag"][j] + 3
               for k in nb):
            iso.append(j)
    if len(iso) < 10:
        return None
    rng = np.random.default_rng(seed)
    iso = np.array(iso)
    if len(iso) > max_stars:
        iso = rng.choice(iso, max_stars, replace=False)
    sig = max(fwhm / 2.3548, 0.5)
    zs = []
    for j in iso:
        xi, yi = int(round(px[j])), int(round(py[j]))
        cut = np.array(reg_mm[:, yi - R:yi + R + 1, xi - R:xi + R + 1], np.float32)
        if cut.shape[1:] != (2 * R + 1, 2 * R + 1):
            continue
        fin = np.isfinite(cut)
        # a saturated core is missing in (almost) every frame; single masked pixels
        # (hot pixels, cosmics) are only missing in a few
        if fin[:, R - 2:R + 3, R - 2:R + 3].mean(axis=0).min() < 0.7:
            continue
        with _quiet():
            st = np.nanmedian(cut, axis=0)
        if not np.isfinite(st).all():
            continue
        gx, gy, flux, ok = SR._fit_gauss(st, R + px[j] - xi, R + py[j] - yi, sig, r=4)
        if not ok or flux <= 0 or math.hypot(gx - (R + px[j] - xi), gy - (R + py[j] - yi)) > 1.5:
            continue
        if flux / (2 * math.pi * sig * sig) < 3.0:   # peak below ~3 σ: too faint for a zero point
            continue
        zs.append(float(cat["mag"][j]) + 2.5 * math.log10(flux))
    return zs, band


def _photometry(cands, known_list, fwhm: float, span_min: float, min_n: int = 3,
                catalog: tuple[list[float], str] | None = None) -> dict | None:
    """Approximate magnitudes of all detections with a sub-stack PSF fit.

    Zero point from the local UCAC4 (APASS magnitudes of isolated field stars, ``catalog``)
    when available, otherwise from the identified known asteroids (predicted V from MPCORB
    H, G; ±0.2–0.4 mag each; colours of asteroids are similar, so the filter offset is
    absorbed in ZP). With the catalogue the known asteroids are the check (measured −
    predicted). Sets ``c.mag`` and ``k.mag_measured``."""
    zk = _zp_known(cands, known_list, fwhm, span_min)
    res, source, band = None, None, None
    if catalog is not None:
        res = _clipped_zp(catalog[0], 10)
        if res is not None:
            source, band = "ucac4", catalog[1]
    if res is None:
        res = _clipped_zp(zk, min_n)
        if res is None:
            return None
        source, band = "known", "V"
    zp, scatter, n_used = res
    for c in cands:
        f = _cand_flux(c)
        c.mag = round(zp - 2.5 * math.log10(f), 2) if f else None
    for k in known_list:
        if k.matched_candidate is not None:
            cm = next((c for c in cands if c.id == k.matched_candidate), None)
            if cm is not None and cm.mag is not None:
                k.mag_measured = cm.mag
    out = {"zp": round(zp, 3), "n": n_used, "scatter": round(max(scatter, 0.02 if source == "ucac4" else 0.05), 2),
           "source": source, "band": band,
           "band_text": (f"UCAC4/APASS {band} ({n_used} hvězd pole)" if source == "ucac4"
                         else "V (nulový bod z předpovědí MPC známých planetek)")}
    if source == "ucac4":
        chk = _clipped_zp(zk, min_n)
        if chk is not None:
            # known asteroids: measured − predicted (V); includes the asteroid colour V − band
            out["known_offset"] = round(zp - chk[0], 2)
            out["known_n"] = chk[2]
    return out


def _forced_noise(f, dts, v, sig, n, fwhm, n_samp: int = 160, seed: int = 7) -> float | None:
    """Robust σ of the forced statistic on random tracks of the same velocity (≥ 0.8)."""
    _, h, w = f.shape
    span_x = abs(v[0]) * float(np.max(np.abs(dts))) + 3 * fwhm + 2
    span_y = abs(v[1]) * float(np.max(np.abs(dts))) + 3 * fwhm + 2
    if w <= 2 * span_x + 8 or h <= 2 * span_y + 8:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_samp):
        xx = float(rng.uniform(span_x, w - span_x))
        yy = float(rng.uniform(span_y, h - span_y))
        acc, used = SR._sum_track(f, dts, xx, yy, v[0], v[1], range(n), 0)
        if used >= 0.7 * n:
            vals.append(float(acc[0, 0]) / sig)
    if len(vals) < 40:
        return None
    a = np.array(vals)
    return max(0.8, 1.4826 * float(np.median(np.abs(a - np.median(a)))))


def _saturated_mover(block, x, y, v, dts, fwhm) -> dict | None:
    """A saturated object on the predicted track: masked (saturated) pixels that move with it.

    Saturated pixels are NaN (non-linear, never measured), so a bright mover has no data on
    its own track and its forced SNR is 0 – (2) Pallas 8.6 mag, Coihueco 30 s: ~30 px missing
    around the prediction in 11 of 12 frames. A slow one overlaps its own holes, so the
    pixel-wise coverage cannot tell it from a saturated star; the *centroid* of the hole can:
    in every frame the masked region at the prediction is located, its centroids must follow
    the predicted motion (a star's hole stays put). Needs ≥ 60 % of the frames and a motion
    of at least 1 px over the series."""
    n, h, w = block.shape
    t = np.asarray(dts, float)
    span = float(t.max() - t.min())
    if math.hypot(*v) * span < 1.0:
        return None
    R = int(max(6, round(3 * fwhm)))
    tol = max(2.0, fwhm)
    ts, cxs, cys = [], [], []
    for i in range(n):
        px, py = x + v[0] * float(t[i]), y + v[1] * float(t[i])
        xi, yi = int(round(px)), int(round(py))
        y0, y1, x0, x1 = max(0, yi - R), min(h, yi + R + 1), max(0, xi - R), min(w, xi + R + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        hole = ~np.isfinite(block[i, y0:y1, x0:x1])
        if hole.sum() < 3:
            continue
        lab, nl = ndimage.label(hole)
        best_k, best_d = 0, 1e9
        for k in range(1, nl + 1):
            yy, xx = np.nonzero(lab == k)
            if len(yy) < 3:
                continue
            dmin = float(np.min(np.hypot(x0 + xx - px, y0 + yy - py)))
            if dmin < best_d:
                best_k, best_d = k, dmin
        if best_k == 0 or best_d > tol:
            continue
        comp = lab == best_k
        # a saturated core is compact, inside the window and surrounded by bright wings; holes at
        # the field edge (no coverage after a large dither) are not – (19707) Tokunai, 2000 QT74,
        # 2000 WO98 (15.5–18 mag, 60–250 px from the edge, 88P/Coihueco 8. 11.) were reported
        # "saturated" from the edge hole
        if (comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
                or comp.sum() > math.pi * (2.5 * fwhm) ** 2):
            continue
        ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
        wing = block[i, y0:y1, x0:x1][ring]
        wing = wing[np.isfinite(wing)]
        if wing.size < 4 or float(np.median(wing)) < 10.0:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        if math.hypot(x0 + cx - px, y0 + cy - py) > 2 * tol:
            continue
        ts.append(float(t[i]))
        cxs.append(x0 + cx)
        cys.append(y0 + cy)
    if len(ts) < max(3, int(math.ceil(0.6 * n))):
        return None
    tt = np.array(ts)
    A = np.stack([np.ones_like(tt), tt], 1)
    (ax, bx), *_ = np.linalg.lstsq(A, np.array(cxs), rcond=None)
    (ay, by), *_ = np.linalg.lstsq(A, np.array(cys), rcond=None)
    dv = math.hypot(bx - v[0], by - v[1]) * span
    if math.hypot(bx, by) * span < 1.0 or dv > max(1.0, 0.5 * math.hypot(*v) * span):
        return None                     # the hole does not move with the object: a saturated star
    return {"frames": len(ts), "dx": float(ax - x), "dy": float(ay - y),
            "rate_px_min": round(math.hypot(bx, by), 4), "pred_rate_px_min": round(math.hypot(*v), 4)}


def _static_peak(static: np.ndarray, x: float, y: float, radius: int) -> tuple[float | None, float | None]:
    """Peak and its distance from a track position in the static template."""
    cx, cy = int(round(x)), int(round(y))
    h, w = static.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    win = static[y0:y1, x0:x1]
    if not win.size or not np.isfinite(win).any():
        return None, None
    iy, ix = np.unravel_index(int(np.nanargmax(win)), win.shape)
    return float(win[iy, ix]), math.hypot((x0 + ix) - x, (y0 + iy) - y)


NEAR_FLAG = "těsně pod prahem"


def _near_second_chance(cands, injections, sp, ctx) -> int:
    """Detections below the SNR threshold that passed every check: a list of their own.

    An unknown one is set aside (status rejected, flag NEAR_FLAG, GUI filter „druhá šance“),
    so the candidate list keeps its noise level; an identified known object stays a candidate
    (the identification confirms it), only flagged. Injections stay as they are – they tell
    how many real objects the band saves."""
    thr = float(sp.snr_threshold)
    near = [c for c in cands if c.snr < thr and c.auto_status == "candidate"]
    if not near:
        return 0
    n_unknown = n_known = n_inj = 0
    for c in near:
        if NEAR_FLAG not in c.flags:
            c.flags.append(NEAR_FLAG)
        if c.injected is not None:
            n_inj += 1
        elif c.known:
            n_known += 1
        else:
            c.auto_status = "auto_rejected"
            n_unknown += 1
    lo = thr - float(getattr(sp, "near_threshold_margin", 0.0))
    n_inj_band = sum(1 for o in (injections or []) if lo <= (o.get("snr_expected", o.get("snr_nominal")) or -1) < thr) if injections else 0
    ctx.log(f"Druhá šance: {n_unknown + n_known} detekcí těsně pod prahem (SNR {lo:g}–{thr:g}) prošlo všemi "
            f"testy – {n_unknown} neznámých je mimo seznam kandidátů pod filtrem „druhá šance“"
            + (f", {n_known} známých ponecháno jako kandidáti" if n_known else "")
            + (f"; injekcí takto zachráněno {n_inj}" + (f" (v pásmu očekávaného SNR jich bylo {n_inj_band})" if n_inj_band else "")
               if injections else "") + ".")
    return n_unknown


def _dedupe_keep_images(cands, fwhm, span_min):
    kept = T.dedupe(sorted(cands, key=lambda c: -c.snr), fwhm, span_min)
    return kept


def _pixel_scale(w) -> float | None:
    if w is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        s = proj_plane_pixel_scales(w)
        return float(np.mean(s) * 3600.0)
    except Exception:
        return None


def _wcs_via_registration(wcs_list, regs, ref_i: int, shape):
    """WCS for the reference grid from the nearest frame that has one.

    Sample a grid of reference pixels, map them into frame j with the star
    registration (``ref_to_frame``), convert to the sky with frame j's WCS and
    fit a TAN-SIP WCS to those points (exact to the registration RMS)."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    import astropy.units as u
    have = [j for j, w in enumerate(wcs_list) if w is not None]
    if not have:
        return None, None
    h, w = shape
    gy, gx = np.mgrid[0:h - 1:12j, 0:w - 1:12j]
    gx, gy = gx.ravel(), gy.ravel()
    for j in sorted(have, key=lambda k: abs(k - ref_i)):
        try:
            fx, fy = regs[j].ref_to_frame(gx, gy)
            ra, dec = wcs_list[j].all_pix2world(fx, fy, 0)
            ok = np.isfinite(ra) & np.isfinite(dec)
            if ok.sum() < 20:
                continue
            sky = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)
            wr = fit_wcs_from_points((gx[ok], gy[ok]), sky, projection="TAN", sip_degree=3)
            wr.pixel_shape = (w, h)
            return wr, j
        except Exception:  # noqa: BLE001 – try the next frame
            continue
    return None, None


def _track(ctx, mm):
    f = getattr(ctx, "track_memmap", None)
    return f(mm) if f is not None else mm


def _release_memmap(mm) -> None:
    """Flush and close the OS mapping behind a numpy memmap.

    On Windows a mapped file cannot be deleted while any view of it is alive;
    closing the underlying mmap releases the file even if a stray slice remains
    (such a slice must not be used afterwards)."""
    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    base = mm
    while base is not None and not hasattr(base, "_mmap"):
        base = getattr(base, "base", None)
    m = getattr(base, "_mmap", None)
    if m is not None:
        try:
            m.close()
        except Exception:
            pass


def _safe_unlink(path: Path, ctx=None, tries: int = 5) -> None:
    """Delete a scratch file; on Windows retry briefly, then give up quietly
    (the whole work directory is removed at the end of the job anyway)."""
    import gc
    for k in range(tries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.2 * (k + 1))
    if ctx is not None:
        ctx.log(f"Pracovní soubor {path.name} nešlo hned smazat (drží ho jiný proces) – smaže se na konci úlohy.")
