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
from .astrometry import find_astap, header_wcs, pixel_rate_to_sky, solve_astap
from .fields import split_fields
from .fitsmeta import list_fits, load_image, parse_section, read_meta
from .gpu import get_backend
from .identify import match_candidates, predict_field
from .imaging import track_crops, write_candidate_images, write_motion_frames, write_motion_gif
from .instruments import InstrumentStore
from .jobs import Job, RunContext
from .masters import MasterLibrary
from .mpcdata import COMETS_NAME, MPCORB_NAME, MpcStore
from .defects import LineProfiler
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
    if len(paths) < 5:
        raise PipelineError(f"V adresáři {job.lights_dir} je jen {len(paths)} FITS snímků (min. 5)"
                            + (" pro toto zorné pole." if job.field else "."))
    metas = [read_meta(p, inst) for p in paths]
    groups = Counter((m.exptime, m.filter, m.xbin, m.ybin, m.width, m.height) for m in metas)
    key, cnt = groups.most_common(1)[0]
    if len(groups) > 1:
        ctx.warn(f"Smíšené snímky ({len(groups)} skupin) – použita největší skupina: "
                 f"{key[0]} s, filtr {key[1]}, bin {key[2]}x{key[3]} ({cnt} snímků).")
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
    dsel = lib.select_dark(ref_meta)
    fsel = lib.select_flat(ref_meta)
    for s in (dsel, fsel):
        ctx.log(f"{s.kind}: {s.path or '—'}  ← {s.reason}")
        for w in s.warnings:
            ctx.warn(w)
    if not dsel.ok:
        raise PipelineError("Master dark: " + dsel.reason)
    if not fsel.ok:
        if inst.require_flat:
            raise PipelineError("Master flat: " + fsel.reason)
        ctx.warn("Bez master flatu – spoléhá se jen na umělý flat.")
    masters = _crop_masters(cal.load_masters(dsel, fsel if fsel.ok else None), sl)
    tick("masters")

    # ---------------------------------------------------- pass 1: calibration
    ctx.stage("Kalibrace, umělý flat, detekce hvězd", 0.03)
    b = max(1, int(inst.software_binning))
    shape0 = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) if sl is not None else (ref_meta.height, ref_meta.width)
    shape = (shape0[0] // b, shape0[1] // b) if b > 1 else shape0
    norm_mm = _track(ctx, np.lib.format.open_memmap(work / "norm.npy", mode="w+", dtype=np.float32, shape=(n,) + shape))
    stars: list = [None] * n
    wcs_list: list = [None] * n
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
            if len(keep) < 5:
                raise PipelineError(f"Po vyřazení nesesazených snímků zbývá jen {len(keep)} (min. 5).")
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
    for y0 in range(0, h, 1024):
        med = SR.temporal_stats(np.array(norm_mm[:, y0:y0 + 1024, :]))[0]
        lines.add(med, y0)
        hot = (med > 6.0) if dithered else np.zeros(med.shape, bool)
        if hot.any():
            lab, nl = ndimage.label(hot)
            sizes = ndimage.sum(hot, lab, index=np.arange(1, nl + 1))
            small = np.isin(lab, np.nonzero(sizes <= 6)[0] + 1)
            small = ndimage.binary_dilation(small, iterations=1)
            for i in range(n):
                view = norm_mm[i, y0:y0 + 1024, :]
                view[small] = np.nan
                del view               # a slice keeps the file mapped (WinError 32)
            hot_total += int(small.sum())
    if dithered:
        ctx.log(f"Zamaskováno {hot_total} px pevně vázaných na čip (horké pixely, defekty).")
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
    cosmic_count = [0] * n
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

    mask_transients = bool(getattr(sp, "mask_transients", False))
    if not mask_transients:
        ctx.log("Maskování kosmiky a stop v jednotlivých snímcích je vypnuté kvůli ochraně úplnosti; "
                "jednosnímkové artefakty omezuje clipping a následné prověření kandidátů.")

    def warp_one(i):
        f = np.array(norm_mm[i])
        if mask_transients:
            cosmic_count[i] = TR.mask_cosmics(f, fw0)
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
        if sum(cosmic_count):
            ctx.log(f"Kosmické záření: zamaskováno {sum(cosmic_count)} px v {sum(1 for c in cosmic_count if c)} snímcích.")
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
    if wcs_ref is None and n_wcs:
        wcs_ref, src_j = _wcs_via_registration(wcs_list, regs, ref_i, shape)
        if wcs_ref is not None:
            ctx.log(f"Referenční snímek nemá WCS – převzat ze snímku {src_j} přes transformaci sesazení.")
    if wcs_ref is not None:
        if wcs_list[ref_i] is not None:
            ctx.log("WCS referenčního snímku převzat z hlavičky.")
    else:
        exe = find_astap(settings.astap.executable)
        if exe:
            fov_h = (inst.pixel_scale_arcsec * ref_meta.height) / 3600.0
            res = solve_astap(ref_meta.path, exe, ref_meta.ra_deg, ref_meta.dec_deg, fov_h,
                              settings.astap.search_radius_deg, settings.astap.database_dir,
                              settings.astap.timeout_s, settings.astap.sip)
            ctx.log(f"{res.message} ({res.seconds:.1f} s)")
            wcs_ref = res.wcs
            if wcs_ref is not None and sl is not None:
                wcs_ref = wcs_ref.slice(sl)
            if wcs_ref is not None and b > 1:
                wcs_ref = wcs_ref.slice((slice(None, None, b), slice(None, None, b)))
        else:
            ctx.warn("ASTAP nenalezen a snímky nemají WCS – bez RA/Dec a bez identifikace známých objektů.")
    scale = _pixel_scale(wcs_ref) or inst.pixel_scale_arcsec * b
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
                    ctx.warn(f"Část pole je v některých snímcích zakrytá nebo silně zeslabená (horizont, kopule, "
                             f"oblačnost). Místa s propustností pod {100 * sp.throughput_min:.0f} % se v těchto "
                             f"snímcích do hledání nepočítají.")

    # ---------------------------------------------------------------- search
    ctx.stage("Synthetic tracking", 0.45)
    with _quiet():
        cy, cx = h // 2, w // 2
        cen = np.nanmedian(reg_mm[:, max(0, cy - 512):cy + 512, max(0, cx - 512):cx + 512], axis=0)
    fwhm = T.estimate_fwhm(np.nan_to_num(cen))
    dts = (mjd - mjd[ref_i]) * 1440.0
    tn = SR.TrackNorm(dts, fwhm)
    vmin_phys = tn.min_speed(sp.min_efficiency)
    # Honour the requested range.  Raising the lower bound to the template's
    # preferred efficiency silently removed ordinary slow main-belt objects
    # (CTA-S1: requested 0.10, actually searched only >= 0.37 arcsec/min).
    # Low efficiency means reduced SNR, not that the object cannot be real.
    rmin_px = sp.rate_min_arcsec_min / scale
    rmax_px = sp.rate_max_arcsec_min / scale
    if vmin_phys * scale > sp.rate_min_arcsec_min:
        eff = tn.efficiency(rmin_px)
        ctx.warn(f"Rychlost {sp.rate_min_arcsec_min:.2f}″/min se hledá podle nastavení, ale za {span_min:.0f} min "
                 f"se objekt posune jen málo vůči FWHM {fwhm:.1f} px (účinnost po odečtu hvězd asi {100 * eff:.0f} %). "
                 "Citlivost na nejpomalejší objekty je nižší; pomůže delší sekvence.")
    grid = T.make_velocity_grid(rmin_px, rmax_px, span_min, sp.step_px)
    if not len(grid):
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
    if int(getattr(sp, "inject_n", 0) or 0) > 0:
        injections = INJ.plan(int(sp.inject_n), (h, w), dts, fwhm, tn, rmin_px, rmax_px, sp.inject_snr_min,
                              sp.inject_snr_max, scale, seed=int(abs(mjd[ref_i]) * 1e4) % (2 ** 31),
                              border=int(4 * fwhm) + 24)
        INJ.add_to_frames(reg_mm, injections, dts, fwhm, maps)
        reg_mm.flush()
        ctx.log(f"Injekce: do snímků přidáno {len(injections)} syntetických objektů (nominální SNR "
                f"{sp.inject_snr_min:g}–{sp.inject_snr_max:g}, {rmin_px * scale:.2f}–{rmax_px * scale:.2f}″/min). "
                f"Nalezené se v seznamu kandidátů neukazují (filtr „injekce“).")

    def tile_one(ti):
        y0, x0 = tiles[ti]
        Y0, X0 = max(0, y0 - margin), max(0, x0 - margin)
        Y1, X1 = min(h, y0 + tile + margin), min(w, x0 + tile + margin)
        block = np.array(reg_mm[:, Y0:Y1, X0:X1])
        if np.isfinite(block).sum() < 0.2 * block.size:
            return []
        # the PSF of wide-field optics changes over the field: FWHM per tile
        fw_t = T.estimate_fwhm(np.nan_to_num(SR.temporal_stats(block)[0]))
        if not (0.6 * fwhm < fw_t < 1.8 * fwhm):
            fw_t = fwhm
        amp_t = maps.tile(Y0, Y1, X0, X1) if maps is not None else None
        pt = SR.prepare_tile(block, fw_t, sp.bright_star_mask_sigma, cap=sp.frame_cap_sigma,
                             amp=amp_t, amp_min=sp.throughput_min)
        tn_t = SR.TrackNorm(dts, fw_t, pt.noise_sigma)
        with gpu_lock:                      # one GPU search at a time, CPU preparation overlaps
            best, bidx = SR.search_tile(be, pt.f, dts, grid, tn_t, should_stop=ctx.should_stop)
        core = (slice(y0 - Y0, min(y0 + tile, h) - Y0), slice(x0 - X0, min(x0 + tile, w) - X0))
        edge = int(2 * fwhm) + 2
        # coarse grid loses up to ~1σ: collect peaks 1σ below the threshold, refine, then cut
        cc = SR.peaks(best, bidx, grid, sp.snr_threshold - 1.0, fw_t, core, sp.max_candidates)
        del best, bidx
        cc = [c for c in cc if edge <= c.x + X0 < w - edge and edge <= c.y + Y0 < h - edge]
        SR.refine(cc, pt.f, dts, grid.dv, tn_t, min_speed=rmin_px, max_speed=rmax_px * 1.1)
        cc = [c for c in cc if c.snr >= sp.snr_threshold]
        # velocity refinement on the un-clipped frames: clipping flattens bright objects
        SR.refine(cc, pt.f_raw, dts, grid.dv / 2, tn_t, min_speed=rmin_px, max_speed=rmax_px * 1.1)
        SR.vet(cc, pt, dts, tn_t, sp.segments, sp.segment_z_max, sp.bright_star_mask_sigma, fw_t)
        size = int(max(48, min(96, 14 * fwhm)))
        HARD = {"nekonzistentní segmenty", "jednosnímkový artefakt", "reziduum hvězdy", "mimo pole v části snímků",
                "nekonzistentní snímky", "bez směru pohybu"}
        for k, c in enumerate(cc):
            c.fwhm = round(fw_t, 2)
            # "reziduum hvězdy" (slow track over a bright star) is decided by the mover test:
            # a real asteroid passing a star (Adelinda, 305/field_01) is clearly a mover
            if "reziduum hvězdy" in c.flags and not (HARD - {"reziduum hvězdy"}) & set(c.flags):
                tsv0 = SR.static_vs_mover(block, dts, c, fw_t)
                if tsv0 is not None and tsv0 >= sp.star_rescue_mover:
                    c.flags.remove("reziduum hvězdy")
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
                            c.x, c.y, c.vx, c.vy = x0f, y0f, vxf, vyf
                            fitted = True
                        else:
                            c.flags.append("astrometrie nejistá")
                c.positions = [{"dt_min": round(p[0], 4), "x": p[1] + X0, "y": p[2] + Y0, "flux": round(p[3], 1),
                                "ok": p[4]} for p in pts]
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
                        c.flags.append("pomalý u hvězdy")
                        c.auto_status = "auto_rejected"
            if "astrometrie nejistá" in c.flags:
                c.auto_status = "auto_rejected"        # fitted track loses the signal: not a coherent mover
            if c.throughput_low_frac is not None and c.throughput_low_frac >= 0.25 and c.snr < 10:
                c.flags.append("zakryté místo")
                c.auto_status = "auto_rejected"
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
            # Thousands of variants of a broad one-frame artifact used to
            # spend minutes encoding identical GIFs. Keep every numeric
            # detection for audit, but skip eager previews only when several
            # independent consistency tests all prove it is not a mover.
            artifact_proofs = {"jednosnímkový artefakt", "nekonzistentní segmenty",
                               "nekonzistentní snímky", "bez směru pohybu"} & set(c.flags)
            obvious_artifact = ("jednosnímkový artefakt" in artifact_proofs
                                and len(artifact_proofs) >= 3 and c.auto_status != "candidate")
            if obvious_artifact:
                c.preview_skipped = "náhled nevytvořen: jednoznačný jednosnímkový artefakt"
            else:
                # stacks of the registered frames (stars visible) + SNR map along the track
                snr_map = (track_crops(pt.f_raw, dts, c.x, c.y, c.vx, c.vy, size).sum(axis=0)
                           / tn_t.sigma(math.hypot(c.vx, c.vy)))
                c.images = write_candidate_images(img_dir, f"t{ti:03d}_{k:04d}",
                                                  track_crops(block, dts, c.x, c.y, c.vx, c.vy, size),
                                                  sp.segments, diff_crops=snr_map, fwhm=fw_t)
                c.images["motion"] = write_motion_gif(img_dir, f"t{ti:03d}_{k:04d}", block, dts,
                                                      c.x, c.y, c.vx, c.vy, size, fwhm=fw_t)
                c.images["frames"] = write_motion_frames(img_dir, f"t{ti:03d}_{k:04d}", block, dts,
                                                         c.x, c.y, c.vx, c.vy, size, amp=amp_t)
            if c.static_star_x is not None:
                c.static_star_x += X0
                c.static_star_y += Y0
            c.x += X0
            c.y += Y0
        ctx.log(f"  dlaždice {ti + 1}/{len(tiles)}: {len(cc)} detekcí, {sum(c.auto_status == 'candidate' for c in cc)} prošlo")
        return cc

    import threading
    gpu_lock = threading.Lock() if be.is_gpu else _NoLock()
    per_tile_mb = (11 if maps is not None else 10) * n * (tile + 2 * margin) ** 2 * 4 / 2**20
    results = _parallel(tile_one, range(len(tiles)), _workers(settings, per_task_mb=per_tile_mb), ctx, 0.45, 0.4)
    cands: list[T.Candidate] = [c for r in results for c in (r or [])]
    for i, c in enumerate(cands):
        c.id = i
    cands = _dedupe_keep_images(cands, fwhm, span_min)
    ctx.log(f"Celkem {len(cands)} detekcí nad SNR {sp.snr_threshold}; "
            f"{sum(c.auto_status == 'candidate' for c in cands)} prošlo automatickým prověřením")
    tick("search")
    known_rejected: list = []
    inj_summary = None
    if injections:
        INJ.match(cands, injections, dts, fwhm)
        inj_summary, lines = INJ.summarise(injections, sp.snr_threshold)
        for line in lines:
            ctx.log(line)

    # ------------------------------------------------------------ candidates
    if wcs_ref is not None:
        for c in cands:
            c.ra, c.dec, c.rate_arcsec_min, c.pa_deg = pixel_rate_to_sky(wcs_ref, c.x, c.y, c.vx, c.vy)
            c.sky_matrix = _local_sky_matrix(wcs_ref, c.x, c.y, c.ra, c.dec)
            for p in c.positions:
                sky = wcs_ref.pixel_to_world(p["x"], p["y"])
                p["ra"], p["dec"] = float(sky.ra.deg), float(sky.dec.deg)
                p["mjd"] = float(mjd[ref_i] + p["dt_min"] / 1440.0)
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
        mp, cm = mstore.load(MPCORB_NAME), mstore.load(COMETS_NAME)
        if mp is None and cm is None:
            ctx.warn("MPCORB ani CometEls nejsou stažené – identifikace přeskočena.")
        else:
            cen_sky = wcs_ref.pixel_to_world((w - 1) / 2, (h - 1) / 2)
            radius = max(cen_sky.separation(wcs_ref.pixel_to_world(px, py)).deg
                         for px, py in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))) * 1.02
            t_ref = Time(mjd[ref_i], format="mjd", scale="utc")
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
            known_list = [k for k in known_list if k.x is not None]
            # wide-field: the tolerance must cover ~2.5 px of astrometric error
            # A faint/slow mover has only a ~2 px trail over this sequence and
            # the three-substack astrometric fit can move either endpoint by
            # about one pixel.  Use the resulting 2 px velocity uncertainty;
            # the full vector is still compared, so an opposite direction is
            # never accepted merely because the speeds are equal.
            match_candidates(cands, known_list, max(sp.identify_tolerance_arcsec, 2.5 * scale),
                             rate_tol_abs=max(0.05, 2.0 * scale / max(span_min, 1.0)))
            # forced measurement on the predicted track of every known object
            todo = [k for k in known_list if k.mag is None or k.mag <= sp.forced_mag_limit]

            def forced_one(j):
                k = todo[j]
                k.forced_snr, k.forced_dxy, k.forced_note = _forced(reg_mm, k.x, k.y, _known_pixel_velocity(wcs_ref, k),
                                                                    dts, fwhm, tn, sp, maps=maps)
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
    tick("identify")

    # --------------------------------------------------------------- outputs
    ctx.stage("Výstupy", 0.96)
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
    shutil.rmtree(work, ignore_errors=True)
    timings["total"] = round(time.time() - t_start, 2)
    job.summary = {
        "n_frames": n, "exptime": key[0], "filter": key[1], "span_min": round(span_min, 2),
        "fwhm_px": round(fwhm, 2), "pixel_scale": round(scale, 4), "grid_size": len(grid),
        "min_rate_arcsec_min": round(rmin_px * scale, 3),
        "max_rate_arcsec_min": round(rmax_px * scale, 3),
        "backend": be.name, "detections": len(cand_dicts),
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
        "injection": inj_summary,
        "masters": {"dark": dsel.to_dict(), "flat": fsel.to_dict()},
        "wcs": wcs_ref is not None,
        "site": [lat, lon, alt], "site_source": site_source, "observatory_code": obs_code,
        "field": job.field or None,
        "mjd_ref": float(mjd[ref_i]),
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


def _known_pixel_velocity(wcs, k) -> tuple[float, float]:
    """Pixel velocity (px/min) of a known object from its rate and position angle."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c0 = SkyCoord(k.ra * u.deg, k.dec * u.deg)
    c1 = c0.directional_offset_by(k.pa_deg * u.deg, (k.rate_arcsec_min * 10) * u.arcsec)
    x0, y0 = wcs.world_to_pixel(c0)
    x1, y1 = wcs.world_to_pixel(c1)
    return (float(x1 - x0) / 10.0, float(y1 - y0) / 10.0)


def _forced(reg_mm, x, y, v, dts, fwhm, tn, sp, half=96, search_px=1.0, maps=None):
    """SNR on the predicted track of a known object (best within ±search_px) and a note.

    The search box is small (predictions are good to < 1 px); a ±2 px box in a
    dense field found star residuals with SNR 30–50 for 18th-mag objects.
    The note says why the value may be unreliable."""
    n, h, w = reg_mm.shape
    xi, yi = int(round(x)), int(round(y))
    Y0, Y1, X0, X1 = max(0, yi - half), min(h, yi + half), max(0, xi - half), min(w, xi + half)
    block = np.array(reg_mm[:, Y0:Y1, X0:X1])
    if block.shape[1] < 32 or block.shape[2] < 32:
        return None, None, "mimo pole"
    amp = maps.tile(Y0, Y1, X0, X1) if maps is not None else None
    pt = SR.prepare_tile(block, fwhm, sp.bright_star_mask_sigma, cap=0, amp=amp, amp_min=sp.throughput_min)
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
        return None, None, "mimo pole v části snímků"
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
    if masked >= 2 and self_like:
        note = "pomalý zdroj ve statické šabloně"
    elif masked >= 2:
        note = "jasná hvězda na stopě"
    elif near > 0.5 * sp.bright_star_mask_sigma:
        note = "hvězda u stopy"
    elif max(abs(bxy[0]), abs(bxy[1])) >= search_px - 1e-6:
        note = "maximum na okraji hledání"
    return round(best, 2), [round(bxy[0], 2), round(bxy[1], 2)], note


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
