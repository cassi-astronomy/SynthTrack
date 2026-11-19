"""Injection–recovery: synthetic movers added to the registered frames.

Used for (a) completeness and false-detection rates per instrument and
sequence and (b) positive training examples for the candidate classifier
(the recovered injections go through exactly the same search and vetting as
real objects). Injected objects are recorded in ``injections.json``; the
detections that recover them carry ``injected = <id>`` and are hidden from
the normal candidate list.

The flux is chosen for a *nominal* SNR of the optimal stack:
    SNR = a · expected(v) / σ_sum(v),   a = F / (2√π σ_psf)
(a = matched-filter amplitude per frame in unit noise, see ``search.TrackNorm``).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def empirical_psf(frame: np.ndarray, stars: np.ndarray, fwhm: float, max_n: int = 150,
                  min_n: int = 15) -> np.ndarray | None:
    """Median profile of isolated, unsaturated stars of one registered frame (sum = 1).

    Injected Gaussians differ from real point sources (real profile: narrower core,
    broader wings); a classifier trained on them learnt "Gaussian = real" and put all
    3 identified asteroids of the 29P field under its cut (7. 11. 2026). ``stars``:
    (N, 3) x, y, flux in the frame's coordinates. None if too few usable stars."""
    if stars is None or len(stars) < min_n:
        return None
    h, w = frame.shape
    r = int(math.ceil(2.5 * fwhm)) + 3
    pad = 2
    xs, ys, fl = (np.asarray(stars[:, k], float) for k in range(3))
    order = np.argsort(fl)[::-1]
    sel = order[int(0.05 * len(order)): int(0.6 * len(order))]          # skip the brightest (saturation)
    others = np.stack([xs, ys], 1)
    yy, xx = np.mgrid[-r - pad:r + pad + 1, -r - pad:r + pad + 1]
    ring = (np.hypot(xx, yy) > r) & (np.hypot(xx, yy) <= r + pad)
    inner = np.hypot(xx, yy) <= r
    cut = []
    for j in sel:
        x, y = xs[j], ys[j]
        xi, yi = int(round(x)), int(round(y))
        if xi - r - pad < 0 or yi - r - pad < 0 or xi + r + pad >= w or yi + r + pad >= h:
            continue
        d = np.hypot(others[:, 0] - x, others[:, 1] - y)
        close = (d > 0.5) & (d < 1.5 * r) & (fl > 0.03 * fl[j])        # a noticeable neighbour in the stamp
        if close.any():
            continue
        c = np.array(frame[yi - r - pad:yi + r + pad + 1, xi - r - pad:xi + r + pad + 1], np.float64)
        if not np.isfinite(c).all():
            continue
        c -= np.median(c[ring])
        pk = c[r + pad, r + pad]
        if not (15.0 < pk < 3000.0):
            continue
        # refine the centre (first moment of the core), centre on the integer pixel
        core = (np.hypot(xx, yy) <= max(2.0, 0.8 * fwhm)) & (c > 0)
        wsum = c[core].sum()
        if wsum <= 0:
            continue
        fx, fy = float((c[core] * xx[core]).sum() / wsum), float((c[core] * yy[core]).sum() / wsum)
        if abs(fx) > 1.0 or abs(fy) > 1.0:
            continue
        c = ndimage.shift(c, (-fy, -fx), order=3, mode="nearest")
        tot = c[inner].sum()
        if tot <= 0:
            continue
        cut.append(c / tot)
        if len(cut) >= max_n:
            break
    if len(cut) < min_n:
        return None
    psf = np.median(np.array(cut), axis=0)[pad:-pad, pad:-pad]
    yy2, xx2 = np.mgrid[-r:r + 1, -r:r + 1]
    psf[np.hypot(xx2, yy2) > r] = 0.0
    psf = np.clip(psf, 0.0, None)
    return (psf / psf.sum()).astype(np.float32)


def psf_gain(psf: np.ndarray | None, fwhm: float) -> float:
    """Matched-filter amplitude per unit flux: Σ psf·g / ‖g‖ (Gaussian g of ``fwhm``)."""
    sp = max(fwhm / 2.3548, 0.5)
    if psf is None:
        return 1.0 / (2.0 * math.sqrt(math.pi) * sp)
    r = psf.shape[0] // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sp * sp))
    return float((psf * g).sum() / math.sqrt((g * g).sum()))


def comet_profile(coma_px: float, nucleus_frac: float, fwhm: float, psf: np.ndarray | None = None) -> np.ndarray:
    """Stamp (sum 1) of a comet: point-like nucleus + coma with surface brightness ∝ 1/ρ.

    The coma is the steady-state 1/ρ profile, softly truncated at ``coma_px``
    (exp(−(ρ/R)²)), convolved with the stars' profile; ``nucleus_frac`` of the light is a
    point source. From strongly condensed (R a few arcsec, half of the light in the
    nucleus) to very diffuse comae of 1–2′ without a strong central condensation."""
    from scipy.signal import fftconvolve
    sp = max(fwhm / 2.3548, 0.5)
    if psf is None:
        rp = int(math.ceil(4 * sp)) + 1
        yy, xx = np.mgrid[-rp:rp + 1, -rp:rp + 1]
        psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sp * sp))
        psf = psf / psf.sum()
    rp = psf.shape[0] // 2
    R = max(float(coma_px), 0.5 * fwhm)
    rc = int(math.ceil(2.2 * R)) + rp
    yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
    rho = np.hypot(xx, yy)
    coma = np.exp(-(rho / R) ** 2) / np.maximum(rho, 0.5)
    coma = fftconvolve(coma / coma.sum(), psf, mode="same")
    coma = np.clip(coma, 0.0, None)
    out = (1.0 - nucleus_frac) * coma / coma.sum()
    out[rc - rp:rc + rp + 1, rc - rp:rc + rp + 1] += nucleus_frac * psf / psf.sum()
    return (out / out.sum()).astype(np.float32)


def profile_gain(prof: np.ndarray, fwhm: float) -> float:
    """Matched-filter amplitude per unit flux of any stamp (Gaussian filter of ``fwhm``)."""
    sp = max(fwhm / 2.3548, 0.5)
    r = prof.shape[0] // 2
    rg = int(math.ceil(4 * sp)) + 1
    yy, xx = np.mgrid[-rg:rg + 1, -rg:rg + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sp * sp))
    core = prof[r - rg:r + rg + 1, r - rg:r + rg + 1] if r >= rg else np.pad(prof, rg - r)
    return float((core * g).sum() / math.sqrt((g * g).sum()))


def plan(n_obj: int, shape: tuple[int, int], dts: np.ndarray, fwhm: float, tn, rmin_px: float,
         rmax_px: float, snr_min: float, snr_max: float, scale_arcsec: float, seed: int = 0,
         border: int = 40, velocities: list | None = None, field_fraction: float = 0.5,
         gain: float | None = None, comet_fraction: float = 0.0, psf: np.ndarray | None = None,
         coma_arcsec: tuple[float, float] = (4.0, 120.0)) -> list[dict]:
    """Random positions, speeds/directions and nominal SNRs.

    ``velocities``: (vx, vy) px/min of the known objects in the field; ``field_fraction``
    of the injections take one of them (±15 % speed, ±10° direction), the rest are
    log-uniform in speed and uniform in direction. Real asteroids of a field share
    a typical motion, which may coincide with the dither drift (29P) – random
    directions alone do not teach that."""
    rng = np.random.default_rng(seed)
    h, w = shape
    sp = max(fwhm / 2.3548, 0.5)
    t0, t1 = float(np.min(dts)), float(np.max(dts))
    out = []
    vel = [(float(a), float(b)) for a, b in (velocities or [])
           if rmin_px <= math.hypot(a, b) <= rmax_px]
    for k in range(int(n_obj)):
        if vel and rng.uniform() < field_fraction:
            a, b = vel[int(rng.integers(len(vel)))]
            v = math.hypot(a, b) * float(rng.uniform(0.85, 1.15))
            pa = math.atan2(b, a) + math.radians(float(rng.uniform(-10, 10)))
            v = min(max(v, rmin_px), rmax_px)
            src = "field"
        else:
            v = float(math.exp(rng.uniform(math.log(max(rmin_px, 1e-4)), math.log(max(rmax_px, rmin_px * 1.01)))))
            pa = float(rng.uniform(0, 2 * math.pi))
            src = "random"
        vx, vy = v * math.cos(pa), v * math.sin(pa)
        # the whole track has to stay inside the frame
        xs = [vx * t0, vx * t1]
        ys = [vy * t0, vy * t1]
        x = float(rng.uniform(border - min(xs), w - border - max(xs)))
        y = float(rng.uniform(border - min(ys), h - border - max(ys)))
        snr = float(rng.uniform(snr_min, snr_max))
        a = snr * tn.sigma(v) / max(tn.expected(v), 1e-6)
        extra = {"kind": "asteroid"}
        g_k = gain
        if comet_fraction > 0 and rng.uniform() < comet_fraction:
            # diffuseness u: 0 = condensed (small coma, bright nucleus) … 1 = diffuse 1–2′ coma
            u = float(rng.uniform())
            lo, hi = coma_arcsec
            coma = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            coma_px = min(coma / max(scale_arcsec, 1e-6), 80.0)
            nuc = round(0.6 * (1.0 - u) ** 2, 3)
            g_k = profile_gain(comet_profile(coma_px, nuc, fwhm, psf), fwhm)
            extra = {"kind": "comet", "coma_arcsec": round(coma, 1), "coma_px": round(coma_px, 2),
                     "nucleus_frac": nuc}
        # nominal SNR = what the (stellar) matched filter sees: a diffuse comet needs far more light
        flux = a / g_k if g_k else a * 2.0 * math.sqrt(math.pi) * sp
        out.append({"id": k, "x": round(x, 3), "y": round(y, 3), "vx": vx, "vy": vy, "velocity": src,
                    "rate_arcsec_min": round(v * scale_arcsec, 4),
                    "pa_px_deg": round(math.degrees(pa) % 360, 1), "snr_nominal": round(snr, 2),
                    "flux_sigma": round(flux, 2), "detected": None, "passed": False, **extra})
    return out


def plan_static(n_obj: int, shape: tuple[int, int], n_frames: int, fwhm: float, snr_min: float,
                snr_max: float, scale_arcsec: float, seed: int = 0, border: int = 100,
                psf: np.ndarray | None = None, comet_fraction: float = 0.5,
                coma_arcsec: tuple[float, float] = (10.0, 120.0)) -> list[dict]:
    """Static sources for the catalogue comparison ("zdroj navíc"): point sources (a nova, a comet
    in outburst, a supernova) and comets with a coma of ``coma_arcsec`` that do not move during
    the series. Nominal SNR = the stellar matched filter on the mean of ``n_frames`` frames
    (σ units per frame): flux = SNR / (gain · √n). Log-uniform in SNR (faint ones matter most)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    gain_pt = psf_gain(psf, fwhm)
    out = []
    for k in range(int(n_obj)):
        x = float(rng.uniform(border, w - border))
        y = float(rng.uniform(border, h - border))
        snr = float(math.exp(rng.uniform(math.log(snr_min), math.log(snr_max))))
        extra = {"kind": "point"}
        g_k = gain_pt
        if comet_fraction > 0 and rng.uniform() < comet_fraction:
            u = float(rng.uniform())
            lo, hi = coma_arcsec
            coma = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            coma_px = min(coma / max(scale_arcsec, 1e-6), 80.0)
            nuc = round(0.4 * (1.0 - u) ** 2, 3)
            g_k = profile_gain(comet_profile(coma_px, nuc, fwhm, psf), fwhm)
            extra = {"kind": "comet", "coma_arcsec": round(coma, 1), "coma_px": round(coma_px, 2),
                     "nucleus_frac": nuc}
        flux = snr / max(g_k * math.sqrt(max(n_frames, 1)), 1e-9)
        out.append({"id": k, "x": round(x, 3), "y": round(y, 3), "vx": 0.0, "vy": 0.0, "velocity": "static",
                    "rate_arcsec_min": 0.0, "pa_px_deg": 0.0, "snr_nominal": round(snr, 2),
                    "flux_sigma": round(flux, 3), "detected": None, "passed": False, **extra})
    return out


def summarise_static(objs: list[dict], dets: list[dict]) -> tuple[dict, list[str]]:
    """Recovery of static injections by the catalogue comparison: per kind and SNR bin, and the
    rules that rejected found ones (the same rules reject a real nova or comet)."""
    from collections import Counter
    by = {}
    for d in dets:                         # a passed match wins over a rejected one at the same injection
        if d.get("injected") is not None and (d["injected"] not in by or d.get("auto_status") == "candidate"):
            by[d["injected"]] = d
    why = Counter()
    for o in objs:
        d = by.get(o["id"])
        o["detected"] = None if d is None else d.get("id", True)
        o["passed"] = bool(d is not None and d.get("auto_status") == "candidate")
        if d is not None:
            o["detected_snr"] = round(float(d.get("snr") or 0), 2)
            o["detected_mag"] = d.get("mag")
            if not o["passed"]:
                why.update(f.split(" (")[0] for f in d.get("flags", [])[1:]
                           if not f.startswith(("Twilight:", "mlhavý objekt (σ")))
    key = "snr_expected" if objs and all("snr_expected" in o for o in objs) else "snr_nominal"
    edges = [0, 8, 12, 20, 1e9]
    lines = [f"Injekce (zdroj navíc): {len(objs)} statických zdrojů, nalezeno "
             f"{sum(o['detected'] is not None for o in objs)}, prošlo {sum(o['passed'] for o in objs)}."]
    kinds = {}
    for kind, label in (("point", "bodové (nova, kometa v outburstu)"), ("comet", "komety s komou"),
                        ("sn", "supernovy v galaxiích")):
        sel = [o for o in objs if o.get("kind") == kind]
        if not sel:
            continue
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = [o for o in sel if lo <= o[key] < hi]
            if b:
                bins.append({"snr_from": lo, "snr_to": None if hi > 1e8 else hi, "n": len(b),
                             "detected": sum(o["detected"] is not None for o in b),
                             "passed": sum(o["passed"] for o in b)})
        mags = sorted(o["detected_mag"] for o in sel if o["passed"] and o.get("detected_mag") is not None)
        kinds[kind] = {"n": len(sel), "detected": sum(o["detected"] is not None for o in sel),
                       "passed": sum(o["passed"] for o in sel), "bins": bins,
                       "faintest_passed_mag": mags[-1] if mags else None}
        lines.append(f"  {label}: nalezeno {kinds[kind]['detected']}/{len(sel)}, prošlo {kinds[kind]['passed']}/{len(sel)}"
                     + (f", nejslabší prošlý ≈ {mags[-1]:.1f} mag" if mags else ""))
        for b in bins:
            hi = "∞" if b["snr_to"] is None else f"{b['snr_to']:g}"
            lines.append(f"    SNR {b['snr_from']:g}–{hi}: nalezeno {b['detected']}/{b['n']}, prošlo {b['passed']}/{b['n']}")
    if why:
        lines.append("  nalezené, ale zamítnuté injekce – důvody: "
                     + ", ".join(f"{k} {v}×" for k, v in why.most_common(6))
                     + " (stejná pravidla by zamítla i skutečnou novu / kometu)")
    return {"n": len(objs), "detected": sum(o["detected"] is not None for o in objs),
            "passed": sum(o["passed"] for o in objs), "kinds": kinds, "static": True,
            "rejected_by": dict(why.most_common(10)), "snr_basis": "expected" if key == "snr_expected" else "nominal"}, lines


def add_to_frames(reg, objs: list[dict], dts: np.ndarray, fwhm: float, maps=None,
                  psf: np.ndarray | None = None) -> None:
    """Add the star profile ``psf`` (or a Gaussian) with total flux ``flux_sigma`` along every track."""
    n, h, w = reg.shape
    sp = max(fwhm / 2.3548, 0.5)
    r0 = psf.shape[0] // 2 if psf is not None else int(math.ceil(4 * sp)) + 1
    comets = {o["id"]: comet_profile(o["coma_px"], o["nucleus_frac"], fwhm, psf)
              for o in objs if o.get("kind") == "comet"}
    tp2 = [[] for _ in objs]
    for i in range(n):
        frame = reg[i]
        for k, o in enumerate(objs):
            x = o["x"] + o["vx"] * float(dts[i])
            y = o["y"] + o["vy"] * float(dts[i])
            xi, yi = int(round(x)), int(round(y))
            prof = comets.get(o["id"]) if comets else None
            r = prof.shape[0] // 2 if prof is not None else r0
            y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            if prof is not None or psf is not None:
                src = prof if prof is not None else psf
                st = ndimage.shift(src, (y - yi, x - xi), order=3, mode="constant", cval=0.0)
                g = st[y0 - (yi - r):y1 - (yi - r), x0 - (xi - r):x1 - (xi - r)]
            else:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sp * sp)) / (2 * math.pi * sp * sp)
            amp = 1.0
            if maps is not None:
                amp = float(maps.tile(yi, yi + 1, xi, xi + 1, frames=[i])[0, 0, 0])
            frame[y0:y1, x0:x1] += np.float32(amp * o["flux_sigma"]) * g.astype(np.float32)
            tp2[k].append(amp * amp)
        del frame
    # expected SNR with the real throughput along the track: the frames are weighted by their
    # throughput A_i (optimal sum), so SNR = nominal · √(Σ A_i² / n). The nominal SNR assumes a
    # clear sky in every frame – at dawn (Coihueco 317, frames down to 31 %) it is ~0.8 × nominal.
    if maps is not None:
        for o, a2 in zip(objs, tp2):
            if a2:
                o["snr_expected"] = round(o["snr_nominal"] * math.sqrt(sum(a2) / n), 2)


def match(cands, objs: list[dict], dts: np.ndarray, fwhm: float) -> None:
    """Mark detections that recover an injected object (position at the reference
    epoch within max(2, FWHM) px and the track end within max(3, 2·FWHM) px)."""
    span = float(np.max(np.abs(dts)))
    tol_p = max(2.0, fwhm)
    tol_v = max(3.0, 2 * fwhm) / max(span, 1e-6)
    for o in objs:
        best, bd = None, 1e9
        for c in cands:
            d = math.hypot(c.x - o["x"], c.y - o["y"])
            dv = math.hypot(c.vx - o["vx"], c.vy - o["vy"])
            if d < tol_p and dv < tol_v and d < bd:
                best, bd = c, d
        if best is not None:
            best.injected = o["id"]
            o["detected"] = best.id
            o["detected_snr"] = round(float(best.snr), 2)
            o["passed"] = best.auto_status == "candidate"


def summarise(objs: list[dict], threshold: float) -> tuple[dict, list[str]]:
    """Recovery per nominal-SNR bin; Czech log lines."""
    edges = [0, 6, 8, 10, 12, 15, 1e9]
    # with throughput maps the bins use the SNR expected under the real sky (twilight, haze)
    key = "snr_expected" if objs and all("snr_expected" in o for o in objs) else "snr_nominal"
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [o for o in objs if lo <= o[key] < hi]
        if not sel:
            continue
        bins.append({"snr_from": lo, "snr_to": None if hi > 1e8 else hi, "n": len(sel),
                     "detected": sum(o["detected"] is not None for o in sel),
                     "passed": sum(bool(o["passed"]) for o in sel)})
    det = sum(o["detected"] is not None for o in objs)
    pas = sum(bool(o["passed"]) for o in objs)
    lines = [f"Injekce: {len(objs)} syntetických objektů, nalezeno {det}, z toho {pas} prošlo prověřením "
             f"(práh SNR {threshold:g})."]
    com = [o for o in objs if o.get("kind") == "comet"]
    comets = None
    if com:
        diffuse = [o for o in com if o.get("coma_arcsec", 0) >= 30]
        comets = {"n": len(com), "detected": sum(o["detected"] is not None for o in com),
                  "passed": sum(bool(o["passed"]) for o in com),
                  "diffuse_n": len(diffuse), "diffuse_passed": sum(bool(o["passed"]) for o in diffuse)}
        lines.append(f"  z toho komet {len(com)} (koma 4–120″): nalezeno {comets['detected']}, prošlo "
                     f"{comets['passed']}; difúzních (koma ≥ 30″) prošlo {comets['diffuse_passed']}/{len(diffuse)}")
    for b in bins:
        hi = "∞" if b["snr_to"] is None else f"{b['snr_to']:g}"
        lines.append(f"  {'očekávané' if key == 'snr_expected' else 'nominální'} SNR {b['snr_from']:g}–{hi}: nalezeno {b['detected']}/{b['n']}, "
                     f"prošlo {b['passed']}/{b['n']}")
    if key == "snr_expected":
        r = sorted(o["snr_expected"] / o["snr_nominal"] for o in objs if o["snr_nominal"] > 0)
        if r:
            lines.insert(1, f"  propustnost snižuje SNR injekcí na {r[len(r) // 2]:.2f} × nominální "
                            f"(rozsah {r[0]:.2f}–{r[-1]:.2f}); třídy níže podle očekávaného SNR")
    return {"n": len(objs), "detected": det, "passed": pas, "bins": bins, "comets": comets,
            "snr_basis": "expected" if key == "snr_expected" else "nominal"}, lines
