"""Supernovae in galaxies without a reference image (a side search of "zdroj navíc").

A supernova in a galaxy is rejected by the general catalogue comparison (the galaxy is a "DSO" or a
PGC galaxy that "explains" the light). Here every catalogued galaxy of the field that the frames can
see is examined on its own:

1. catalogue stars (UCAC4 + bright stars) around it are subtracted with their expected flux;
2. the galaxy is subtracted from itself turned by 180° about its catalogued nucleus: a galaxy is
   roughly point-symmetric, a supernova off the nucleus is not – it leaves a positive point (and a
   negative one in the mirrored place) even when it is much fainter than the galaxy;
3. the residual is filtered with the stars' profile; a peak far above the noise *of this galaxy's own
   asymmetry* (spiral arms, dust lanes) that is point-like and present in every frame is reported;
4. a supernova on the nucleus (within ~1 px) leaves no asymmetry: only a galaxy much brighter than its
   catalogue magnitude can show it (OpenNGC magnitudes only; PGC has none here).

At 14″/px most galaxies are a few pixels: a supernova close to the nucleus of a small galaxy cannot
be told from it. Injected supernovae (``plan_injections``) measure what is found.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def galaxies(pgc: dict | None, dso: dict | None, to_pix, shape, scale_as: float, fwhm: float,
             lim_mag: float, mag_of_d25, border: int = 40, max_n: int = 3000) -> list[dict]:
    """Catalogued galaxies inside the frame that the frames could see (estimated mag ≤ limit + 1).
    PGC (positions, D25) and OpenNGC galaxies (mag, axes) merged by position."""
    h, w = shape
    out = []
    if pgc is not None and len(pgc.get("ra", [])):
        ra, de, d25 = (np.asarray(pgc[k], float) for k in ("ra", "dec", "d25"))
        m = np.array([mag_of_d25(float(v)) for v in d25])
        sel = np.nonzero(m <= lim_mag + 1.0)[0]
        if len(sel):
            x, y = to_pix(ra[sel], de[sel])
            for i, xi, yi in zip(sel, x, y):
                if np.isfinite(xi) and border <= xi < w - border and border <= yi < h - border:
                    nm = f"PGC{int(pgc['pgc'][i])}" if np.isfinite(pgc["pgc"][i]) else "PGC"
                    out.append({"name": nm, "ra": float(ra[i]), "dec": float(de[i]), "x": float(xi), "y": float(yi),
                                "d25": float(d25[i]) if np.isfinite(d25[i]) else 0.3, "mag_est": float(m[i]),
                                "mag_cat": None})
    if dso:
        typ = np.asarray(dso.get("type", []))
        isg = np.array([str(t).startswith("G") for t in typ])
        if isg.any():
            idx = np.nonzero(isg)[0]
            x, y = to_pix(np.asarray(dso["ra"], float)[idx], np.asarray(dso["dec"], float)[idx])
            for i, xi, yi in zip(idx, x, y):
                if not (np.isfinite(xi) and border <= xi < w - border and border <= yi < h - border):
                    continue
                mg = float(dso["mag"][i]) if np.isfinite(dso["mag"][i]) else None
                maj = float(dso["maj"][i]) if np.isfinite(dso["maj"][i]) else None
                hit = next((g for g in out if math.hypot(g["x"] - xi, g["y"] - yi) < max(1.5, 20 / scale_as)), None)
                if hit is not None:
                    hit["name"] = f"{dso['name'][i]} = {hit['name']}"
                    hit["mag_cat"] = mg
                    if maj:
                        hit["d25"] = max(hit["d25"], maj)
                elif mg is not None and mg <= lim_mag + 1.0:
                    out.append({"name": str(dso["name"][i]), "ra": float(dso["ra"][i]), "dec": float(dso["dec"][i]),
                                "x": float(xi), "y": float(yi), "d25": maj or 0.5, "mag_est": mg, "mag_cat": mg})
    out.sort(key=lambda g: g["mag_cat"] if g["mag_cat"] is not None else g["mag_est"])
    return out[:max_n]


def radius_px(g: dict, scale_as: float, fwhm: float) -> float:
    """Search radius: the visible disc (~0.75 D25/2) plus one FWHM, at least 2 FWHM."""
    return max(2.0 * fwhm, 0.75 * 0.5 * g["d25"] * 60.0 / scale_as + fwhm)


def plan_injections(gals: list[dict], n: int, n_frames: int, fwhm: float, scale_as: float, snr_min: float,
                    snr_max: float, gain: float, seed: int = 0) -> list[dict]:
    """Point sources placed in or next to catalogued galaxies: 0.3–1.2 × the galaxy's search radius
    from the nucleus, nominal SNR log-uniform (as the other static injections)."""
    rng = np.random.default_rng(seed)
    if not gals:
        return []
    out = []
    pick = rng.choice(len(gals), size=min(n, len(gals)), replace=len(gals) < n)
    for k, gi in enumerate(pick):
        g = gals[int(gi)]
        r = radius_px(g, scale_as, fwhm) * float(rng.uniform(0.3, 1.0))
        r = max(r, 0.6 * fwhm)
        a = float(rng.uniform(0, 2 * math.pi))
        snr = float(math.exp(rng.uniform(math.log(snr_min), math.log(snr_max))))
        flux = snr / max(gain * math.sqrt(max(n_frames, 1)), 1e-9)
        out.append({"id": None, "x": round(g["x"] + r * math.cos(a), 3), "y": round(g["y"] + r * math.sin(a), 3),
                    "vx": 0.0, "vy": 0.0, "velocity": "static", "rate_arcsec_min": 0.0, "pa_px_deg": 0.0,
                    "snr_nominal": round(snr, 2), "flux_sigma": round(flux, 3), "detected": None, "passed": False,
                    "kind": "sn", "galaxy": g["name"], "offset_px": round(r, 2)})
    return out


def flux_scale(img: np.ndarray, cxy: np.ndarray, exp_flux: np.ndarray, fwhm: float, max_n: int = 400) -> float:
    """Peak of the PSF-filtered image (Gaussian σ = FWHM/2.355) above the local sky per unit expected flux,
    median over unsaturated, isolated catalogue stars. With it the star model, the measured flux and the
    magnitude all use the same scale – whatever the image units and however the real PSF differs from
    the Gaussian (undersampled stacks)."""
    h, w = img.shape
    if not len(cxy):
        return float("nan")
    sg = max(fwhm / 2.3548, 0.6)
    rb = max(4.0 * fwhm, 6.0)
    H = int(math.ceil(rb + 2))
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
    rr = np.hypot(xx, yy)
    bg = (rr > rb - 1.5) & (rr <= rb + 1.5)
    order = np.argsort(-exp_flux)
    lo, hi = int(0.25 * len(order)), int(0.75 * len(order))           # skip the brightest (saturated) and faintest
    from scipy.spatial import cKDTree
    T = cKDTree(cxy)
    vals = []
    for i in order[lo:hi]:
        x, y = cxy[i]
        xi, yi = int(round(x)), int(round(y))
        if not (H <= xi < w - H and H <= yi < h - H) or len(T.query_ball_point([x, y], rb + 1)) > 1:
            continue
        c = np.asarray(img[yi - H:yi + H + 1, xi - H:xi + H + 1], float)
        if not np.isfinite(c).all() or exp_flux[i] <= 0:
            continue
        f = ndimage.gaussian_filter(c, sg)
        v = float(ndimage.map_coordinates(f, [[y - yi + H], [x - xi + H]], order=1)[0]) - float(np.median(f[bg]))
        vals.append(v / exp_flux[i])
        if len(vals) >= max_n:
            break
    return float(np.median(vals)) if len(vals) >= 20 else float("nan")


def _fill(cut: np.ndarray, ann: np.ndarray, max_bad: float = 0.25):
    """Cut-out minus its sky – a plane fitted to the annulus (twilight frames have strong gradients, and
    a gradient minus itself turned by 180° is a doubled gradient, not zero); masked (non-finite) pixels
    set to the sky. Returns (cut, bad mask or None), (None, None) when too much is masked."""
    fin = np.isfinite(cut)
    if fin.mean() < 1 - max_bad or (fin & ann).sum() < 30:
        return None, None
    H, W = cut.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    sel = ann & fin
    v = cut[sel]
    keep = np.abs(v - np.median(v)) <= 3 * 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12
    A = np.stack([np.ones(int(keep.sum())), xx[sel][keep] - W / 2, yy[sel][keep] - H / 2], 1)
    try:
        cf = np.linalg.lstsq(A, v[keep], rcond=None)[0]
    except Exception:                      # noqa: BLE001
        cf = np.array([float(np.median(v)), 0.0, 0.0])
    c = cut - (cf[0] + cf[1] * (xx - W / 2) + cf[2] * (yy - H / 2))
    if fin.all():
        return c, None
    c[~fin] = 0.0
    return c, ~fin


def _bad_zone(bad: np.ndarray, cx: float, cy: float, fwhm: float) -> np.ndarray:
    """Pixels within ~1 FWHM of a masked pixel or of a masked pixel's 180° mirror image."""
    H, W = bad.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    mir = ndimage.map_coordinates(bad.astype(float), [2 * cy - yy, 2 * cx - xx], order=0, mode="constant", cval=0.0) > 0
    return ndimage.binary_dilation(bad | mir, iterations=max(1, int(round(fwhm))))


def _gauss(sig: float, r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig * sig))
    return k / k.sum()


def _residual(cut: np.ndarray, cx: float, cy: float, stars, sg: float) -> np.ndarray:
    """cut − stars − cut turned by 180° about (cx, cy) (local coordinates)."""
    img = cut.copy()
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    for sx, sy, f in stars:
        img -= f * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sg * sg)) / (2 * math.pi * sg * sg)
    rot = ndimage.map_coordinates(img, [2 * cy - yy, 2 * cx - xx], order=3, mode="nearest")
    return img - rot


def search(cube, img, sig: float, gals: list[dict], to_pix, cxy, cat_mag, exp_flux, zp, fwhm: float,
           scale_as: float, wcs, params, known=None, sig_f=None) -> tuple[list[dict], dict]:
    """Candidates (twilight candidate dicts, kind "supernova") and stats."""
    P = params
    n = cube.shape[0]
    h, w = img.shape
    sg = max(fwhm / 2.3548, 0.6)
    ctree = None
    ok_c = np.isfinite(cxy).all(axis=1) if len(cxy) else np.zeros(0, bool)
    cxy, cat_mag, exp_flux = cxy[ok_c], np.asarray(cat_mag)[ok_c], np.asarray(exp_flux)[ok_c]
    if len(cxy):
        from scipy.spatial import cKDTree
        ctree = cKDTree(cxy)
    k = flux_scale(np.asarray(img), cxy, exp_flux, fwhm) if len(cxy) else float("nan")
    if not math.isfinite(k) or k <= 0:
        k = 1.0 / (sig * 4 * math.pi * sg * sg)                 # nominal: stars in σ units, Gaussian PSF
    kf = []
    for i in range(n):
        ki = flux_scale(np.asarray(cube[i]), cxy, exp_flux, fwhm, max_n=150) if len(cxy) else float("nan")
        kf.append(ki if math.isfinite(ki) and ki > 0 else k / max(n, 1))
    gxy = np.array([[g["x"], g["y"]] for g in gals], float) if gals else np.zeros((0, 2))
    okg = np.isfinite(gxy).all(axis=1)
    from scipy.spatial import cKDTree as _KD
    gtree = _KD(gxy[okg]) if okg.any() else None
    gidx = np.nonzero(okg)[0]
    out, stats = [], {"galaxies": len(gals), "examined": 0, "peaks": 0}
    stats["flux_scale"] = round(k, 4)
    A4 = 4 * math.pi * sg * sg
    for g in gals:
        if not (math.isfinite(g["x"]) and math.isfinite(g["y"])):
            continue
        R = radius_px(g, scale_as, fwhm)
        H = int(math.ceil(1.5 * R + 4 * fwhm + 4))
        xi, yi = int(round(g["x"])), int(round(g["y"]))
        y0, y1, x0, x1 = yi - H, yi + H + 1, xi - H, xi + H + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        cut = np.asarray(img[y0:y1, x0:x1], float)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
        rr = np.hypot(xx - g["x"], yy - g["y"])
        ann = (rr > 1.5 * R + 2 * fwhm) & (rr <= H)
        cut, bad = _fill(cut, ann)
        if cut is None or ann.sum() < 30:
            continue
        stats["examined"] += 1
        # catalogue stars in the cut-out (flux in img units)
        stars = []
        if ctree is not None:
            for j in ctree.query_ball_point([g["x"], g["y"]], H * 1.5):
                stars.append((cxy[j, 0] - x0, cxy[j, 1] - y0, exp_flux[j]))
        # model total flux chosen so that its filtered peak equals the stars' measured one
        res = _residual(cut, g["x"] - x0, g["y"] - y0, [(a_, b_, k * A4 * f_) for a_, b_, f_ in stars], sg)
        S1 = ndimage.gaussian_filter(res, sg)
        S2 = ndimage.gaussian_filter(res, 2 * sg)
        noise_sky = 1.4826 * float(np.median(np.abs(S1[ann] - np.median(S1[ann])))) + 1e-9
        inner = (rr <= 1.5 * R) & (rr >= 0.5)
        if bad is not None:                 # masked pixels, or their mirror images, are no measurement
            inner &= ~_bad_zone(bad, g["x"] - x0, g["y"] - y0, fwhm)
        # the galaxy's own asymmetry: robust spread of the filtered residual inside it (peaks excluded)
        vin = S1[inner]
        noise_gal = 1.4826 * float(np.median(np.abs(vin - np.median(vin)))) if vin.size >= 20 else 0.0
        noise = max(noise_sky, noise_gal)
        S = np.where(inner, S1, -np.inf)
        # up to 3 local maxima ≥ 2 FWHM apart: the highest may be a residual of something else
        mx = (S == ndimage.maximum_filter(S, size=max(3, int(2 * fwhm) | 1))) & np.isfinite(S) & (S > P.sn_z_min * noise_sky)
        pk = sorted(zip(*np.nonzero(mx)), key=lambda t: -S[t])[:3]
        gout = []
        for py, px in pk:
            py, px = int(py), int(px)
            # the galaxy's own asymmetry without this peak and its negative mirror image (they would inflate it)
            lx, ly = xx - x0, yy - y0
            mcx, mcy = 2 * (g["x"] - x0) - px, 2 * (g["y"] - y0) - py
            zone = (np.hypot(lx - px, ly - py) <= 2 * fwhm) | (np.hypot(lx - mcx, ly - mcy) <= 2 * fwhm)
            vin = S1[inner & ~zone]
            noise_gal = 1.4826 * float(np.median(np.abs(vin - np.median(vin)))) if vin.size >= 20 else 0.0
            noise = max(noise_sky, noise_gal)
            z = float(S1[py, px] / noise)
            if not np.isfinite(z) or z < P.sn_z_min:
                continue
            stats["peaks"] += 1 if not gout else 0
            sharp = float(S1[py, px] / S2[py, px]) if S2[py, px] > 0 else float("inf")
            # light of the galaxy itself under the peak (before the self-subtraction)
            G1 = float(ndimage.gaussian_filter(cut, sg)[py, px])
            light = G1 / max(float(S1[py, px]), 1e-9)
            # sub-pixel position (parabola)
            def par(a, b, c):
                d = a - 2 * b + c
                return float(np.clip(0.5 * (a - c) / d, -1.0, 1.0)) if d < 0 else 0.0
            fx = px + (par(S1[py, px - 1], S1[py, px], S1[py, px + 1]) if 0 < px < S1.shape[1] - 1 else 0.0)
            fy = py + (par(S1[py - 1, px], S1[py, px], S1[py + 1, px]) if 0 < py < S1.shape[0] - 1 else 0.0)
            X, Y = x0 + fx, y0 + fy
            off = math.hypot(X - g["x"], Y - g["y"])
            flux = float(S1[py, px]) / k                                  # expected-flux units (as exp_flux)
            mag = float(zp(X, Y)) - 2.5 * math.log10(max(flux, 1e-9))
            flags, status = [], "candidate"
            flags.append(f"v galaxii {g['name']} – supernova? (odstup {off * scale_as:.0f}″ od jádra, {z:.0f}σ nad "
                         f"nesouměrností galaxie)")
            if sharp > getattr(P, "sn_sharp_max", 4.5):
                flags.append("ostřejší než hvězda – horký pixel nebo zbytek po odečtení")
                status = "auto_rejected"
            elif sharp < P.sn_sharp_min:
                flags.append(f"rozlehlé (ostrost {sharp:.2f}) – nesouměrnost galaxie, ne bodový zdroj")
                status = "auto_rejected"
            if gtree is not None:              # the nucleus of a neighbouring galaxy (compact groups, pairs)
                for jj in gtree.query_ball_point([X, Y], max(1.2 * fwhm, 2.0)):
                    o = gals[int(gidx[jj])]
                    if o is not g:
                        flags.append(f"jádro sousední galaxie {o['name']}")
                        status = "auto_rejected"
                        break
            if light < 0.6:
                flags.append("přebytek jen proti zrcadlovému místu (tam díra / maska), v obraze sám není")
                status = "auto_rejected"
            elif light > P.sn_light_max and z < getattr(P, "sn_light_strong_z", 15.0):
                flags.append(f"galaxie je pod ním {light - 1:.1f}× jasnější než přebytek – spíš uzel galaxie nebo hvězda "
                             f"pod mezí UCAC4")
                status = "auto_rejected"
            elif light > P.sn_light_max:            # strong: a supernova on the bright body is possible
                flags.append(f"na jasné části galaxie (světlo pod ním {light - 1:.1f}× přebytek), ale silný – "
                             f"ověřit v DSS (uzel galaxie?)")
            elif light > P.sn_light_warn:
                flags.append(f"leží na jasné části galaxie (světlo pod ním {light - 1:.1f}× přebytek) – ověřit v DSS")
            if off < P.sn_min_offset_fwhm * fwhm:
                flags.append("u jádra – nesouměrnost nelze odlišit")
                status = "auto_rejected"
            # a catalogue star on the spot: its subtraction residual
            if ctree is not None:
                for jj in ctree.query_ball_point([X, Y], max(1.0 * fwhm, 1.5)):
                    if (math.hypot(cxy[jj, 0] - g["x"], cxy[jj, 1] - g["y"]) <= max(fwhm, 1.5)
                            and math.hypot(cxy[jj, 0] - X, cxy[jj, 1] - Y) > 0.5 * fwhm):
                        continue                  # the galaxy's nucleus listed in UCAC4, and the excess is beside it
                    if exp_flux[jj] > 0.2 * flux:
                        flags.append(f"na hvězdě UCAC4 {cat_mag[jj]:.1f} mag – zbytek po odečtení hvězdy")
                        status = "auto_rejected"
                        break
            # in every frame: the same residual measured per frame at the peak
            fs = []
            for i in range(n):
                ci, bi = _fill(np.asarray(cube[i, y0:y1, x0:x1], float), ann)
                if ci is None or (bi is not None and _bad_zone(bi, g["x"] - x0, g["y"] - y0, fwhm)[py, px]):
                    fs.append(None)
                    continue
                si = sig_f[i] if sig_f is not None else 1.0
                ri = _residual(ci, g["x"] - x0, g["y"] - y0, [(a, b, kf[i] * A4 * f) for a, b, f in stars], sg)
                Si = ndimage.gaussian_filter(ri, sg)
                ni = 1.4826 * float(np.median(np.abs(Si[ann] - np.median(Si[ann])))) + 1e-9
                fs.append(round(float(Si[py, px] / max(ni, 1e-9)), 1))
            good = [v for v in fs if v is not None]
            if not good:
                flags.append("nelze ověřit po snímcích (saturace / okraj)")
                status = "auto_rejected"
            need = P.few_frame_snr_min if n <= 4 else 2.5
            if good and (sum(v >= need for v in good) < max(2, math.ceil(0.75 * len(good)))):
                flags.append(f"jen v {sum(v >= need for v in good)}/{len(good)} snímcích")
                status = "auto_rejected"
            kn = None
            for item in (known or []):
                if math.hypot(float(item[1]) - X, float(item[2]) - Y) * scale_as <= P.known_radius_arcsec:
                    kn = item[0]
                    break
            ra, dec = (float(v) for v in wcs.all_pix2world([[X, Y]], 0)[0])
            gout.append({"x": X, "y": Y, "vx": 0.0, "vy": 0.0, "ra": ra, "dec": dec, "mag": round(mag, 2),
                        "snr": round(z, 1), "kind": "supernova", "catalog_mag": None, "nearest_catalog_px": None,
                        "frames_used": len(good), "obj_fwhm": None, "fwhm": round(fwhm, 2), "rate_arcsec_min": None,
                        "pa_deg": None, "motion_sigma": 0.0, "frame_snr": fs, "flags": flags, "auto_status": status,
                        "known": {"name": kn} if kn else None, "galaxy": g["name"], "galaxy_offset_arcsec": round(off * scale_as, 1),
                        "sn_sharp": round(sharp, 2) if math.isfinite(sharp) else None, "sn_light": round(light, 2), "sn_z": round(z, 1),
                        "sn_noise_gal": round(noise_gal / max(noise_sky, 1e-9), 2)})
        if gout:                           # the best of this galaxy: a passed one, else the strongest
            gout.sort(key=lambda c: (c["auto_status"] != "candidate", -c["snr"]))
            out.append(gout[0])
    stats["candidates"] = sum(1 for c in out if c["auto_status"] == "candidate")
    return out, stats
