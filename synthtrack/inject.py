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


def plan(n_obj: int, shape: tuple[int, int], dts: np.ndarray, fwhm: float, tn, rmin_px: float,
         rmax_px: float, snr_min: float, snr_max: float, scale_arcsec: float, seed: int = 0,
         border: int = 40) -> list[dict]:
    """Random positions, log-uniform speeds, uniform directions and nominal SNRs."""
    rng = np.random.default_rng(seed)
    h, w = shape
    sp = max(fwhm / 2.3548, 0.5)
    t0, t1 = float(np.min(dts)), float(np.max(dts))
    out = []
    for k in range(int(n_obj)):
        v = float(math.exp(rng.uniform(math.log(max(rmin_px, 1e-4)), math.log(max(rmax_px, rmin_px * 1.01)))))
        pa = float(rng.uniform(0, 2 * math.pi))
        vx, vy = v * math.cos(pa), v * math.sin(pa)
        # the whole track has to stay inside the frame
        xs = [vx * t0, vx * t1]
        ys = [vy * t0, vy * t1]
        x = float(rng.uniform(border - min(xs), w - border - max(xs)))
        y = float(rng.uniform(border - min(ys), h - border - max(ys)))
        snr = float(rng.uniform(snr_min, snr_max))
        a = snr * tn.sigma(v) / max(tn.expected(v), 1e-6)
        flux = a * 2.0 * math.sqrt(math.pi) * sp
        out.append({"id": k, "x": round(x, 3), "y": round(y, 3), "vx": vx, "vy": vy,
                    "rate_arcsec_min": round(v * scale_arcsec, 4),
                    "pa_px_deg": round(math.degrees(pa) % 360, 1), "snr_nominal": round(snr, 2),
                    "flux_sigma": round(flux, 2), "detected": None, "passed": False})
    return out


def add_to_frames(reg, objs: list[dict], dts: np.ndarray, fwhm: float, maps=None) -> None:
    """Add Gaussian PSFs (total flux ``flux_sigma``) along every track into ``reg`` in place."""
    n, h, w = reg.shape
    sp = max(fwhm / 2.3548, 0.5)
    r = int(math.ceil(4 * sp)) + 1
    for i in range(n):
        frame = reg[i]
        for o in objs:
            x = o["x"] + o["vx"] * float(dts[i])
            y = o["y"] + o["vy"] * float(dts[i])
            xi, yi = int(round(x)), int(round(y))
            y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sp * sp)) / (2 * math.pi * sp * sp)
            amp = 1.0
            if maps is not None:
                amp = float(maps.tile(yi, yi + 1, xi, xi + 1, frames=[i])[0, 0, 0])
            frame[y0:y1, x0:x1] += np.float32(amp * o["flux_sigma"]) * g.astype(np.float32)
        del frame


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
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [o for o in objs if lo <= o["snr_nominal"] < hi]
        if not sel:
            continue
        bins.append({"snr_from": lo, "snr_to": None if hi > 1e8 else hi, "n": len(sel),
                     "detected": sum(o["detected"] is not None for o in sel),
                     "passed": sum(bool(o["passed"]) for o in sel)})
    det = sum(o["detected"] is not None for o in objs)
    pas = sum(bool(o["passed"]) for o in objs)
    lines = [f"Injekce: {len(objs)} syntetických objektů, nalezeno {det}, z toho {pas} prošlo prověřením "
             f"(práh SNR {threshold:g})."]
    for b in bins:
        hi = "∞" if b["snr_to"] is None else f"{b['snr_to']:g}"
        lines.append(f"  nominální SNR {b['snr_from']:g}–{hi}: nalezeno {b['detected']}/{b['n']}, "
                     f"prošlo {b['passed']}/{b['n']}")
    return {"n": len(objs), "detected": det, "passed": pas, "bins": bins}, lines
