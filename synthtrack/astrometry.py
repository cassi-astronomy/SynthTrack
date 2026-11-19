"""Plate solving (ASTAP, offline) and WCS based registration.

ASTAP is used to *identify* the field. Precise astrometry for MPC reports
(refit of WCS+SIP against Gaia DR3 with proper motions) is a planned module
(``astrometry_refine``) that will sit on top of this.

Solver backends are pluggable – ``SOLVERS`` maps a name to a callable so that
astrometry.net (local/online) or other tools can be added later.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy import ndimage

from .gpu import NO_WINDOW

WINDOWS_ASTAP = [r"C:\Program Files\astap\astap.exe", r"C:\Program Files\astap\astap_cli.exe",
                 r"C:\Program Files (x86)\astap\astap.exe"]


@dataclass
class SolveResult:
    ok: bool
    wcs: WCS | None
    message: str
    seconds: float = 0.0


def header_wcs(header) -> WCS | None:
    """WCS already present in the header (e.g. solved by the capture software)."""
    if "CTYPE1" not in header or "CRVAL1" not in header:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(header, naxis=2)
        return w if w.has_celestial else None
    except Exception:
        return None


def find_astap(configured: str = "") -> str | None:
    if configured and Path(configured).exists():
        return configured
    for name in ("astap_cli", "astap"):
        p = shutil.which(name)
        if p:
            return p
    if os.name == "nt":
        for p in WINDOWS_ASTAP:
            if Path(p).exists():
                return p
    return None


def solve_astap(path: str | Path, exe: str, ra_deg: float | None, dec_deg: float | None,
                fov_height_deg: float, radius_deg: float = 10.0, db_dir: str = "",
                timeout_s: int = 120, sip: bool = True, downsample: int = 0) -> SolveResult:
    import time

    path = Path(path)
    args = [exe, "-f", str(path), "-fov", f"{fov_height_deg:.4f}", "-r", f"{radius_deg:g}",
            "-z", str(downsample), "-wcs"]
    if ra_deg is not None and dec_deg is not None:
        args += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}"]
    if db_dir:
        args += ["-d", db_dir]
    if sip:
        args += ["-sip"]          # newer ASTAP versions; ignored by older ones
    t0 = time.time()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s,
                              stdin=subprocess.DEVNULL, **NO_WINDOW)
    except subprocess.TimeoutExpired:
        return SolveResult(False, None, f"ASTAP timeout {timeout_s} s", time.time() - t0)
    except OSError as e:
        return SolveResult(False, None, f"ASTAP nelze spustit: {e}", time.time() - t0)
    dt = time.time() - t0
    wcs_file = path.with_suffix(".wcs")
    ini_file = path.with_suffix(".ini")
    if wcs_file.exists():
        try:
            hdr = fits.Header.fromfile(str(wcs_file), sep="", endcard=True, padding=False)
        except Exception:
            hdr = fits.Header.fromtextfile(str(wcs_file))
        w = header_wcs(hdr)
        if w is not None:
            return SolveResult(True, w, "ASTAP OK", dt)
    msg = "ASTAP: řešení nenalezeno"
    if ini_file.exists():
        for line in ini_file.read_text(errors="replace").splitlines():
            if line.startswith("ERROR") or line.startswith("WARNING"):
                msg += f" ({line.strip()})"
    if proc.stderr.strip():
        msg += f" – {proc.stderr.strip()[:200]}"
    return SolveResult(False, None, msg, dt)


SOLVERS = {"astap": solve_astap}


# --------------------------------------------------------------------------
# WCS based warping onto the reference grid
# --------------------------------------------------------------------------

def warp_to_reference(img: np.ndarray, wcs_img: WCS, wcs_ref: WCS, shape_ref: tuple[int, int],
                      step: int = 32, order: int = 1) -> np.ndarray:
    """Resample ``img`` onto the pixel grid of the reference frame.

    The coordinate mapping is evaluated on a coarse grid and interpolated,
    which is exact enough for smooth distortions and much faster than a
    per-pixel WCS transform.
    """
    h, w = shape_ref
    gy = np.unique(np.append(np.arange(0, h, step), h - 1))
    gx = np.unique(np.append(np.arange(0, w, step), w - 1))
    GX, GY = np.meshgrid(gx, gy)
    sky = wcs_ref.pixel_to_world(GX, GY)
    px, py = wcs_img.world_to_pixel(sky)
    # interpolate coarse maps to full resolution
    fy = np.interp(np.arange(h), gy, np.arange(len(gy)))
    fx = np.interp(np.arange(w), gx, np.arange(len(gx)))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    mx = ndimage.map_coordinates(px, [FY, FX], order=1)
    my = ndimage.map_coordinates(py, [FY, FX], order=1)
    out = ndimage.map_coordinates(img, [my, mx], order=order, mode="constant", cval=np.nan, prefilter=False)
    return out.astype(np.float32)


def pixel_rate_to_sky(wcs: WCS, x: float, y: float, vx: float, vy: float) -> tuple[float, float, float, float]:
    """RA, Dec at (x, y) and rate ″/min, PA° for a pixel velocity (px/min)."""
    from .identify import _pa_deg, _sep_deg

    s1 = wcs.pixel_to_world(x, y)
    s2 = wcs.pixel_to_world(x + vx, y + vy)
    ra1, de1 = s1.ra.deg, s1.dec.deg
    ra2, de2 = s2.ra.deg, s2.dec.deg
    return float(ra1), float(de1), float(_sep_deg(ra1, de1, ra2, de2) * 3600), float(_pa_deg(ra1, de1, ra2, de2))
