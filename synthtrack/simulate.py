"""Synthetic data: test sequences and injection of fake movers into real frames.

Used by the tests and (later) by injection–recovery runs that measure the
completeness and false-alarm rate of each instrument.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta


def gaussian_psf(img: np.ndarray, x: float, y: float, flux: float, fwhm: float) -> None:
    s = fwhm / 2.3548
    r = int(4 * s) + 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - r), min(img.shape[0], yi + r + 1)
    x0, x1 = max(0, xi - r), min(img.shape[1], xi + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += flux / (2 * np.pi * s * s) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


@dataclass
class Mover:
    x0: float          # px at first frame
    y0: float
    vx: float          # px/min
    vy: float
    flux: float        # total ADU per frame


def make_sequence(out_dir: str | Path, n: int = 20, shape=(256, 256), exptime: float = 30.0,
                  cadence_s: float = 40.0, sky: float = 1000.0, read_noise: float = 5.0,
                  fwhm: float = 3.0, n_stars: int = 80, movers: list[Mover] | None = None,
                  drift_px_per_frame: tuple[float, float] = (0.0, 0.0), gradient: float = 0.0,
                  vignetting: float = 0.0, dark_level: float = 200.0, filter_name: str = "L",
                  start: str = "2026-10-20T22:00:00", seed: int = 1, hot_pixels: int = 0,
                  wcs_center: tuple[float, float] | None = None, scale_arcsec: float = 1.0,
                  throughput=None) -> list[Path]:
    """Write a calibratable 16-bit FITS sequence plus matching master dark & flat.

    With ``wcs_center`` (RA, Dec in degrees) every frame gets a TAN WCS with
    north up and east left; the WCS follows the simulated drift.
    ``throughput(i, xx, yy)`` (optional) multiplies the light of stars and
    movers in frame i (extinction gradient, obstruction, cirrus); the sky is
    left as it is.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "lights").mkdir(parents=True, exist_ok=True)
    (out / "masters").mkdir(parents=True, exist_ok=True)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.5 * max(h, w)) ** 2
    flat = (1.0 - vignetting * r2).astype(np.float32)
    flat *= (1 + 0.01 * rng.standard_normal(shape)).astype(np.float32)  # pixel response
    dark = (dark_level + 2 * rng.standard_normal(shape)).astype(np.float32)
    hot = rng.integers(0, [h, w], size=(hot_pixels, 2)) if hot_pixels else np.zeros((0, 2), int)
    for (hy, hx) in hot:
        dark[hy, hx] += 3000
    sx = rng.uniform(5, w - 5, n_stars)
    sy = rng.uniform(5, h - 5, n_stars)
    sf = 10 ** rng.uniform(3, 5.5, n_stars)
    t0 = Time(start, scale="utc")
    paths = []
    for i in range(n):
        dy, dx = drift_px_per_frame[1] * i, drift_px_per_frame[0] * i
        sky_img = np.full(shape, sky, np.float32) + gradient * (xx / w) * sky
        star_img = np.zeros(shape, np.float32)
        for x, y, f in zip(sx, sy, sf):
            gaussian_psf(star_img, x + dx, y + dy, f, fwhm)
        tmin = i * cadence_s / 60.0 + exptime / 120.0
        for m in movers or []:
            gaussian_psf(star_img, m.x0 + m.vx * tmin + dx, m.y0 + m.vy * tmin + dy, m.flux, fwhm)
        if throughput is not None:
            star_img *= np.asarray(throughput(i, xx, yy), np.float32)
        signal = (sky_img + star_img) * flat
        img = rng.poisson(np.clip(signal, 0, None)).astype(np.float32)
        img += dark + read_noise * rng.standard_normal(shape).astype(np.float32)
        img = np.clip(img, 0, 65535).astype(np.uint16)
        hdr = fits.Header()
        hdr["IMAGETYP"] = "Light Frame"
        hdr["EXPTIME"] = exptime
        hdr["FILTER"] = filter_name
        hdr["CCD-TEMP"] = -10.0
        hdr["XBINNING"] = 1
        hdr["YBINNING"] = 1
        hdr["DATE-OBS"] = (t0 + TimeDelta(i * cadence_s, format="sec")).isot
        if wcs_center is not None:
            sc = scale_arcsec / 3600.0
            hdr.update({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
                        "CRVAL1": wcs_center[0], "CRVAL2": wcs_center[1],
                        "CRPIX1": (w - 1) / 2 + 1 + dx, "CRPIX2": (h - 1) / 2 + 1 + dy,
                        "CD1_1": -sc, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": sc})
        p = out / "lights" / f"light_{i:03d}.fits"
        fits.PrimaryHDU(img, hdr).writeto(p, overwrite=True)
        paths.append(p)
    hd = fits.Header({"IMAGETYP": "Master Dark", "EXPTIME": exptime, "CCD-TEMP": -10.0,
                      "XBINNING": 1, "YBINNING": 1, "DATE-OBS": t0.isot})
    fits.PrimaryHDU(dark, hd).writeto(out / "masters" / f"master_dark_{int(exptime)}s.fits", overwrite=True)
    hf = fits.Header({"IMAGETYP": "Master Flat", "FILTER": filter_name, "XBINNING": 1, "YBINNING": 1,
                      "DATE-OBS": t0.isot})
    fits.PrimaryHDU((flat * 30000).astype(np.float32), hf).writeto(out / "masters" / f"master_flat_{filter_name}.fits",
                                                                   overwrite=True)
    return paths
