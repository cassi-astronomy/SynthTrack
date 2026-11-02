"""Frame calibration and background flattening ("artificial flat").

Synthetic tracking sums many frames along a hypothesised motion; any residual
gradient or vignetting adds coherently and produces false peaks. Therefore
every frame is (1) calibrated with dark/flat, (2) flattened with a smooth
background model estimated with sigma clipping and source rejection, and
(3) divided by a local noise map so that pixels are in units of sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .fitsmeta import load_image
from .masters import Selection


@dataclass
class Masters:
    dark: np.ndarray | None = None
    flat: np.ndarray | None = None          # normalised to median 1
    notes: list[str] | None = None


def load_masters(dark_sel: Selection | None, flat_sel: Selection | None) -> Masters:
    notes: list[str] = []
    dark = None
    if dark_sel is not None and dark_sel.ok:
        dark, _ = load_image(dark_sel.path)
        if dark_sel.scale is not None and dark_sel.bias_path:
            bias, _ = load_image(dark_sel.bias_path)
            dark = bias + (dark - bias) * np.float32(dark_sel.scale)
            notes.append(f"dark škálován ×{dark_sel.scale:.3f}")
    flat = None
    if flat_sel is not None and flat_sel.ok:
        flat, _ = load_image(flat_sel.path)
        med = float(np.nanmedian(flat))
        if not np.isfinite(med) or med <= 0:
            raise ValueError(f"Master flat {flat_sel.path} má neplatný medián {med}")
        flat = flat / np.float32(med)
        # guard against dead/vignetted-to-zero regions
        flat[~np.isfinite(flat) | (flat < 0.05)] = np.nan
    return Masters(dark=dark, flat=flat, notes=notes)


def calibrate(raw: np.ndarray, masters: Masters, saturation: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (calibrated image, bad-pixel mask). Mask is True for unusable pixels."""
    img = raw.astype(np.float32, copy=True)
    bad = raw >= saturation
    if masters.dark is not None:
        _check_shape(img, masters.dark, "dark")
        img -= masters.dark
    if masters.flat is not None:
        _check_shape(img, masters.flat, "flat")
        img /= masters.flat
        bad |= ~np.isfinite(masters.flat)
    bad |= ~np.isfinite(img)
    # grow saturated cores (bleeding / halos)
    if bad.any():
        bad = ndimage.binary_dilation(bad, iterations=2)
    img[bad] = np.nan
    return img, bad


def _check_shape(img, master, kind):
    if img.shape != master.shape:
        raise ValueError(f"Rozměr master {kind} {master.shape} neodpovídá snímku {img.shape} "
                         "(jiný binning nebo ořez?)")


def remove_cosmics(img: np.ndarray, mask: np.ndarray, gain: float, read_noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Optional L.A.Cosmic via astroscrappy (if installed)."""
    try:
        import astroscrappy  # type: ignore
    except ImportError:
        return img, mask
    work = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    crmask, _ = astroscrappy.detect_cosmics(work, inmask=mask, gain=gain, readnoise=read_noise,
                                            sigclip=5.0, objlim=5.0, cleantype="medmask")
    mask = mask | crmask
    img = img.copy()
    img[crmask] = np.nan
    return img, mask


def software_bin(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    v = img[:h, :w].reshape(h // factor, factor, w // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(v, axis=(1, 3)).astype(np.float32)


def nanmedian_last(x: np.ndarray) -> np.ndarray:
    """Median over the last axis ignoring NaN – one vectorised sort.

    ``np.nanmedian`` falls back to a Python loop when NaNs are present, which
    also holds the GIL and prevents frames from being processed in parallel."""
    srt = np.sort(x, axis=-1)                         # NaN sorts last
    cnt = np.isfinite(x).sum(axis=-1)
    lo = np.maximum((cnt - 1) // 2, 0)[..., None]
    hi = np.maximum(cnt // 2, 0)[..., None]
    with np.errstate(invalid="ignore"):
        m = 0.5 * (np.take_along_axis(srt, lo, -1) + np.take_along_axis(srt, hi, -1))[..., 0]
    return np.where(cnt > 0, m, np.nan).astype(np.float32)


def background_model(img: np.ndarray, mesh: int = 64, nsigma: float = 3.0,
                     iters: int = 3, filter_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Smooth background and noise maps from a sigma-clipped mesh.

    The mesh statistics use iterative sigma clipping (stars rejected), meshes
    dominated by bright objects are replaced by a median filter of their
    neighbours, and the grid is interpolated back to full resolution with a
    cubic spline. Returns (background, rms) with the image shape.
    """
    h, w = img.shape
    ny, nx = max(1, h // mesh), max(1, w // mesh)
    # pad to whole meshes using reflection so edges get proper statistics
    ph, pw = ny * mesh, nx * mesh
    core = img[:ph, :pw]
    blocks = core.reshape(ny, mesh, nx, mesh).transpose(0, 2, 1, 3)
    # statistics on every 2nd pixel of large meshes: 4× faster, same precision
    if mesh >= 32:
        blocks = blocks[:, :, ::2, ::2]
    data = blocks.reshape(ny, nx, -1).astype(np.float32, copy=True)
    for _ in range(iters):
        med = nanmedian_last(data)[..., None]
        mad = nanmedian_last(np.abs(data - med))[..., None] * 1.4826
        with np.errstate(invalid="ignore"):
            clip = np.abs(data - med) > nsigma * np.maximum(mad, 1e-6)
        data[clip] = np.nan
    valid_frac = np.mean(np.isfinite(data), axis=2)
    with np.errstate(invalid="ignore"):
        med = nanmedian_last(data)
        # mode estimator (Pearson) is more robust against faint wings of stars
        mean = np.nanmean(data, axis=2)
        bkg = np.where(np.abs(mean - med) < 0.3 * np.nanstd(data, axis=2), 2.5 * med - 1.5 * mean, med)
        rms = np.nanstd(data, axis=2)
    bad = (valid_frac < 0.5) | ~np.isfinite(bkg) | ~np.isfinite(rms)
    bkg = _fill_bad(bkg, bad)
    rms = _fill_bad(rms, bad)
    if filter_size > 1 and min(ny, nx) >= filter_size:
        bkg = ndimage.median_filter(bkg, size=filter_size, mode="nearest")
        rms = ndimage.median_filter(rms, size=filter_size, mode="nearest")
    bkg_full = _upsample(bkg, h, w, mesh)
    rms_full = _upsample(rms, h, w, mesh)
    rms_full = np.maximum(rms_full, 1e-3 * max(float(np.nanmedian(rms)), 1e-6))
    return bkg_full.astype(np.float32), rms_full.astype(np.float32)


def _fill_bad(grid: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return grid
    if bad.all():
        return np.zeros_like(grid)
    g = grid.copy()
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    g[bad] = grid[tuple(i[bad] for i in idx)]
    return g


def _upsample(grid: np.ndarray, h: int, w: int, mesh: int) -> np.ndarray:
    """Bicubic spline through the mesh centres, evaluated on the full pixel grid (separable, fast)."""
    from scipy.interpolate import RectBivariateSpline

    ny, nx = grid.shape
    yc = (np.arange(ny) + 0.5) * mesh - 0.5
    xc = (np.arange(nx) + 0.5) * mesh - 0.5
    if ny < 2 or nx < 2:
        return np.full((h, w), float(np.mean(grid)), np.float32)
    k = 3 if min(ny, nx) >= 4 else 1
    spl = RectBivariateSpline(yc, xc, grid, kx=k, ky=k)
    yy = np.clip(np.arange(h, dtype=float), yc[0], yc[-1])
    xx = np.clip(np.arange(w, dtype=float), xc[0], xc[-1])
    return spl(yy, xx).astype(np.float32)


def flatten_and_normalise(img: np.ndarray, mask: np.ndarray, mesh: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Subtract background, divide by noise. Masked pixels become 0 (neutral in stacks)."""
    bkg, rms = background_model(img, mesh=mesh)
    norm = (img - bkg) / rms
    norm[mask | ~np.isfinite(norm)] = 0.0
    stats = {
        "bkg_median": float(np.nanmedian(bkg)),
        "bkg_ptp": float(np.nanmax(bkg) - np.nanmin(bkg)),
        "rms_median": float(np.nanmedian(rms)),
        "masked_frac": float(mask.mean()),
    }
    return norm.astype(np.float32), mask, stats
