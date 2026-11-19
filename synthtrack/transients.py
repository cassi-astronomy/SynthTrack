"""Masking of single-frame transients: cosmic rays and satellite / aircraft trails.

Both appear in one frame only. The search itself copes with them (per-frame
clip, "jednosnímkový artefakt"), but each produces dozens of detections that
clutter the rejected list. Masked pixels become NaN = no data in that frame.

* cosmic rays – in the σ-normalised frame *before* resampling (sharp: the
  peak is far above every one of its 8 neighbours),
* trails – after registration, in the residual frame − temporal median:
  connected components > 4 σ that are long (≥ ``min_len_px``) and thin
  (aspect ≥ 4), dilated by ~2 FWHM. A real mover is never elongated in a
  single frame at the searched rates (≪ 1 px per exposure).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def mask_cosmics(img: np.ndarray, fwhm: float, nsigma: float = 8.0,
                 max_fraction: float = 0.001) -> int:
    """Mask isolated cosmic-ray pixels in a σ-unit frame in place.

    The operation deliberately favours recall.  A hit must exceed the
    strongest of all eight neighbours, and an implausibly large mask is
    refused entirely.  This prevents undersampled bright stars from turning
    into thousands of square holes when the optional masker is enabled.
    """
    sig = max(fwhm / 2.3548, 0.6)
    psf_ratio = math.exp(1.0 / (2 * sig * sig))
    lim = max(4.0, 2.5 * psf_ratio)
    a = np.nan_to_num(img, nan=0.0)
    neighbours = [np.roll(np.roll(a, dy, axis=0), dx, axis=1)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    nb_max = np.maximum.reduce(neighbours)
    hit = (a > nsigma) & (a > lim * np.maximum(nb_max, 0.5))
    hit[:1] = hit[-1:] = False
    hit[:, :1] = hit[:, -1:] = False
    if not hit.any():
        return 0
    finite_count = max(1, int(np.isfinite(img).sum()))
    if int(hit.sum()) > max(1, int(max_fraction * finite_count)):
        return 0
    # Grow only into significant charge directly attached to the sharp core;
    # unconditional 3x3 dilation was the source of conspicuous white squares.
    grown = hit | (ndimage.binary_dilation(hit, structure=np.ones((3, 3), bool))
                   & (a > max(0.5 * nsigma, 4.0)))
    if int(grown.sum()) > max(1, int(max_fraction * finite_count)):
        return 0
    img[grown] = np.nan
    return int(grown.sum())


def mask_trails(reg, fwhm: float, band: int = 512, nsigma: float = 4.0, min_len_px: float = 25.0,
                min_aspect: float = 4.0, max_frame_fraction: float = 0.01) -> tuple[int, list[int]]:
    """Mask satellite trails in registered frames (memmap, in place).

    Returns (masked pixels, frames with a trail)."""
    n, h, w = reg.shape
    grow = max(2, int(round(2 * fwhm)))
    total, frames = 0, set()
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        cube = np.array(reg[:, y0:y1, :], np.float32)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(cube, axis=0)
                res = cube - med[None]
                sig = 1.4826 * np.nanmedian(np.abs(res), axis=0)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig))
        for i in range(n):
            m = np.nan_to_num(res[i] / sig) > nsigma
            if m.sum() < min_len_px:
                continue
            lab, nl = ndimage.label(m, structure=np.ones((3, 3), bool))
            if nl == 0:
                continue
            bad = np.zeros_like(m)
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                ext = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
                if ext < min_len_px:
                    continue
                ys, xs = np.nonzero(lab[sl] == k)
                if len(xs) < 0.5 * min_len_px:
                    continue
                cov = np.cov(np.vstack([xs, ys]).astype(float))
                ev = np.sort(np.linalg.eigvalsh(cov))
                length = 4.0 * math.sqrt(max(ev[1], 0.0))
                aspect = math.sqrt(max(ev[1], 0.0) / max(ev[0], 0.25))
                if length >= min_len_px and aspect >= min_aspect:
                    bad[sl] |= lab[sl] == k
            if bad.any():
                bad = ndimage.binary_dilation(bad, iterations=grow)
                # A genuine narrow trail occupies only a small part of one
                # exposure.  Refuse a broad connected pattern (usually star
                # residuals or a bad temporal-noise estimate) instead of
                # destructively removing a material part of the image.
                if int(bad.sum()) > max(4096, int(max_frame_fraction * h * w)):
                    continue
                view = reg[i, y0:y1, :]
                view[bad] = np.nan
                del view
                total += int(bad.sum())
                frames.add(i)
        del cube, res
    return total, sorted(frames)
