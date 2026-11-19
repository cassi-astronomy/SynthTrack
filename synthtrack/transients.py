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
                 max_fraction: float = 0.001, apply: bool = True):
    """Mask isolated cosmic-ray pixels in a σ-unit frame in place.

    The operation deliberately favours recall.  A hit must exceed the
    strongest of all eight neighbours, and an implausibly large mask is
    refused entirely.  This prevents undersampled bright stars from turning
    into thousands of square holes when the optional masker is enabled.
    """
    sig = max(fwhm / 2.3548, 0.6)
    psf_ratio = math.exp(1.0 / (2 * sig * sig))
    lim = max(4.0, 2.5 * psf_ratio)
    a = np.nan_to_num(img, nan=0.0).astype(np.float32, copy=False)
    # maximum of the 8 neighbours, accumulated in place: 3 frames of memory instead
    # of 16 (8 shifted copies + their stack = 1 GB per 4k frame and thread, which
    # failed with MemoryError on Windows while frames were warped in parallel)
    h, w = a.shape
    p = np.pad(a, 1, mode="edge")
    nb_max = np.full_like(a, -np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                np.maximum(nb_max, p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], out=nb_max)
    del p
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
    if not apply:
        return grown
    img[grown] = np.nan
    return int(grown.sum())


def confirm_cosmics(reg, order: list[int], hits: dict, nsigma: float = 8.0, rel: float = 0.3,
                    win: int = 2, grow: int = 1) -> tuple[int, int]:
    """Mask cosmic-ray candidates in the registered frames only if they are single-frame.

    ``hits[i]`` = (y, x) sky (reference-grid) coordinates of the candidate pixels of frame
    ``i``. A candidate whose peak also appears – ≥ ``rel`` × its value within ±``win`` px –
    in the time-adjacent frames is a real source and keeps its data: in frames with better
    seeing an undersampled star or a slow asteroid has a peak > 4× its neighbours and was
    masked whole (88P/Coihueco 9. 11.: 21 % of the star cores and (149) Medusa, (419)
    Aurelia, (494) Virtus in 3–5 frames each → "nekonzistentní segmenty"). Returns
    (masked pixels, kept candidates)."""
    n, h, w = reg.shape
    pos = {f: k for k, f in enumerate(order)}
    masked = kept = 0
    st = np.ones((3, 3), bool)
    for i, (ys, xs) in hits.items():
        if len(ys) == 0:
            continue
        m = np.zeros((h, w), bool)
        m[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] = True
        lab, nl = ndimage.label(m, structure=st)
        k = pos[i]
        nbr = [order[j] for j in (k - 1, k + 1) if 0 <= j < len(order)]
        bad = np.zeros((h, w), bool)
        for sl_i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            y0, y1 = max(0, sl[0].start - win), min(h, sl[0].stop + win)
            x0, x1 = max(0, sl[1].start - win), min(w, sl[1].stop + win)
            here = np.array(reg[i, y0:y1, x0:x1])
            peak = float(np.nanmax(here)) if np.isfinite(here).any() else 0.0
            real = False
            for j in nbr:
                there = np.array(reg[j, y0:y1, x0:x1])
                if np.isfinite(there).any() and float(np.nanmax(there)) >= max(rel * peak, 0.5 * nsigma):
                    real = True
                    break
            if real:
                kept += 1
            else:
                bad[sl] |= lab[sl] == sl_i
        if bad.any():
            bad = ndimage.binary_dilation(bad, structure=st, iterations=grow)
            view = reg[i]
            view[bad] = np.nan
            del view
            masked += int(bad.sum())
    return masked, kept


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
