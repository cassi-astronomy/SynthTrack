"""Registration, static-sky removal and the synthetic tracking search.

Conventions
  * cube[i] is frame i, registered to the reference frame, in sigma units.
  * times are minutes relative to the reference epoch t_ref (sequence middle).
  * a hypothesis is a velocity (vx, vy) in px/min on the registered grid; the
    object is at (x, y) at t_ref and at (x + vx*dt, y + vy*dt) in frame i.
  * stacking along (vx, vy) = sum_i cube[i] shifted by (-vx*dt_i, -vy*dt_i).

This is the reference implementation written against the NumPy API so that
it runs unchanged on CuPy (GPU). Fused CUDA kernels and tiling for very large
frames are planned optimisations behind the same function signatures.
"""
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .gpu import Backend


@contextlib.contextmanager
def _quiet():
    """Silence NaN-slice warnings of nanmedian (no coverage at frame edges is expected)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

# --------------------------------------------------------------------------
# Registration (translation) – fallback when frames are not plate solved
# --------------------------------------------------------------------------


def measure_offset(ref: np.ndarray, img: np.ndarray, clip: float = 30.0) -> tuple[float, float]:
    """Sub-pixel translation (dy, dx) of ``img`` relative to ``ref`` by phase correlation."""
    a = np.clip(ref, 0, clip)
    b = np.clip(img, 0, clip)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-12)
    cc = np.fft.irfft2(r, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)

    def _sub(c, i, n, axis):
        im, ip = (i - 1) % n, (i + 1) % n
        if axis == 0:
            y0, y1, y2 = c[im, ix], c[i, ix], c[ip, ix]
        else:
            y0, y1, y2 = c[iy, im], c[iy, i], c[iy, ip]
        d = (y0 - 2 * y1 + y2)
        return 0.0 if d == 0 else 0.5 * (y0 - y2) / d

    dy = iy + _sub(cc, iy, cc.shape[0], 0)
    dx = ix + _sub(cc, ix, cc.shape[1], 1)
    if dy > cc.shape[0] / 2:
        dy -= cc.shape[0]
    if dx > cc.shape[1] / 2:
        dx -= cc.shape[1]
    # shift that maps img onto ref
    return float(dy), float(dx)


def apply_shift(img: np.ndarray, dy: float, dx: float, order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Shift an image; pixels coming from outside the frame get ``cval``
    (use NaN to mark missing coverage for the static-sky template)."""
    if abs(dy) < 1e-3 and abs(dx) < 1e-3:
        return img
    return ndimage.shift(img, (dy, dx), order=order, mode="constant", cval=cval,
                         prefilter=order > 1).astype(np.float32)


# --------------------------------------------------------------------------
# Static sky
# --------------------------------------------------------------------------


def detector_hot_pixels(det_cube: np.ndarray, reg_median: np.ndarray, offsets: list[tuple[float, float]],
                        nsigma: float = 6.0) -> np.ndarray | None:
    """Pixels bright in detector coordinates but not on the sky (hot pixels / defects).

    Only meaningful when the field drifted by more than a few pixels during the
    sequence – otherwise hot pixels are static and removed with the stars.
    """
    if not offsets:
        return None
    spread = max(math.hypot(dy - offsets[0][0], dx - offsets[0][1]) for dy, dx in offsets)
    if spread < 3.0:
        return None
    # residual of every frame after removing the (re-projected) static sky,
    # median in detector coordinates: only detector-fixed defects survive
    resid = np.empty_like(det_cube)
    for i, (dy, dx) in enumerate(offsets):
        resid[i] = det_cube[i] - apply_shift(reg_median, -dy, -dx)
    resid = np.nan_to_num(resid)
    hot = np.median(resid, axis=0) > nsigma
    return ndimage.binary_dilation(hot, iterations=1) if hot.any() else None


def remove_static(cube: np.ndarray, bright_sigma: float, dts: np.ndarray | None = None,
                  exclusion_min: float = 0.0, min_frames: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the static sky; mask residuals of bright/saturated stars.

    A plain median over all frames also contains part of every slow mover and
    eats its signal. With ``exclusion_min`` > 0 the template for frame i is the
    median of only those frames taken at least ``exclusion_min`` minutes away
    from frame i (the object has moved by > ~1.5 FWHM in that time), with at
    least ``min_frames`` frames (the most distant ones if necessary).

    Returns (subtracted cube, static median image, bright mask).
    """
    with _quiet():
        med = np.nan_to_num(np.nanmedian(cube, axis=0)).astype(np.float32)
    n = cube.shape[0]
    if dts is not None and exclusion_min > 0 and n >= min_frames + 2:
        sub = np.empty_like(cube, dtype=np.float32)
        cache: dict[tuple, np.ndarray] = {}
        for i in range(n):
            far = np.abs(dts - dts[i])
            sel = np.nonzero(far >= exclusion_min)[0]
            if len(sel) < min_frames:
                sel = np.argsort(far)[::-1][:min_frames]
            key = tuple(sorted(int(s) for s in sel))
            if key not in cache:
                if len(cache) > 4:
                    cache.clear()
                with _quiet():
                    cache[key] = np.nanmedian(cube[list(key)], axis=0)
            sub[i] = cube[i] - cache[key]
    else:
        sub = cube - med[None]
    sub = np.nan_to_num(sub, nan=0.0)        # no coverage -> neutral
    bright = med > bright_sigma
    if bright.any():
        # TODO: per-star mask radius growing with brightness (halos, spikes)
        bright = ndimage.binary_dilation(bright, iterations=3)
        sub[:, bright] = 0.0
    return sub.astype(np.float32), med, bright


# --------------------------------------------------------------------------
# Matched filter / PSF
# --------------------------------------------------------------------------


def estimate_fwhm(img: np.ndarray, nmax: int = 60) -> float:
    """Rough FWHM (px) from second moments of moderately bright isolated stars."""
    sm = ndimage.gaussian_filter(img, 1.0)
    peaks = (sm == ndimage.maximum_filter(sm, size=9)) & (sm > 15) & (sm < 300)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return 2.5
    order = np.argsort(sm[ys, xs])[::-1][:nmax]
    vals = []
    r = 5
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for y, x in zip(ys[order], xs[order]):
        if y < r or x < r or y >= img.shape[0] - r or x >= img.shape[1] - r:
            continue
        c = np.clip(img[y - r:y + r + 1, x - r:x + r + 1], 0, None)
        s = c.sum()
        if s <= 0:
            continue
        sig2 = ((c * (yy ** 2 + xx ** 2)).sum() / s) / 2.0
        if sig2 > 0:
            vals.append(2.3548 * math.sqrt(sig2))
    return float(np.clip(np.median(vals), 1.2, 15.0)) if vals else 2.5


def matched_filter(cube_sub: np.ndarray, fwhm: float) -> np.ndarray:
    """Gaussian matched filter per frame, renormalised to unit robust sigma."""
    sigma = max(fwhm / 2.3548, 0.5)
    out = np.empty_like(cube_sub)
    for i, f in enumerate(cube_sub):
        g = ndimage.gaussian_filter(f, sigma, mode="constant")
        mad = np.median(np.abs(g[g != 0])) * 1.4826 if np.any(g != 0) else 1.0
        out[i] = g / max(mad, 1e-6)
    return out


# --------------------------------------------------------------------------
# Velocity grid
# --------------------------------------------------------------------------


@dataclass
class VelocityGrid:
    vx: np.ndarray          # px/min
    vy: np.ndarray
    dv: float
    span_min: float

    def __len__(self):
        return len(self.vx)


def make_velocity_grid(rate_min_px: float, rate_max_px: float, span_min: float, step_px: float) -> VelocityGrid:
    """Cartesian grid with max trail error ``step_px`` anywhere in the sequence.

    The worst case grid error per axis is dv/2 at |dt| = span/2, so dv = 4*step/span.
    """
    span = max(span_min, 1e-3)
    dv = 4.0 * step_px / span
    n = int(math.ceil(rate_max_px / dv))
    ax = np.arange(-n, n + 1) * dv
    VX, VY = np.meshgrid(ax, ax)
    r = np.hypot(VX, VY)
    keep = (r >= max(rate_min_px - dv / 2, 0.5 * rate_min_px, 1e-9)) & (r <= rate_max_px + dv / 2)
    return VelocityGrid(VX[keep].astype(np.float32), VY[keep].astype(np.float32), float(dv), float(span))


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def _shift_add(xp, acc, frame, sy: int, sx: int):
    """acc[y, x] += frame[y + sy, x + sx] (zero outside)."""
    h, w = frame.shape
    y0d, y1d = max(0, -sy), min(h, h - sy)
    x0d, x1d = max(0, -sx), min(w, w - sx)
    if y1d <= y0d or x1d <= x0d:
        return
    acc[y0d:y1d, x0d:x1d] += frame[y0d + sy:y1d + sy, x0d + sx:x1d + sx]


def stack_along(xp, cube, dts, vx: float, vy: float, frames=None):
    idx = range(cube.shape[0]) if frames is None else frames
    acc = xp.zeros(cube.shape[1:], dtype=xp.float32)
    n = 0
    for i in idx:
        sy = int(round(vy * dts[i]))
        sx = int(round(vx * dts[i]))
        _shift_add(xp, acc, cube[i], sy, sx)
        n += 1
    return acc, n


def search(backend: Backend, cube_f: np.ndarray, dts: np.ndarray, grid: VelocityGrid,
           progress=None, should_stop=None) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel best SNR and the index of the velocity that produced it."""
    xp = backend.xp
    dcube = backend.asarray(cube_f)
    n = cube_f.shape[0]
    norm = 1.0 / math.sqrt(n)
    best = xp.full(cube_f.shape[1:], -1e9, dtype=xp.float32)
    bidx = xp.zeros(cube_f.shape[1:], dtype=xp.int32)
    total = len(grid)
    report_every = max(1, total // 100)
    for k in range(total):
        acc, _ = stack_along(xp, dcube, dts, float(grid.vx[k]), float(grid.vy[k]))
        acc *= norm
        better = acc > best
        best = xp.where(better, acc, best)
        bidx = xp.where(better, xp.int32(k), bidx)
        if progress and (k % report_every == 0 or k == total - 1):
            progress((k + 1) / total)
        if should_stop and k % report_every == 0 and should_stop():
            break
    return backend.asnumpy(best), backend.asnumpy(bidx)


# --------------------------------------------------------------------------
# Candidates and vetting
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    snr: float
    seg_snr: list[float] = field(default_factory=list)
    median_ratio: float = 0.0
    frames_used: int = 0
    star_contaminated: bool = False
    snr_raw: float = 0.0             # significance without per-frame clipping
    fwhm: float | None = None        # local PSF FWHM used for this candidate
    chi2: float = 0.0                # per-frame consistency (≈1 for a real mover)
    snr_reverse: float = 0.0         # significance for the opposite motion
    positions: list = field(default_factory=list)   # measured sub-stack positions (astrometry)
    mover_vs_static: float | None = None  # (E_mover − E_static)/E_mover, ≤ 0 = rather a variable star
    snr_check: float | None = None        # independent SNR (median template, no search)
    frames_with_signal: float | None = None
    star_ratio: float | None = None       # static sky under the track / peak of the object itself
    static_star_x: float | None = None    # nearest/brightest static-star peak, for lossless GUI grouping
    static_star_y: float | None = None
    throughput_low_frac: float | None = None   # share of frames with throughput < 0.6 on the track
    throughput_min: float | None = None   # lowest frame throughput along the track (None = no maps)
    ml_score: float | None = None         # probability of a real object from the trained classifier
    injected: int | None = None           # id of the synthetic object this detection recovered
    flags: list[str] = field(default_factory=list)
    auto_status: str = "candidate"   # candidate | auto_rejected
    status: str = "unreviewed"       # unreviewed | accepted | unsure | rejected
    # filled later
    ra: float | None = None
    dec: float | None = None
    rate_arcsec_min: float | None = None
    pa_deg: float | None = None
    mag: float | None = None
    known: dict | None = None
    sky_matrix: list | None = None       # local [[dξ/dx,dξ/dy],[dη/dx,dη/dy]] in deg/px
    images: dict = field(default_factory=dict)
    preview_skipped: str | None = None   # obvious multi-test artifact; numeric record is retained

    def to_dict(self) -> dict:
        return asdict(self)


def find_peaks(best: np.ndarray, bidx: np.ndarray, grid: VelocityGrid, threshold: float,
               fwhm: float, max_n: int, edge: int) -> list[Candidate]:
    size = max(5, int(2 * fwhm) | 1)
    mx = ndimage.maximum_filter(best, size=size, mode="constant", cval=-1e9)
    pk = (best == mx) & (best >= threshold)
    pk[:edge, :] = pk[-edge:, :] = False
    pk[:, :edge] = pk[:, -edge:] = False
    ys, xs = np.nonzero(pk)
    order = np.argsort(best[ys, xs])[::-1][:max_n]
    out = []
    for j, o in enumerate(order):
        y, x = int(ys[o]), int(xs[o])
        k = int(bidx[y, x])
        out.append(Candidate(id=j, x=float(x), y=float(y), vx=float(grid.vx[k]), vy=float(grid.vy[k]),
                             snr=float(best[y, x])))
    return out


def vet(cands: list[Candidate], cube_f: np.ndarray, dts: np.ndarray, static_med: np.ndarray,
        bright_mask: np.ndarray, segments: int, seg_min: float, bright_sigma: float, fwhm: float = 3.0) -> None:
    """Consistency tests that separate real movers from artefacts.

    * segment SNR: the track must be present in every time segment
      (a cosmic ray / satellite glint lives in one frame only)
    * median ratio: per-frame values along the track – median vs mean;
      a single bright frame gives a high mean but low median
    * star contamination: a bright static source close to the track; a slow
      "mover" sitting on a star is almost always a subtraction residual
    """
    n = cube_f.shape[0]
    segs = [list(a) for a in np.array_split(np.arange(n), max(1, segments))]
    h, w = cube_f.shape[1:]
    for c in cands:
        # --- per-frame samples along the track
        vals = []
        for i in range(n):
            yy = int(round(c.y + c.vy * dts[i]))
            xx = int(round(c.x + c.vx * dts[i]))
            if 0 <= yy < h and 0 <= xx < w:
                vals.append(float(cube_f[i, yy, xx]))
        c.frames_used = len(vals)
        if vals:
            v = np.array(vals)
            mean = v.mean()
            c.median_ratio = float(np.median(v) / mean) if mean > 0 else 0.0
        # --- segment SNRs
        c.seg_snr = []
        for s in segs:
            acc = 0.0
            for i in s:
                yy = int(round(c.y + c.vy * dts[i]))
                xx = int(round(c.x + c.vx * dts[i]))
                if 0 <= yy < h and 0 <= xx < w:
                    acc += float(cube_f[i, yy, xx])
            c.seg_snr.append(round(acc / math.sqrt(max(len(s), 1)), 2))
        # --- star contamination: static source near the track (start, middle, end)
        r = max(2, int(round(1.5 * fwhm)))
        near = 0.0
        for i in (0, n // 2, n - 1):
            yy = int(round(c.y + c.vy * dts[i]))
            xx = int(round(c.x + c.vx * dts[i]))
            y0, y1, x0, x1 = max(0, yy - r), min(h, yy + r + 1), max(0, xx - r), min(w, xx + r + 1)
            if y1 > y0 and x1 > x0:
                near = max(near, float(static_med[y0:y1, x0:x1].max()))
                if bright_mask[y0:y1, x0:x1].any():
                    near = max(near, bright_sigma)
        c.star_contaminated = near > 0.5 * bright_sigma
        motion = math.hypot(c.vx, c.vy) * (dts[-1] - dts[0])
        # --- decide
        flags = []
        if min(c.seg_snr) < seg_min:
            flags.append("slabý segment")
        if c.median_ratio < 0.3:
            flags.append("jednosnímkový artefakt")
        if c.star_contaminated:
            flags.append("u hvězdy")
        if c.frames_used < 0.7 * n:
            flags.append("mimo pole v části snímků")
        c.flags = flags
        if c.star_contaminated and motion < 3 * fwhm:
            flags.append("reziduum hvězdy")
        hard = {"slabý segment", "jednosnímkový artefakt", "reziduum hvězdy"}
        c.auto_status = "auto_rejected" if hard.intersection(flags) else "candidate"


def refine_centroid(stack: np.ndarray, x: float, y: float, r: int = 3) -> tuple[float, float]:
    h, w = stack.shape
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
    sub = np.clip(stack[y0:y1, x0:x1], 0, None)
    s = sub.sum()
    if s <= 0:
        return x, y
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((sub * xx).sum() / s), float((sub * yy).sum() / s)


def refine_velocity(cands: list[Candidate], cube_f: np.ndarray, dts: np.ndarray, dv: float,
                    sub: int = 4, win: int = 2) -> None:
    """Local fine search around the coarse grid velocity (in place).

    Evaluates (2*sub+1)^2 velocities within ±dv and ±``win`` px around the
    candidate using only small cut-outs, so it is cheap even on the CPU. This
    improves rate/PA accuracy and recovers SNR lost to the coarse grid.
    """
    n, h, w = cube_f.shape
    norm = 1.0 / math.sqrt(n)
    steps = np.arange(-sub, sub + 1) * (dv / sub)
    size = 2 * win + 1
    for c in cands:
        best = (c.snr, c.vx, c.vy, c.x, c.y)
        for ddx in steps:
            for ddy in steps:
                vx, vy = c.vx + ddx, c.vy + ddy
                acc = np.zeros((size, size), np.float32)
                for i in range(n):
                    cx = int(round(c.x + vx * dts[i]))
                    cy = int(round(c.y + vy * dts[i]))
                    y0, x0 = cy - win, cx - win
                    if y0 < 0 or x0 < 0 or y0 + size > h or x0 + size > w:
                        continue
                    acc += cube_f[i, y0:y0 + size, x0:x0 + size]
                acc *= norm
                k = int(np.argmax(acc))
                val = float(acc.flat[k])
                if val > best[0]:
                    oy, ox = divmod(k, size)
                    best = (val, vx, vy, c.x + ox - win, c.y + oy - win)
        c.snr, c.vx, c.vy, c.x, c.y = best


def dedupe(cands: list[Candidate], fwhm: float, span_min: float) -> list[Candidate]:
    """Merge detections of the same object found at neighbouring velocities.

    Two detections are the same object if their tracks stay within 2 FWHM of
    each other over the whole sequence. The highest SNR one is kept; the
    survivors are renumbered by SNR.
    """
    order = sorted(cands, key=lambda c: -c.snr)
    kept: list[Candidate] = []
    half = span_min / 2
    lim = 2.0 * fwhm
    for c in order:
        dup = False
        for k in kept:
            for t in (-half, 0.0, half):
                d = math.hypot((c.x + c.vx * t) - (k.x + k.vx * t), (c.y + c.vy * t) - (k.y + k.vy * t))
                if d > lim:
                    break
            else:
                dup = True
            if dup:
                break
        if not dup:
            kept.append(c)
    for i, c in enumerate(kept):
        c.id = i
    return kept
