"""Synthetic tracking search on one tile of the registered stack.

Detection statistic (optimal for an unknown static sky)
--------------------------------------------------------
Model of every pixel p in frame i:   d_i(p) = S(p) + F · P(p − x_i) + noise,
with an unknown static sky S, a mover of flux F on the track x_i = x + v·t_i
and a PSF P. Eliminating S by least squares gives

    F̂ ∝ Σ_i [ (d_i − d̄) ⊛ P ](x_i)

i.e. subtract the per-pixel *mean* over time (here sigma-clipped against
outliers), matched-filter every frame and shift-and-add along v. The
crucial part is the noise of that sum, which is NOT √n·σ: removing the mean
also removes the part of the mover that overlaps between frames, so

    σ²_sum(S) = |S| − (1/n) Σ_{i,j∈S} ρ_ij ,   ρ_ij = exp(−|v|²(t_i − t_j)² / 4σ²_psf)

(frames normalised to unit noise; S = set of frames summed). For fast movers
ρ_ij → δ_ij and the usual √n is recovered. For slow movers – a few pixels
over the whole sequence, typical for wide-field lenses – the usual √n
underestimates the significance by up to a factor 2–3, which is why such
objects are missed by standard synthetic tracking.

Noise is measured *in time* for every pixel (robust MAD of d_i − d̄), so star
cores, imperfect registration and variable seeing are automatically down-
weighted instead of producing false detections.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .gpu import Backend
from .tracking import Candidate, VelocityGrid, _quiet, _shift_add


# --------------------------------------------------------------------------
# Preparation of a tile
# --------------------------------------------------------------------------

@dataclass
class PreparedTile:
    f: np.ndarray            # (n, h, w) filtered, unit-noise frames, clipped to ±cap (search statistic)
    f_raw: np.ndarray        # same without clipping (photometry, vetting)
    static: np.ndarray       # (h, w) static sky (clipped mean), sigma units
    noise: np.ndarray        # (h, w) temporal noise map of the unfiltered frames
    bright: np.ndarray       # (h, w) mask of bright/saturated stars
    coverage: np.ndarray     # (h, w) number of frames with data
    noise_sigma: float = 0.0 # Gaussian width of the measured noise autocorrelation (px)
    valid: np.ndarray | None = None  # (n, h, w) frame has data here (no mask, not blocked)


def temporal_stats(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median and robust σ (from the inter-quartile range) over time.

    One sort along the time axis gives both (NaN = no data sorts last); this
    is ~6× faster than nanmedian + a second nanmedian for the MAD."""
    srt = np.sort(cube, axis=0)
    cnt = np.isfinite(cube).sum(axis=0)
    last = np.maximum(cnt - 1, 0)

    def q(frac):
        pos = frac * last
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, np.maximum(last, 0))
        w = (pos - lo).astype(np.float32)
        vlo = np.take_along_axis(srt, lo[None], axis=0)[0]
        vhi = np.take_along_axis(srt, hi[None], axis=0)[0]
        return vlo * (1 - w) + vhi * w

    med = q(0.5)
    sig = (q(0.75) - q(0.25)) / 1.349
    med = np.where(cnt > 0, med, np.nan)
    sig = np.where(cnt > 2, sig, np.nan)
    good = np.isfinite(sig) & (sig > 0)
    fill = float(np.median(sig[good])) if good.any() else 1.0
    sig = np.where(good, sig, fill)
    return med.astype(np.float32), sig.astype(np.float32)


def clipped_mean(cube: np.ndarray, k: float = 4.0, iters: int = 2, stats=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel sigma-clipped mean over time and robust temporal sigma."""
    med, s = stats if stats is not None else temporal_stats(cube)
    mean = med
    with _quiet():
        for _ in range(iters):
            ok = np.abs(cube - mean[None]) <= k * s[None]          # NaN compares False
            cnt = ok.sum(axis=0)
            tot = np.where(ok, cube, 0).sum(axis=0)
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), med)
    return mean.astype(np.float32), s


def amp_clipped_mean(cube: np.ndarray, amp: np.ndarray, k: float = 4.0, iters: int = 2,
                     amp_min: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Static sky S for the model d_i = A_i·S + noise (A_i = throughput map of frame i).

    Weighted least squares Σ A_i d_i / Σ A_i² over the frames that agree within
    k σ; frames with A_i < ``amp_min`` (blocked) do not define the template.
    Returns S and the robust temporal σ of the residual d_i − A_i·S (same
    meaning as the σ of ``clipped_mean`` when A ≡ 1)."""
    good = np.isfinite(cube) & (amp >= amp_min)
    with _quiet():
        ratio = np.where(good, cube / np.maximum(amp, 1e-3), np.nan).astype(np.float32)
    med, s = temporal_stats(ratio)
    del ratio
    mean = med
    with _quiet():
        for _ in range(iters):
            ok = good & (np.abs(cube - amp * mean[None]) <= k * s[None])
            num = np.where(ok, amp * cube, 0).sum(axis=0)
            den = np.where(ok, amp * amp, 0).sum(axis=0)
            mean = np.where(den > 1e-3, num / np.maximum(den, 1e-3), med)
        _, s_res = temporal_stats((cube - amp * mean[None]).astype(np.float32))
    return mean.astype(np.float32), s_res


def _smooth_lowres(img: np.ndarray, sigma: float, block: int = 8) -> np.ndarray:
    """Gaussian smoothing with large σ via a block-averaged image (~20× faster)."""
    h, w = img.shape
    hb, wb = -(-h // block), -(-w // block)
    pad = np.pad(img, ((0, hb * block - h), (0, wb * block - w)), mode="edge")
    small = pad.reshape(hb, block, wb, block).mean(axis=(1, 3))
    small = ndimage.gaussian_filter(small, sigma / block, mode="nearest")
    yy = (np.arange(h) + 0.5) / block - 0.5
    xx = (np.arange(w) + 0.5) / block - 0.5
    from scipy.interpolate import RectBivariateSpline
    if hb < 2 or wb < 2:
        return np.full_like(img, float(small.mean()))
    spl = RectBivariateSpline(np.arange(hb), np.arange(wb), small, kx=1, ky=1)
    return spl(np.clip(yy, 0, hb - 1), np.clip(xx, 0, wb - 1)).astype(np.float32)


def noise_model(s_raw: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Temporal noise as a function of the static brightness: s² = a + b·S + c·S².

    a – sky, b – Poisson noise of the star itself, c – residuals that scale
    with brightness (registration, seeing and focus changes). Fitted on the
    tile from binned medians; it fills in the star cores where 18 samples are
    too few for a reliable per-pixel estimate.
    """
    S = np.clip(np.nan_to_num(static), 0, None).ravel()
    v = (s_raw.ravel()) ** 2
    edges = np.concatenate([[0, 1, 2, 4], np.geomspace(8, max(16.0, float(S.max()) + 1), 14)])
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (S >= lo) & (S < hi)
        if m.sum() >= 20:
            xs.append(float(np.median(S[m])))
            ys.append(float(np.median(v[m])))
    a = float(np.median(v[S < 1.0])) if (S < 1.0).sum() > 100 else float(np.median(v))
    if len(xs) < 3:
        return np.full_like(s_raw, math.sqrt(a))
    from scipy.optimize import nnls
    xs, ys = np.array(xs), np.array(ys) - a
    wgt = 1.0 / np.maximum(ys + a, 1e-3)                  # relative errors: equal weight per bin
    (b, c), _ = nnls(np.stack([xs, xs * xs], axis=1) * wgt[:, None], np.maximum(ys, 0) * wgt)
    Sm = np.clip(np.nan_to_num(static), 0, None)
    return np.sqrt(np.maximum(a + b * Sm + c * Sm * Sm, 1e-6)).astype(np.float32)


def prepare_tile(reg: np.ndarray, fwhm: float, bright_sigma: float, clip_k: float = 4.0,
                 highpass_px: float = 12.0, photometric: bool = True, cap: float = 4.0,
                 amp: np.ndarray | None = None, amp_min: float = 0.5,
                 chip_shift: np.ndarray | None = None) -> PreparedTile:
    """Static-sky removal, noise normalisation and matched filtering of one tile.

    Besides the clipped-mean template two effects that would otherwise
    masquerade as slow movers are removed:
      * transparency / airmass changes – every frame is scaled to the template
        using the stars of the tile (cirrus, twilight, extinction),
      * slow background trends – a heavily smoothed version of every
        difference frame (objects clipped) is subtracted.

    ``amp`` (n, h, w): throughput maps (``throughput.ThroughputMaps.tile``).
    When given, the template is fitted as d_i = A_i·S (extinction gradients,
    obstruction, cirrus) instead of one scale factor per frame and tile, and
    pixels of a frame with A_i < ``amp_min`` do not enter the search at all:
    a partly blocked entrance pupil also shifts and deforms the stars, which
    no flux scaling can subtract.

    ``chip_shift`` (n, 2): (dy, dx) of every frame's detector position relative
    to the registered grid in this tile (from the registration). When given,
    the pattern fixed on the detector (flat/dark residuals, warm pixels below
    the hot-pixel limit, fixed-pattern noise) is estimated as the median of the
    filtered frames in detector coordinates and subtracted. With a small or
    drift-like dithering this pattern does not average out: it moves over the
    sky exactly like an object at the drift velocity (161P, Coihueco: 0.3 σ per
    pixel, split-half correlation 0.65; most of 2 305 passing candidates had
    the drift velocity). A real mover lies on a detector pixel only in a few
    frames and survives the median.
    """
    n = reg.shape[0]
    reg = reg.astype(np.float32, copy=True)
    valid = np.isfinite(reg)
    coverage = valid.sum(axis=0)
    if amp is not None:
        amp = np.clip(np.nan_to_num(amp, nan=1.0), 0.0, 3.0).astype(np.float32)
        static, s_t = amp_clipped_mean(reg, amp, clip_k)
        photometric = False
    else:
        stats = temporal_stats(reg)
        static, s_t = clipped_mean(reg, clip_k, stats=stats)
    if photometric:
        stars = (static > 15) & (static < 0.6 * bright_sigma) & (coverage == n)
        if stars.sum() > 50:
            scaled = False
            for i in range(n):
                with _quiet():
                    a_i = float(np.nanmedian(reg[i][stars] / static[stars]))
                if np.isfinite(a_i) and 0.3 < a_i < 3.0 and abs(a_i - 1) > 1e-3:
                    reg[i] /= a_i
                    scaled = True
            if scaled:   # re-iterate the clipped mean from the previous solution (no new sort)
                static, _ = clipped_mean(reg, clip_k, stats=(static, s_t))
    dm = reg - (static[None] if amp is None else amp * static[None])   # NaN where no data
    # temporal noise: robust σ over time (≈ σ of d_i − d̄ corrected for the removed mean)
    s_raw = s_t
    floor = float(np.median(s_raw[::4, ::4]))
    s = np.maximum(ndimage.median_filter(s_raw, size=5), noise_model(s_raw, static))
    s = np.maximum(s, 0.8 * floor)
    u = np.nan_to_num(dm / s[None], nan=0.0)
    del dm
    if amp is not None and amp_min > 0:
        blocked = amp < amp_min
        u[blocked] = 0.0                          # blocked / heavily dimmed: no data
        valid = valid & ~blocked                  # …and not part of the noise normalisation below
    if highpass_px > 0:
        for i in range(n):
            u[i] -= _smooth_lowres(np.clip(u[i], -2.5, 2.5), highpass_px)
        u[~valid] = 0.0
    # Gaussian matched filter, analytic unit-noise normalisation (white noise in, unit std out)
    sig = max(fwhm / 2.3548, 0.5)
    gnorm = 1.0 / (2.0 * math.sqrt(math.pi) * sig)       # ||g||_2 for a unit-sum Gaussian
    f = np.empty_like(u)
    for i in range(n):
        f[i] = ndimage.gaussian_filter(u[i], sig, mode="constant") / gnorm
        # resampling (registration) correlates neighbouring pixels, so the analytic
        # white-noise normalisation underestimates the filtered noise: re-normalise
        # every frame empirically (robust MAD over the tile)
        nz = f[i][::3, ::3][valid[i][::3, ::3]]          # every 9th pixel is plenty
        if nz.size > 1000:
            mad = 1.4826 * float(np.median(np.abs(nz - np.median(nz))))
            if mad > 0:
                f[i] /= mad
    if chip_shift is not None and n >= 6:
        f = subtract_chip_pattern(f, np.asarray(chip_shift, float), valid)
    # saturated stars: their masked cores (NaN) move with the dithering and leave
    # holes of varying shape – mask them generously and treat as no data
    # A saturated core is missing in most frames. A pixel missing in only one or
    # a few frames is a bad column or hot pixel moved by the dithering, a cosmic
    # ray or a trail – treating it as saturation zeroed bright asteroids crossed
    # by a bad column (305/pole 1: Bamberga, Dembowska, Nephthys, forced SNR 0).
    holes = (n - coverage) >= max(3, n // 3)
    edge = np.zeros_like(holes)
    eb = 24
    edge[:eb], edge[-eb:], edge[:, :eb], edge[:, -eb:] = True, True, True, True
    # only holes at bright stars (saturated cores) are masked generously; isolated
    # holes (hot pixels, defects – masked per frame) are simply no data
    # A hole counts as a saturated core when it touches bright *measured* sky: a
    # saturated star is ringed by bright wings. An all-frames hole with nothing bright
    # around it (a masked hot pixel or defect) is just missing data – treating NaN as
    # bright zeroed ~8 px around every such hole (161P/Hartley-IRAS: forced SNR 0).
    near_bright = ndimage.maximum_filter(np.where(np.isfinite(static), static, -np.inf),
                                         size=2 * int(fwhm) + 3) > 0.5 * bright_sigma
    # …and only when the hole has a core missing in (almost) every frame: saturation is
    # fixed on the sky with the star, so its centre is never measured. Pixels missing in
    # a few frames next to a bright *unsaturated* source are other masks (hot pixels moved
    # by the dither, trails) – (7) Iris 11.4 mag, 29P field: 3–10 px missing per frame
    # around it, its own brightness made them "saturated", the track was zeroed (forced 0).
    hb = holes & ~edge
    if hb.any():
        lab, nl = ndimage.label(hb, structure=np.ones((3, 3), bool))
        core = coverage <= max(1, n // 6)
        hit = np.intersect1d(np.unique(lab[hb & near_bright]), np.unique(lab[hb & core]))
        hb = np.isin(lab, hit[hit > 0])
        # saturation is fixed on the sky: the hole of a saturated star stays within a few
        # FWHM of its never-measured core. A component reaching further is connected to other
        # missing data – the field edge that a large dither leaves without coverage in part of
        # the frames (88P/Coihueco 8. 11.: 13 of 30 frames 25–43 px away; (4298) Jorgenunez,
        # (19707) Tokunai, (2664) Everhart near the edge were zeroed, forced SNR 0)
        if hb.any():
            hb &= ndimage.binary_dilation(hb & core, iterations=int(3 * fwhm) + 4)
    sat = ndimage.binary_dilation(hb, iterations=int(2 * fwhm) + 3) if hb.any() else np.zeros_like(holes)
    # missing template: saturated core → bright; a masked hot pixel / defect → empty sky
    static = np.where(np.isfinite(static), static, np.where(sat, 2.0 * bright_sigma, 0.0)).astype(np.float32)
    bright = ndimage.binary_dilation(static > bright_sigma, iterations=2) | sat
    f[:, sat] = 0.0
    f_raw = f.astype(np.float32)
    # one frame can contribute at most ±cap σ: a faint mover has ~1–3 σ per frame,
    # a cosmic ray or a hot pixel 10 σ+ in a single frame
    fc = np.clip(f_raw, -cap, cap) if cap and cap > 0 else f_raw
    return PreparedTile(fc, f_raw, static, s.astype(np.float32), bright, coverage,
                        noise_acf_sigma(f_raw, valid & ~bright[None], sig), valid=valid)


def chip_path(f: np.ndarray, dts: np.ndarray, c, shift: np.ndarray, fwhm: float) -> tuple[float | None, float]:
    """(signal on the best detector-fixed path / linear track, overlap of the two paths).

    The detector-fixed path is anchored at the track point of *every* frame (±0.5 px):
    a dither that is drift-like only in part of the series (88P/Coihueco 8. 11. 2026: 13
    frames jumping by 25 px, then 17 frames drifting 6 px) puts a warm pixel on a linear
    track in those frames only; the old anchor at the mean position missed it (ratio 0
    everywhere, 1 993 unknowns at PA 300–330°). ``overlap`` = share of frames in which the
    best detector path is within 1 FWHM of the linear track – a warm pixel can only
    imitate the track where they coincide."""
    n = f.shape[0]
    idx = np.arange(n)
    xs, ys = c.x + c.vx * dts, c.y + c.vy * dts
    lin = float(_bilinear(f, idx, xs, ys).sum())
    if lin <= 0:
        return None, 0.0
    best, bx, by = -np.inf, 0.0, 0.0
    anchors = {(round(float(xs[i] + shift[i, 1]) * 2) / 2, round(float(ys[i] + shift[i, 0]) * 2) / 2)
               for i in range(n)}
    for ax, ay in anchors:
        for ddy in (-0.5, 0.0, 0.5):
            for ddx in (-0.5, 0.0, 0.5):
                v = float(_bilinear(f, idx, ax + ddx - shift[:, 1], ay + ddy - shift[:, 0]).sum())
                if v > best:
                    best, bx, by = v, ax + ddx, ay + ddy
    ov = float(np.mean(np.hypot(bx - shift[:, 1] - xs, by - shift[:, 0] - ys) <= fwhm))
    return round(best / lin, 3), round(ov, 3)


def chip_path_ratio(f: np.ndarray, dts: np.ndarray, c, shift: np.ndarray) -> float | None:
    """Signal on the best detector-fixed path / signal on the linear track.

    A source fixed on the detector (warm or flickering pixel) moves over the sky
    by −shift[i] (registration offsets); with a small, drift-like dithering a
    linear track can follow it. 161P (Coihueco, lower half of the GSENSE4040):
    a cluster of detections at the drift velocity had this ratio ~0.85 (median),
    injected movers ~0.2, the comet 0.54. ≈1 = the detector path explains the
    signal as well as a moving object. Positions in tile coordinates."""
    n = f.shape[0]
    idx = np.arange(n)
    xs, ys = c.x + c.vx * dts, c.y + c.vy * dts
    lin = float(_bilinear(f, idx, xs, ys).sum())
    if lin <= 0:
        return None
    cx, cy = float(np.mean(xs + shift[:, 1])), float(np.mean(ys + shift[:, 0]))
    best = -np.inf
    for ddy in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for ddx in (-1.0, -0.5, 0.0, 0.5, 1.0):
            best = max(best, float(_bilinear(f, idx, cx + ddx - shift[:, 1], cy + ddy - shift[:, 0]).sum()))
    return round(best / lin, 3)


def subtract_chip_pattern(f: np.ndarray, shift: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Remove the detector-fixed part of filtered unit-noise frames (in place) and re-normalise.

    Frame i is moved to detector coordinates (``shift[i]`` = detector − grid,
    bilinear), the median over frames is the fixed pattern, which is moved back
    to every frame and subtracted. Pixels without data stay at 0."""
    n = f.shape[0]
    chip = np.empty_like(f)
    for i in range(n):
        chip[i] = ndimage.shift(f[i], shift[i], order=1, mode="constant", cval=np.nan)
    with _quiet():
        fp = np.nanmedian(np.where(chip == 0.0, np.nan, chip), axis=0)   # 0 = no data
    del chip
    fp = np.nan_to_num(fp, nan=0.0).astype(np.float32)
    for i in range(n):
        back = ndimage.shift(fp, -shift[i], order=1, mode="constant", cval=0.0)
        has = f[i] != 0.0
        f[i][has] -= back[has]
        ok = has if valid is None else (has & valid[i])
        nz = f[i][::3, ::3][ok[::3, ::3]]
        if nz.size > 1000:
            mad = 1.4826 * float(np.median(np.abs(nz - np.median(nz))))
            if mad > 0:
                f[i] /= mad
    return f


def noise_acf_sigma(f: np.ndarray, ok: np.ndarray, default: float, lag: int = 2) -> float:
    """Width s of the filtered-noise autocorrelation, ρ(d) = exp(−d²/4s²), measured at lag d.

    Resampling during registration correlates neighbouring pixels, so the
    noise after the matched filter is smoother than for white noise (s > σ_filter).
    The noise of a slow track depends on exactly this correlation."""
    num = den = 0.0
    for i in range(f.shape[0]):
        a = f[i]
        m = ok[i] & (np.abs(a) < 4)
        for sl0, sl1 in (((slice(None), slice(lag, None)), (slice(None), slice(None, -lag))),
                         ((slice(lag, None), slice(None)), (slice(None, -lag), slice(None)))):
            mm = m[sl0] & m[sl1]
            if mm.sum() < 1000:
                continue
            x, y = a[sl0][mm], a[sl1][mm]
            num += float(np.sum(x * y))
            den += float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if den <= 0:
        return default
    rho = num / den
    if not (0.01 < rho < 0.999):
        return default
    return float(math.sqrt(-lag * lag / (4.0 * math.log(rho))))


# --------------------------------------------------------------------------
# Noise of the stacked sum
# --------------------------------------------------------------------------

class TrackNorm:
    """σ of the shift-and-add sum for a given speed (see module docstring)."""

    def __init__(self, dts: np.ndarray, fwhm: float, noise_sigma: float | None = None,
                 weights: np.ndarray | None = None, amps: np.ndarray | None = None):
        self.dts = np.asarray(dts, float)
        # throughput A_i the static template was fitted with (S = Σ A_j d_j / Σ A_j², residual
        # d_i − A_i·S, ``amp_clipped_mean``); only the segment test uses it (seg_sigma/seg_expected)
        self.a = None if amps is None else np.asarray(amps, float)
        self.n = len(self.dts)
        # frame weights w_i (the frames f_i are multiplied by w_i before summing). With
        # throughput weights w_i = A_i the sum is the optimal one: a frame at 30 % adds
        # 30 % of the signal and only 0.3² of the noise variance instead of a full share.
        #   σ²(S) = Σ w_i² − (1/n) Σ_ij w_i w_j ρ_ij      (template = plain mean of the frames)
        #   E(S)  = Σ_i w_i (a_i − (1/n) Σ_j a_j ρ_ij),   object amplitude a_i ∝ A_i = w_i
        self.w = np.ones(self.n) if weights is None else np.asarray(weights, float)
        self.s2 = (max(fwhm / 2.3548, 0.5)) ** 2                 # signal: PSF ⊛ filter overlap
        self.n2 = noise_sigma ** 2 if noise_sigma else self.s2    # noise: measured autocorrelation
        self.dt2 = (self.dts[:, None] - self.dts[None, :]) ** 2

    def rho(self, speed_px_min: float) -> np.ndarray:
        return np.exp(-(speed_px_min ** 2) * self.dt2 / (4.0 * self.s2))

    def rho_noise(self, speed_px_min: float) -> np.ndarray:
        return np.exp(-(speed_px_min ** 2) * self.dt2 / (4.0 * self.n2))

    def sigma(self, speed: float, subset=None) -> float:
        r = self.rho_noise(speed)
        w = self.w
        if subset is None:
            return math.sqrt(max(float(w @ w) - float(w @ r @ w) / self.n, 1e-6))
        s = np.asarray(subset)
        ws = w[s]
        return math.sqrt(max(float(ws @ ws) - float(ws @ r[np.ix_(s, s)] @ ws) / self.n, 1e-6))

    def expected(self, speed: float, subset=None) -> float:
        """Expected sum (in units of the per-frame template-free signal) over ``subset``.

        Frame i keeps the fraction 1 − c_i, c_i = mean_j ρ_ij: frames near the
        middle of a slow track overlap most with the template and carry least."""
        idx = np.arange(self.n) if subset is None else np.asarray(subset)
        return float(np.sum(self.frame_weights(speed)[idx]))

    def frame_weights(self, speed: float) -> np.ndarray:
        w = self.w
        return w * (w - (self.rho(speed) @ w) / self.n)

    # ---------------------------------------------------- segment test (time subsets)
    # With a template fitted by least squares with the throughput A_i the object's own light
    # is removed from frame i in proportion to A_i (and to the A_j² of the frames it overlaps):
    #   E_i  = w_i A_i (1 − Σ_j A_j² ρ_ij / Σ A²)
    #   σ²_s = Σ_s w_i² − u_s ρ_ss u_s / Σ A²,   u = w·A
    # The plain-mean model (frame_weights/sigma) predicts almost nothing in frames at 30 %
    # throughput, measured slow tracks keep a third of the signal there (Coihueco 317 dawn:
    # 3rd segment z +2.1 median, 29 % of slow injections |z| > 3; this model: z ≈ 0, 0 %).
    # Measured per-segment noise matches within ~5 %. The *total* σ of slow tracks is not
    # taken from here: the LS residual is orthogonal to A, so for ρ → 1 the model σ → 0,
    # while real data keep clipping/map-error noise (×1.3–1.6 at 0.03–0.1 px/min).
    def _seg_ok(self, speed: float) -> bool:
        if self.a is None:
            return False
        e = self.w * self.a * (1.0 - (self.rho(speed) @ (self.a * self.a)) / max(float(self.a @ self.a), 1e-9))
        return float(e.sum()) > 0.05 * float((self.w * self.a).sum())

    def seg_frame_weights(self, speed: float) -> np.ndarray:
        if not self._seg_ok(speed):
            return self.frame_weights(speed)
        a = self.a
        return self.w * a * (1.0 - (self.rho(speed) @ (a * a)) / float(a @ a))

    def seg_expected(self, speed: float, subset=None) -> float:
        idx = np.arange(self.n) if subset is None else np.asarray(subset)
        return float(np.sum(self.seg_frame_weights(speed)[idx]))

    def seg_sigma(self, speed: float, subset=None) -> float:
        if not self._seg_ok(speed):
            return self.sigma(speed, subset)
        r = self.rho_noise(speed)
        u = self.w * self.a
        idx = np.arange(self.n) if subset is None else np.asarray(subset)
        ws, us = self.w[idx], u[idx]
        return math.sqrt(max(float(ws @ ws) - float(us @ r[np.ix_(idx, idx)] @ us) / float(self.a @ self.a), 1e-6))

    def min_speed(self, eff_min: float) -> float:
        """Slowest speed whose efficiency reaches ``eff_min`` (bisection)."""
        lo, hi = 0.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.efficiency(mid) < eff_min:
                lo = mid
            else:
                hi = mid
        return hi

    def efficiency(self, speed: float) -> float:
        """Fraction of the template-free signal that survives (0 static … 1 fast)."""
        return math.sqrt(max(1.0 - self.rho(speed).sum() / self.n ** 2, 0.0))


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def search_tile(backend: Backend, f: np.ndarray, dts: np.ndarray, grid: VelocityGrid, tn: TrackNorm,
                progress=None, should_stop=None, scales: np.ndarray | None = None,
                noise_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shift-and-add over the velocity grid; best SNR and its velocity index per pixel.

    Velocities whose integer shift pattern is identical (common for slow
    movers on a coarse grid) are stacked only once; of such a group the one
    with the smallest σ_sum is kept (the sub-pixel refinement follows anyway).

    With ``scales`` (array of len(grid), filled in place) every velocity's SNR map is
    divided by its own empirical noise, 1.4826 · MAD over the tile (sources are sparse),
    never below 1. The model σ (TrackNorm) assumes independent frames; a weak fixed
    pattern on the detector adds coherently along the velocity at which the detector
    moves over the sky (88P/Coihueco 8. 11.: 17 drifting frames → σ 1.50 at the drift
    velocity, 1 900 SNR 6–7 detections there, 1.00 for fast tracks) and slow tracks have
    ~1.3 (star residuals, integer shifts). ``noise_mask`` (True = sky) keeps the
    neighbourhood of stars out of that estimate: their residuals are judged by the
    vetting, they must not raise the threshold for faint movers elsewhere.
    """
    xp = backend.xp
    d = backend.asarray(f)
    n = f.shape[0]
    shifts = np.stack([np.rint(np.outer(grid.vy, dts)), np.rint(np.outer(grid.vx, dts))], axis=2).astype(np.int32)
    sig = np.array([tn.sigma(math.hypot(float(vx), float(vy))) for vx, vy in zip(grid.vx, grid.vy)])
    keys = shifts.reshape(len(grid), -1)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    reps = []
    for g in range(len(first)):
        members = np.nonzero(inv == g)[0]
        reps.append(int(members[np.argmin(sig[members])]))
    best = xp.full(f.shape[1:], -1e9, dtype=xp.float32)
    bidx = xp.zeros(f.shape[1:], dtype=xp.int32)
    acc = xp.empty(f.shape[1:], dtype=xp.float32)
    mask = xp.empty(f.shape[1:], dtype=bool)
    total = len(reps)
    step = max(1, total // 50)

    pre = None
    if scales is not None:
        # empirical noise of every velocity from a 1/16 subsample of its stack (≈ 6 % extra
        # work), relative to the typical noise of velocities of the same *speed*: slow tracks
        # are noisier in every direction (template, star residuals) – that is left alone; an
        # excess in one direction (detector pattern moving with a dither drift) is scaled out
        h, w = f.shape[1:]
        M = int(np.abs(shifts[reps]).max()) + 1
        if h - 2 * M > 64 and w - 2 * M > 64:
            sub_mask = (backend.asarray(noise_mask[M:h - M:4, M:w - M:4]) if noise_mask is not None else None)
            raw = np.full(len(reps), np.nan)
            sub = xp.zeros(((h - 2 * M + 3) // 4, (w - 2 * M + 3) // 4), dtype=xp.float32)
            for jj, k in enumerate(reps):
                sub.fill(0)
                for i in range(n):
                    sy, sx = int(shifts[k, i, 0]), int(shifts[k, i, 1])
                    sub += d[i, M + sy:h - M + sy:4, M + sx:w - M + sx:4]
                smp = sub[(sub != 0) & sub_mask] if sub_mask is not None else sub[sub != 0]
                if int(smp.size) > 200:
                    med = xp.median(smp)
                    raw[jj] = float(1.4826 * xp.median(xp.abs(smp - med))) / sig[k]
            speed = np.hypot(grid.vx[reps], grid.vy[reps])
            lsp = np.log(np.maximum(speed, 1e-6))
            pre = np.ones(len(reps))
            for jj in range(len(reps)):
                if not np.isfinite(raw[jj]):
                    continue
                near = np.abs(lsp - lsp[jj]) <= 0.15            # ±16 % in speed
                ref = raw[near & np.isfinite(raw)]
                if ref.size >= 8:
                    pre[jj] = min(max(raw[jj] / float(np.median(ref)), 1.0), 4.0)

    for j, k in enumerate(reps):
        acc.fill(0)
        for i in range(n):
            _shift_add(xp, acc, d[i], int(shifts[k, i, 0]), int(shifts[k, i, 1]))
        acc *= xp.float32(1.0 / sig[k])
        if pre is not None:
            sc = float(pre[j])
            if sc > 1.0:
                acc *= xp.float32(1.0 / sc)
            scales[inv == inv[k]] = sc
        xp.greater(acc, best, out=mask)
        xp.copyto(best, acc, where=mask)
        bidx[mask] = k
        if progress and j % step == 0:
            progress((j + 1) / total)
        if should_stop and j % step == 0 and should_stop():
            break
    return backend.asnumpy(best), backend.asnumpy(bidx)


def _sum_track(f, dts, x, y, vx, vy, frames, win=0):
    n, h, w = f.shape
    size = 2 * win + 1
    acc = np.zeros((size, size), np.float32)
    used = 0
    for i in frames:
        cx = int(round(x + vx * dts[i]))
        cy = int(round(y + vy * dts[i]))
        if cy - win < 0 or cx - win < 0 or cy + win >= h or cx + win >= w:
            continue
        acc += f[i, cy - win:cy + win + 1, cx - win:cx + win + 1]
        used += 1
    return acc, used


def _bilinear(f: np.ndarray, frames: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Sample f[frame, Y, X] at fractional positions (zero outside)."""
    n, h, w = f.shape
    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    out = np.zeros(X.shape, np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            out += np.where(ok, f[frames, np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], 0.0) * wx * wy
    return out


def refine(cands: list[Candidate], f: np.ndarray, dts: np.ndarray, dv: float, tn: TrackNorm,
           sub: int = 4, win: float = 1.0, min_speed: float = 0.0, max_speed: float = 1e9) -> None:
    """Fine velocity/position search around the coarse grid point (in place, vectorised).

    The track is sampled at sub-pixel positions (bilinear). With integer
    rounding a slow track of a few pixels is identical for a range of speeds
    and the slowest one (smallest noise) would always win – a 10–20 % bias.
    """
    n, h, w = f.shape
    steps = np.arange(-sub, sub + 1) * (dv / sub)
    DX, DY = np.meshgrid(steps, steps)
    DX, DY = DX.ravel(), DY.ravel()
    offs = np.arange(-win, win + 1e-6, 0.5)
    OX, OY = np.meshgrid(offs, offs)
    OX, OY = OX.ravel(), OY.ravel()                                   # (P,)
    frames = np.arange(n)[None, :, None]
    for c in cands:
        vx = c.vx + DX
        vy = c.vy + DY
        speed = np.hypot(vx, vy)
        ok = (speed >= min_speed) & (speed <= max_speed)
        if not ok.any():
            continue
        vx, vy, speed = vx[ok], vy[ok], speed[ok]
        X = c.x + vx[:, None, None] * dts[None, :, None] + OX[None, None, :]   # (V, n, P)
        Y = c.y + vy[:, None, None] * dts[None, :, None] + OY[None, None, :]
        vals = _bilinear(f, frames, X, Y).sum(axis=1)                         # (V, P)
        sig = np.array([tn.sigma(s) for s in speed])
        snr = vals / sig[:, None]
        k = int(np.argmax(snr))
        iv, ip = np.unravel_index(k, snr.shape)
        if True:   # bilinear SNR replaces the integer-grid value
            c.snr = float(snr.flat[k])
            c.vx, c.vy = float(vx[iv]), float(vy[iv])
            c.x, c.y = c.x + float(OX[ip]), c.y + float(OY[ip])


def vet(cands: list[Candidate], tile: PreparedTile, dts: np.ndarray, tn: TrackNorm, segments: int,
        seg_z_max: float, bright_sigma: float, fwhm: float) -> None:
    """Consistency tests (in place).

    * segments: the signal in every time segment must agree with the
      expectation for a real mover of that speed (z-score ≤ ``seg_z_max``);
      for slow movers the middle segment legitimately carries less signal,
    * median vs mean of per-frame samples, each divided by its expected
      share (single-frame artefacts: cosmics, satellite glints),
    * a bright static source close to a slow track = subtraction residual,
    * enough frames covering the track.
    """
    f = tile.f_raw
    n, h, w = f.shape
    segs = [np.array(a) for a in np.array_split(np.arange(n), max(1, segments))]
    rv = max(1, int(round(0.6 * fwhm)))
    for c in cands:
        speed = math.hypot(c.vx, c.vy)
        wts = tn.frame_weights(speed)
        # frames without data at the object (a dithered bad column or hot pixel, a
        # masked cosmic or trail crossing the core) carry no signal by construction;
        # they must not count as "signal missing" (305/pole 1: Dembowska, column
        # through the core in frames 10–12 → inconsistent segments, χ² 5.8)
        has = np.ones(n, bool)
        if tile.valid is not None:
            for i in range(n):
                cx = int(round(c.x + c.vx * dts[i]))
                cy = int(round(c.y + c.vy * dts[i]))
                win = tile.valid[i, max(0, cy - rv):cy + rv + 1, max(0, cx - rv):cx + rv + 1]
                has[i] = bool(win.size) and bool(win.all())
            if has.sum() < max(3, n // 2):
                has[:] = True                  # mostly masked: judge as before
        # report the un-clipped significance
        tot, _ = _sum_track(f, dts, c.x, c.y, c.vx, c.vy, range(n), 0)
        c.snr_raw = float(tot[0, 0]) / tn.sigma(speed)
        samples = []
        for i in range(n):
            cx = int(round(c.x + c.vx * dts[i]))
            cy = int(round(c.y + c.vy * dts[i]))
            if (0 <= cy < h and 0 <= cx < w and tile.coverage[cy, cx] > 0 and has[i]
                    and wts[i] >= 0.5 * wts.max()):
                samples.append(float(f[i, cy, cx]) / wts[i])
        c.frames_used = int(sum(1 for i in range(n) if 0 <= int(round(c.y + c.vy * dts[i])) < h
                                and 0 <= int(round(c.x + c.vx * dts[i])) < w))
        v = np.array(samples) if samples else np.zeros(1)
        mean = v.mean()
        c.median_ratio = float(np.median(v) / mean) if mean > 0 else 0.0
        # per-frame consistency: every frame should carry amp·w_i ± 1 (unit noise)
        amp0 = float(tot[0, 0]) / max(tn.expected(speed), 1e-6)
        res = []
        for i in range(n):
            cx = int(round(c.x + c.vx * dts[i]))
            cy = int(round(c.y + c.vy * dts[i]))
            if 0 <= cy < h and 0 <= cx < w and tile.coverage[cy, cx] > 0 and has[i]:
                e_i = amp0 * wts[i]
                res.append((float(f[i, cy, cx]) - e_i) ** 2 / (1.0 + (0.3 * e_i) ** 2))
        # trimmed: the largest max(1, n/10) terms are left out – one frame hit by a cosmic or a
        # star residual must not reject a real object (11P/Coihueco 8. 11.: SNR 39, one frame
        # at 23 σ instead of 7 σ → χ² 5.4 → "nekonzistentní snímky"); single-frame artefacts
        # are judged by the median/mean ratio and the segments
        if res:
            rs = np.sort(np.array(res))
            k_out = max(1, len(rs) // 10) if len(rs) >= 8 else 0
            c.chi2 = float(rs[:len(rs) - k_out].mean()) / _trim_factor(k_out / len(rs))
        else:
            c.chi2 = 0.0
        # direction: the reversed motion must not fit as well (symmetric residuals do)
        rev, _ = _sum_track(f, dts, c.x, c.y, -c.vx, -c.vy, range(n), 0)
        c.snr_reverse = float(rev[0, 0]) / tn.sigma(speed)
        amp = float(tot[0, 0]) / max(tn.seg_expected(speed), 1e-6)     # per-frame amplitude
        if not has.all():
            # frames without data (field edge after a large dither, masks) carry no signal: the
            # amplitude comes from the frames with data, a segment without any is not judged
            # (88P/Coihueco 9. 11.: (4298) Jorgenunez at the edge, segments [0, 8.8, 7.4] → rejected)
            hi = np.nonzero(has)[0]
            th, _ = _sum_track(f, dts, c.x, c.y, c.vx, c.vy, hi, 0)
            amp = float(th[0, 0]) / max(tn.seg_expected(speed, hi), 1e-6)
        c.seg_snr, zs = [], []
        for sgm0 in segs:
            if not has[sgm0].any() and has.sum() >= max(3, n // 2):
                c.seg_snr.append(0.0)
                continue
            sgm = sgm0[has[sgm0]] if has[sgm0].any() else sgm0
            acc, _ = _sum_track(f, dts, c.x, c.y, c.vx, c.vy, sgm, 0)
            sig = tn.seg_sigma(speed, sgm)
            c.seg_snr.append(round(float(acc[0, 0]) / sig, 2))
            exp_s = amp * tn.seg_expected(speed, sgm)
            # statistical z-score, but tolerate 35 % model mismatch for bright objects
            # (PSF changes, slight velocity error, photometric variation)
            zs.append((float(acc[0, 0]) - exp_s) / math.hypot(sig, 0.35 * exp_s))
        r = max(2, int(round(1.5 * fwhm)))
        near = 0.0
        star_xy = None
        for i in (0, n // 2, n - 1):
            cx = int(round(c.x + c.vx * dts[i]))
            cy = int(round(c.y + c.vy * dts[i]))
            y0, y1, x0, x1 = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
            if y1 > y0 and x1 > x0:
                sw = tile.static[y0:y1, x0:x1]
                peak = float(sw.max())
                if peak > near:
                    py, px = np.unravel_index(int(np.argmax(sw)), sw.shape)
                    near, star_xy = peak, (x0 + px, y0 + py)
                if tile.bright[y0:y1, x0:x1].any():
                    near = max(near, bright_sigma)
        # peak brightness of the mover itself in σ units (it also leaves a smear in the template)
        sig_psf = max(fwhm / 2.3548, 0.5)
        yi, xi = min(max(int(round(c.y)), 0), h - 1), min(max(int(round(c.x)), 0), w - 1)
        own_peak = max(amp, 0.0) * float(tile.noise[yi, xi]) / (math.sqrt(math.pi) * sig_psf)
        c.star_contaminated = near > max(0.5 * bright_sigma, 4.0 * own_peak)
        if c.star_contaminated and star_xy is not None:
            c.static_star_x, c.static_star_y = float(star_xy[0]), float(star_xy[1])
        motion = speed * (dts[-1] - dts[0])
        flags = []
        if zs and max(abs(z) for z in zs) > seg_z_max:
            flags.append("nekonzistentní segmenty")
        if c.median_ratio < 0.3:
            flags.append("jednosnímkový artefakt")
        if c.star_contaminated:
            flags.append("u hvězdy")
        if c.star_contaminated and motion < 3 * fwhm:
            flags.append("reziduum hvězdy")
        if c.frames_used < 0.7 * n:
            flags.append("mimo pole v části snímků")
        # per-frame consistency; a bright object (SNR ≥ 30) has a per-frame
        # signal of 5–15 σ, where seeing, focus and the slight velocity error
        # of a slow mover exceed the 30 % model tolerance (Eleonora, 305/pole 1:
        # SNR 64, χ² 2.55) – allow more model mismatch there
        # A bright object moving < 1.5 FWHM is largely absorbed by the static
        # template in the middle frames, much more than the per-frame model
        # predicts (Dembowska 0.8 FWHM: measured ~0 σ vs. 5–7 σ expected, χ² 5.8).
        # There χ² carries no information; mover-vs-static and segments decide.
        slow_bright = motion < 1.5 * fwhm and c.snr_raw >= 20
        if c.chi2 > (4.0 if c.snr_raw >= 30 else 2.5) and not slow_bright:
            flags.append("nekonzistentní snímky")
        if c.snr_raw > 0 and c.snr_reverse > 0.7 * c.snr_raw and motion > 0.5 * fwhm:
            flags.append("bez směru pohybu")
        c.flags = flags
        hard = {"nekonzistentní segmenty", "jednosnímkový artefakt", "reziduum hvězdy", "mimo pole v části snímků",
                "nekonzistentní snímky", "bez směru pohybu"}
        c.auto_status = "auto_rejected" if hard.intersection(flags) else "candidate"


def _trim_factor(p: float) -> float:
    """Mean of the lower (1 − p) of a χ²₁ distribution: the trimmed χ² is divided by it, so
    pure noise still gives ≈ 1 and the limits keep their meaning."""
    if p <= 0:
        return 1.0
    from scipy.stats import chi2 as _chi2
    q = float(_chi2.ppf(1.0 - p, 1))
    return float((1.0 - _chi2.sf(q, 3)) / (1.0 - p))


def peaks(best: np.ndarray, bidx: np.ndarray, grid: VelocityGrid, threshold: float, fwhm: float,
          core: tuple[slice, slice], max_n: int) -> list[Candidate]:
    """Local maxima above threshold inside the tile core (margins belong to neighbours)."""
    size = max(5, int(2 * fwhm) | 1)
    mx = ndimage.maximum_filter(best, size=size, mode="constant", cval=-1e9)
    pk = (best == mx) & (best >= threshold)
    mask = np.zeros_like(pk)
    mask[core] = True
    pk &= mask
    ys, xs = np.nonzero(pk)
    order = np.argsort(best[ys, xs])[::-1][:max_n]
    out = []
    for o in order:
        y, x = int(ys[o]), int(xs[o])
        k = int(bidx[y, x])
        out.append(Candidate(id=0, x=float(x), y=float(y), vx=float(grid.vx[k]), vy=float(grid.vy[k]),
                             snr=float(best[y, x])))
    return out


# --------------------------------------------------------------------------
# Astrometry on sub-stacks
# --------------------------------------------------------------------------

def _fit_gauss(img: np.ndarray, x0: float, y0: float, sigma: float, r: int = 4):
    """Least-squares fit of a 2-D Gaussian + plane in a (2r+1)² window. Returns (x, y, flux, ok)."""
    from scipy.optimize import least_squares

    h, w = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
        return x0, y0, 0.0, False
    cut = img[yi - r:yi + r + 1, xi - r:xi + r + 1]
    if not np.isfinite(cut).all():
        return x0, y0, 0.0, False
    yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1]

    def model(p):
        x, y, a, b0, bx, by = p
        return a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)) + b0 + bx * (xx - xi) + by * (yy - yi)
    p0 = [x0, y0, max(float(cut.max() - np.median(cut)), 1e-3), float(np.median(cut)), 0.0, 0.0]
    try:
        res = least_squares(lambda p: (model(p) - cut).ravel(), p0,
                            bounds=([x0 - 2, y0 - 2, 0, -np.inf, -np.inf, -np.inf],
                                    [x0 + 2, y0 + 2, np.inf, np.inf, np.inf, np.inf]))
    except Exception:
        return x0, y0, 0.0, False
    x, y, a = res.x[:3]
    flux = 2 * math.pi * sigma * sigma * a
    return float(x), float(y), float(flux), bool(res.success and a > 0)


def measure_substacks(block: np.ndarray, dts: np.ndarray, c: Candidate, segments: int, fwhm: float,
                      amp: np.ndarray | None = None) -> list:
    """Positions of the object in ``segments`` sub-stacks of star-subtracted frames.

    Each sub-stack is the mean of its frames shifted along the track to the
    sub-stack's mean time; the object is fitted with a Gaussian + plane.
    The static sky (stars) is removed first with a *local* template that
    leaves out, for every pixel, the frames in which the object itself is
    near that pixel – so the template contains no object and the residual no
    stars. Fitting the raw frames (stars left in) let the fit latch on to a
    neighbouring star in dense fields: wrong rate/direction, O−C of 1–2 px
    and real asteroids rejected as static sources.
    Returns [(t_mean, x, y, flux, ok), ...] in block coordinates.
    """
    sig = max(fwhm / 2.3548, 0.6)
    r = 6
    out = []
    for seg in np.array_split(np.arange(block.shape[0]), max(1, segments)):
        tm = float(np.mean(dts[seg]))
        xc, yc = c.x + c.vx * tm, c.y + c.vy * tm
        st = residual_stack(block, dts, c, seg, fwhm, amp, r)
        gx, gy, flux, ok = _fit_gauss(st, r, r, sig, r=4)
        out.append((tm, xc + (gx - r), yc + (gy - r), flux, ok))
    return out


def residual_stack(block: np.ndarray, dts: np.ndarray, c: Candidate, frames, fwhm: float,
                   amp: np.ndarray | None = None, r: int = 6) -> np.ndarray:
    """Star-free stack of ``frames`` along the track, (2r+1)² centred on the object.

    Every frame is placed so that the object lands on the centre; the static sky
    is removed with a *local* template that, for every pixel, leaves out the
    frames in which the object is closer than 1.5 FWHM (at least the n/4
    frames where it is farthest). Shared by the sub-stack astrometry and the
    sharpness test (whose forward model repeats exactly these steps)."""
    n = block.shape[0]
    excl = 1.5 * fwhm                        # object closer than this to a pixel: frame not in its template
    keep_min = max(4, n // 4)
    gx_, gy_ = np.meshgrid(np.arange(-r, r + 1, dtype=float), np.arange(-r, r + 1, dtype=float))
    track_x = c.x + c.vx * dts                # (n,)
    track_y = c.y + c.vy * dts
    allf = np.arange(n)[:, None, None]
    acc = np.zeros((2 * r + 1, 2 * r + 1), np.float64)
    cnt = np.zeros_like(acc)
    for i in frames:
        # place frame i so that the object (at x_i) lands on the stack centre
        X = track_x[i] + gx_
        Y = track_y[i] + gy_
        Xb = np.broadcast_to(X, (n,) + X.shape)
        Yb = np.broadcast_to(Y, (n,) + Y.shape)
        V = _bilinear(block, np.broadcast_to(allf, Xb.shape), Xb, Yb).astype(np.float64)   # all frames, same sky pixels
        if amp is not None:
            A = amp[:, np.clip(np.rint(Y).astype(int), 0, amp.shape[1] - 1),
                    np.clip(np.rint(X).astype(int), 0, amp.shape[2] - 1)].astype(np.float64)
        else:
            A = np.ones_like(V)
        use = _template_use(X, Y, track_x, track_y, excl, keep_min)
        use &= np.isfinite(V) & (A > 0.25)
        with _quiet():
            ratio = np.where(use, V / np.maximum(A, 1e-3), np.nan)
            S = _nanmedian0(ratio) if use.any() else np.full(X.shape, np.nan)
        v = V[i] - A[i] * S
        okv = np.isfinite(v)
        acc[okv] += v[okv]
        cnt[okv] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def _nanmedian0(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.nanmedian`` along ``axis`` as one vectorised sort (same values, float64).

    ``np.nanmedian`` with NaNs present runs a Python loop per output pixel; the
    forward model of ``sharpness`` called it 5·n times per detection and took
    ~80 % of the vetting time (11P/Coihueco 8. 11.: 15 min per tile)."""
    # contiguous, NaN → +inf: the plain float sort path. np.sort of a non-contiguous view with
    # NaNs crashed the process (Windows access violation, NumPy 2.2.6, 8. 11. 2026)
    x = np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1))
    fin = np.isfinite(x)
    cnt = fin.sum(axis=-1)
    srt = np.sort(np.where(fin, x, np.inf), axis=-1)
    lo = np.maximum((cnt - 1) // 2, 0)[..., None]
    hi = np.maximum(cnt // 2, 0)[..., None]
    with np.errstate(invalid="ignore"):
        m = 0.5 * (np.take_along_axis(srt, lo, -1) + np.take_along_axis(srt, hi, -1))[..., 0]
    return np.where(cnt > 0, m, np.nan)


def _template_use(X, Y, track_x, track_y, excl, keep_min):
    """Frames that may enter the static template at each window pixel (object farther than ``excl``)."""
    n = len(track_x)
    dist = np.hypot(X[None] - track_x[:, None, None], Y[None] - track_y[:, None, None])
    use = dist > excl
    few = use.sum(axis=0) < keep_min
    if few.any():                     # slow mover: take the frames where it is farthest away
        kth = np.sort(dist, axis=0)[n - keep_min]
        use = np.where(few[None], dist >= kth[None], use)
    return use


def sharpness(block: np.ndarray, dts: np.ndarray, c: Candidate, fwhm: float,
              amp: np.ndarray | None = None) -> tuple[float, float] | None:
    """Width of the object vs. the stars: (object FWHM in px, Δχ² of "star PSF" vs. best width).

    Hot / warm pixels, cosmic-ray remnants and resampling noise are 1–2 px wide;
    a real asteroid or comet is at least as wide as the stars (trailing and
    track errors only widen it). Measured on 600 mm / IMX533 (FWHM 3.5 px,
    !Hunting/308): 72 of 77 passed false candidates 1.0–1.9 px – warm pixels
    following the drift of the pointing.

    The star-free residual stack is *forward modelled*: for every trial width
    the same template subtraction (``residual_stack``) is applied to a noiseless
    object on this track. A slow object (< 1 FWHM per sequence) partly ends in
    its own template and its residual looks narrower than it is – the model
    has exactly the same loss, so it is not mistaken for a sharp artefact.
    Δχ² is in units of the stack noise; for undersampled stars the two models
    hardly differ and Δχ² stays small, so the test cannot reject there."""
    n = block.shape[0]
    sig_star = max(fwhm / 2.3548, 0.5)
    r = int(min(12, max(6, math.ceil(2.5 * fwhm))))
    st = residual_stack(block, dts, c, range(n), fwhm, amp, r)
    ok = np.isfinite(st)
    if ok.sum() < 0.6 * st.size:
        return None
    gx, gy = np.meshgrid(np.arange(-r, r + 1, dtype=float), np.arange(-r, r + 1, dtype=float))
    # noise of the stack from its outskirts (object-free), never below the white-noise expectation
    far = ok & (np.hypot(gx, gy) > max(1.6 * fwhm, 3.0))
    if far.sum() >= 30:
        d = st[far] - np.median(st[far])
        var = float((1.4826 * np.median(np.abs(d))) ** 2)
    else:
        var = 0.0
    var = max(var, 0.5 / n)
    tx, ty = c.vx * dts, c.vy * dts           # track relative to the object at t = 0
    keep_min = max(4, n // 4)
    excl = 1.5 * fwhm
    # template membership depends only on the geometry: the same for every trial width
    uses = []
    for i in range(n):
        X, Y = tx[i] + gx, ty[i] + gy
        uses.append(_template_use(X, Y, tx, ty, excl, keep_min))
    sigs = np.unique(np.r_[np.linspace(0.35, 1.4 * sig_star, 16), sig_star])
    k_psf = int(np.argmin(np.abs(sigs - sig_star)))
    inv2s2 = (1.0 / (2.0 * sigs * sigs))[:, None, None, None]
    y = st[ok]
    ym = y.mean()
    best_free, best_psf, best_sig = None, None, sig_star
    for ox, oy in ((0.0, 0.0), (-0.4, 0.0), (0.4, 0.0), (0.0, -0.4), (0.0, 0.4)):
        acc = np.zeros((len(sigs),) + gx.shape)
        for i in range(n):
            # model frames j at these sky pixels: object centred at track_j + offset, all widths at
            # once; the Gaussian is separable – two small exp() tables instead of one per pixel
            ex = np.exp(-((tx[i] + gx[0])[None, None] - tx[None, :, None] - ox) ** 2 * inv2s2[:, :, 0])  # (w, n, nx)
            ey = np.exp(-((ty[i] + gy[:, 0])[None, None] - ty[None, :, None] - oy) ** 2 * inv2s2[:, :, 0])
            G = ey[..., :, None] * ex[..., None, :]                # (widths, frames, ny, nx)
            S = _nanmedian0(np.where(uses[i][None], G, np.nan), axis=1)
            acc += G[:, i] - S
        M = (acc / n)[:, ok]                                      # (widths, pixels)
        mm = M.mean(axis=1, keepdims=True)
        dm = M - mm
        den = (dm * dm).sum(axis=1)
        a = np.where(den > 0, (dm * (y - ym)).sum(axis=1) / np.maximum(den, 1e-30), 0.0)
        e = ((y - ym)[None] - a[:, None] * dm) ** 2
        e = e.sum(axis=1)
        e = np.where(a > 0, e, np.inf)                             # flux must be positive
        k = int(np.argmin(e))
        if np.isfinite(e[k]) and (best_free is None or e[k] < best_free):
            best_free, best_sig = float(e[k]), float(sigs[k])
        if np.isfinite(e[k_psf]) and (best_psf is None or e[k_psf] < best_psf):
            best_psf = float(e[k_psf])
    if best_free is None:
        return None
    if best_psf is None:                      # the star PSF does not even fit with a positive flux
        best_psf = float(((y - y.mean()) ** 2).sum())
    return 2.3548 * float(best_sig), (best_psf - best_free) / var


def track_snr(f: np.ndarray, dts: np.ndarray, x: float, y: float, vx: float, vy: float, tn: TrackNorm) -> float:
    """Significance of a sub-pixel track (bilinear) in unit-noise filtered frames."""
    n = f.shape[0]
    X = (x + vx * dts)[:, None]
    Y = (y + vy * dts)[:, None]
    v = _bilinear(f, np.arange(n)[:, None], X, Y)
    return float(v.sum()) / tn.sigma(math.hypot(vx, vy))


def fit_motion(points: list) -> tuple[float, float, float, float] | None:
    """Linear motion (x0, y0 at t=0, vx, vy) from sub-stack positions."""
    pts = [p for p in points if p[4]]
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    A = np.stack([np.ones_like(t), t], axis=1)
    (x0, vx), *_ = np.linalg.lstsq(A, x, rcond=None)
    (y0, vy), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(x0), float(y0), float(vx), float(vy)


def independent_check(block: np.ndarray, dts: np.ndarray, c: Candidate, fwhm: float) -> tuple[float, float] | None:
    """Second, independent measurement of the track: (SNR, share of frames with signal).

    Static sky = per-pixel median of all frames in a window around the track
    (no clipped mean, no matched-filter normalisation, no velocity search);
    in every frame the Gaussian-weighted residual at the object's position is
    taken in units of the window noise. A real mover shows up in most frames
    and its SNR agrees with the search; star-residual and noise-tail
    detections in dense fields mostly do not (measured on CTA-S1 304/4–5)."""
    n, h, w = block.shape
    sg = max(fwhm / 2.3548, 0.6)
    tx, ty = c.x + c.vx * dts, c.y + c.vy * dts
    pad = int(math.ceil(3 * fwhm)) + 2
    x0, x1 = int(math.floor(tx.min())) - pad, int(math.ceil(tx.max())) + pad + 1
    y0, y1 = int(math.floor(ty.min())) - pad, int(math.ceil(ty.max())) + pad + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    cut = np.asarray(block[:, y0:y1, x0:x1], np.float32)
    with _quiet():
        med = np.nanmedian(cut, axis=0)
        res = cut - med[None]
        noise = 1.4826 * float(np.nanmedian(np.abs(res - np.nanmedian(res))))
    if not np.isfinite(noise) or noise <= 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    amps = []
    for i in range(n):
        d2 = (xx - tx[i]) ** 2 + (yy - ty[i]) ** 2
        m = d2 <= (2 * fwhm) ** 2
        wgt = np.exp(-d2 / (2 * sg * sg))[m]
        v = res[i][m]
        ok = np.isfinite(v)
        if ok.sum() < 0.5 * m.sum():
            continue
        amps.append(float((v[ok] * wgt[ok]).sum() / math.sqrt(float((wgt[ok] ** 2).sum())) / noise))
    if len(amps) < max(4, n // 2):
        return None
    a = np.array(amps)
    return float(a.sum() / math.sqrt(len(a))), float(np.mean(a > 0.5))


def static_vs_mover(block: np.ndarray, dts: np.ndarray, c: Candidate, fwhm: float) -> float | None:
    """Mover vs. static source with variable brightness: T = (E_mover − E_static) / E_mover.

    Both hypotheses get a free (non-negative) flux in every frame – real
    objects in a low, partly obstructed series vary a lot, too – so only the
    geometry differs: PSF following the track, or PSF at one fixed position
    (best within the track's extent). E = Σ_i max(projection_i, 0)² of the
    median-subtracted frames. A star that brightens or changes its shape
    during the sequence is explained at least as well by the static model
    (T ≤ 0); a mover moving ≳ 1 FWHM is not. CTA-S1 304/4–5: false median
    T −0.3 … −1.1, real ones +0.4 (none below −0.55)."""
    n, h, w = block.shape
    sg = max(fwhm / 2.3548, 0.6)
    tx, ty = c.x + c.vx * dts, c.y + c.vy * dts
    cx, cy = float(tx.mean()), float(ty.mean())
    half = max(1.0, 0.5 * math.hypot(tx[-1] - tx[0], ty[-1] - ty[0])) + 1.0
    W = int(half + 3 * fwhm + 2)
    x0, x1, y0, y1 = int(cx) - W, int(cx) + W + 1, int(cy) - W, int(cy) + W + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    cut = np.asarray(block[:, y0:y1, x0:x1], np.float32)
    with _quiet():
        R = cut - np.nanmedian(cut, axis=0)[None]
        noise = 1.4826 * float(np.nanmedian(np.abs(R - np.nanmedian(R))))
    if not np.isfinite(noise) or noise <= 0:
        return None
    R = np.nan_to_num(R / noise)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    def g(px, py):
        k = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sg * sg))
        return k / math.sqrt(float((k * k).sum()))
    # both energies leave out the largest max(1, n/10) frame terms: one frame with a cosmic or a
    # satellite next to the track must not decide (88P/Coihueco 11. 11.: (3312) Pedersen, an
    # unmasked 48 σ hit 3 px from the track in frame 26 gave the static model 2 343 vs 998 → T −1.35,
    # "proměnná hvězda"; without it the static model has ~150)
    k_out = max(1, n // 10) if n >= 8 else 0

    def energy(p):
        e = np.sort(np.clip(p, 0, None) ** 2)
        return float(e[:len(e) - k_out].sum())

    pb = np.array([float((R[i] * g(tx[i], ty[i])).sum()) for i in range(n)])
    wp = np.clip(pb, 0, None)
    e_mov = energy(pb)
    if e_mov <= 0:
        return -10.0
    # the static source must sit where the track's signal is: near the signal-weighted
    # centre of the track (a real mover has it in the middle, a track crossing a
    # variable star has it on the star) – another variable star elsewhere along a
    # long track must not count
    wc = wp ** 2 / max(float((wp ** 2).sum()), 1e-12)
    sx, sy = float((wc * tx).sum()), float((wc * ty).sum())
    lim = max(1.5 * fwhm, 2.0)
    e_st = 0.0
    for ox in np.arange(-half, half + 1e-6, 0.5):
        for oy in np.arange(-half, half + 1e-6, 0.5):
            px, py = cx + ox, cy + oy
            if math.hypot(px - sx, py - sy) > lim:
                continue
            pa = np.tensordot(R, g(px, py), axes=([1, 2], [0, 1]))
            e_st = max(e_st, energy(pa))
    return (e_mov - e_st) / e_mov


# --------------------------------------------------------------------------
# Empirical noise threshold (time-shuffled null search)
# --------------------------------------------------------------------------

def noise_threshold(detect, vet_full, dts: np.ndarray, fwhm: float, noise_sigma: float, sp,
                    area_mpx: float, seed: int = 0, weights=None) -> tuple[float, dict]:
    """SNR threshold of one tile from the same search run on time-shuffled frames.

    A fixed SNR threshold assumes Gaussian noise. In dense fields (Milky Way)
    star residuals – seeing, guiding, throughput – give a heavy tail: 161P,
    Coihueco 6. 11. 2026: 14 470 detections, 2 305 passing all checks, a
    handful real. Shuffling which frame belongs to which time scrambles every
    real mover, while the per-frame noise and the star residuals stay where
    they are (a slow shuffled track still covers the same pixels). Detections
    that pass the full vetting on shuffled frames are therefore false by
    construction; their SNR distribution gives the expected number of false
    candidates above a threshold. The tile threshold is the lowest SNR (≥ the
    base threshold) at which this expectation is ≤ ``null_false_per_mpx`` ×
    tile area.

    The TrackNorm σ does not change under a permutation (it depends only on
    the multiset of time differences), so SNRs are on the same scale. To keep
    the cost low only a stratified sample of null detections is fully vetted
    (all above base + 2, at most ``null_sample_per_bin`` per 0.5-σ bin below),
    each weighted by its bin's sampling fraction.

    ``detect(dts, tn)`` and ``vet_full(cands, dts, tn)`` are the pipeline's own
    tile functions, so the null detections go through exactly the same rules."""
    base = float(sp.snr_threshold)
    k_perm = max(1, int(getattr(sp, "null_permutations", 2)))
    per_bin = max(5, int(getattr(sp, "null_sample_per_bin", 25)))
    max_add = float(getattr(sp, "null_max_add", 4.0))
    target = float(getattr(sp, "null_false_per_mpx", 1.0)) * max(area_mpx, 1e-3)
    rng = np.random.default_rng(seed)
    n = len(dts)
    passed_snr, passed_w = [], []
    n_null = 0
    order = np.argsort(dts)
    for _ in range(k_perm):
        # a random permutation that moves frames far in time: no frame keeps a near neighbour
        for _try in range(20):
            perm = rng.permutation(n)
            pos = np.empty(n, int)
            pos[order] = np.arange(n)              # time rank of every frame
            if np.mean(np.abs(pos - pos[perm])) > n / 4:
                break
        d_p = np.asarray(dts, float)[perm]
        tn_p = TrackNorm(d_p, fwhm, noise_sigma, weights=weights, amps=weights)   # weights belong to frames, not times
        cn = detect(d_p, tn_p)
        cn = [c for c in cn if c.snr >= base]          # detect() also returns the "druhá šance" band below
        n_null += len(cn)
        sample, weight = [], []
        hi = [c for c in cn if c.snr >= base + 2.0]
        sample += hi
        weight += [1.0] * len(hi)
        lo = [c for c in cn if c.snr < base + 2.0]
        bins = np.floor((np.array([c.snr for c in lo]) - base) / 0.5).astype(int) if lo else np.array([], int)
        for b in np.unique(bins):
            idx = np.nonzero(bins == b)[0]
            take = idx if len(idx) <= per_bin else rng.choice(idx, per_bin, replace=False)
            sample += [lo[i] for i in take]
            weight += [len(idx) / len(take)] * len(take)
        vet_full(sample, d_p, tn_p)
        for c, wgt in zip(sample, weight):
            if c.auto_status == "candidate":
                passed_snr.append(c.snr)
                passed_w.append(wgt)
    ps, pw = np.array(passed_snr, float), np.array(passed_w, float)

    def expected(s: float) -> float:
        return float(pw[ps >= s].sum()) / k_perm if len(ps) else 0.0

    thr = base
    for s in np.arange(base, base + max_add + 1e-9, 0.1):
        thr = float(s)
        if expected(s) <= target:
            break
    info = {"threshold": round(thr, 2), "expected_at_base": round(expected(base), 2),
            "expected_at_threshold": round(expected(thr), 2), "target": round(target, 2),
            "null_detections": round(n_null / k_perm, 1), "permutations": k_perm,
            "null_passed_sampled": int(len(ps))}
    return thr, info
