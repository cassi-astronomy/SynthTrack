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
                 amp: np.ndarray | None = None, amp_min: float = 0.5) -> PreparedTile:
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
    # saturated stars: their masked cores (NaN) move with the dithering and leave
    # holes of varying shape – mask them generously and treat as no data
    holes = coverage < n
    edge = np.zeros_like(holes)
    eb = 24
    edge[:eb], edge[-eb:], edge[:, :eb], edge[:, -eb:] = True, True, True, True
    # only holes at bright stars (saturated cores) are masked generously; isolated
    # holes (hot pixels, defects – masked per frame) are simply no data
    near_bright = ndimage.maximum_filter(np.where(np.isfinite(static), static, 2.0 * bright_sigma),
                                         size=2 * int(fwhm) + 3) > 0.5 * bright_sigma
    hb = holes & ~edge & near_bright
    sat = ndimage.binary_dilation(hb, iterations=int(2 * fwhm) + 3) if hb.any() else np.zeros_like(holes)
    static = np.where(np.isfinite(static), static, 2.0 * bright_sigma).astype(np.float32)
    bright = ndimage.binary_dilation(static > bright_sigma, iterations=2) | sat
    f[:, sat] = 0.0
    f_raw = f.astype(np.float32)
    # one frame can contribute at most ±cap σ: a faint mover has ~1–3 σ per frame,
    # a cosmic ray or a hot pixel 10 σ+ in a single frame
    fc = np.clip(f_raw, -cap, cap) if cap and cap > 0 else f_raw
    return PreparedTile(fc, f_raw, static, s.astype(np.float32), bright, coverage,
                        noise_acf_sigma(f_raw, valid & ~bright[None], sig))


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

    def __init__(self, dts: np.ndarray, fwhm: float, noise_sigma: float | None = None):
        self.dts = np.asarray(dts, float)
        self.n = len(self.dts)
        self.s2 = (max(fwhm / 2.3548, 0.5)) ** 2                 # signal: PSF ⊛ filter overlap
        self.n2 = noise_sigma ** 2 if noise_sigma else self.s2    # noise: measured autocorrelation
        self.dt2 = (self.dts[:, None] - self.dts[None, :]) ** 2

    def rho(self, speed_px_min: float) -> np.ndarray:
        return np.exp(-(speed_px_min ** 2) * self.dt2 / (4.0 * self.s2))

    def rho_noise(self, speed_px_min: float) -> np.ndarray:
        return np.exp(-(speed_px_min ** 2) * self.dt2 / (4.0 * self.n2))

    def sigma(self, speed: float, subset=None) -> float:
        r = self.rho_noise(speed)
        if subset is None:
            return math.sqrt(max(self.n - r.sum() / self.n, 1e-6))
        s = np.asarray(subset)
        return math.sqrt(max(len(s) - r[np.ix_(s, s)].sum() / self.n, 1e-6))

    def expected(self, speed: float, subset=None) -> float:
        """Expected sum (in units of the per-frame template-free signal) over ``subset``.

        Frame i keeps the fraction 1 − c_i, c_i = mean_j ρ_ij: frames near the
        middle of a slow track overlap most with the template and carry least."""
        c = self.rho(speed).mean(axis=1)
        idx = np.arange(self.n) if subset is None else np.asarray(subset)
        return float(np.sum(1.0 - c[idx]))

    def frame_weights(self, speed: float) -> np.ndarray:
        return 1.0 - self.rho(speed).mean(axis=1)

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
                progress=None, should_stop=None) -> tuple[np.ndarray, np.ndarray]:
    """Shift-and-add over the velocity grid; best SNR and its velocity index per pixel.

    Velocities whose integer shift pattern is identical (common for slow
    movers on a coarse grid) are stacked only once; of such a group the one
    with the smallest σ_sum is kept (the sub-pixel refinement follows anyway).
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
    for j, k in enumerate(reps):
        acc.fill(0)
        for i in range(n):
            _shift_add(xp, acc, d[i], int(shifts[k, i, 0]), int(shifts[k, i, 1]))
        acc *= xp.float32(1.0 / sig[k])
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
    for c in cands:
        speed = math.hypot(c.vx, c.vy)
        wts = tn.frame_weights(speed)
        # report the un-clipped significance
        tot, _ = _sum_track(f, dts, c.x, c.y, c.vx, c.vy, range(n), 0)
        c.snr_raw = float(tot[0, 0]) / tn.sigma(speed)
        samples = []
        for i in range(n):
            cx = int(round(c.x + c.vx * dts[i]))
            cy = int(round(c.y + c.vy * dts[i]))
            if 0 <= cy < h and 0 <= cx < w and tile.coverage[cy, cx] > 0 and wts[i] >= 0.5 * wts.max():
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
            if 0 <= cy < h and 0 <= cx < w and tile.coverage[cy, cx] > 0:
                e_i = amp0 * wts[i]
                res.append((float(f[i, cy, cx]) - e_i) ** 2 / (1.0 + (0.3 * e_i) ** 2))
        c.chi2 = float(np.mean(res)) if res else 0.0
        # direction: the reversed motion must not fit as well (symmetric residuals do)
        rev, _ = _sum_track(f, dts, c.x, c.y, -c.vx, -c.vy, range(n), 0)
        c.snr_reverse = float(rev[0, 0]) / tn.sigma(speed)
        amp = float(tot[0, 0]) / max(tn.expected(speed), 1e-6)         # per-frame amplitude
        c.seg_snr, zs = [], []
        for sgm in segs:
            acc, _ = _sum_track(f, dts, c.x, c.y, c.vx, c.vy, sgm, 0)
            sig = tn.sigma(speed, sgm)
            c.seg_snr.append(round(float(acc[0, 0]) / sig, 2))
            exp_s = amp * tn.expected(speed, sgm)
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
        if max(abs(z) for z in zs) > seg_z_max:
            flags.append("nekonzistentní segmenty")
        if c.median_ratio < 0.3:
            flags.append("jednosnímkový artefakt")
        if c.star_contaminated:
            flags.append("u hvězdy")
        if c.star_contaminated and motion < 3 * fwhm:
            flags.append("reziduum hvězdy")
        if c.frames_used < 0.7 * n:
            flags.append("mimo pole v části snímků")
        if c.chi2 > 2.5:
            flags.append("nekonzistentní snímky")
        if c.snr_raw > 0 and c.snr_reverse > 0.7 * c.snr_raw and motion > 0.5 * fwhm:
            flags.append("bez směru pohybu")
        c.flags = flags
        hard = {"nekonzistentní segmenty", "jednosnímkový artefakt", "reziduum hvězdy", "mimo pole v části snímků",
                "nekonzistentní snímky", "bez směru pohybu"}
        c.auto_status = "auto_rejected" if hard.intersection(flags) else "candidate"


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
    n = block.shape[0]
    sig = max(fwhm / 2.3548, 0.6)
    excl = 1.5 * fwhm                        # object closer than this to a pixel: frame not in its template
    keep_min = max(4, n // 4)
    r = 6
    gx_, gy_ = np.meshgrid(np.arange(-r, r + 1, dtype=float), np.arange(-r, r + 1, dtype=float))
    track_x = c.x + c.vx * dts                # (n,)
    track_y = c.y + c.vy * dts
    allf = np.arange(n)[:, None, None]
    out = []
    for seg in np.array_split(np.arange(n), max(1, segments)):
        tm = float(np.mean(dts[seg]))
        xc, yc = c.x + c.vx * tm, c.y + c.vy * tm
        acc = np.zeros((2 * r + 1, 2 * r + 1), np.float64)
        cnt = np.zeros_like(acc)
        for i in seg:
            # place frame i so that the object (at x_i) lands on the sub-stack centre
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
            # distance of every pixel of the window to the object in every frame
            dist = np.hypot(X[None] - track_x[:, None, None], Y[None] - track_y[:, None, None])
            use = dist > excl
            few = use.sum(axis=0) < keep_min
            if few.any():                     # slow mover: take the frames where it is farthest away
                kth = np.sort(dist, axis=0)[n - keep_min]
                use = np.where(few[None], dist >= kth[None], use)
            use &= np.isfinite(V) & (A > 0.25)
            with _quiet():
                ratio = np.where(use, V / np.maximum(A, 1e-3), np.nan)
                S = np.nanmedian(ratio, axis=0) if use.any() else np.full(X.shape, np.nan)
            v = V[i] - A[i] * S
            okv = np.isfinite(v)
            acc[okv] += v[okv]
            cnt[okv] += 1
        st = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        gx, gy, flux, ok = _fit_gauss(st, r, r, sig, r=4)
        out.append((tm, xc + (gx - r), yc + (gy - r), flux, ok))
    return out


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
    pb = np.array([float((R[i] * g(tx[i], ty[i])).sum()) for i in range(n)])
    wp = np.clip(pb, 0, None)
    e_mov = float((wp ** 2).sum())
    if e_mov <= 0:
        return -10.0
    # the static source must sit where the track's signal is: near the signal-weighted
    # centre of the track (a real mover has it in the middle, a track crossing a
    # variable star has it on the star) – another variable star elsewhere along a
    # long track must not count
    wc = wp ** 2 / e_mov
    sx, sy = float((wc * tx).sum()), float((wc * ty).sum())
    lim = max(1.5 * fwhm, 2.0)
    e_st = 0.0
    for ox in np.arange(-half, half + 1e-6, 0.5):
        for oy in np.arange(-half, half + 1e-6, 0.5):
            px, py = cx + ox, cy + oy
            if math.hypot(px - sx, py - sy) > lim:
                continue
            pa = np.tensordot(R, g(px, py), axes=([1, 2], [0, 1]))
            e_st = max(e_st, float((np.clip(pa, 0, None) ** 2).sum()))
    return (e_mov - e_st) / e_mov
