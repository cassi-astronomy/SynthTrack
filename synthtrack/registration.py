"""Star-based registration with a polynomial distortion model.

Wide-field optics (e.g. a 135 mm lens, 15°×15°) are strongly distorted and
dithering moves the stars by hundreds of pixels, so a pure translation leaves
residuals of several pixels at the field edges – fatal for star subtraction.
Each frame is therefore matched to the reference frame star by star:

1. stars are detected on the (background-flattened, σ-normalised) frames,
2. the initial offset comes from the WCS of both frames if present, otherwise
   from a vote over all brightest-star pair differences (robust to dithering
   of any size, rotation must be small – the same mount/camera orientation),
3. stars are paired by nearest neighbour and a 2-D polynomial (default 3rd
   order) is fitted with iterative sigma clipping, in both directions
   (reference→frame for warping, frame→reference for detector-fixed masks),
4. the frame is resampled onto the reference pixel grid (cubic spline).

The fit residual RMS is reported per frame; with good data it is ~0.05 px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Star detection
# --------------------------------------------------------------------------

def detect_stars(img: np.ndarray, nsigma: float = 8.0, max_n: int = 4000, sat_sigma: float | None = None,
                 border: int = 8) -> np.ndarray:
    """Centroids of point sources (N, 3): x, y, flux. ``img`` in sigma units."""
    sm = ndimage.gaussian_filter(np.nan_to_num(img), 1.0)
    mx = ndimage.maximum_filter(sm, size=7)
    pk = (sm == mx) & (sm > nsigma)
    pk[:border] = pk[-border:] = False
    pk[:, :border] = pk[:, -border:] = False
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return np.zeros((0, 3))
    vals = sm[ys, xs]
    order = np.argsort(vals)[::-1]
    if sat_sigma is not None:
        order = order[vals[order] < sat_sigma]
    order = order[: max_n * 2]
    ys, xs = ys[order], xs[order]
    r = 2
    ok = (ys >= r) & (xs >= r) & (ys < img.shape[0] - r) & (xs < img.shape[1] - r)
    ys, xs = ys[ok], xs[ok]
    if len(ys) == 0:
        return np.zeros((0, 3))
    # vectorised 5×5 centroids around all peaks at once
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    cut = np.nan_to_num(img[ys[:, None, None] + oy[None], xs[:, None, None] + ox[None]])
    cut = np.clip(cut - np.median(cut.reshape(len(ys), -1), axis=1)[:, None, None], 0, None)
    tot = cut.sum(axis=(1, 2))
    good = tot > 0
    cx = xs[good] + (cut[good] * ox).sum(axis=(1, 2)) / tot[good]
    cy = ys[good] + (cut[good] * oy).sum(axis=(1, 2)) / tot[good]
    out = np.stack([cx, cy, tot[good]], axis=1)[:max_n]
    return out


# --------------------------------------------------------------------------
# Polynomial transform
# --------------------------------------------------------------------------

def _terms(u, v, order):
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append(u ** i * v ** j)
    return np.stack(cols, axis=-1)


@dataclass
class PolyTransform:
    """(x, y) -> (x', y') as a polynomial in normalised coordinates."""
    order: int
    cx: float
    cy: float
    scale: float
    coef_x: np.ndarray
    coef_y: np.ndarray
    rms: float = 0.0
    n_used: int = 0

    def __call__(self, x, y):
        u = (np.asarray(x, float) - self.cx) / self.scale
        v = (np.asarray(y, float) - self.cy) / self.scale
        T = _terms(u, v, self.order)
        return T @ self.coef_x, T @ self.coef_y

    @classmethod
    def fit(cls, src, dst, order, center, scale, clip=3.0, iters=5) -> "PolyTransform":
        u = (src[:, 0] - center[0]) / scale
        v = (src[:, 1] - center[1]) / scale
        T = _terms(u, v, order)
        keep = np.ones(len(src), bool)
        for _ in range(iters):
            cx, *_ = np.linalg.lstsq(T[keep], dst[keep, 0], rcond=None)
            cy, *_ = np.linalg.lstsq(T[keep], dst[keep, 1], rcond=None)
            res = np.hypot(T @ cx - dst[:, 0], T @ cy - dst[:, 1])
            sig = 1.4826 * np.median(res[keep]) + 1e-3
            new = res < max(clip * sig, 0.05)
            if np.array_equal(new, keep):
                break
            keep = new
            if keep.sum() < T.shape[1] * 3:
                break
        rms = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.any() else float("inf")
        return cls(order, center[0], center[1], scale, cx, cy, rms, int(keep.sum()))

    @classmethod
    def identity(cls, shape):
        h, w = shape
        t = cls.fit(np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2], [w / 3, h / 4]], float),
                    np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2], [w / 3, h / 4]], float),
                    1, ((w - 1) / 2, (h - 1) / 2), max(w, h) / 2)
        return t


@dataclass
class FrameRegistration:
    ref_to_frame: PolyTransform
    frame_to_ref: PolyTransform
    n_matched: int
    rms_px: float
    method: str
    offset: tuple[float, float] = (0.0, 0.0)      # mean (dx, dy) frame - ref, informative
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"method": self.method, "n": self.n_matched, "rms_px": round(self.rms_px, 3),
                "dx": round(self.offset[0], 2), "dy": round(self.offset[1], 2)}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def vote_offset(ref: np.ndarray, cur: np.ndarray, n: int = 250, bin_px: float = 2.0,
                max_shift: float | None = None) -> tuple[float, float, int]:
    """Translation cur - ref from a 2-D histogram of all bright-star pair differences."""
    a = ref[np.argsort(-ref[:, 2])[:n], :2]
    b = cur[np.argsort(-cur[:, 2])[:n], :2]
    d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
    if max_shift:
        d = d[np.all(np.abs(d) < max_shift, axis=1)]
    if len(d) == 0:
        return 0.0, 0.0, 0
    lim = np.abs(d).max() + bin_px
    nb = int(2 * lim / bin_px) + 1
    H, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=nb, range=[[-lim, lim], [-lim, lim]])
    H = ndimage.uniform_filter(H, 2) * 4      # tolerate a peak split between bins
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    x0, y0 = 0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])
    sel = np.hypot(d[:, 0] - x0, d[:, 1] - y0) < 2 * bin_px
    if sel.sum() == 0:
        return x0, y0, 0
    return float(np.median(d[sel, 0])), float(np.median(d[sel, 1])), int(sel.sum())


def match_frame(ref: np.ndarray, cur: np.ndarray, shape: tuple[int, int], order: int = 3,
                initial=None, tol_px: float = 4.0) -> FrameRegistration:
    """Fit ref->frame and frame->ref polynomials from two star lists.

    ``initial``: optional callable mapping reference pixels to approximate
    frame pixels (e.g. from the two WCS solutions).
    """
    h, w = shape
    center, scale = ((w - 1) / 2, (h - 1) / 2), max(w, h) / 2
    notes = []
    if initial is None:
        dx, dy, votes = vote_offset(ref, cur)
        method = "stars"
        guess = lambda x, y: (x + dx, y + dy)                         # noqa: E731
        if votes < 8:
            notes.append(f"slabá shoda posunu ({votes} párů)")
    else:
        guess = initial
        method = "stars+wcs"
    tree = cKDTree(cur[:, :2])
    src = ref[:, :2]
    model = None
    for it, (o, tol) in enumerate(((1, tol_px), (min(2, order), tol_px * 0.6), (order, tol_px * 0.4),
                                   (order, tol_px * 0.3))):
        px, py = guess(src[:, 0], src[:, 1]) if model is None else model(src[:, 0], src[:, 1])
        dist, idx = tree.query(np.stack([px, py], axis=1), distance_upper_bound=tol)
        ok = np.isfinite(dist)
        # one-to-one: drop frame stars claimed twice
        u, cnt = np.unique(idx[ok], return_counts=True)
        dup = set(u[cnt > 1])
        ok &= np.array([i not in dup for i in idx])
        need = 3 * (o + 1) * (o + 2) // 2
        if ok.sum() < need:
            if model is None:
                raise RuntimeError(f"Registrace selhala: spárováno jen {int(ok.sum())} hvězd")
            break
        model = PolyTransform.fit(src[ok], cur[idx[ok], :2], o, center, scale)
    pairs_src, pairs_dst = src[ok], cur[idx[ok], :2]
    inv = PolyTransform.fit(pairs_dst, pairs_src, model.order, center, scale)
    off = np.median(pairs_dst - pairs_src, axis=0)
    return FrameRegistration(model, inv, model.n_used, model.rms, method, (float(off[0]), float(off[1])), notes)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def _coord_maps(tr: PolyTransform, shape, step: int = 32):
    """Full-resolution coordinate maps of ``tr``: exact on a coarse grid, bicubic in between.

    A cubic polynomial is smooth enough that a 32-px grid interpolated with a
    separable bicubic spline reproduces it to < 1e-4 px, ~10× faster than
    evaluating the polynomial (or map_coordinates) on every pixel."""
    from scipy.interpolate import RectBivariateSpline

    h, w = shape
    gy = np.unique(np.append(np.arange(0, h, step), h - 1)).astype(float)
    gx = np.unique(np.append(np.arange(0, w, step), w - 1)).astype(float)
    GX, GY = np.meshgrid(gx, gy)
    mx, my = tr(GX.ravel(), GY.ravel())
    yy, xx = np.arange(h, dtype=float), np.arange(w, dtype=float)
    MX = RectBivariateSpline(gy, gx, mx.reshape(GX.shape), kx=3, ky=3)(yy, xx).astype(np.float32)
    MY = RectBivariateSpline(gy, gx, my.reshape(GX.shape), kx=3, ky=3)(yy, xx).astype(np.float32)
    return MY, MX


def warp(img: np.ndarray, tr: PolyTransform, out_shape, order: int = 3, cval: float = np.nan) -> np.ndarray:
    """Sample ``img`` at tr(x_out, y_out) for every output pixel."""
    MY, MX = _coord_maps(tr, out_shape)
    bad = ~np.isfinite(img)
    src = np.where(bad, 0.0, img) if bad.any() else img
    out = ndimage.map_coordinates(src, [MY, MX], order=order, mode="constant", cval=np.nan,
                                  prefilter=order > 1)
    if bad.any():
        # masked pixels (saturated cores, hot pixels) stay masked after resampling –
        # filling them with 0 turned saturated stars into dark-centred rings that
        # moved with the dithering
        mb = ndimage.map_coordinates(bad.astype(np.float32), [MY, MX], order=1, mode="constant", cval=1.0)
        out[mb > 0.01] = np.nan
    if not np.isnan(cval):
        out = np.nan_to_num(out, nan=cval)
    return out.astype(np.float32)


def antialias(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian pre-smoothing of an undersampled σ-unit frame before resampling.

    Stars with FWHM ≲ 2.5 px change shape with the sub-pixel phase of any
    interpolation (spline overshoot = dark rings, peak ±30 %), which leaves
    star residuals after the template subtraction. Smoothing to FWHM ≈ 3 px
    cuts that 5–10×. NaNs are respected (normalised convolution) and the
    result is rescaled to unit white noise again."""
    if sigma <= 0:
        return img
    bad = ~np.isfinite(img)
    src = np.where(bad, 0.0, img).astype(np.float32)
    num = ndimage.gaussian_filter(src, sigma, mode="nearest")
    if bad.any():
        den = ndimage.gaussian_filter((~bad).astype(np.float32), sigma, mode="nearest")
        num = np.where(den > 0.3, num / np.maximum(den, 1e-3), np.nan)
        num[bad] = np.nan
    r = int(math.ceil(4 * sigma))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma))
    k /= k.sum()
    gain = 1.0 / float(np.sum(k * k))          # 2-D noise std after smoothing = Σk² (1-D kernel)
    return (num * gain).astype(np.float32)


def register_stack(frames: list[np.ndarray], ref_i: int, order: int = 3, wcs_list=None,
                   sat_sigma: float | None = None, log=None) -> list[FrameRegistration]:
    """Registrations of all frames to frame ``ref_i``."""
    shape = frames[ref_i].shape
    stars = [detect_stars(f, sat_sigma=sat_sigma) for f in frames]
    return register_from_stars(stars, shape, ref_i, order, wcs_list, log)


def match_quality(reg: FrameRegistration, ref: np.ndarray, cur: np.ndarray, shape,
                  max_offset_px: float | None = None, tol_px: float = 1.2) -> tuple[bool, str]:
    """Is ``reg`` a real match of the same sky, or chance pairs of two different fields?

    In a dense field a few stars always fall within the matching tolerance by
    chance, so a fit "succeeds" even between unrelated frames. A true match
    pairs far more stars than chance would (``expected``) and a decent share
    of the smaller star list; the offset must stay within the dithering limit."""
    h, w = shape
    n_ref, n_cur = len(ref), len(cur)
    if reg.rms_px > 1.0:
        return False, f"RMS {reg.rms_px:.2f} px – náhodné páry hvězd, ne stejné pole"
    # chance pairs within the radius the matched pairs actually spread over
    tol = max(tol_px, 2.5 * reg.rms_px)
    expected = n_ref * (n_cur / float(h * w)) * math.pi * tol ** 2
    need = max(12.0, 8.0 * expected, 0.15 * min(n_ref, n_cur))
    if reg.n_matched < need:
        return False, f"spárováno {reg.n_matched} hvězd z {min(n_ref, n_cur)} (náhodou ~{expected:.0f})"
    off = math.hypot(*reg.offset)
    if max_offset_px is not None and off > max_offset_px:
        return False, f"posun {off:.0f} px je větší než dithering v jednom poli (max {max_offset_px:.0f} px)"
    return True, ""


def register_from_stars(stars: list[np.ndarray], shape, ref_i: int, order: int = 3, wcs_list=None,
                        log=None, failed: list | None = None,
                        max_offset_px: float | None = None) -> list[FrameRegistration | None]:
    """Register every frame to ``stars[ref_i]``.

    With ``failed`` (a list) a frame that cannot be matched – or whose match
    does not pass ``match_quality`` (a frame of another field) – gets ``None``
    and ``(index, reason)`` is appended to ``failed``; without it a failure
    raises as before."""
    ref = stars[ref_i]
    if len(ref) < 30:
        raise RuntimeError(f"Na referenčním snímku nalezeno jen {len(ref)} hvězd")
    regs = []
    for i, cur in enumerate(stars):
        if i == ref_i:
            ident = PolyTransform.identity(shape)
            regs.append(FrameRegistration(ident, ident, len(ref), 0.0, "reference"))
            continue
        initial = None
        if wcs_list is not None and wcs_list[i] is not None and wcs_list[ref_i] is not None:
            wr, wi = wcs_list[ref_i], wcs_list[i]

            def initial(x, y, wr=wr, wi=wi):
                sky = wr.pixel_to_world(x, y)
                return wi.world_to_pixel(sky)
        try:
            try:
                reg = match_frame(ref, cur, shape, order, initial)
            except RuntimeError:
                if initial is None:
                    raise
                reg = match_frame(ref, cur, shape, order, None)
        except RuntimeError as e:
            if failed is None:
                raise
            failed.append((i, str(e)))
            regs.append(None)
            if log:
                log(f"  snímek {i:2d}: NESESAZEN – {e}")
            continue
        if failed is not None:
            ok, why = match_quality(reg, ref, cur, shape, max_offset_px)
            if not ok:
                failed.append((i, why))
                regs.append(None)
                if log:
                    log(f"  snímek {i:2d}: NESESAZEN – {why}")
                continue
        regs.append(reg)
        if log:
            log(f"  snímek {i:2d}: {reg.n_matched} hvězd, RMS {reg.rms_px:.3f} px, "
                f"posun ({reg.offset[0]:+.1f}, {reg.offset[1]:+.1f}) px")
    return regs


def max_shift(regs: list[FrameRegistration]) -> float:
    return max(math.hypot(*r.offset) for r in regs) if regs else 0.0
