"""Bad columns and rows of the detector (CCD charge traps, hot/cold columns).

A column that is only ~0.3–2 σ above (or below) the sky in every pixel is
far below the hot-pixel threshold, but it is 4000 pixels long. With
dithering it moves on the registered grid, and when the dither drifts
(tracking, field rotation) a straight track along the column fits it
perfectly – dozens of false candidates on one column. Detected here from
the temporal median of the *unregistered* frames (detector coordinates),
averaged along the column in blocks, so levels of ~0.1 σ are significant.
Partial columns (a trap that starts at some row) are found per block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class LineDefects:
    block: int
    columns: list[int] = field(default_factory=list)            # whole columns
    column_blocks: list[tuple[int, int]] = field(default_factory=list)   # (column, block index)
    rows: list[int] = field(default_factory=list)
    row_blocks: list[tuple[int, int]] = field(default_factory=list)      # (row, block index)
    levels: dict = field(default_factory=dict)                   # "c275" -> mean level (σ)

    def apply(self, img: np.ndarray, y0: int = 0) -> int:
        """Set the defects to NaN in `img` (rows y0 … y0+len(img) of the frame). Returns #pixels."""
        h, w = img.shape
        b = self.block
        n0 = int(np.isnan(img).sum())
        for c in self.columns:
            img[:, c] = np.nan
        for c, k in self.column_blocks:
            a, z = max(k * b - y0, 0), min((k + 1) * b - y0, h)
            if a < z:
                img[a:z, c] = np.nan
        for r in self.rows:
            if y0 <= r < y0 + h:
                img[r - y0, :] = np.nan
        for r, k in self.row_blocks:
            if y0 <= r < y0 + h:
                img[r - y0, k * b:(k + 1) * b] = np.nan
        return int(np.isnan(img).sum()) - n0

    def describe(self) -> str:
        parts = []
        if self.columns:
            parts.append(f"{len(self.columns)} celých sloupců")
        if self.column_blocks:
            parts.append(f"{len({c for c, _ in self.column_blocks})} částečných sloupců")
        if self.rows:
            parts.append(f"{len(self.rows)} řádků")
        if self.row_blocks:
            parts.append(f"{len({r for r, _ in self.row_blocks})} částečných řádků")
        return ", ".join(parts) if parts else "žádné"

    def worst(self, k: int = 5) -> str:
        items = sorted(self.levels.items(), key=lambda kv: -abs(kv[1]))[:k]
        return ", ".join(f"{'sloupec' if n[0] == 'c' else 'řádek'} {n[1:]} ({v:+.2f} σ)" for n, v in items)


class LineProfiler:
    """Accumulates block medians of the temporal-median image chunk by chunk."""

    def __init__(self, shape: tuple[int, int], block: int = 128):
        self.h, self.w = shape
        self.block = block
        self.col = np.full((-(-self.h // block), self.w), np.nan, np.float32)   # (row blocks, columns)
        self.row = np.full((self.h, -(-self.w // block)), np.nan, np.float32)   # (rows, column blocks)

    def add(self, med: np.ndarray, y0: int) -> None:
        """`med` = temporal median of rows y0 … y0+len(med); y0 must be a multiple of `block`."""
        b = self.block
        assert y0 % b == 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in range(0, med.shape[0], b):
                self.col[(y0 + k) // b] = np.nanmedian(med[k:k + b], axis=0)
            for j in range(0, self.w, b):
                self.row[y0:y0 + med.shape[0], j // b] = np.nanmedian(med[:, j:j + b], axis=1)

    def find(self, z_block: float = 6.0, z_line: float = 6.0, min_level: float = 0.08,
             max_fraction: float = 0.03) -> LineDefects:
        out = LineDefects(self.block)
        for axis in ("c", "r"):
            P = self.col if axis == "c" else self.row.T        # (blocks, lines)
            if P.shape[1] < 32:
                continue
            filled = np.where(np.isfinite(P), P, np.nanmedian(P))
            R = P - ndimage.median_filter(filled, size=(1, 15), mode="nearest")
            with np.errstate(invalid="ignore"):
                sig_b = 1.4826 * np.nanmedian(np.abs(R), axis=1, keepdims=True) + 1e-4
                mean = np.nanmean(R, axis=0)
            nb = np.isfinite(R).sum(axis=0)
            sig_l = 1.4826 * np.nanmedian(np.abs(mean)) + 1e-4
            whole = np.isfinite(mean) & (np.abs(mean) > z_line * sig_l) & (np.abs(mean) > min_level) & (nb >= 2)
            part = np.isfinite(R) & (np.abs(R) > z_block * sig_b) & (np.abs(R) > 2 * min_level)
            part[:, whole] = False
            # a trap starts somewhere and continues: a lone block is kept only if strong
            runs = ndimage.convolve(part.astype(np.uint8), np.ones((3, 1), np.uint8), mode="constant") >= 2
            part &= runs | (np.abs(np.nan_to_num(R)) > 3 * z_block * sig_b)
            part = ndimage.binary_dilation(part, structure=np.ones((3, 1), bool))   # ±1 block margin
            nlines = P.shape[1]
            if whole.sum() > max_fraction * nlines:          # e.g. a gradient pattern: keep the strongest
                keep = np.argsort(-np.abs(np.nan_to_num(mean)))[: int(max_fraction * nlines)]
                m = np.zeros_like(whole)
                m[keep] = True
                whole &= m
            lines = [int(i) for i in np.nonzero(whole)[0]]
            blocks = [(int(i), int(k)) for k, i in zip(*np.nonzero(part))]
            for i in lines:
                out.levels[f"{axis}{i}"] = float(mean[i])
            for i, k in blocks:
                key = f"{axis}{i}"
                if abs(float(R[k, i])) > abs(out.levels.get(key, 0.0)):
                    out.levels[key] = float(R[k, i])
            if axis == "c":
                out.columns, out.column_blocks = lines, blocks
            else:
                out.rows, out.row_blocks = lines, blocks
        return out


def chip_fixed(cube: np.ndarray, med: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
               level_frac: float = 0.5, min_level: float = 3.0, min_excess: float = 0.2) -> np.ndarray:
    """Which bright pixels of a temporal median in chip coordinates are fixed on the chip.

    The temporal median in chip coordinates was assumed to contain only hot
    pixels once the frames are dithered. That holds for a wide random
    dithering, not for a small or clustered one (161P, Coihueco: 18 frames in
    three groups 2–4 px apart): a star sits on the same chip pixel in more
    than half of the frames, its median peak is 7–10 σ and it was masked as a
    "hot pixel" in every frame – the hole then zeroed a comet crossing it.

    Two hypotheses are compared for every candidate pixel p (``cand`` = (m, 2)
    rows, cols in ``cube``):
      * fixed on the chip – bright at p in every frame,
      * fixed on the sky – bright at p + d_i − d_j in frame i, i.e. a star that
        lies on p in frame j (``offsets`` (n, m, 2): chip position of the
        candidate's sky point in frame i minus p, from the registration).
    F = fraction of frames brighter than ``level_frac`` × median value
    (3×3 maximum, sub-pixel registration errors); the best j is taken for the
    sky hypothesis. A pixel is chip-fixed when F_chip − F_sky ≥ ``min_excess``.
    Ties (no dithering between the frames that matter) keep the data."""
    n, h, w = cube.shape
    m = len(cand)
    if m == 0:
        return np.zeros(0, bool)
    ys, xs = cand[:, 0], cand[:, 1]
    lev = np.maximum(min_level, level_frac * med[ys, xs])
    d = np.rint(offsets).astype(int)                       # (n, m, 2)
    f_chip = np.zeros(m, np.float32)
    hits = np.zeros((n, m), np.float32)                    # sky hypothesis j, summed over frames i
    for i in range(n):
        fr = np.nan_to_num(cube[i], nan=-1e30, posinf=-1e30, neginf=-1e30)
        mx = ndimage.maximum_filter(fr, size=3, mode="nearest")
        f_chip += mx[ys, xs] > lev
        for j in range(n):
            py = np.clip(ys + d[i, :, 0] - d[j, :, 0], 0, h - 1)
            px = np.clip(xs + d[i, :, 1] - d[j, :, 1], 0, w - 1)
            hits[j] += mx[py, px] > lev
    f_sky = hits.max(axis=0) / n
    return (f_chip / n - f_sky) >= min_excess


def chip_fixed_weak(cube: np.ndarray, med: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
                    level_frac: float = 0.5, min_frac: float = 0.5) -> np.ndarray:
    """Weak warm pixels (temporal chip median ~1.5–6 σ) that ``chip_fixed`` cannot see.

    ``chip_fixed`` works with a fixed 3 σ level and a 3×3 maximum; for a 2–4 σ pixel
    the sky hypothesis is then satisfied by *any* star within the dither range
    (11P/Coihueco 8. 11. 2026: 28 frames, a 11 px drift-like dither, 7 400 of
    11 600 passing detections moved exactly with the drift and sat on single
    pixels 2–6 σ in the chip median – below the 6 σ hot-pixel cut).

    Here every frame is tested on its own: frame i counts as *unexplained* when
    the chip pixel p is above ``level_frac`` × median there but the static sky
    under p in that frame – the median over the other frames k of the chip pixel
    p − d_i + d_k, i.e. the same sky point – is below that level. A star explains
    its own brightness (the sky point under p is bright), a warm pixel does not.
    Fixed on the chip when ≥ ``min_frac`` of the frames are unexplained.
    ``cand`` (m, 2) rows, cols; ``offsets`` (n, m, 2) as in ``chip_fixed``."""
    n, h, w = cube.shape
    m = len(cand)
    if m == 0:
        return np.zeros(0, bool)
    ys, xs = cand[:, 0], cand[:, 1]
    lev = (level_frac * med[ys, xs]).astype(np.float32)
    d = np.rint(offsets).astype(int)                       # (n, m, 2)
    unexplained = np.zeros(m, np.float32)
    vals = np.empty((n, m), np.float32)
    for i in range(n):
        for k in range(n):
            if k == i:
                vals[k] = np.nan                           # the tested value itself stays out
                continue
            py = np.clip(ys - d[i, :, 0] + d[k, :, 0], 0, h - 1)
            px = np.clip(xs - d[i, :, 1] + d[k, :, 1], 0, w - 1)
            vals[k] = cube[k, py, px]
        sky = _nanmedian_rows(vals)
        ci = cube[i, ys, xs]
        with np.errstate(invalid="ignore"):
            unexplained += (ci > lev) & ~(sky >= lev)       # NaN sky (no data) does not explain it
    return unexplained / n >= min_frac


def _nanmedian_rows(v: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring NaN (vectorised; NaN where no value)."""
    t = np.ascontiguousarray(v.T)                     # sort along the last, contiguous axis
    fin = np.isfinite(t)
    cnt = fin.sum(axis=1)
    srt = np.sort(np.where(fin, t, np.inf), axis=1)   # no NaN in the sort (see search._nanmedian0)
    lo = np.maximum((cnt - 1) // 2, 0)[:, None]
    hi = np.maximum(cnt // 2, 0)[:, None]
    with np.errstate(invalid="ignore"):
        m = 0.5 * (np.take_along_axis(srt, lo, 1) + np.take_along_axis(srt, hi, 1))[:, 0]
    return np.where(cnt > 0, m, np.nan)


def inpaint_pixels(frame: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> None:
    """Replace the listed pixels by the median of their unmasked 8 neighbours (in place).

    Used for isolated hot / warm pixels instead of NaN: ``warp`` turns one missing input
    pixel into ~4 missing output pixels, and with 1–2 % of the chip masked every star and
    asteroid lost its core in ~30 % of the frames (88P/Coihueco 8. 11.: (149) Medusa, (419)
    Aurelia, (494) Virtus 12–14 mag locked onto a wrong, slow velocity – "nekonzistentní
    segmenty"). A neighbour median keeps the frame usable; the pixel's own (hot) value is
    never used."""
    h, w = frame.shape
    if len(ys) == 0:
        return
    bad = np.zeros((h, w), bool)
    bad[ys, xs] = True
    vals = np.full((8, len(ys)), np.nan, np.float32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            yy, xx = np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)
            v = frame[yy, xx].astype(np.float32)
            v[~ok | bad[yy, xx]] = np.nan
            vals[k] = v
            k += 1
    frame[ys, xs] = _nanmedian_rows(vals)
