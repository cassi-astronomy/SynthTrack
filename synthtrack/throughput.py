"""Per-frame throughput maps (transparency × gain of the σ normalisation).

Low-altitude fields, dome or horizon obstruction, passing cirrus and the
dark-sky → twilight change make the signal of a star differ between frames
*and across the field*: at airmass 3 the extinction gradient over a 16°
field is tens of per cent and an obstructed corner can go from 0 to 100 %
during the sequence. A single scale factor per tile cannot remove that, the
static-sky template then leaves star residuals that the shift-and-add sums
into slow "movers".

The map A_i(p) is the ratio of star fluxes in registered frame i to their
typical flux, median-binned on a coarse grid and interpolated bilinearly.
The search then models every pixel as d_i = A_i·S + mover + noise
(``search.amp_clipped_mean``), i.e. the residual d_i − A_i·S stays unit noise
even where a frame is (partly) blocked.

Pure NumPy; works on a memmap (only star stamps are read).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ThroughputMaps:
    cell: int                 # grid cell size in px
    grid: np.ndarray          # (n, gy, gx) float32, throughput of frame i in cell (≈ 1)
    shape: tuple[int, int]    # (h, w) of the registered frames
    n_stars: int = 0

    # -------------------------------------------------------------- queries
    def spread(self) -> float:
        """Largest spatial variation within one frame (p95 − p5 of the cells)."""
        g = self.grid.reshape(self.grid.shape[0], -1)
        return float(np.max(np.percentile(g, 95, axis=1) - np.percentile(g, 5, axis=1)))

    def per_frame_min(self) -> np.ndarray:
        return self.grid.reshape(self.grid.shape[0], -1).min(axis=1)

    def tile(self, y0: int, y1: int, x0: int, x1: int, frames=None) -> np.ndarray:
        """(n, y1−y0, x1−x0) float32 bilinear interpolation of the cell centres."""
        n, gy, gx = self.grid.shape
        idx = range(n) if frames is None else frames
        u = np.clip((np.arange(x0, x1) + 0.5) / self.cell - 0.5, 0, gx - 1)
        v = np.clip((np.arange(y0, y1) + 0.5) / self.cell - 0.5, 0, gy - 1)
        ux0 = np.floor(u).astype(np.intp)
        ux1 = np.minimum(ux0 + 1, gx - 1)
        wx = (u - ux0).astype(np.float32)
        vy0 = np.floor(v).astype(np.intp)
        vy1 = np.minimum(vy0 + 1, gy - 1)
        wy = (v - vy0).astype(np.float32)[:, None]
        out = np.empty((len(idx), y1 - y0, x1 - x0), np.float32)
        for k, i in enumerate(idx):
            g = self.grid[i]
            rows = g[:, ux0] * (1 - wx) + g[:, ux1] * wx          # (gy, w)
            out[k] = rows[vy0] * (1 - wy) + rows[vy1] * wy
        return out

    def describe(self, mjd_minutes=None) -> list[str]:
        """Human readable (Czech) summary of strongly affected frames."""
        lines = []
        mins = self.per_frame_min()
        g = self.grid.reshape(self.grid.shape[0], -1)
        for i, m in enumerate(mins):
            if m < 0.7:
                frac = float(np.mean(g[i] < 0.5))
                lines.append(f"snímek {i:2d}: propustnost klesá až na {100 * max(m, 0):.0f} %"
                             + (f", {100 * frac:.0f} % pole pod 50 %" if frac > 0 else ""))
        return lines

    def to_summary(self) -> dict:
        return {"cell_px": self.cell, "stars": self.n_stars, "spread": round(self.spread(), 3),
                "frame_min": [round(float(x), 3) for x in self.per_frame_min()]}


def _fill_holes(g: np.ndarray) -> np.ndarray:
    """Fill NaN cells from finite 8-neighbours (repeated), rest = median."""
    g = g.copy()
    for _ in range(max(g.shape)):
        bad = ~np.isfinite(g)
        if not bad.any():
            break
        p = np.pad(g, 1, constant_values=np.nan)
        stack = np.stack([p[1 + dy:1 + dy + g.shape[0], 1 + dx:1 + dx + g.shape[1]]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        cnt = np.isfinite(stack).sum(axis=0)
        mean = np.where(cnt > 0, np.nansum(np.nan_to_num(stack), axis=0) / np.maximum(cnt, 1), np.nan)
        upd = bad & (cnt > 0)
        if not upd.any():
            break
        g[upd] = mean[upd]
    if not np.isfinite(g).all():
        fill = float(np.median(g[np.isfinite(g)])) if np.isfinite(g).any() else 1.0
        g[~np.isfinite(g)] = fill
    return g


def measure(reg, stars: np.ndarray, cell: int = 256, r: int = 2, min_stars: int = 4,
            flux_min: float = 40.0) -> ThroughputMaps | None:
    """Throughput maps from aperture photometry of the reference-frame stars.

    ``reg``: (n, h, w) registered frames in σ units (memmap is fine),
    ``stars``: (N, 3) x, y, flux of the reference frame (same pixel grid).
    Returns None when there are too few usable stars."""
    n, h, w = reg.shape
    if stars is None or len(stars) < 20:
        return None
    st = np.asarray(stars, float)
    st = st[np.isfinite(st).all(axis=1)]
    # skip the brightest 3 % (saturation, blooming) and the faintest (noisy)
    st = st[st[:, 2] <= np.percentile(st[:, 2], 97)]
    xi = np.rint(st[:, 0]).astype(np.intp)
    yi = np.rint(st[:, 1]).astype(np.intp)
    m = r + 4
    ok = (xi >= m) & (yi >= m) & (xi < w - m) & (yi < h - m)
    xi, yi = xi[ok], yi[ok]
    if len(xi) < 20:
        return None
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    ring = np.zeros((2 * r + 9, 2 * r + 9), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ry, rx = np.nonzero(ring)
    ry, rx = ry - (r + 4), rx - (r + 4)
    F = np.full((n, len(xi)), np.nan, np.float32)
    for i in range(n):
        img = reg[i]
        ap = np.asarray(img[yi[:, None] + oy.ravel()[None], xi[:, None] + ox.ravel()[None]], np.float32)
        bg = np.asarray(img[yi[:, None] + ry[None], xi[:, None] + rx[None]], np.float32)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            F[i] = ap.sum(axis=1) - ap.shape[1] * np.nanmedian(bg, axis=1)
        del img
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typ = np.nanpercentile(F, 70, axis=0)
    use = np.isfinite(typ) & (typ > flux_min) & (np.isfinite(F).sum(axis=0) >= max(3, n // 2))
    if use.sum() < 20:
        return None
    ratio = np.clip(F[:, use] / typ[use], -0.5, 3.0)
    xs, ys = xi[use], yi[use]
    gy, gx = max(1, -(-h // cell)), max(1, -(-w // cell))
    cid = (ys // cell) * gx + (xs // cell)
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    bounds = np.searchsorted(cid_s, np.arange(gy * gx + 1))
    grid = np.full((n, gy * gx), np.nan, np.float32)
    for c in range(gy * gx):
        a, b = bounds[c], bounds[c + 1]
        if b - a >= min_stars:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                grid[:, c] = np.nanmedian(ratio[:, order[a:b]], axis=1)
    if not np.isfinite(grid).any():
        return None
    grid = grid.reshape(n, gy, gx)
    for i in range(n):
        grid[i] = _fill_holes(grid[i])
    return ThroughputMaps(cell, np.clip(grid, 0.0, 3.0).astype(np.float32), (h, w), int(use.sum()))
