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
