import numpy as np

from synthtrack.defects import LineProfiler
from synthtrack.identify import comet_total_mag


def test_bad_columns_found_and_masked():
    rng = np.random.default_rng(1)
    h, w = 1024, 512
    med = rng.normal(0, 1 / np.sqrt(20), (h, w)).astype(np.float32)   # temporal median of 20 frames
    med[:, 100] += 0.4                 # weak hot column (way below the hot-pixel threshold)
    med[:, 300] -= 0.9                 # cold column
    med[:384, 200] += 1.3              # partial column: a trap that ends at row ~384
    med[500:520, 400:420] += 30        # a star must not flag anything
    lp = LineProfiler((h, w), block=128)
    lp.add(med, 0)
    d = lp.find()
    assert 100 in d.columns and 300 in d.columns
    assert 200 in d.columns or {k for c, k in d.column_blocks if c == 200} >= {0, 1, 2}
    flagged = set(d.columns) | {c for c, _ in d.column_blocks}
    assert flagged <= {99, 100, 101, 199, 200, 201, 299, 300, 301}, flagged
    assert not d.rows and not d.row_blocks
    img = np.zeros((h, w), np.float32)
    d.apply(img, 0)
    assert np.isnan(img[:, 100]).all() and np.isnan(img[:, 300]).all() and np.isnan(img[:300, 200]).all()
    assert np.isfinite(img[:, 50]).all()


def test_comet_magnitude_uses_2_5_k():
    # C/1995 O1 (Hale-Bopp) in CometEls: H = -2.0, K = 4.0; r ≈ 50 au, Δ ≈ 50 au in 2026
    m = float(comet_total_mag(-2.0, 4.0, 50.0, 50.0))
    assert abs(m - (-2.0 + 5 * np.log10(50) + 10 * np.log10(50))) < 1e-6
    assert m > 22
    assert np.isnan(comet_total_mag(np.nan, 4.0, 2.0, 1.0))
