"""Throughput-weighted shift-and-add (TrackNorm weights)."""
import math

import numpy as np

from synthtrack import search as SR

DTS = np.linspace(-20, 20, 15)


def _sim(amp, weights, v=(0.3, 0.1), trials=300, seed=0):
    """Monte-Carlo σ of the weighted, template-subtracted track sum on pure noise, and the
    mean sum for a mover whose per-frame amplitude is ``amp`` (unit-noise frames)."""
    rng = np.random.default_rng(seed)
    n = len(DTS)
    sums_noise, sums_obj = [], []
    s = 2.85 / 2.3548
    for _ in range(trials):
        f = rng.normal(0, 1, (n, 48, 48))
        from scipy import ndimage
        f = np.array([ndimage.gaussian_filter(x, s) for x in f])
        f /= f.std(axis=(1, 2), keepdims=True)
        sig = np.zeros_like(f)
        yy, xx = np.mgrid[:48, :48]
        for i, t in enumerate(DTS):
            sig[i] = amp[i] * np.exp(-((xx - 24 - v[0] * t) ** 2 + (yy - 24 - v[1] * t) ** 2) / (4 * s * s))
        for data, out in ((f, sums_noise), (f + sig, sums_obj)):
            u = data - data.mean(axis=0)
            u = u * weights[:, None, None]
            out.append(sum(SR._bilinear(u, np.array([i]), np.array([24 + v[0] * t]),
                                        np.array([24 + v[1] * t]))[0] for i, t in enumerate(DTS)))
    return np.std(sums_noise), np.mean(sums_obj) - np.mean(sums_noise)


def test_weighted_sigma_matches_simulation_and_beats_equal_weights():
    amp = np.r_[np.ones(10), np.full(5, 0.3)]               # last third at 30 % (dawn)
    tn_w = SR.TrackNorm(DTS, 2.85, weights=amp)
    tn_1 = SR.TrackNorm(DTS, 2.85)
    sw, ew = _sim(amp, amp, trials=150)
    s1, e1 = _sim(amp, np.ones(15), trials=150, seed=1)
    v = math.hypot(0.3, 0.1)
    assert abs(sw / tn_w.sigma(v) - 1) < 0.2
    assert abs(s1 / tn_1.sigma(v) - 1) < 0.2
    assert ew / sw > e1 / s1                                  # weighting raises the SNR
    assert tn_w.expected(v) / tn_w.sigma(v) > SR.TrackNorm(DTS, 2.85).expected(v) / tn_1.sigma(v) * 0.6
