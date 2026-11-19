"""Segment test of slow movers under falling throughput (dawn): LS template model."""
import math

import numpy as np

from synthtrack import search as SR
from synthtrack.tracking import Candidate


def _sim(seed, amps_model):
    rng = np.random.default_rng(seed)
    n, H = 17, 96
    dts = np.linspace(-18, 14, n)
    A = np.linspace(1.1, 0.32, n)                   # 317 dawn profile
    fwhm = 3.0
    sp = fwhm / 2.3548
    vx, vy = 0.06, 0.03                             # ~1 FWHM over the series
    yy, xx = np.mgrid[0:H, 0:H]
    cube = np.empty((n, H, H), np.float32)
    flux = 60.0
    for i in range(n):
        x, y = 48 + vx * dts[i], 48 + vy * dts[i]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sp * sp)) / (2 * math.pi * sp * sp)
        cube[i] = rng.normal(0, 1, (H, H)) + A[i] * flux * g
    amp = np.repeat(A, H * H).reshape(n, H, H).astype(np.float32)
    pt = SR.prepare_tile(cube, fwhm, 50.0, cap=0, amp=amp, amp_min=0.3)
    w = A.copy()
    pt.f_raw = pt.f_raw * w[:, None, None]
    pt.f = pt.f_raw
    tn = SR.TrackNorm(dts, fwhm, pt.noise_sigma, weights=w, amps=A if amps_model else None)
    c = Candidate(id=0, x=48.0, y=48.0, vx=vx, vy=vy, snr=10.0)
    SR.vet([c], pt, dts, tn, 3, 3.0, 50.0, fwhm)
    return c


def test_slow_mover_segments_consistent_at_dawn():
    bad_new = sum("nekonzistentní segmenty" in _sim(s, True).flags for s in range(12))
    assert bad_new <= 1


def test_plain_mean_model_was_wrong_at_dawn():
    # documents the bug fixed by the LS-template model (without ``amps`` the old model is used)
    bad_old = sum("nekonzistentní segmenty" in _sim(s, False).flags for s in range(12))
    assert bad_old >= 6
