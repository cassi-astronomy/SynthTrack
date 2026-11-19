"""Detector-fixed pattern with a drift-like dithering (161P, Coihueco 6. 11. 2026)."""
import math

import numpy as np

from synthtrack import search as SR

DTS = np.array([-26.954, -24.707, -22.471, -20.223, -17.98, -15.744, -13.513, -4.483, -2.234, 0.0,
                2.234, 4.498, 6.729, 8.965, 11.198, 13.434, 15.67, 17.943])
# measured registration offsets (dy, dx) of the real sequence: detector − grid
D = np.array([[2.33, 2.15], [2.22, 2.09], [2.45, 2.2], [2.57, 2.34], [2.55, 2.35], [2.6, 2.57], [1.33, 0.97],
              [1.3, 1.45], [-0.24, -0.13], [0.0, 0.0], [0.04, -0.02], [0.14, 0.05], [0.52, 0.11],
              [-0.87, -1.48], [-1.02, -1.18], [-0.41, -1.46], [0.13, -1.28], [0.62, -0.95]])


def _shift(a, dy, dx):
    from scipy import ndimage
    return ndimage.shift(a, (dy, dx), order=1, mode="nearest")


def _cube(seed, movers=(), h=200, w=200, fp_amp=0.4):
    rng = np.random.default_rng(seed)
    n = len(DTS)
    fp = rng.normal(0, fp_amp, (h, w))                       # fixed on the detector
    yy, xx = np.mgrid[:h, :w]
    s = 2.85 / 2.3548
    cube = np.empty((n, h, w), np.float32)
    for i in range(n):
        frame = rng.normal(0, 1, (h, w)) + _shift(fp, -D[i, 0], -D[i, 1])   # pattern seen on the grid
        for x0, y0, vx, vy, a in movers:
            frame += a * np.exp(-((xx - x0 - vx * DTS[i]) ** 2 + (yy - y0 - vy * DTS[i]) ** 2) / (2 * s * s))
        cube[i] = frame
    return cube


def _track(f, x, y, vx, vy):
    return sum(float(SR._bilinear(f, np.array([i]), np.array([x + vx * t]), np.array([y + vy * t]))[0])
               for i, t in enumerate(DTS))


def test_chip_pattern_removed_at_drift_velocity_movers_kept():
    drift = (0.093, 0.074)                                    # px/min, from D over the sequence
    movers = [(60.0, 60.0, 0.35, -0.2, 1.2), (140.0, 70.0, -0.15, 0.3, 1.2), (70.0, 140.0, 0.6, 0.1, 1.2)]
    cube = _cube(1, movers)
    tn = SR.TrackNorm(DTS, 2.85)
    sig = tn.sigma(math.hypot(*drift))
    plain = SR.prepare_tile(cube, 2.85, 40.0, cap=0)
    corr = SR.prepare_tile(cube, 2.85, 40.0, cap=0, chip_shift=D)
    grid = [(x, y) for x in range(30, 170, 7) for y in range(30, 170, 7)]
    a = np.array([_track(plain.f_raw, x, y, *drift) for x, y in grid]) / sig
    b = np.array([_track(corr.f_raw, x, y, *drift) for x, y in grid]) / sig
    assert a.std() > 1.4 * b.std()                            # the drift-coherent pattern is gone
    for x0, y0, vx, vy, _ in movers:
        s0 = _track(plain.f_raw, x0, y0, vx, vy)
        s1 = _track(corr.f_raw, x0, y0, vx, vy)
        assert s1 > 0.8 * s0, (s0, s1)


def test_chip_path_ratio_separates_warm_pixel_from_mover():
    from synthtrack.tracking import Candidate
    rng = np.random.default_rng(4)
    n, h, w = len(DTS), 120, 120
    yy, xx = np.mgrid[:h, :w]
    s = 2.85 / 2.3548
    cube = rng.normal(0, 1, (n, h, w)).astype(np.float32)
    for i in range(n):
        # warm pixel at detector (60, 40): on the grid at detector − shift
        cube[i] += 1.5 * np.exp(-((xx - 40 + D[i, 1]) ** 2 + (yy - 60 + D[i, 0]) ** 2) / (2 * s * s))
        cube[i] += 1.5 * np.exp(-((xx - 80 - 0.3 * DTS[i]) ** 2 + (yy - 60 - 0.2 * DTS[i]) ** 2) / (2 * s * s))
    pt = SR.prepare_tile(cube, 2.85, 40.0, cap=0)
    # best linear track through the warm pixel's sky path
    A = np.vstack([np.ones(n), DTS]).T
    px = np.linalg.lstsq(A, 40 - D[:, 1], rcond=None)[0]
    py = np.linalg.lstsq(A, 60 - D[:, 0], rcond=None)[0]
    warm = Candidate(id=0, x=px[0], y=py[0], vx=px[1], vy=py[1], snr=8.0)
    mover = Candidate(id=1, x=80.0, y=60.0, vx=0.3, vy=0.2, snr=8.0)
    assert SR.chip_path_ratio(pt.f_raw, DTS, warm, D) > 0.9
    assert SR.chip_path_ratio(pt.f_raw, DTS, mover, D) < 0.5
