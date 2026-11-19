"""Realistic injections: the frame's star profile and the field's motion (inject.py)."""
import math

import numpy as np

from synthtrack import inject as INJ


def _moffat(img, x, y, flux, alpha=2.65, beta=2.5):     # FWHM ≈ 3 px, wings unlike a Gaussian
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    p = (1 + ((xx - x) ** 2 + (yy - y) ** 2) / alpha ** 2) ** (-beta)
    img += flux * p / p.sum()


def test_empirical_psf_matches_star_profile_and_injection_keeps_flux():
    rng = np.random.default_rng(2)
    h = w = 600
    frame = rng.normal(0, 1, (h, w)).astype(np.float32)
    stars = []
    for k in range(80):
        x, y = rng.uniform(30, w - 30), rng.uniform(30, h - 30)
        f = rng.uniform(800, 3000)
        _moffat(frame, x, y, f)
        stars.append((x, y, f))
    stars = np.array(stars)
    psf = INJ.empirical_psf(frame, stars, 2.9, min_n=10)
    assert psf is not None and abs(psf.sum() - 1) < 1e-4
    ref = np.zeros((psf.shape[0] + 10,) * 2)
    _moffat(ref, ref.shape[1] // 2, ref.shape[0] // 2, 1.0)
    c = ref.shape[0] // 2
    r = psf.shape[0] // 2
    # core of the empirical profile follows the true (non-Gaussian) one
    assert abs(psf[r, r] - ref[c, c]) / ref[c, c] < 0.15
    # injected flux is preserved
    reg = np.zeros((3, 120, 120), np.float32)
    objs = [{"x": 60.3, "y": 59.6, "vx": 0.5, "vy": 0.0, "flux_sigma": 100.0}]
    INJ.add_to_frames(reg, objs, np.array([-10.0, 0.0, 10.0]), 2.9, psf=psf)
    assert all(abs(reg[i].sum() - 100.0) < 2.0 for i in range(3))
    assert INJ.psf_gain(psf, 2.9) > 0


def test_plan_takes_half_of_the_motions_from_the_field():
    class TN:
        def sigma(self, v):
            return 4.0

        def expected(self, v):
            return 4.0
    vel = [(0.2, -0.1)] * 10
    objs = INJ.plan(200, (1000, 1000), np.linspace(-15, 15, 13), 2.9, TN(), 0.05, 0.8, 5, 15, 6.4,
                    velocities=vel)
    field = [o for o in objs if o["velocity"] == "field"]
    assert 70 < len(field) < 130
    for o in field:
        assert abs(math.hypot(o["vx"], o["vy"]) / math.hypot(0.2, -0.1) - 1) <= 0.16


def test_expected_snr_follows_throughput():
    """With throughput maps the injections carry the SNR expected under the real sky."""
    import numpy as np
    from synthtrack import inject as INJ
    from synthtrack.throughput import ThroughputMaps
    n = 9
    reg = np.zeros((n, 64, 64), np.float32)
    tp = np.linspace(1.0, 0.3, n)
    maps = ThroughputMaps(cell=32, grid=np.repeat(tp, 4).reshape(n, 2, 2).astype(np.float32), shape=(64, 64))
    objs = [{"id": 0, "x": 32.0, "y": 32.0, "vx": 0.0, "vy": 0.0, "snr_nominal": 10.0, "flux_sigma": 5.0}]
    INJ.add_to_frames(reg, objs, np.arange(n, dtype=float), 2.5, maps)
    assert abs(objs[0]["snr_expected"] - 10.0 * float(np.sqrt((tp ** 2).mean()))) < 0.02
    summ, lines = INJ.summarise([dict(objs[0], detected=None, passed=False)], 6.0)
    assert summ["snr_basis"] == "expected" and "očekávané" in "\n".join(lines)
