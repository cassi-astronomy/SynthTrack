"""Per-run ranking calibrated by injections (rank.py)."""
import numpy as np

from synthtrack import rank as RK


class C:
    def __init__(self, **kw):
        self.flags, self.known, self.injected, self.auto_status = [], None, None, "candidate"
        self.__dict__.update(kw)


def _make(rng, real, n):
    out = []
    for _ in range(n):
        if real:
            out.append(C(snr=rng.uniform(6, 14), chi2=rng.uniform(0.4, 1.0), obj_fwhm=rng.normal(2.9, 0.2),
                         fwhm=2.84, snr_reverse=rng.normal(-2.5, 1), frames_with_signal=rng.uniform(0.8, 1),
                         mover_vs_static=rng.uniform(0.2, 0.8), median_ratio=rng.normal(1, 0.1),
                         star_ratio=rng.uniform(0, 5), seg_snr=list(rng.uniform(3, 6, 3))))
        else:
            out.append(C(snr=rng.uniform(6, 8), chi2=rng.uniform(0.8, 2.2), obj_fwhm=rng.normal(2.3, 0.4),
                         fwhm=2.84, snr_reverse=rng.normal(0, 1), frames_with_signal=rng.uniform(0.6, 0.9),
                         mover_vs_static=rng.uniform(-0.3, 0.5), median_ratio=rng.normal(0.95, 0.2),
                         star_ratio=rng.uniform(0, 12), seg_snr=list(rng.uniform(0, 5, 3))))
    return out


def test_calibrate_keeps_injections_and_known_sets_aside_noise():
    rng = np.random.default_rng(0)
    inj = _make(rng, True, 80)
    for c in inj:
        c.injected = 1
    known = _make(rng, True, 5)
    for c in known:
        c.known = {"name": "test"}
    noise = _make(rng, False, 600)
    r = RK.calibrate(inj + known + noise, 2.84, keep=0.9)
    assert r is not None and r["cv_auc"] > 0.9
    assert sum(c.auto_status == "candidate" for c in inj) == 80          # injections never set aside
    assert all(c.auto_status == "candidate" for c in known)             # known never set aside
    assert r["remaining"] < 0.3 * len(noise)
    assert all(("nízké skóre podle injekcí" in c.flags) or ("hraniční skóre podle injekcí" in c.flags)
               for c in noise if c.auto_status != "candidate")


def test_calibrate_needs_enough_injections():
    rng = np.random.default_rng(1)
    inj = _make(rng, True, 5)
    for c in inj:
        c.injected = 1
    assert RK.calibrate(inj + _make(rng, False, 100), 2.84) is None
