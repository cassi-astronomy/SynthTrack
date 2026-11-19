import math

import numpy as np

from synthtrack import sngal as S
from synthtrack.twilight import TwilightParams


class _W:                                   # minimal WCS stand-in for search()
    def all_pix2world(self, xy, o):
        return np.asarray(xy, float) * 0.004


def _scene(sn=True, neighbour=False, n=2, seed=1):
    rng = np.random.default_rng(seed)
    h = w = 160
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sg = 2.4 / 2.3548
    star = lambda x, y, f: f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sg * sg)) / (2 * math.pi * sg * sg)
    gal = 400 * np.exp(-(((xx - 80) / 3.0) ** 2 + ((yy - 80) / 1.8) ** 2) / 2)     # point-symmetric disc
    sky = 0.05 * (xx - 80) + 0.03 * (yy - 80)                                       # twilight gradient
    cxy, ef = [], []
    for _ in range(120):                                                           # catalogue stars
        x, y = rng.uniform(5, w - 5, 2)
        if math.hypot(x - 80, y - 80) < 20:
            continue
        cxy.append((x, y)); ef.append(float(rng.uniform(20, 400)))
    base = gal + sum(star(x, y, f) for (x, y), f in zip(cxy, ef))
    if sn:
        base = base + star(84.0, 81.5, 120.0)
    if neighbour:
        base = base + 150 * np.exp(-((xx - 86) ** 2 + (yy - 78) ** 2) / (2 * 1.2 ** 2))
    cube = np.stack([base + sky + rng.normal(0, 1.0, (h, w)) for _ in range(n)]).astype(np.float32)
    img = np.median(cube, axis=0) - sky
    gals = [{"name": "G1", "x": 80.0, "y": 80.0, "d25": 1.0, "mag_est": 13.0, "mag_cat": None}]
    if neighbour:
        gals.append({"name": "G2", "x": 86.0, "y": 78.0, "d25": 0.3, "mag_est": 15.0, "mag_cat": None})
    return cube, img, gals, np.array(cxy), np.array(ef)


def _run(**kw):
    cube, img, gals, cxy, ef = _scene(**kw)
    P = TwilightParams()
    cs, st = S.search(cube, img, 1.0, gals, None, cxy, 15 - 2.5 * np.log10(ef), ef, lambda x, y: 20.0, 2.4,
                      14.0, _W(), P, known=None, sig_f=[1.0] * cube.shape[0])
    return [c for c in cs if c["galaxy"] == "G1"], st


def test_supernova_found_despite_gradient():
    cs, st = _run(sn=True)
    assert cs and cs[0]["auto_status"] == "candidate"
    assert math.hypot(cs[0]["x"] - 84.0, cs[0]["y"] - 81.5) < 1.0
    assert all(v is not None and v >= 4 for v in cs[0]["frame_snr"])
    assert math.isfinite(st["flux_scale"])


def test_symmetric_galaxy_gives_nothing():
    cs, _ = _run(sn=False)
    assert not [c for c in cs if c["auto_status"] == "candidate"]


def test_neighbour_nucleus_rejected():
    cs, _ = _run(sn=False, neighbour=True)
    assert not [c for c in cs if c["auto_status"] == "candidate"]


def test_nan_positions_skipped():
    cube, img, gals, cxy, ef = _scene(sn=True)
    gals.append({"name": "far", "x": float("nan"), "y": float("nan"), "d25": 1.0, "mag_est": 13.0, "mag_cat": None})
    cxy = np.vstack([cxy, [[np.nan, np.nan]]]); ef = np.append(ef, 10.0)
    cs, st = S.search(cube, img, 1.0, gals, None, cxy, 15 - 2.5 * np.log10(ef), ef, lambda x, y: 20.0, 2.4, 14.0,
                      _W(), TwilightParams(), sig_f=[1.0, 1.0])
    assert st["examined"] == 1
