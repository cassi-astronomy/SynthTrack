"""Regression tests for the Coihueco / FRAM runs of 8. 11. 2026 (Test1–4).

* weak warm pixels following a slow dither drift (11P: 7 400 false detections),
* the drift-velocity rule,
* slow known objects with an overestimated rate and comets far from their prediction,
* a saturated known mover (Pallas) – holes in the data that move with it,
* comet injections, approximate photometry from the field's known asteroids.
"""
from __future__ import annotations

import math

import numpy as np

from synthtrack import inject as INJ
from synthtrack.defects import chip_fixed_weak
from synthtrack.identify import KnownObject, match_candidates
from synthtrack.pipeline import _drift_velocity, _photometry, _saturated_mover
from synthtrack.tracking import Candidate


def _gauss(h, w, x, y, amp, sig=1.1):
    yy, xx = np.mgrid[0:h, 0:w]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig * sig))


def test_weak_warm_pixel_masked_star_kept():
    rng = np.random.default_rng(3)
    n, h, w = 28, 64, 64
    # slow linear drift of the pointing (like 11P: ~11 px over the series)
    d = np.stack([np.linspace(-5, 5, n), np.linspace(2, -2, n)], 1)       # (dy, dx) frame − ref
    cube = rng.standard_normal((n, h, w)).astype(np.float32)
    for i in range(n):
        # a star fixed on the sky at (40, 20) → on the chip at sky + d_i
        cube[i] += _gauss(h, w, 40 + d[i, 1], 20 + d[i, 0], 4.0).astype(np.float32)
        # a neighbouring star 3 px away from the warm pixel's sky track (would fool chip_fixed)
        cube[i] += _gauss(h, w, 22 + d[i, 1], 33 + d[i, 0], 6.0).astype(np.float32)
    cube[:, 30, 20] += 3.0                                                # warm pixel, fixed on the chip
    med = np.median(cube, axis=0)
    cand = np.array([[30, 20], [int(round(20 + np.median(d[:, 0]))), int(round(40 + np.median(d[:, 1])))]])
    offs = np.broadcast_to(d[:, None, :], (n, len(cand), 2)).astype(np.float32)
    fixed = chip_fixed_weak(cube, med, cand, offs)
    assert fixed[0], "warm pixel (3 σ) must be recognised as fixed on the chip"
    assert not fixed[1], "a star (fixed on the sky) must be kept"


def test_drift_velocity_linear_and_random():
    t = np.linspace(-35, 35, 28)
    cs = np.stack([0.16 * t + 0.1, -0.05 * t], 1)                         # (dy, dx) per frame
    v = _drift_velocity(cs, t, 2.4)
    assert v is not None and abs(v[0] - 0.05) < 1e-6 and abs(v[1] + 0.16) < 1e-6
    rnd = np.random.default_rng(1).uniform(-15, 15, (12, 2))
    assert _drift_velocity(rnd, np.linspace(-30, 30, 12), 2.8) is None


def _cand(i, ra, dec, rate, pa, snr=8.0):
    c = Candidate(id=i, x=0.0, y=0.0, vx=0.0, vy=0.0, snr=snr)
    c.ra, c.dec, c.rate_arcsec_min, c.pa_deg = ra, dec, rate, pa
    return c


def test_slow_object_rate_tolerance():
    # (2218) Wotho: right on the prediction, rate fitted 0.50 vs 0.27″/min (slow, ~1 FWHM per series)
    k = KnownObject("(2218) Wotho", "asteroid", 0.8661, -21.6328, 16.0, 0.274, 331.5, 2.7, 2.0)
    c = _cand(0, 0.8661, -21.6328, 0.498, 325.0)
    scale, fwhm, span = 6.39, 2.83, 64.0
    match_candidates([c], [k], 30.0, rate_tol_abs=max(0.05, 2.0 * scale / span))
    assert c.known is None                                               # the old tolerance missed it
    match_candidates([c], [k], 30.0, rate_tol_abs=max(0.05, max(2.0, 1.2 * fwhm) * scale / span))
    assert c.known and c.known["name"] == "(2218) Wotho"


def test_comet_wider_tolerance_but_motion_must_agree():
    k = KnownObject("11P/Tempel-Swift-LINEAR", "comet", 41.3520, 27.9021, 16.2, 1.557, 159.5, 1.39, 0.40)
    good = _cand(0, 41.3781, 27.9047, 1.628, 161.4, snr=39)              # 84″ off, same motion
    wrong = _cand(1, 41.3700, 27.9100, 1.60, 250.0, snr=50)               # nearby, other direction
    match_candidates([wrong, good], [k], 30.0, comet_tol_arcsec=180.0)
    assert good.known and good.known["name"].startswith("11P")
    assert wrong.known is None
    a = KnownObject("(1) Test", "asteroid", 41.3520, 27.9021, 15.0, 1.557, 159.5, 2.0, 1.0)
    c2 = _cand(2, 41.3781, 27.9047, 1.628, 161.4)
    match_candidates([c2], [a], 30.0, comet_tol_arcsec=180.0)
    assert c2.known is None                                              # asteroids keep the tight tolerance


def test_saturated_mover_found_star_hole_not():
    rng = np.random.default_rng(0)
    n, h, w = 12, 80, 80
    dts = np.linspace(-32, 32, n)
    v = (0.06, 0.03)                                                       # px/min, ~4.3 px over the series
    cube = rng.standard_normal((n, h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        x, y = 40 + v[0] * dts[i], 40 + v[1] * dts[i]
        cube[i] += _gauss(h, w, x, y, 400.0, 2.0).astype(np.float32)      # bright wings
        cube[i][np.hypot(xx - x, yy - y) < 3.0] = np.nan                   # saturated core moving with it
        cube[i] += _gauss(h, w, 15, 60, 400.0, 2.0).astype(np.float32)
        cube[i][np.hypot(xx - 15, yy - 60) < 3.0] = np.nan                 # saturated star (fixed)
        cube[i][:, :int(round(6 + 3 * np.sin(i)))] = np.nan               # field edge moving with the dither
    sat = _saturated_mover(cube, 40.0, 40.0, v, dts, 2.8)
    assert sat is not None and sat["frames"] >= 10 and math.hypot(sat["dx"], sat["dy"]) < 0.5
    assert _saturated_mover(cube, 15.0, 60.0, v, dts, 2.8) is None        # a star's hole does not move
    # a faint object next to the moving field edge (no bright wings) is not "saturated"
    assert _saturated_mover(cube, 8.0 + v[0] * 0, 20.0, v, dts, 2.8) is None


def test_comet_injections_profile_and_plan():
    p = INJ.comet_profile(15.0, 0.0, 2.5)
    assert abs(float(p.sum()) - 1.0) < 1e-4 and p.shape[0] > 60
    # a diffuse coma needs far more light for the same matched-filter SNR
    assert INJ.profile_gain(p, 2.5) < 0.2 * INJ.psf_gain(None, 2.5)
    from synthtrack.search import TrackNorm
    dts = np.linspace(-30, 30, 20)
    tn = TrackNorm(dts, 2.5)
    objs = INJ.plan(60, (512, 512), dts, 2.5, tn, 0.05, 0.8, 6, 12, 6.4, seed=2, comet_fraction=0.3)
    com = [o for o in objs if o["kind"] == "comet"]
    assert 8 <= len(com) <= 30
    assert all(4.0 <= o["coma_arcsec"] <= 120.0 and 0 <= o["nucleus_frac"] <= 0.6 for o in com)
    reg = np.zeros((20, 512, 512), np.float32)
    INJ.add_to_frames(reg, com[:3], dts, 2.5)
    o = com[0]
    assert abs(float(reg[10].sum()) - sum(c["flux_sigma"] for c in com[:3])) < 0.05 * sum(c["flux_sigma"] for c in com[:3]) + 1
    assert np.isfinite(reg).all() and reg[10, int(round(o["y"])), int(round(o["x"]))] > 0


def test_photometry_zero_point_from_known():
    known, cands = [], []
    zp_true = 19.0
    for i, mag in enumerate([14.5, 15.2, 15.9, 16.4, 13.8]):
        k = KnownObject(f"({i + 1}) A", "asteroid", 0, 0, mag, 0.5, 90, 2, 1, matched_candidate=i)
        c = Candidate(id=i, x=0, y=0, vx=0.1, vy=0.0, snr=20.0)
        c.known = {"name": k.name, "kind": "asteroid"}
        f = 10 ** (0.4 * (zp_true - mag)) * (1.0 + 0.05 * (-1) ** i)
        c.positions = [{"flux": f, "ok": True}] * 3
        known.append(k)
        cands.append(c)
    unk = Candidate(id=9, x=0, y=0, vx=0.1, vy=0, snr=9.0)
    unk.positions = [{"flux": 10 ** (0.4 * (zp_true - 17.3)), "ok": True}]
    cands.append(unk)
    ph = _photometry(cands, known, fwhm=2.5, span_min=60.0)
    assert ph is not None and abs(ph["zp"] - zp_true) < 0.06
    assert abs(unk.mag - 17.3) < 0.1 and known[0].mag_measured is not None


def _write_ucac4(root, stars):
    """stars: list of (ra, dec, V, r) → minimal UCAC4 zone files (sorted by RA)."""
    from synthtrack import ucac4 as U4
    u4b = root / "u4b"
    u4b.mkdir(parents=True)
    zones = {}
    for ra, dec, v, r in stars:
        z = int(math.floor((dec + 90) / 0.2)) + 1
        zones.setdefault(z, []).append((ra, dec, v, r))
    for z, ss in zones.items():
        rec = np.zeros(len(ss), U4.REC)
        ss.sort()
        for k, (ra, dec, v, r) in enumerate(ss):
            rec[k]["ra"] = int(round(ra * U4.MAS))
            rec[k]["spd"] = int(round((dec + 90) * U4.MAS))
            rec[k]["magm"] = int(v * 1000)
            rec[k]["apasm"] = [20000, int(v * 1000), 20000, int(r * 1000), 20000]
        rec.tofile(u4b / f"z{z:03d}")
    return u4b


def test_ucac4_query_wrap_and_bands(tmp_path):
    from synthtrack import ucac4 as U4
    _write_ucac4(tmp_path, [(359.9, 10.05, 12.0, 11.6), (0.1, 10.1, 13.0, 12.7), (180, 10.1, 11, 11),
                            (0.05, 10.5, 14.0, 13.5)])
    root = U4.find_root(tmp_path)
    assert root is not None and root.name == "u4b"
    c = U4.query(root, 359.8, 0.2, 10.0, 10.2, "r")
    assert sorted(np.round(c["mag"], 2).tolist()) == [11.6, 12.7]
    assert U4.band_for_filter("R") == "r" and U4.band_for_filter("V") == "V" and U4.band_for_filter("N") == "r"
    assert U4.find_root(tmp_path / "nothing") is None


def test_photometry_zero_point_from_ucac4(tmp_path):
    from astropy.wcs import WCS
    from synthtrack.pipeline import _zp_ucac4
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = [128.5, 128.5]
    w.wcs.cdelt = [-6.4 / 3600, 6.4 / 3600]
    rng = np.random.default_rng(5)
    n, h, ww = 10, 256, 256
    cube = rng.standard_normal((n, h, ww)).astype(np.float32) * 0.3
    zp_true, sig = 21.0, 2.6 / 2.3548
    stars = []
    for k in range(40):
        x, y = rng.uniform(20, 236, 2)
        r = rng.uniform(12, 15)
        ra, dec = (float(a) for a in w.pixel_to_world_values(x, y))
        stars.append((ra, dec, r + 0.4, r))
        flux = 10 ** (0.4 * (zp_true - r))
        cube += _gauss(h, ww, x, y, flux / (2 * math.pi * sig * sig), sig).astype(np.float32)[None]
    _write_ucac4(tmp_path, stars)
    from synthtrack import ucac4 as U4
    zs, band = _zp_ucac4(U4.find_root(tmp_path), cube, w, 61352.0, 2.6, "R")
    assert band == "r" and len(zs) >= 15
    assert abs(float(np.median(zs)) - zp_true) < 0.05


def test_chip_path_partial_drift():
    """Warm pixel on a linear track only in the drift part of the series (88P-like dither)."""
    from synthtrack.search import chip_path
    n, h, w = 30, 120, 120
    t = np.linspace(-40, 32, n)
    sh = np.zeros((n, 2))
    sh[:13] = [[-12, 25]] * 13                                            # first frames 25 px away
    sh[13:, 0] = np.linspace(0.4, -6.3, 17)                               # then a slow drift in y
    f = np.random.default_rng(2).standard_normal((n, h, w)).astype(np.float32) * 0.3
    px, py = 60.0, 60.0                                                   # chip position of the warm pixel
    for i in range(n):
        f[i] += _gauss(h, w, px - sh[i, 1], py - sh[i, 0], 1.5).astype(np.float32)
    vy = -(sh[-1, 0] - sh[13, 0]) / (t[-1] - t[13])
    c = Candidate(id=0, x=px - sh[15, 1], y=py - sh[15, 0] - vy * t[15], vx=0.0, vy=vy, snr=7)
    r, ov = chip_path(f, t, c, sh, 2.3)
    assert r is not None and r >= 0.9 and ov >= 0.5
    # a real mover with a random dither: the detector path meets the track only by chance
    rnd = np.random.default_rng(3).uniform(-15, 15, (n, 2))
    g = np.random.default_rng(4).standard_normal((n, h, w)).astype(np.float32) * 0.3
    for i in range(n):
        g[i] += _gauss(h, w, 50 + 0.3 * t[i], 60 + 0.1 * t[i], 1.5).astype(np.float32)
    m = Candidate(id=1, x=50, y=60, vx=0.3, vy=0.1, snr=7)
    r2, ov2 = chip_path(g, t, m, rnd, 2.3)
    assert ov2 < 0.3


def test_velocity_noise_norm_directional_excess():
    """A weak detector pattern seen through a drift: excess noise at one velocity only."""
    from synthtrack.gpu import get_backend
    from synthtrack.search import TrackNorm, search_tile
    from synthtrack.tracking import make_velocity_grid
    rng = np.random.default_rng(8)
    n, h, w = 20, 256, 256
    dts = np.linspace(-30, 30, n)
    pattern = rng.standard_normal((h + 40, w + 40)).astype(np.float32) * 0.5
    f = rng.standard_normal((n, h, w)).astype(np.float32)
    vd = (0.0, 0.25)                                       # pattern moves over the "sky" with the drift
    for i in range(n):
        sy = int(round(vd[1] * dts[i]))
        f[i] += pattern[20 - sy:20 - sy + h, 20:20 + w]
    tn = TrackNorm(dts, 2.5)
    grid = make_velocity_grid(0.1, 0.6, 60.0, 0.7)
    sc = np.ones(len(grid))
    search_tile(get_backend("numpy", 0, 0.5), f, dts, grid, tn, scales=sc)
    k = int(np.argmin((grid.vx - vd[0]) ** 2 + (grid.vy - vd[1]) ** 2))
    far = np.hypot(grid.vx - vd[0], grid.vy - vd[1]) > 0.15
    assert sc[k] > 1.5
    assert np.median(sc[far]) < 1.05


def test_sharp_rescue_calibrated_by_injections():
    from synthtrack.pipeline import _sharp_rescue
    from synthtrack.settings import SearchSettings
    cands = []
    for i in range(30):
        c = Candidate(id=i, x=0, y=0, vx=0.1, vy=0, snr=8)
        c.injected, c.fwhm, c.obj_fwhm = i, 2.6, 2.6 * (0.66 + 0.01 * (i % 10))
        cands.append(c)
    real = Candidate(id=100, x=0, y=0, vx=0.1, vy=0, snr=7, fwhm=2.6, obj_fwhm=1.63)
    real.flags, real.auto_status = ["ostřejší než hvězdy"], "auto_rejected"
    pix = Candidate(id=101, x=0, y=0, vx=0.1, vy=0, snr=7, fwhm=2.6, obj_fwhm=0.9)
    pix.flags, pix.auto_status = ["ostřejší než hvězdy"], "auto_rejected"
    assert _sharp_rescue(cands + [real, pix], SearchSettings()) == 1
    assert real.auto_status == "candidate" and "užší než hvězdy?" in real.flags
    assert pix.auto_status == "auto_rejected"


def test_inpaint_hot_pixels_keeps_star_core():
    from synthtrack.defects import inpaint_pixels
    img = _gauss(21, 21, 10, 10, 50.0, 1.0).astype(np.float32)
    ref = img.copy()
    img[10, 10] = 900.0                                    # hot pixel on the star's peak
    img[3, 3] = 40.0                                       # hot pixel on empty sky
    inpaint_pixels(img, np.array([10, 3]), np.array([10, 3]))
    assert np.isfinite(img).all()
    assert 0.3 * ref[10, 10] < img[10, 10] < ref[10, 10]   # neighbour median: lower, but data
    assert abs(img[3, 3]) < 0.5


def test_field_edge_without_coverage_not_treated_as_saturation():
    """Large dither: a strip at the edge is missing in 13 of 30 frames next to a saturated star."""
    from synthtrack.search import prepare_tile
    rng = np.random.default_rng(1)
    n, h, w = 30, 160, 160
    reg = rng.standard_normal((n, h, w)).astype(np.float32)
    star = _gauss(h, w, 120, 80, 3000.0, 1.6).astype(np.float32)
    reg += star[None]
    yy, xx = np.mgrid[0:h, 0:w]
    reg[:, np.hypot(xx - 120, yy - 80) < 3] = np.nan       # saturated core, every frame
    reg[:13, :, 121:] = np.nan                             # edge strip without coverage (jump frames)
    # a faint mover in the strip, visible in the other 17 frames
    for i in range(n):
        reg[i] += _gauss(h, w, 140 + 0.05 * i, 30 + 0.05 * i, 8.0, 1.1).astype(np.float32)
    reg[:13, :, 121:] = np.nan
    pt = prepare_tile(reg, 2.6, 40.0)
    assert np.abs(pt.f_raw[13:, 28:34, 138:144]).max() > 1.0, "the strip must keep its data"
    assert np.abs(pt.f_raw[:, 78:83, 118:123]).max() == 0.0, "the saturated core stays masked"


def test_confirm_cosmics_keeps_sharp_star_masks_single_hit():
    from synthtrack.transients import confirm_cosmics, mask_cosmics
    rng = np.random.default_rng(0)
    n, h, w = 5, 200, 200
    reg = rng.standard_normal((n, h, w)).astype(np.float32)
    for i in range(n):
        reg[i] += _gauss(h, w, 20, 20, 60.0, 1.1 if i != 2 else 0.45).astype(np.float32)   # frame 2: sharp
    reg[3, 8, 30] += 200.0                                                             # a cosmic in frame 3
    hits = {}
    for i in range(n):
        g = mask_cosmics(reg[i].copy(), 2.6, apply=False)
        if isinstance(g, np.ndarray) and g.any():
            hits[i] = np.nonzero(g)
    assert 2 in hits and 3 in hits                       # the sharp star looks like a cosmic on its own
    px, kept = confirm_cosmics(reg, list(range(n)), hits)
    assert kept >= 1 and np.isfinite(reg[2, 18:23, 18:23]).all()
    assert np.isnan(reg[3, 8, 30])


def test_substack_rescue_slow_object_at_dawn():
    from synthtrack.pipeline import _substack_rescue
    dts = np.linspace(-11.6, 11.6, 13)
    tp = np.array([1.075, 1.062, 1.039, 1.042, 0.998, 0.982, 0.909, 0.859, 0.765, 0.669, 0.643, 0.567, 0.506])
    amp = np.broadcast_to(tp[:, None, None], (13, 64, 64)).astype(np.float32)
    c = Candidate(id=12, x=30, y=30, vx=0.033, vy=0.15, snr=15.0)
    c.flags, c.auto_status = ["nekonzistentní segmenty"], "auto_rejected"
    c.positions = [{"flux": 42.5, "ok": True}, {"flux": 31.5, "ok": True}, {"flux": 25.6, "ok": True}]
    _substack_rescue(c, {"nekonzistentní segmenty", "jednosnímkový artefakt"}, dts, 3, amp, 2.66)
    assert c.auto_status == "candidate" and "segmenty nesouhlasí, podstacky souhlasí" in c.flags
    # a fast object with the same numbers keeps the rejection (no self-subtraction to blame)
    f = Candidate(id=13, x=30, y=30, vx=1.0, vy=0.0, snr=15.0)
    f.flags, f.auto_status = ["nekonzistentní segmenty"], "auto_rejected"
    f.positions = c.positions
    _substack_rescue(f, {"nekonzistentní segmenty"}, dts, 3, amp, 2.66)
    assert f.auto_status == "auto_rejected"


def test_twilight_frames_sun_altitude():
    from synthtrack.pipeline import _twilight_frames
    dawn = 61353.332654 + np.arange(19) * 0.0013437          # Coihueco 314, 10. 11. 2026: Sun −17 … −11°
    tw = _twilight_frames(dawn, -35.1167, -69.6, 1650.0)
    assert tw is not None and tw.all()
    night = 61353.15 + np.arange(10) * 0.002                   # ~ local 00:36: deep night
    assert not _twilight_frames(night, -35.1167, -69.6, 1650.0).any()


def test_catalog_star_reject_slow_on_bright_star():
    from synthtrack.pipeline import _catalog_stars_reject
    cat = {"x": np.array([100.0, 300.0]), "y": np.array([100.0, 300.0]), "mag": np.array([9.0, 15.5])}
    star = Candidate(id=1, x=100.5, y=99.8, vx=0.04, vy=0.0, snr=39.0); star.mag = 9.3
    faint = Candidate(id=2, x=302.8, y=300.1, vx=0.04, vy=0.0, snr=8.0); faint.mag = 16.0   # off-centre, star only 0.5 mag brighter
    centred = Candidate(id=5, x=300.3, y=300.1, vx=0.04, vy=0.0, snr=8.0); centred.mag = 15.3  # the star itself
    fast = Candidate(id=3, x=100.2, y=100.1, vx=0.8, vy=0.0, snr=9.0); fast.mag = 16.0
    known = Candidate(id=4, x=100.0, y=100.0, vx=0.04, vy=0.0, snr=50.0); known.mag = 12.0
    known.known = {"name": "(1) X"}
    n = _catalog_stars_reject([star, faint, fast, known, centred], cat, fwhm=2.2, span_min=70.0)
    assert n == 2 and star.auto_status == centred.auto_status == "auto_rejected"
    assert faint.auto_status == fast.auto_status == known.auto_status == "candidate"


def test_static_vs_mover_ignores_single_frame_hit():
    from synthtrack.search import static_vs_mover
    rng = np.random.default_rng(4)
    n, h, w = 30, 60, 60
    dts = np.linspace(-35, 35, n)
    cube = rng.standard_normal((n, h, w)).astype(np.float32)
    for i in range(n):
        cube[i] += _gauss(h, w, 30 + 0.085 * dts[i], 30 + 0.02 * dts[i], 6.0, 0.93).astype(np.float32)
    c = Candidate(id=0, x=30, y=30, vx=0.085, vy=0.02, snr=19)
    t0 = static_vs_mover(cube, dts, c, 2.2)
    cube[26, 28, 27] += 60.0                                       # a cosmic 3 px from the track
    cube[26] += _gauss(h, w, 27.5, 28.0, 40.0, 0.7).astype(np.float32)
    t1 = static_vs_mover(cube, dts, c, 2.2)
    assert t0 > 0.3 and t1 > 0.3
