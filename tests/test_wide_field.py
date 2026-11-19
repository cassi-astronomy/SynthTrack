"""Wide-field specifics: distortion-aware registration, slow movers, header quirks."""
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from synthtrack import search as SR
from synthtrack import tracking as T
from synthtrack.fitsmeta import _sane_temp, parse_section, read_meta
from synthtrack.gpu import get_backend
from synthtrack.identify import predict_field
from synthtrack.instruments import Instrument
from synthtrack.mpcdata import parse_mpcorb
from synthtrack.registration import detect_stars, match_frame, warp
from synthtrack.simulate import gaussian_psf
from test_identify import _mpcorb_line


def _field(rng, shape, stars, fwhm, noise=1.0):
    img = rng.normal(0, noise, shape).astype(np.float32)
    for x, y, f in stars:
        gaussian_psf(img, x, y, f, fwhm)
    return img


def test_registration_with_distortion_and_dithering():
    rng = np.random.default_rng(0)
    h = w = 600
    n = 400
    xs, ys = rng.uniform(10, w - 10, n), rng.uniform(10, h - 10, n)
    fl = 10 ** rng.uniform(2, 3.5, n)

    # frame = reference shifted by 17 px dither + barrel distortion (several px at the corners)
    def distort(x, y):
        u, v = (x - w / 2) / (w / 2), (y - h / 2) / (h / 2)
        r2 = u * u + v * v
        return x + 17.3 + 3.0 * u * r2, y - 11.6 + 3.0 * v * r2
    dx, dy = distort(xs, ys)
    ref = _field(rng, (h, w), zip(xs, ys, fl), 2.6)
    cur = _field(rng, (h, w), zip(dx, dy, fl), 2.6)
    reg = match_frame(detect_stars(ref), detect_stars(cur), (h, w), order=3)
    assert reg.n_matched > 200
    assert reg.rms_px < 0.1, reg.rms_px
    px, py = reg.ref_to_frame(xs, ys)
    assert np.median(np.hypot(px - dx, py - dy)) < 0.05
    out = warp(cur, reg.ref_to_frame, (h, w))
    core = (slice(40, h - 40), slice(40, w - 40))
    resid = np.nan_to_num(out - ref)[core]
    assert np.std(resid) < 2.0          # stars subtract cleanly (noise √2 ≈ 1.4)


def test_slow_mover_detected_with_correct_noise_normalisation():
    """A mover that travels only ~1 FWHM during the sequence (typical for a 135 mm lens).

    The template removes part of its signal, but the noise of the sum shrinks
    as well; with the correct normalisation (TrackNorm) it is detected, while
    the naive √n normalisation underestimates its significance ~2×.
    """
    rng = np.random.default_rng(5)
    n, h, w, fwhm = 18, 160, 160, 3.0
    dts = np.linspace(-28.8, 28.8, n)
    v = (0.045, 0.02)                         # px/min -> ~2.8 px over 57.6 min
    stars = []
    while len(stars) < 40:     # keep the mover's neighbourhood free of stars
        x, y = rng.uniform(5, w - 5), rng.uniform(5, h - 5)
        if math.hypot(x - 80, y - 70) > 12:
            stars.append((x, y, 10 ** rng.uniform(2, 3.3)))
    cube = np.empty((n, h, w), np.float32)
    for i, t in enumerate(dts):
        cube[i] = _field(rng, (h, w), stars, fwhm)
        gaussian_psf(cube[i], 80 + v[0] * t, 70 + v[1] * t, 22.0, fwhm)
    tn = SR.TrackNorm(dts, fwhm)
    assert 0.3 < tn.efficiency(math.hypot(*v)) < 0.6
    pt = SR.prepare_tile(cube, fwhm, 40.0)
    grid = T.make_velocity_grid(tn.min_speed(0.3), 0.2, dts[-1] - dts[0], 0.7)
    best, bidx = SR.search_tile(get_backend("numpy"), pt.f, dts, grid, tn)
    cands = SR.peaks(best, bidx, grid, 5.0, fwhm, (slice(8, h - 8), slice(8, w - 8)), 20)
    SR.refine(cands, pt.f_raw, dts, grid.dv, tn, min_speed=tn.min_speed(0.3))
    SR.vet(cands, pt, dts, tn, 3, 3.0, 40.0, fwhm)
    hit = [c for c in cands if math.hypot(c.x - 80, c.y - 70) < 2.5]
    assert hit and hit[0].auto_status == "candidate", [(c.x, c.y, c.snr, c.flags) for c in cands[:5]]
    assert hit[0].snr > 7
    naive = hit[0].snr * tn.sigma(math.hypot(hit[0].vx, hit[0].vy)) / math.sqrt(n)
    assert naive < 0.7 * hit[0].snr


def test_header_quirks():
    assert parse_section("[31:4126,12:4107]", (4127, 4144)) == (slice(11, 4107), slice(30, 4126))
    assert parse_section(None, (10, 10)) is None
    assert _sane_temp(-274.0) is None and _sane_temp(-19.9) == -19.9


def test_static_peak_distinguishes_slow_object_from_nearby_star():
    from synthtrack.pipeline import _static_peak
    static = np.zeros((31, 31), float)
    static[15, 15] = 100
    peak, offset = _static_peak(static, 15.1, 14.9, 5)
    assert peak == 100 and offset < 0.2
    peak, offset = _static_peak(static, 11.0, 15.0, 5)
    assert peak == 100 and offset == 4.0


@pytest.mark.parametrize("time_key", ["DATE-OBS", "MJD-OBS"])
def test_mid_exposure_time_applies_instrument_clock_offset(tmp_path, time_key):
    """DATE-OBS and MJD-OBS must describe the same corrected mid-exposure time."""
    start = Time("2026-10-20T22:00:00", scale="utc")
    header = fits.Header({"EXPTIME": 30.0, "NAXIS1": 20, "NAXIS2": 10})
    header[time_key] = start.isot if time_key == "DATE-OBS" else start.mjd
    inst = Instrument(id="clock", date_obs_reference="start", time_offset_s=2.5)
    meta = read_meta(tmp_path / "unused.fits", inst, header)
    assert meta.mjd_mid == pytest.approx(start.mjd + 17.5 / 86400.0)


def test_old_epoch_objects_do_not_disturb_others():
    """Regression: an orbit with an epoch outside the packaged DE421 span used to
    inject a dummy Pluto at the barycentre and shift every prediction by ~40″."""
    a = _mpcorb_line(name="(1) A", epoch="K26AK", M=100.0, a=2.6, e=0.12, inc=6.0,
                     n=math.degrees(0.01720209895 / 2.6 ** 1.5))
    old = _mpcorb_line(name="(2) Old", epoch="K1029", M=10.0, a=2.3, e=0.1, inc=3.0,
                       n=math.degrees(0.01720209895 / 2.3 ** 1.5))
    t = Time("2026-10-21T08:00:00", scale="utc")
    one = predict_field(parse_mpcorb([a]), None, t, 0, 0, 180, -24.6, -70.4, 2400, mag_limit=99)
    both = predict_field(parse_mpcorb([a, old]), None, t, 0, 0, 180, -24.6, -70.4, 2400, mag_limit=99)
    k1 = [k for k in one if k.name == "(1) A"][0]
    k2 = [k for k in both if k.name == "(1) A"][0]
    assert math.hypot((k1.ra - k2.ra) * math.cos(math.radians(k1.dec)), k1.dec - k2.dec) * 3600 < 0.01
