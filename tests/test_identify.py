"""Parsing of MPC element files and helpers shared by the precision tests."""
import math

import numpy as np

from synthtrack.identify import EPS_J2000, K_GAUSS, KnownObject, match_candidates
from synthtrack.mpcdata import jd_from_ymd, parse_comets, parse_mpcorb, unpack_epoch
from synthtrack.tracking import Candidate


def rv_to_elements(r, v):
    """Heliocentric ecliptic r [AU], v [AU/day] -> a, e, i, node, peri, M (deg)."""
    mu = K_GAUSS ** 2
    h = np.cross(r, v)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1 / (2 / rn - vn ** 2 / mu)
    evec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(evec)
    i = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
    nvec = np.cross([0, 0, 1], h)
    node = math.degrees(math.atan2(nvec[1], nvec[0])) % 360
    peri = math.degrees(math.acos(np.dot(nvec, evec) / (np.linalg.norm(nvec) * e)))
    if evec[2] < 0:
        peri = 360 - peri
    nu = math.acos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
    if np.dot(r, v) < 0:
        nu = 2 * math.pi - nu
    E = 2 * math.atan(math.tan(nu / 2) * math.sqrt((1 - e) / (1 + e)))
    M = math.degrees(E - e * math.sin(E)) % 360
    n = math.degrees(K_GAUSS / a ** 1.5)
    return dict(a=a, e=e, inc=i, node=node, peri=peri, M=M, n=n)


def eq_to_ecl(x):
    ce, se = math.cos(EPS_J2000), math.sin(EPS_J2000)
    return np.array([x[0], ce * x[1] + se * x[2], -se * x[1] + ce * x[2]])


def test_packed_epoch():
    assert unpack_epoch("K2555") == (2025, 5, 5)
    assert unpack_epoch("K24AV") == (2024, 10, 31)
    assert abs(jd_from_ymd(2000, 1, 1.5) - 2451545.0) < 1e-9


def _mpcorb_line(name="(99999) Testik", H=15.2, G=0.15, epoch="K2555", M=10.0, peri=20.0, node=30.0,
                 inc=5.0, e=0.1, n=0.25, a=2.5):
    line = [" "] * 202
    def put(a1, b1, text, right=True):
        w = b1 - a1 + 1
        s = text.rjust(w) if right else text.ljust(w)
        line[a1 - 1:b1] = list(s[:w])
    put(1, 7, "99999", right=False)
    put(9, 13, f"{H:5.2f}"); put(15, 19, f"{G:5.2f}"); put(21, 25, epoch, right=False)
    put(27, 35, f"{M:9.5f}"); put(38, 46, f"{peri:9.5f}"); put(49, 57, f"{node:9.5f}")
    put(60, 68, f"{inc:9.5f}"); put(71, 79, f"{e:9.7f}"); put(81, 91, f"{n:11.8f}"); put(93, 103, f"{a:11.7f}")
    put(118, 122, "  120"); put(162, 165, "0000"); put(167, 194, name, right=False)
    return "".join(line)


def test_parse_mpcorb_roundtrip():
    lines = ["MPCORB header", "-" * 160, _mpcorb_line(), _mpcorb_line(name="2026 AB1", M=200.0)]
    d = parse_mpcorb(lines)
    assert list(d["name"]) == ["(99999) Testik", "2026 AB1"]
    assert abs(d["a"][0] - 2.5) < 1e-9 and abs(d["M"][1] - 200.0) < 1e-9
    assert abs(d["H"][0] - 15.2) < 1e-5


def test_parse_comets():
    line = list(" " * 170)
    def put(a1, b1, text):
        w = b1 - a1 + 1
        line[a1 - 1:b1] = list(text.rjust(w)[:w])
    put(1, 4, "0001"); put(5, 5, "P")
    put(15, 18, "2061"); put(20, 21, "07"); put(23, 29, "28.0000")
    put(31, 39, "0.583000"); put(42, 49, "0.967000"); put(52, 59, "111.8700"); put(62, 69, "59.1000"); put(72, 79, "162.2000")
    put(92, 95, " 5.5"); put(97, 100, " 8.0")
    s = "".join(line)
    s = s[:102] + "1P/Halley".ljust(56) + s[158:]
    d = parse_comets([s])
    assert d["name"][0] == "1P/Halley"
    assert abs(d["q"][0] - 0.583) < 1e-9 and abs(d["e"][0] - 0.967) < 1e-9


def test_known_match_requires_compatible_motion_direction():
    known = [KnownObject("Test", "asteroid", 10.0, 20.0, 18.0, 1.0, 90.0, 2.0, 1.0)]
    opposite = Candidate(1, 10, 10, 0, 0, 12.0, ra=10.0, dec=20.0,
                         rate_arcsec_min=1.0, pa_deg=270.0)
    match_candidates([opposite], known, tol_arcsec=5)
    assert opposite.known is None
    same = Candidate(2, 10, 10, 0, 0, 10.0, ra=10.0, dec=20.0,
                     rate_arcsec_min=1.05, pa_deg=92.0)
    match_candidates([same], known, tol_arcsec=5)
    assert same.known and same.known["name"] == "Test"


def test_slow_known_match_allows_two_pixel_velocity_uncertainty():
    """CTA-S1 regression: short slow tracks have uncertain fitted velocity."""
    known = [KnownObject("(1184) Gaea", "asteroid", 10.0, 20.0, 15.6,
                         0.3854, 40.96, 2.0, 1.0)]
    measured = Candidate(1, 10, 10, 0, 0, 9.0, ra=10.0, dec=20.0,
                         rate_arcsec_min=0.7893, pa_deg=60.86)
    match_candidates([measured], known, tol_arcsec=35, rate_tol_abs=0.47)
    assert measured.known and measured.known["name"] == "(1184) Gaea"

    known[0].matched_candidate = None
    opposite = Candidate(2, 10, 10, 0, 0, 9.0, ra=10.0, dec=20.0,
                         rate_arcsec_min=0.7893, pa_deg=220.86)
    match_candidates([opposite], known, tol_arcsec=35, rate_tol_abs=0.47)
    assert opposite.known is None
