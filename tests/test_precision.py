"""Precision of positions of known objects – target well below 1″.

Error budget of a predicted position = orbit (MPCORB elements, not tested
here) + propagation + ephemeris + observation geometry. The last three are
verified here:

* integrator vs. analytic Kepler (no perturbers)                    < 1 m
* integrator vs. JPL DE421 (Mercury, Venus, Mars as test particles,
  ±1 year)                                                           < 5 km
* geometry (light time, topocentre, frames) vs. astropy              < 0.01″
* real asteroids vs. JPL Horizons (tests/data/horizons_*.json)       < 1″
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time

from synthtrack import ephem as E
from synthtrack import nbody
from synthtrack.ephem import AU_KM, BodyTable, BuiltinEphemeris, LegacyDeEphemeris

ARCSEC = 1 / 206264.806
DATA = Path(__file__).parent / "data"


def _de421():
    from synthtrack.ephem import PACKAGED_DE421
    for p in (os.environ.get("SYNTHTRACK_DE421"), PACKAGED_DE421):
        if p and (Path(p) / "constants.npy").exists():
            return LegacyDeEphemeris(p)
    return None


class _Massive:
    """Treat one body as the test particle: drop it from the perturbers, add its mass to the Sun."""

    def __init__(self, drop, extra_gm):
        self.drop, self.extra = drop, extra_gm

    def __enter__(self):
        self.saved, self.k2 = dict(E.GM), nbody.K2
        for d in self.drop:
            E.GM.pop(d)
        nbody.K2 = E.K2 + self.extra

    def __exit__(self, *a):
        E.GM.clear()
        E.GM.update(self.saved)
        nbody.K2 = self.k2


def _state(eph, body_fn, t, h=0.05):
    """Position and 5-point-stencil velocity. A small h is *worse*: a JD stored as a
    double carries ~40 µs of rounding, i.e. ~1 m of position noise per evaluation."""
    r = body_fn(t)
    v = (-body_fn(t + 2 * h) + 8 * body_fn(t + h) - 8 * body_fn(t - h) + body_fn(t - 2 * h)) / (12 * h)
    return r[None], v[None]


def test_integrator_matches_kepler_without_perturbers():
    q, e, inc, node, peri, tp = 2.1, 0.15, 12.0, 80.0, 150.0, 2461300.0
    t0, t1 = 2461330.5, 2461630.5
    r0, v0 = nbody.conic_state(np.array([q]), np.array([e]), inc, node, peri, tp, t0)
    rt, _ = nbody.conic_state(np.array([q]), np.array([e]), inc, node, peri, tp, t1)
    saved = dict(E.GM)
    E.GM.clear()
    try:
        R, _ = nbody.propagate(r0, v0, t0, [t1], BuiltinEphemeris(), gr=False)
    finally:
        E.GM.update(saved)
    assert np.linalg.norm(R[0, 0] - rt[0]) * AU_KM * 1000 < 1.0     # metres


@pytest.mark.skipif(_de421() is None, reason="DE421 files not available")
@pytest.mark.parametrize("body", ["mercury", "venus", "mars"])
def test_integrator_vs_de421(body):
    """A planet treated as a test particle must follow JPL DE421 for a year.

    Mercury and Venus also verify the relativistic term (without it Mercury is
    off by ~50 km after a year)."""
    de = _de421()
    t0 = 2461300.5
    f = lambda t: de.heliocentric(body, np.atleast_1d(t))[:, 0]          # noqa: E731
    r0, v0 = _state(de, f, t0, h=0.02)
    targets = np.array([t0 + 100, t0 + 365, t0 - 200])
    with _Massive([body], E.GM[body]):
        R, _ = nbody.propagate(r0, v0, t0, targets, de)
    truth = de.heliocentric(body, targets).T
    err_km = np.linalg.norm(R[:, 0] - truth, axis=1) * AU_KM
    assert np.all(err_km < 5.0), err_km      # 5 km ≈ 0.007″ at 1 AU


@pytest.mark.parametrize("body,iso", [("mars", "2026-10-21T21:00:00"), ("jupiter", "2027-03-05T02:30:00"),
                                      ("mars", "2027-03-05T02:30:00")])
def test_geometry_vs_astropy(body, iso):
    """Our astrometric position + aberration + light deflection == astropy apparent position."""
    import erfa

    eph = BuiltinEphemeris()
    loc = EarthLocation.from_geodetic(14.8 * u.deg, 49.9 * u.deg, 450 * u.m)
    t = Time(iso, scale="utc")
    jd = t.tdb.jd
    tab = BodyTable.build(eph, jd - 1, jd + 1)
    r, v = _state(eph, lambda x: tab.helio(body, x), jd, h=0.01)
    g, gv = loc.get_gcrs_posvel(t)
    obs = eph.barycentric("earth", [jd])[:, 0] + g.get_xyz().to_value(u.au)
    vec, _, _ = nbody.astrometric(r, v, jd, obs, tab)
    p = vec[0] / np.linalg.norm(vec[0])
    e_vec = obs - tab.bary("sun", jd)
    em = np.linalg.norm(e_vec)
    p = erfa.ldsun(p, e_vec / em, em)
    v_e = (eph.barycentric("earth", [jd + 1e-3]) - eph.barycentric("earth", [jd - 1e-3]))[:, 0] / 2e-3
    vo = (v_e + gv.get_xyz().to_value(u.au / u.day)) / E.C_AU_DAY
    p = erfa.ab(p, vo, em, math.sqrt(1 - vo @ vo))
    with solar_system_ephemeris.set("builtin"):
        tv = get_body(body, t, loc).cartesian.xyz.value
    sep = math.acos(min(1.0, float(p @ (tv / np.linalg.norm(tv))))) / ARCSEC
    assert sep < 0.01, sep


HORIZONS = sorted(DATA.glob("horizons_*.json"))


@pytest.mark.skipif(not HORIZONS, reason="no Horizons reference data")
@pytest.mark.parametrize("path", HORIZONS, ids=[p.stem for p in HORIZONS])
def test_against_jpl_horizons(path):
    """End-to-end: MPCORB line -> N-body -> topocentric astrometric RA/Dec vs JPL Horizons."""
    from synthtrack.identify import predict_field
    from synthtrack.mpcdata import parse_mpcorb

    ref = json.loads(path.read_text())
    el = parse_mpcorb([ref["mpcorb_line"]])
    de = _de421()
    for row in ref["ephemeris"]:
        t = Time(row["utc"], scale="utc")
        k = predict_field(el, None, t, row["ra"], row["dec"], 1.0, ref["lat"], ref["lon"], ref["alt_m"],
                          mag_limit=99, eph=de or BuiltinEphemeris())
        assert k, "object not predicted in the field"
        d = math.hypot((k[0].ra - row["ra"]) * math.cos(math.radians(row["dec"])), k[0].dec - row["dec"]) * 3600
        assert d < ref.get("tolerance_arcsec", 1.0), f"{row['utc']}: {d:.3f}″"
