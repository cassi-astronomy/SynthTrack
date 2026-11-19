"""Statistical ranging: Kepler propagation and coverage of a low-elongation parabolic comet."""
import math

import numpy as np

from synthtrack import ranging as RG

SITE = (-35.1167, -69.6, 1650.0)       # Coihueco
MJD0 = 61356.3446


def test_kepler_period_and_hyperbolic_energy():
    r0 = np.array([[2.0, 0.3, 0.1], [0.5, 0.2, -0.1]])
    v0 = np.array([[0.0, 0.011, 0.002], [0.0, 0.035, 0.004]])        # elliptic, hyperbolic
    a = 1 / (2 / np.linalg.norm(r0[0]) - v0[0] @ v0[0] / RG.MU)
    P = 2 * math.pi * math.sqrt(a ** 3 / RG.MU)
    r = RG.kepler(r0[:1], v0[:1], P)
    assert np.allclose(r, r0[:1], atol=1e-9)
    # energy conservation via a small step on the hyperbolic one
    def en(r, v):
        return v @ v / 2 - RG.MU / np.linalg.norm(r)
    h = 1e-3
    r1, r2 = RG.kepler(r0[1:], v0[1:], 10.0)[0], RG.kepler(r0[1:], v0[1:], 10.0 + h)[0]
    rm = RG.kepler(r0[1:], v0[1:], 10.0 - h)[0]
    assert abs(en(r1, (r2 - rm) / (2 * h)) - en(r0[1], v0[1])) < 1e-9


def _comet_candidate():
    R, V = RG.observer_state(MJD0, SITE)
    sun = -R / np.linalg.norm(R)
    perp = np.cross(sun, [0, 0, 1.0]); perp /= np.linalg.norm(perp)
    el = math.radians(35)                                            # low elongation (twilight)
    u = math.cos(el) * sun + math.sin(el) * perp
    r = R + 1.1 * u
    rn = np.linalg.norm(r)
    t = np.cross(r, [0.2, 0.3, 1.0]); t /= np.linalg.norm(t)
    d = -0.6 * r / rn + 0.8 * t                                      # inbound, parabolic speed
    v = d / np.linalg.norm(d) * math.sqrt(2 * RG.MU / rn)
    pos = []
    for dt in (-13.0, -1.0, 0.0, 9.9):
        ra, de = RG.predict(r[None], v[None], MJD0, MJD0 + dt / 1440, SITE)
        pos.append((dt, math.degrees(ra[0]), math.degrees(de[0])))
    (_, ra1, de1), (_, ra0, de0), (_, ra2, de2) = pos[0], pos[2], pos[3]
    cd = math.cos(math.radians(de0))
    dx, dy = ((ra2 - ra1 + 540) % 360 - 180) * cd * 3600, (de2 - de1) * 3600
    rate = math.hypot(dx, dy) / 22.9
    c = {"id": 1, "ra": ra0, "dec": de0, "rate_arcsec_min": rate, "pa_deg": math.degrees(math.atan2(dx, dy)) % 360,
         "fwhm": 2.8, "snr": 20.0,
         "positions": [{"dt_min": p[0], "ra": p[1], "dec": p[2], "ok": True} for k, p in enumerate(pos) if k != 2]}
    return c, r, v


def _inside(c, r, v, model):
    res = RG.run(c, MJD0, SITE, hours=(24,), model=model, pixel_scale=6.4, n=2000)
    T = res["times"][0]
    ra_t, de_t = RG.predict(r[None], v[None], MJD0, T["mjd"], SITE)
    xi, eta = RG.tangent(ra_t, de_t, math.radians(T["ra"]), math.radians(T["dec"]))
    p = np.array([xi[0], eta[0]]) * 180 * 60 / math.pi
    pts = np.array(T["points"])
    return float(np.min(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]))), T


def test_parabolic_comet_low_elongation_in_cloud():
    c, r, v = _comet_candidate()
    d_comet, Tc = _inside(c, r, v, "comet")
    d_all, Ta = _inside(c, r, v, "all")
    # the true position lies within the sampled cloud (nearest sample ≪ cloud size)
    assert d_comet < 0.05 * Tc["major_arcmin"] + 1.0
    assert d_all < 0.05 * Ta["major_arcmin"] + 1.0


def test_rho_min_shrinks_region_and_keeps_comet():
    c, r, v = _comet_candidate()
    full = RG.run(c, MJD0, SITE, hours=(24,), model="comet", pixel_scale=6.4, n=2000)["times"][0]
    cut = RG.run(c, MJD0, SITE, hours=(24,), model="comet", pixel_scale=6.4, n=2000, rho_min=0.1)
    T = cut["times"][0]
    assert T["major_arcmin"] < 0.5 * full["major_arcmin"]
    ra_t, de_t = RG.predict(r[None], v[None], MJD0, T["mjd"], SITE)
    xi, eta = RG.tangent(ra_t, de_t, math.radians(T["ra"]), math.radians(T["dec"]))
    p = np.array([xi[0], eta[0]]) * 180 * 60 / math.pi
    pts = np.array(T["points"])
    assert float(np.min(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]))) < 0.05 * T["major_arcmin"] + 1.0
