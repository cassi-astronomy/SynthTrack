"""Statistical ranging: where can a short-arc discovery be in 12–48 hours?

One night's series gives the *attributable* – position (α, δ) and angular rate
(α̇, δ̇) at the reference epoch – but not the topocentric distance ρ nor the radial
rate ρ̇. For every pair (ρ, ρ̇) the heliocentric state is

    r = R_obs + ρ·u,      v = V_obs + ρ̇·u + ρ·u̇

(u unit vector to the object, u̇ from α̇, δ̇). Only pairs giving an admissible
orbit are kept: heliocentric energy ≤ 0 (e ≤ 1), not bound to the Earth, plus the
constraints of the chosen orbit class. Random samples of that region (log-uniform
in ρ, uniform in ρ̇ between the e = 1 limits, the attributable varied within its
errors) are propagated (two-body, universal variables, light time) and give a
cloud of possible positions – the idea of the MPC NEOCP uncertainty maps
(systematic/statistical ranging, Virtanen et al. 2001; Milani et al. 2004).

The density of the cloud depends on the sampling prior; its *extent* is the
answer to "how big a field do I need tomorrow". Pure NumPy, ~0.1 s per object.
"""
from __future__ import annotations

import math

import numpy as np

MU = 2.9591220828559115e-4          # k², AU³/day² (Sun)
MU_EARTH = MU / 332946.0487
C_AU_DAY = 173.1446326846693
ARCSEC = math.pi / (180 * 3600)

MODELS = {
    "all": "všechny dráhy s e ≤ 1",
    "mb": "planetka hlavního pásu (a 1,8–4,6 AU, e < 0,4, i < 40°)",
    "neo": "blízkozemní objekt (q < 1,3 AU)",
    "comet": "kometa (parabolická dráha, q < 6 AU)",
}


# ----------------------------------------------------------------------------- Kepler

def _stumpff(z):
    c = np.empty_like(z)
    s = np.empty_like(z)
    pos, neg = z > 1e-8, z < -1e-8
    mid = ~(pos | neg)
    sz = np.sqrt(z[pos])
    c[pos] = (1 - np.cos(sz)) / z[pos]
    s[pos] = (sz - np.sin(sz)) / sz ** 3
    sz = np.sqrt(-z[neg])
    c[neg] = (np.cosh(sz) - 1) / -z[neg]
    s[neg] = (np.sinh(sz) - sz) / sz ** 3
    zm = z[mid]
    c[mid] = 0.5 - zm / 24
    s[mid] = 1 / 6 - zm / 120
    return c, s


def kepler(r0: np.ndarray, v0: np.ndarray, dt, mu: float = MU) -> np.ndarray:
    """Two-body positions after ``dt`` days; r0, v0 (N, 3) AU, AU/day; dt scalar or (N,)."""
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    dt = np.broadcast_to(np.asarray(dt, float), (len(r0),)).copy()
    rn = np.linalg.norm(r0, axis=1)
    vr = np.einsum("ij,ij->i", r0, v0) / rn
    alpha = 2 / rn - np.einsum("ij,ij->i", v0, v0) / mu          # 1/a
    sq = math.sqrt(mu)
    x = sq * np.abs(alpha) * dt                                   # initial guess (elliptic)
    big = np.abs(alpha) < 1e-6
    x[big] = sq * dt[big] / rn[big]
    for _ in range(60):
        z = alpha * x * x
        c, s = _stumpff(z)
        f = rn * vr / sq * x * x * c + (1 - alpha * rn) * x ** 3 * s + rn * x - sq * dt
        fp = rn * vr / sq * x * (1 - z * s) + (1 - alpha * rn) * x * x * c + rn
        dx = f / fp
        x -= dx
        if np.all(np.abs(dx) < 1e-12 * (1 + np.abs(x))):
            break
    z = alpha * x * x
    c, s = _stumpff(z)
    fl = 1 - x * x / rn * c
    g = dt - x ** 3 / sq * s
    return fl[:, None] * r0 + g[:, None] * v0


def elements(r: np.ndarray, v: np.ndarray, mu: float = MU) -> dict:
    """a, e, q, i (deg) of heliocentric states (N, 3) in the *equatorial* frame."""
    rn = np.linalg.norm(r, axis=1)
    v2 = np.einsum("ij,ij->i", v, v)
    en = v2 / 2 - mu / rn
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    evec = np.cross(v, h) / mu - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    with np.errstate(divide="ignore"):
        a = np.where(np.abs(en) > 1e-12, -mu / (2 * en), np.inf)
    q = hn * hn / (mu * (1 + e))
    # inclination to the ecliptic (ε J2000)
    eps = math.radians(84381.448 / 3600)
    hz_ecl = -math.sin(eps) * h[:, 1] + math.cos(eps) * h[:, 2]
    inc = np.degrees(np.arccos(np.clip(hz_ecl / hn, -1, 1)))
    return {"a": a, "e": e, "q": q, "i": inc}


# ----------------------------------------------------------------------------- geometry

def unit(ra, dec):
    ca, sa, cd, sd = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
    u = np.stack([cd * ca, cd * sa, sd], axis=-1)
    du_da = np.stack([-cd * sa, cd * ca, np.zeros_like(ra)], axis=-1)
    du_dd = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
    return u, du_da, du_dd


def radec(vec):
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    d = np.sqrt(x * x + y * y + z * z)
    return np.arctan2(y, x) % (2 * math.pi), np.arcsin(np.clip(z / d, -1, 1)), d


def tangent(ra, dec, ra0, dec0):
    """Gnomonic offsets (ξ east, η north) in radians around (ra0, dec0)."""
    cd, sd, cd0, sd0 = np.cos(dec), np.sin(dec), math.cos(dec0), math.sin(dec0)
    da = ra - ra0
    den = sd * sd0 + cd * cd0 * np.cos(da)
    return cd * np.sin(da) / den, (sd * cd0 - cd * sd0 * np.cos(da)) / den


# ----------------------------------------------------------------------------- attributable

def attributable(c: dict, pixel_scale: float | None = None) -> dict | None:
    """α, δ (rad), α̇, δ̇ (rad/day) at the reference epoch and their 1σ errors."""
    if c.get("ra") is None or c.get("rate_arcsec_min") is None or c.get("pa_deg") is None:
        return None
    ra, dec = math.radians(c["ra"]), math.radians(c["dec"])
    rate = c["rate_arcsec_min"] * ARCSEC * 1440.0                  # rad/day
    pa = math.radians(c["pa_deg"])
    dadt = rate * math.sin(pa) / max(math.cos(dec), 1e-6)
    dddt = rate * math.cos(pa)
    # scatter of the sub-stack positions around a straight line (as the GUI's linear ephemeris);
    # floor: 0.25 px, and the centroid error of one sub-stack ≈ FWHM / SNR_substack. Three points
    # hardly measure their own scatter: (218) Bianca (317/1, SNR 10, slow) had its rate 39 % off
    # while the scatter promised 7 %; with the FWHM/SNR floor and Student-t tails it is ~2σ.
    floor = 0.25 * float(pixel_scale or 1.0)
    if c.get("fwhm") and c.get("snr") and pixel_scale:
        n_sub = max(1, len(c.get("positions") or []) or 3)
        floor = max(floor, float(c["fwhm"]) * float(pixel_scale) / max(float(c["snr"]) / math.sqrt(n_sub), 1.0))
    pts = [p for p in (c.get("positions") or []) if p.get("ok", True) and p.get("ra") is not None
           and p.get("dt_min") is not None]
    sig_pos, sig_v = floor, None
    if len(pts) >= 2:
        t = np.array([p["dt_min"] for p in pts], float)
        cd = math.cos(dec)
        x = np.array([((p["ra"] - c["ra"] + 540) % 360 - 180) * cd * 3600 for p in pts])
        y = np.array([(p["dec"] - c["dec"]) * 3600 for p in pts])
        tm = t.mean()
        sxx = float(((t - tm) ** 2).sum())
        rss = 0.0
        for v in (x, y):
            b = float(((t - tm) * (v - v.mean())).sum() / sxx) if sxx > 0 else 0.0
            rss += float(((v - v.mean() - b * (t - tm)) ** 2).sum())
        sig_pos = max(floor, math.sqrt(rss / max(1, 2 * len(pts) - 4)))
        if sxx > 0:
            sig_v = sig_pos / math.sqrt(sxx)                        # ″/min
    if sig_v is None:
        sig_v = sig_pos / 10.0
    return {"ra": ra, "dec": dec, "dadt": dadt, "dddt": dddt,
            "sig_pos": sig_pos * ARCSEC, "sig_v": sig_v * ARCSEC * 1440.0}


def observer_state(mjd: float, site) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric equatorial position (AU) and velocity (AU/day) of the topocentric observer."""
    from astropy.time import Time
    from .identify import observer_heliocentric
    lat, lon, alt = site
    h = 0.002                                                       # ~3 min
    t = Time([mjd - h, mjd, mjd + h], format="mjd", scale="utc")
    p = np.array([observer_heliocentric(t[i], lat, lon, alt) for i in range(3)])
    return p[1], (p[2] - p[0]) / (2 * h)


# ----------------------------------------------------------------------------- ranging

def sample(att: dict, R: np.ndarray, V: np.ndarray, model: str = "all", n: int = 4000,
           rho_range=(0.003, 100.0), seed: int = 0, max_batches: int = 40, r_range=None):
    """Admissible heliocentric states (r, v) at the reference epoch and their ρ."""
    rng = np.random.default_rng(seed)
    out_r, out_v, out_rho = [], [], []
    got = 0
    lo, hi = math.log10(rho_range[0]), math.log10(rho_range[1])
    batch = max(2000, 3 * n)
    for _ in range(max_batches):
        m = batch
        ra = att["ra"] + rng.normal(0, att["sig_pos"], m) / max(math.cos(att["dec"]), 1e-6)
        de = att["dec"] + rng.normal(0, att["sig_pos"], m)
        # heavy tails (Student t, 3 dof, scaled to σ): a rate from three sub-stacks is sometimes far off
        dadt = att["dadt"] + rng.standard_t(3, m) * (att["sig_v"] / math.sqrt(3)) / max(math.cos(att["dec"]), 1e-6)
        dddt = att["dddt"] + rng.standard_t(3, m) * (att["sig_v"] / math.sqrt(3))
        u, ua, ud = unit(ra, de)
        udot = ua * dadt[:, None] + ud * dddt[:, None]
        rho = 10 ** rng.uniform(lo, hi, m)
        r = R[None, :] + rho[:, None] * u
        rn = np.linalg.norm(r, axis=1)
        W = V[None, :] + rho[:, None] * udot
        b = np.einsum("ij,ij->i", W, u)
        disc = b * b - np.einsum("ij,ij->i", W, W) + 2 * MU / rn      # e ≤ 1 limits of ρ̇
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        if model == "comet":                                          # parabolic: on the limit
            side = rng.integers(0, 2, m) * 2 - 1
            rdot = -b + side * sq * (1 - 1e-9)
        else:
            rdot = -b + sq * rng.uniform(-1, 1, m)
        v = W + rdot[:, None] * u
        # not bound to the Earth (satellites, the Moon's neighbourhood)
        vrel = v - V[None, :]
        ok &= np.einsum("ij,ij->i", vrel, vrel) > 2 * MU_EARTH / rho
        ok &= rn > 0.01
        if r_range is not None:                                       # heliocentric distance now
            ok &= (rn >= r_range[0]) & (rn <= r_range[1])
        if model != "all":
            el = elements(r[ok], v[ok])
            sel = np.ones(int(ok.sum()), bool)
            if model == "mb":
                sel = (el["a"] > 1.8) & (el["a"] < 4.6) & (el["e"] < 0.4) & (el["i"] < 40)
            elif model == "neo":
                sel = el["q"] < 1.3
            elif model == "comet":
                sel = el["q"] < 6.0
            idx = np.nonzero(ok)[0]
            ok[:] = False
            ok[idx[sel]] = True
        out_r.append(r[ok])
        out_v.append(v[ok])
        out_rho.append(rho[ok])
        got += int(ok.sum())
        if got >= n:
            break
    if not got:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    r, v, rho = np.concatenate(out_r)[:n], np.concatenate(out_v)[:n], np.concatenate(out_rho)[:n]
    return r, v, rho


def predict(r0, v0, t0_mjd: float, mjd: float, site) -> tuple[np.ndarray, np.ndarray]:
    """Astrometric RA/Dec (rad) at ``mjd`` of the sampled states (light time iterated)."""
    R, _ = observer_state(mjd, site)
    dt = np.full(len(r0), mjd - t0_mjd)
    tau = np.zeros(len(r0))
    for _ in range(3):
        r = kepler(r0, v0, dt - tau)
        d = r - R[None, :]
        tau = np.linalg.norm(d, axis=1) / C_AU_DAY
    ra, dec, _ = radec(d)
    return ra, dec


def run(c: dict, mjd_ref: float, site, hours=(12, 18, 24, 36, 48), model: str = "all", n: int = 3000,
        pixel_scale: float | None = None, seed: int = 0, n_points: int = 600, r_range=None,
        rho_min: float = 0.003) -> dict:
    """Ranging summary for one candidate (JSON-ready)."""
    att = attributable(c, pixel_scale)
    if att is None:
        raise ValueError("kandidát nemá polohu nebo pohyb")
    R, V = observer_state(mjd_ref, site)
    r0, v0, rho = sample(att, R, V, model=model, n=n, seed=seed, r_range=r_range,
                         rho_range=(max(0.001, float(rho_min)), 100.0))
    if not len(r0):
        return {"model": model, "n": 0, "note": "Pro tento předpoklad neexistuje žádná přípustná dráha."}
    # the sampled states must reproduce the attributable at t0 (check of the geometry)
    el = elements(r0, v0)
    out = {"model": model, "model_label": MODELS.get(model, model), "n": int(len(r0)),
           "rho_au": [float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))],
           "r_au": [float(np.percentile(np.linalg.norm(r0, axis=1), p)) for p in (2.5, 50, 97.5)],
           "r_range": list(r_range) if r_range is not None else None,
           "a_au": [float(np.nanpercentile(np.where(np.isfinite(el["a"]), el["a"], np.nan), p)) for p in (5, 50, 95)],
           "e": [float(np.percentile(el["e"], p)) for p in (5, 50, 95)],
           "q_au": [float(np.percentile(el["q"], p)) for p in (5, 50, 95)],
           "times": []}
    rng = np.random.default_rng(seed + 1)
    pick = rng.choice(len(r0), size=min(n_points, len(r0)), replace=False)
    rate = c["rate_arcsec_min"] * ARCSEC * 60 * 24                   # rad per day
    for h in hours:
        mjd = mjd_ref + h / 24.0
        ra, dec = predict(r0, v0, mjd_ref, mjd, site)
        # linear (great-circle) prediction for comparison
        pa = math.radians(c["pa_deg"])
        rho_l = rate * h / 24.0
        d0 = att["dec"]
        sd = math.sin(d0) * math.cos(rho_l) + math.cos(d0) * math.sin(rho_l) * math.cos(pa)
        dl = math.asin(max(-1, min(1, sd)))
        al = att["ra"] + math.atan2(math.sin(pa) * math.sin(rho_l) * math.cos(d0), math.cos(rho_l) - math.sin(d0) * sd)
        # centre: median in the tangent plane around the linear prediction
        xi, eta = tangent(ra, dec, al, dl)
        cx, cy = float(np.median(xi)), float(np.median(eta))
        X = np.stack([xi - cx, eta - cy], 1)
        cov = np.cov(X.T) if len(X) > 2 else np.eye(2) * 1e-12
        w, vec = np.linalg.eigh(cov)
        major = vec[:, 1]
        pmaj = X @ major
        pmin = X @ vec[:, 0]
        lo_M, hi_M = np.percentile(pmaj, [2.5, 97.5])
        lo_m, hi_m = np.percentile(pmin, [2.5, 97.5])
        if hi_m - lo_m > hi_M - lo_M:              # heavy tails: the covariance axis is not the long one
            major = vec[:, 0]
            lo_M, hi_M, lo_m, hi_m = lo_m, hi_m, lo_M, hi_M
        # centre back to RA/Dec (inverse gnomonic)
        rr = math.hypot(cx, cy)
        if rr > 0:
            cc = math.atan(rr)
            dec_c = math.asin(math.cos(cc) * math.sin(dl) + cy * math.sin(cc) * math.cos(dl) / rr)
            ra_c = al + math.atan2(cx * math.sin(cc), rr * math.cos(dl) * math.cos(cc) - cy * math.sin(dl) * math.sin(cc))
        else:
            ra_c, dec_c = al, dl
        to_am = 180 * 60 / math.pi
        out["times"].append({
            "hours": h, "mjd": mjd,
            "ra": math.degrees(ra_c) % 360, "dec": math.degrees(dec_c),
            "major_arcmin": float((hi_M - lo_M) * to_am), "minor_arcmin": float((hi_m - lo_m) * to_am),
            "major_pa_deg": float(math.degrees(math.atan2(major[0], major[1])) % 180),
            "linear_ra": math.degrees(al) % 360, "linear_dec": math.degrees(dl),
            "linear_offset_arcmin": float(math.hypot(cx, cy) * to_am),
            # cloud for the plot: offsets (ξ east, η north) from the centre, arcmin
            "points": np.round(X[pick] * to_am, 3).tolist(),
            "linear_xy": [round(-cx * to_am, 3), round(-cy * to_am, 3)],
        })
    return out
