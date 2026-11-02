"""Identification of detections with known asteroids and comets (offline).

A fast two-body pass over the whole catalogue selects objects near the
field; these are then integrated with the full N-body model (``nbody.py``:
planets, Moon, Pluto, solar GR) from their element epoch to the observation
time, with iterated light time and a topocentric observer. The result is the
astrometric ICRF position used by the MPC. The integration and geometry are
verified to milli-arcseconds against JPL DE421 and astropy
(``tests/test_precision.py``); the remaining error is the orbit itself.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric
from astropy.time import Time

K_GAUSS = 0.01720209895          # rad/day
C_AU_DAY = 173.1446326846693
EPS_J2000 = math.radians(84381.448 / 3600.0)
AU_KM = 149597870.7


@dataclass
class KnownObject:
    name: str
    kind: str                # asteroid | comet
    ra: float
    dec: float
    mag: float | None
    rate_arcsec_min: float
    pa_deg: float
    r_au: float
    delta_au: float
    x: float | None = None   # pixel position in the reference frame, if inside
    y: float | None = None
    matched_candidate: int | None = None
    forced_snr: float | None = None      # SNR measured on the predicted track
    forced_dxy: list | None = None       # offset (px) of the best forced position
    forced_note: str | None = None       # why the forced SNR may be unreliable (star on the track …)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Kepler / conics
# --------------------------------------------------------------------------

def _solve_elliptic(M, e):
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(50):
        f = E - e * np.sin(E) - M
        d = 1 - e * np.cos(E)
        dE = f / d
        E = E - dE
        if np.all(np.abs(dE) < 1e-12):
            break
    return E


def _solve_hyperbolic(M, e):
    H = np.arcsinh(M / e)
    for _ in range(60):
        f = e * np.sinh(H) - H - M
        d = e * np.cosh(H) - 1
        dH = f / d
        H = H - dH
        if np.all(np.abs(dH) < 1e-12):
            break
    return H


def _pq(peri, node, inc):
    w, O, i = np.radians(peri), np.radians(node), np.radians(inc)
    cw, sw, cO, sO, ci, si = np.cos(w), np.sin(w), np.cos(O), np.sin(O), np.cos(i), np.sin(i)
    P = np.stack([cw * cO - sw * sO * ci, cw * sO + sw * cO * ci, sw * si])
    Q = np.stack([-sw * cO - cw * sO * ci, -sw * sO + cw * cO * ci, cw * si])
    return P, Q


def _ecl_to_eq(v):
    ce, se = math.cos(EPS_J2000), math.sin(EPS_J2000)
    x, y, z = v
    return np.stack([x, ce * y - se * z, se * y + ce * z])


def asteroid_positions(el: dict, jd_tt: float, idx=None) -> np.ndarray:
    """Heliocentric equatorial (ICRS-aligned) positions [3, N] in AU."""
    s = slice(None) if idx is None else idx
    a, e = el["a"][s], el["e"][s]
    M = np.radians(el["M"][s] + el["n"][s] * (jd_tt - el["epoch_jd"][s]))
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = _solve_elliptic(M, e)
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(np.clip(1 - e * e, 0, None)) * np.sin(E)
    P, Q = _pq(el["peri"][s], el["node"][s], el["inc"][s])
    return _ecl_to_eq(xp * P + yp * Q)


def comet_positions(el: dict, jd_tt: float, idx=None) -> np.ndarray:
    s = slice(None) if idx is None else idx
    q, e = el["q"][s].astype(float), el["e"][s].astype(float)
    dt = jd_tt - el["Tp_jd"][s]
    xp = np.zeros_like(q)
    yp = np.zeros_like(q)
    ell = e < 0.9999
    hyp = e > 1.0001
    par = ~(ell | hyp)
    if ell.any():
        a = q[ell] / (1 - e[ell])
        n = K_GAUSS / a ** 1.5
        M = np.mod(n * dt[ell] + np.pi, 2 * np.pi) - np.pi
        E = _solve_elliptic(M, e[ell])
        xp[ell] = a * (np.cos(E) - e[ell])
        yp[ell] = a * np.sqrt(1 - e[ell] ** 2) * np.sin(E)
    if hyp.any():
        a = q[hyp] / (e[hyp] - 1)
        n = K_GAUSS / a ** 1.5
        H = _solve_hyperbolic(n * dt[hyp], e[hyp])
        xp[hyp] = a * (e[hyp] - np.cosh(H))
        yp[hyp] = a * np.sqrt(e[hyp] ** 2 - 1) * np.sinh(H)
    if par.any():
        W = 3 * K_GAUSS / np.sqrt(2 * q[par] ** 3) * dt[par]
        Y = np.cbrt(W / 2 + np.sqrt(W * W / 4 + 1))
        sv = Y - 1 / Y          # tan(nu/2)
        xp[par] = q[par] * (1 - sv * sv)
        yp[par] = 2 * q[par] * sv
    P, Q = _pq(el["peri"][s], el["node"][s], el["inc"][s])
    return _ecl_to_eq(xp * P + yp * Q)


# --------------------------------------------------------------------------
# Observer
# --------------------------------------------------------------------------

def observer_heliocentric(t: Time, lat: float, lon: float, alt_m: float) -> np.ndarray:
    earth = get_body_barycentric("earth", t)
    sun = get_body_barycentric("sun", t)
    helio = (earth - sun).get_xyz().to_value(u.au)
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, alt_m * u.m)
    gcrs, _ = loc.get_gcrs_posvel(t)
    return helio + gcrs.get_xyz().to_value(u.au)


def _radec(vec):
    x, y, z = vec
    d = np.sqrt(x * x + y * y + z * z)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z / d, -1, 1)))
    return ra, dec, d


def _sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1))))


def _pa_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    y = np.sin(r2 - r1) * np.cos(d2)
    x = np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(r2 - r1)
    return np.degrees(np.arctan2(y, x)) % 360.0


def comet_total_mag(M1, K, r, delta):
    """Total magnitude of a comet from CometEls.txt elements.

    CometEls stores the absolute magnitude H (cols 92-95) and the *slope
    parameter* K (cols 97-100); the MPC formula is
    m = H + 5 log Δ + 2.5·K·log r, i.e. the usual "K1" coefficient is 2.5·K
    (K = 4 → 10 log r). Using K directly made distant comets ~12 mag too
    bright (C/1995 O1 at r ≈ 50 au: 13.4 instead of ≈ 23.5)."""
    M1 = np.asarray(M1, float)
    k = 2.5 * np.nan_to_num(np.asarray(K, float), nan=4.0)
    return np.where(np.isfinite(M1), M1 + 5 * np.log10(delta) + k * np.log10(r), np.nan)


def hg_mag(H, G, r, delta, phase_rad):
    t = np.tan(phase_rad / 2)
    phi1 = np.exp(-3.33 * t ** 0.63)
    phi2 = np.exp(-1.87 * t ** 1.22)
    return H + 5 * np.log10(r * delta) - 2.5 * np.log10(np.clip((1 - G) * phi1 + G * phi2, 1e-6, None))


# --------------------------------------------------------------------------
# Field prediction
# --------------------------------------------------------------------------

def _observer_bary(eph, t_utc: Time, loc: EarthLocation) -> np.ndarray:
    jd = t_utc.tdb.jd
    g, _ = loc.get_gcrs_posvel(t_utc)
    return eph.barycentric("earth", [jd])[:, 0] + g.get_xyz().to_value(u.au)


def _prefilter(pos_helio, obs_helio, ra0, dec0, radius):
    ra, dec, _ = _radec(pos_helio - obs_helio[:, None])
    return _sep_deg(ra0, dec0, ra, dec) < radius


def _propagate_groups(states_fn, epochs, idx, targets, eph, table, q=None, nea_batch: int = 64,
                      twobody_fn=None, log=None):
    """N-body propagate rows ``idx`` (grouped by element epoch) to ``targets``.

    Objects that can approach the Earth or the Sun closely (q < 1.3 AU) force
    tiny integrator steps; they are integrated in small separate batches so
    the main-belt majority keeps its large steps (≈ 10× faster overall)."""
    R = np.empty((len(targets), len(idx), 3))
    V = np.empty_like(R)
    ep = epochs[idx]
    qq = q[idx] if q is not None else np.full(len(idx), 9.9)
    for e in np.unique(ep):
        g_all = np.nonzero(ep == e)[0]
        near = g_all[qq[g_all] < 1.3]
        far = g_all[qq[g_all] >= 1.3]
        batches = ([far] if far.size else []) + [near[i:i + nea_batch] for i in range(0, near.size, nea_batch)]
        for g in batches:
            _propagate_robust(g, idx, float(e), states_fn, targets, eph, table, R, V, twobody_fn, log)
    return R, V


def _propagate_robust(g, idx, epoch, states_fn, targets, eph, table, R, V, twobody_fn, log):
    """Integrate group ``g``; if the integrator fails (a badly determined orbit
    running into a planet/Moon makes the step size collapse), bisect the group
    so one bad orbit does not take the others down. A single failing orbit is
    propagated two-body and reported."""
    from .nbody import propagate
    try:
        r0, v0 = states_fn(idx[g])
        R[:, g], V[:, g] = propagate(r0, v0, epoch, targets, eph, table)
        return
    except RuntimeError:
        if g.size > 1:
            h = g.size // 2
            _propagate_robust(g[:h], idx, epoch, states_fn, targets, eph, table, R, V, twobody_fn, log)
            _propagate_robust(g[h:], idx, epoch, states_fn, targets, eph, table, R, V, twobody_fn, log)
            return
        if twobody_fn is None:
            raise
        for k, t in enumerate(targets):
            R[k, g], V[k, g] = twobody_fn(idx[g], t)
        if log is not None:
            name = getattr(twobody_fn, "names", None)
            label = str(name[idx[g][0]]) if name is not None else f"č. {int(idx[g][0])}"
            log(f"  N-body selhalo pro {label} (nejspíš špatně určená dráha) – použita dvoutělesová.")


def predict_field(mpcorb: dict | None, comets: dict | None, t_utc: Time, ra0: float, dec0: float,
                  radius_deg: float, lat: float, lon: float, alt_m: float,
                  mag_limit: float = 24.0, rate_dt_min: float = 10.0, eph=None,
                  precise: bool = True, max_nbody_span_days: float = 800.0,
                  log=None) -> list[KnownObject]:
    """All known objects within ``radius_deg`` of the field centre at ``t_utc``.

    Positions are topocentric astrometric (ICRF, light-time corrected, no
    aberration) – the convention of MPC observations and Gaia star positions.
    With ``precise`` the orbits are integrated with the N-body model from
    their element epoch (``nbody.py``); otherwise two-body (fast preview).

    Orbits whose epoch is more than ``max_nbody_span_days`` from the
    observation (lost / one-opposition objects, epochs back to the 1940s in a
    full MPCORB) are propagated two-body: their orbit uncertainty dwarfs the
    perturbations, and integrating them would mean one integration per stale
    epoch plus a planet table spanning decades (minutes instead of seconds).
    """
    from .ephem import BodyTable, load_ephemeris
    from .nbody import asteroid_state, astrometric, comet_state

    eph = eph or load_ephemeris("auto")
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, alt_m * u.m)
    t2 = t_utc + rate_dt_min * u.min
    jds = np.array([t_utc.tdb.jd, t2.tdb.jd])
    obs = [_observer_bary(eph, t_utc, loc), _observer_bary(eph, t2, loc)]
    sun0 = eph.barycentric("sun", [jds[0]])[:, 0]
    obs_helio = obs[0] - sun0
    out: list[KnownObject] = []

    for kind, el in (("asteroid", mpcorb), ("comet", comets)):
        if el is None or len(el["name"]) == 0:
            continue
        # ---- 1) coarse two-body pass over the whole catalogue
        if kind == "asteroid":
            p = asteroid_positions(el, jds[0])
            q = el["a"] * (1 - el["e"])
            sel = _prefilter(p, obs_helio, ra0, dec0, radius_deg + 2.0)
            # NEAs: perturbations since the element epoch can move them – a margin
            # around the field for all, the wide 45° net only for the nearby ones
            # (thousands of NEAs otherwise, minutes of integration near the ecliptic)
            near_q = q < 1.3
            sel |= near_q & _prefilter(p, obs_helio, ra0, dec0, radius_deg + 5.0)
            delta = np.linalg.norm(p - obs_helio[:, None], axis=0) if p.shape[0] == 3 \
                else np.linalg.norm(p - obs_helio[None, :], axis=1)
            sel |= near_q & (delta < 0.2) & _prefilter(p, obs_helio, ra0, dec0, 45.0)
            epochs = el["epoch_jd"]
        else:
            p = comet_positions(el, jds[0])
            sel = _prefilter(p, obs_helio, ra0, dec0, radius_deg + 5.0)
            epochs = el["epoch_jd"] if "epoch_jd" in el else np.full(len(el["name"]), np.nan)
        idx = np.nonzero(sel)[0]
        if idx.size == 0:
            continue
        # ---- 2) precise states at observation times
        if kind == "asteroid":
            fn = lambda ii: asteroid_state(el, ii)                                  # noqa: E731
        else:
            fn = lambda ii: comet_state(el, ii, epochs[ii])                         # noqa: E731
        perturbed = np.isfinite(epochs[idx]) if precise else np.zeros(idx.size, bool)
        if precise:
            stale = perturbed & (np.abs(np.nan_to_num(epochs[idx], nan=jds[0]) - jds[0]) > max_nbody_span_days)
            if stale.any():
                perturbed &= ~stale
                if log is not None:
                    log(f"  {kind}: {int(stale.sum())} drah se starou epochou (> {max_nbody_span_days:.0f} d) "
                        f"jen dvoutělesově – jejich nejistota je stejně velká.")
        R = np.empty((2, idx.size, 3))
        V = np.empty_like(R)
        integrated_epochs = epochs[idx[perturbed]]
        lo = min(np.nanmin(integrated_epochs) if integrated_epochs.size else jds[0], jds[0])
        hi = max(np.nanmax(integrated_epochs) if integrated_epochs.size else jds[1], jds[1])
        table = BodyTable.build(eph, lo, hi, bodies=None) if perturbed.any() else BodyTable.build(eph, jds[0], jds[1], bodies=("sun",))
        if perturbed.any():
            pi = idx[perturbed]
            qarr = el["a"] * (1 - el["e"]) if kind == "asteroid" else el["q"]
            if kind == "asteroid":
                tb = lambda ii, t: asteroid_state(el, ii, t)                         # noqa: E731
            else:
                tb = lambda ii, t: comet_state(el, ii, t)                            # noqa: E731
            tb.names = el["name"]
            Rp, Vp = _propagate_groups(fn, epochs, pi, jds, eph, table, q=qarr, twobody_fn=tb, log=log)
            R[:, perturbed], V[:, perturbed] = Rp, Vp
        if (~perturbed).any():
            ui = idx[~perturbed]
            for k in range(2):
                if kind == "asteroid":
                    R[k, ~perturbed], V[k, ~perturbed] = asteroid_state(el, ui, jds[k])
                else:
                    R[k, ~perturbed], V[k, ~perturbed] = comet_state(el, ui, jds[k])
        res = []
        for k in range(2):
            vec, delta, r_ret = astrometric(R[k], V[k], jds[k], obs[k], table)
            ra, dec, _ = _radec(vec.T)
            res.append((ra, dec, delta, r_ret))
        (ra, dec, delta, r_ret), (ra2, dec2, _, _) = res
        inside = _sep_deg(ra0, dec0, ra, dec) < radius_deg
        if not inside.any():
            continue
        sel_i = idx[inside]
        ra, dec, delta, ra2, dec2, r_ret, vec = ra[inside], dec[inside], delta[inside], ra2[inside], dec2[inside], r_ret[inside], None
        rate = _sep_deg(ra, dec, ra2, dec2) * 3600 / rate_dt_min
        pa = _pa_deg(ra, dec, ra2, dec2)
        r = np.linalg.norm(r_ret, axis=1)
        # phase angle Sun–object–observer
        to_obs = (obs[0] - sun0)[None, :] - r_ret
        cosph = np.sum(-r_ret * to_obs, axis=1) / (r * np.linalg.norm(to_obs, axis=1))
        phase = np.arccos(np.clip(cosph, -1, 1))
        if kind == "asteroid":
            mag = hg_mag(el["H"][sel_i], el["G"][sel_i], r, delta, phase)
        else:
            mag = comet_total_mag(el["M1"][sel_i], el["K1"][sel_i], r, delta)
        for j in range(sel_i.size):
            m = float(mag[j]) if np.isfinite(mag[j]) else None
            if m is not None and m > mag_limit:
                continue
            out.append(KnownObject(str(el["name"][sel_i[j]]), kind, float(ra[j]), float(dec[j]), m,
                                   float(rate[j]), float(pa[j]), float(r[j]), float(delta[j])))
    out.sort(key=lambda k: (k.mag if k.mag is not None else 99))
    return out


def match_candidates(cands, known: list[KnownObject], tol_arcsec: float, rate_tol_frac: float = 0.35,
                     rate_tol_abs: float = 0.05) -> None:
    """Attach the nearest compatible known object to each candidate (in place)."""
    if not known:
        return
    # strongest real candidates claim known objects first
    cands = sorted(cands, key=lambda c: (c.auto_status != "candidate", -c.snr))
    kra = np.array([k.ra for k in known])
    kdec = np.array([k.dec for k in known])
    for c in cands:
        if c.ra is None:
            continue
        sep = _sep_deg(c.ra, c.dec, kra, kdec) * 3600
        order = np.argsort(sep)
        for j in order[:5]:
            if sep[j] > tol_arcsec:
                break
            k = known[j]
            if c.rate_arcsec_min is not None and c.pa_deg is not None:
                cpa, kpa = math.radians(c.pa_deg), math.radians(k.pa_deg)
                # Compare the full sky-plane velocity vector. Equal speed in the
                # opposite direction used to produce a false identification.
                dr = math.hypot(c.rate_arcsec_min * math.sin(cpa) - k.rate_arcsec_min * math.sin(kpa),
                                c.rate_arcsec_min * math.cos(cpa) - k.rate_arcsec_min * math.cos(kpa))
                if dr > max(rate_tol_abs, rate_tol_frac * k.rate_arcsec_min):
                    continue
            c.known = {"name": k.name, "kind": k.kind, "o_c_arcsec": round(float(sep[j]), 2),
                       "pred_mag": None if k.mag is None else round(k.mag, 2),
                       "pred_rate": round(k.rate_arcsec_min, 3), "pred_pa": round(k.pa_deg, 1)}
            if k.matched_candidate is None:
                k.matched_candidate = c.id
            break
