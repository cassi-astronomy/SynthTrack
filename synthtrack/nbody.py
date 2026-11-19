"""High-precision propagation of asteroid/comet orbits (N-body, vectorised).

Model (heliocentric frame, ICRF axes, AU, days, TDB):
  * Sun (Gaussian k² as used by the MPC elements),
  * direct + indirect perturbations of Mercury … Neptune, Moon and Pluto
    from the ephemeris (Earth and Moon separately – matters for NEAs),
  * 1PN general-relativistic correction of the Sun,
  * massless test particles; all objects of one epoch are integrated
    together with a shared adaptive DOP853 step (rtol 1e-12).

Accuracy against JPL DE ephemerides is at the metre level for the
integration itself; in practice the error budget is dominated by the
quality of the orbit (MPCORB elements) – see tests/test_precision.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .ephem import C_AU_DAY, GM, K2, BodyTable, Ephemeris

EPS_J2000 = math.radians(84381.448 / 3600.0)
_CE, _SE = math.cos(EPS_J2000), math.sin(EPS_J2000)
ECL2EQ = np.array([[1, 0, 0], [0, _CE, -_SE], [0, _SE, _CE]])


# --------------------------------------------------------------------------
# Elements -> state
# --------------------------------------------------------------------------

def _kepler_E(M, e):
    E = np.where(e < 0.8, M, np.pi * np.sign(M) + 0 * M)
    E = np.where(E == 0, M, E)
    for _ in range(60):
        f = E - e * np.sin(E) - M
        dE = f / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < 1e-14):
            break
    return E


def _kepler_H(M, e):
    H = np.arcsinh(M / e)
    for _ in range(80):
        dH = (e * np.sinh(H) - H - M) / (e * np.cosh(H) - 1)
        H = H - dH
        if np.all(np.abs(dH) < 1e-14):
            break
    return H


def conic_state(q, e, inc, node, peri, tp, t, mu=K2):
    """Heliocentric equatorial state (N,3),(N,3) at time ``t`` for conic elements.

    Angles in degrees (ecliptic J2000), ``tp`` = time of perihelion (days).
    Works for elliptic, parabolic and hyperbolic orbits.
    """
    q, e = np.asarray(q, float), np.asarray(e, float)
    dt = np.asarray(t, float) - np.asarray(tp, float)
    dt = np.broadcast_to(dt, q.shape).astype(float)
    nu = np.empty_like(q)
    ell = e < 1 - 1e-8
    hyp = e > 1 + 1e-8
    par = ~(ell | hyp)
    if ell.any():
        a = q[ell] / (1 - e[ell])
        M = np.sqrt(mu / a ** 3) * dt[ell]
        M = np.mod(M + np.pi, 2 * np.pi) - np.pi
        E = _kepler_E(M, e[ell])
        nu[ell] = 2 * np.arctan2(np.sqrt(1 + e[ell]) * np.sin(E / 2), np.sqrt(1 - e[ell]) * np.cos(E / 2))
    if hyp.any():
        a = q[hyp] / (e[hyp] - 1)
        M = np.sqrt(mu / a ** 3) * dt[hyp]
        H = _kepler_H(M, e[hyp])
        nu[hyp] = 2 * np.arctan(np.sqrt((e[hyp] + 1) / (e[hyp] - 1)) * np.tanh(H / 2))
    if par.any():
        W = 3 * np.sqrt(mu / (2 * q[par] ** 3)) * dt[par]
        Y = np.cbrt(W / 2 + np.sqrt(W * W / 4 + 1))
        nu[par] = 2 * np.arctan(Y - 1 / Y)
    p = q * (1 + e)
    r = p / (1 + e * np.cos(nu))
    xp, yp = r * np.cos(nu), r * np.sin(nu)
    f = np.sqrt(mu / p)
    vxp, vyp = -f * np.sin(nu), f * (e + np.cos(nu))
    w, O, i = np.radians(peri), np.radians(node), np.radians(inc)
    cw, sw, cO, sO, ci, si = np.cos(w), np.sin(w), np.cos(O), np.sin(O), np.cos(i), np.sin(i)
    P = np.stack([cw * cO - sw * sO * ci, cw * sO + sw * cO * ci, sw * si], axis=-1)
    Q = np.stack([-sw * cO - cw * sO * ci, -sw * sO + cw * cO * ci, cw * si], axis=-1)
    pos = xp[:, None] * P + yp[:, None] * Q
    vel = vxp[:, None] * P + vyp[:, None] * Q
    return pos @ ECL2EQ.T, vel @ ECL2EQ.T


def asteroid_state(el: dict, idx, t=None):
    """State at the element epoch (or at ``t`` two-body) for MPCORB rows ``idx``."""
    a, e = el["a"][idx], el["e"][idx]
    n = np.sqrt(K2 / a ** 3)                              # rad/day, consistent with mu
    tp = el["epoch_jd"][idx] - np.radians(el["M"][idx]) / n
    tt = el["epoch_jd"][idx] if t is None else t
    return conic_state(a * (1 - e), e, el["inc"][idx], el["node"][idx], el["peri"][idx], tp, tt)


def comet_state(el: dict, idx, t):
    return conic_state(el["q"][idx], el["e"][idx], el["inc"][idx], el["node"][idx], el["peri"][idx],
                       el["Tp_jd"][idx], t)


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------

def acceleration(t, r, v, table: BodyTable, gr: bool = True):
    """Heliocentric acceleration (N,3) of massless particles."""
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    acc = -K2 * r / rn ** 3
    if table.combined is not None:
        P = table.helio_all(t)                                       # (B, 3), one spline call
        sel = [j for j, b in enumerate(table.names) if b in GM]
        if sel:
            P = P[sel]
            gm = np.array([GM[table.names[j]] for j in sel])           # GM may be edited (tests)
            d = P[None, :, :] - r[:, None, :]                          # (N, B, 3)
            dn3 = np.einsum("nbk,nbk->nb", d, d) ** 1.5
            direct = np.einsum("nbk,nb->nk", d, gm[None, :] / dn3)
            indirect = (gm[:, None] * P / (np.einsum("bk,bk->b", P, P) ** 1.5)[:, None]).sum(axis=0)
            acc += direct - indirect[None, :]
    else:
        for body, gm in GM.items():
            if body not in table.splines:
                continue
            rp = table.helio(body, t)
            d = rp[None, :] - r
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            acc += gm * (d / dn ** 3 - rp / np.linalg.norm(rp) ** 3)
    if gr:
        c2 = C_AU_DAY ** 2
        v2 = np.sum(v * v, axis=1, keepdims=True)
        rv = np.sum(r * v, axis=1, keepdims=True)
        acc += K2 / (c2 * rn ** 3) * ((4 * K2 / rn - v2) * r + 4 * rv * v)
    return acc


def propagate(r0: np.ndarray, v0: np.ndarray, t0: float, targets, eph: Ephemeris,
              table: BodyTable | None = None, rtol: float = 1e-12, atol: float = 1e-14,
              gr: bool = True, max_nfev: int | None = 30000) -> tuple[np.ndarray, np.ndarray]:
    """Integrate states (N,3) from ``t0`` to every time in ``targets``.

    ``max_nfev`` caps the right-hand-side evaluations: a (usually badly
    determined) orbit passing through a planet or the Moon makes the step size
    collapse and would otherwise stall for minutes; the caller then splits the
    batch or falls back to two-body.

    Returns positions and velocities with shape (len(targets), N, 3).
    """
    targets = np.atleast_1d(np.asarray(targets, float))
    n = r0.shape[0]
    if table is None:
        table = BodyTable.build(eph, min(t0, targets.min()), max(t0, targets.max()))
    out_r = np.empty((len(targets), n, 3))
    out_v = np.empty((len(targets), n, 3))

    nfev = [0]

    def rhs(t, y):
        nfev[0] += 1
        if max_nfev is not None and nfev[0] > max_nfev:
            raise RuntimeError(f"příliš mnoho kroků (> {max_nfev}), těsné přiblížení k tělesu")
        s = y.reshape(n, 6)
        r, v = s[:, :3], s[:, 3:]
        a = acceleration(t, r, v, table, gr)
        return np.concatenate([v, a], axis=1).ravel()

    y0 = np.concatenate([r0, v0], axis=1).ravel()
    for direction in (1, -1):
        sel = np.nonzero((targets - t0) * direction > 0)[0]
        if sel.size == 0:
            continue
        order = sel[np.argsort((targets[sel] - t0) * direction)]
        tev = targets[order]
        sol = solve_ivp(rhs, (t0, tev[-1]), y0, method="DOP853", t_eval=tev, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError("N-body integrace selhala: " + sol.message)
        Y = sol.y.T.reshape(len(tev), n, 6)
        out_r[order], out_v[order] = Y[:, :, :3], Y[:, :, 3:]
    same = np.nonzero(targets == t0)[0]
    out_r[same], out_v[same] = r0, v0
    return out_r, out_v


# --------------------------------------------------------------------------
# Observation geometry
# --------------------------------------------------------------------------

def astrometric(r_helio, v_helio, t_obs: float, obs_bary: np.ndarray, table: BodyTable, iters: int = 4):
    """Topocentric astrometric direction (ICRF) with light-time correction.

    ``r_helio``/``v_helio`` (N,3) at ``t_obs``. The object is moved back by the
    light time τ with a second-order Taylor step (exact to < 1 m for τ < 1 h).
    Returns (vector (N,3) object-minus-observer at retarded time, distance, r_helio_retarded).
    """
    rn = np.linalg.norm(r_helio, axis=1, keepdims=True)
    a = -K2 * r_helio / rn ** 3
    sun = table.bary("sun", t_obs)
    v_sun = (table.bary("sun", t_obs + 0.01) - table.bary("sun", t_obs - 0.01)) / 0.02
    tau = np.linalg.norm(r_helio + sun - obs_bary, axis=1, keepdims=True) / C_AU_DAY
    for _ in range(iters):
        r_ret = r_helio - v_helio * tau + 0.5 * a * tau ** 2
        sun_ret = sun - v_sun * tau            # Sun moves ~13 m/s – linear is exact enough
        vec = r_ret + sun_ret - obs_bary
        tau = np.linalg.norm(vec, axis=1, keepdims=True) / C_AU_DAY
    return vec, tau[:, 0] * C_AU_DAY, r_ret
