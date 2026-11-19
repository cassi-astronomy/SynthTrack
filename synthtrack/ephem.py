"""Planetary ephemerides: Sun, planets, Earth and Moon (barycentric ICRF, AU, TDB).

Sources, in order of preference (``Settings.ephemeris = "auto"``):

1. JPL SPK kernel (``*.bsp``, e.g. ``de440s.bsp`` from NAIF, 32 MB) placed in
   the catalogue folder, read by ``jplephem`` – reference quality, 1849–2150.
2. A legacy JPL DE4xx folder (``jpl-*.npy`` Chebyshev arrays) in that folder.
3. The packaged JPL DE421 excerpt 2020–2035 (1.2 MB, ships with SynthTrack) –
   reference quality; outside that span it falls back to
4. Astropy built-in (ERFA epv00 for Earth/Sun – a few km; plan94 for the
   planets – up to ~1e5 km, which only matters slightly as a perturber).

All sources return barycentric positions in AU on the ICRF axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

AU_KM = 149597870.700
C_AU_DAY = 173.1446326846693
K2 = 0.01720209895 ** 2          # Gaussian GM_sun used by the MPC elements (AU^3/d^2)

# GM in AU^3/day^2 (DE440). Earth and Moon split with EMRAT.
_GMB = 8.9970116036316091e-10
_EMRAT = 81.3005682214972154
GM = {
    "mercury": 4.9125001948893182e-11,
    "venus": 7.2434523326441187e-10,
    "earth": _GMB * _EMRAT / (1 + _EMRAT),
    "moon": _GMB / (1 + _EMRAT),
    "mars": 9.5495488297258119e-11,
    "jupiter": 2.8253458252257917e-07,
    "saturn": 8.4597059933762903e-08,
    "uranus": 1.2920265649682399e-08,
    "neptune": 1.5243573478851939e-08,
    "pluto": 2.1750964648933581e-12,
}
PERTURBERS = tuple(GM.keys())
BODIES = ("sun",) + PERTURBERS


class Ephemeris:
    name = "abstract"
    reference_quality = False
    bodies = BODIES

    def barycentric(self, body: str, jd_tdb) -> np.ndarray:
        """Position of ``body`` relative to the solar-system barycentre, shape (3, T) in AU."""
        raise NotImplementedError

    def heliocentric(self, body: str, jd_tdb) -> np.ndarray:
        return self.barycentric(body, jd_tdb) - self.barycentric("sun", jd_tdb)


# --------------------------------------------------------------------------
class SpkEphemeris(Ephemeris):
    """JPL SPK kernel via jplephem (DE430/DE440/DE441, de440s.bsp …)."""
    reference_quality = True
    _SEG = {"mercury": 1, "venus": 2, "mars": 4, "jupiter": 5, "saturn": 6, "uranus": 7,
            "neptune": 8, "pluto": 9, "sun": 10}

    def __init__(self, path: str | Path):
        from jplephem.spk import SPK

        self.path = str(path)
        self.kernel = SPK.open(self.path)
        self.name = f"JPL {Path(path).name}"
        self._pairs = {(s.center, s.target): s for s in self.kernel.segments}

    def _seg(self, c, t, jd):
        return np.asarray(self._pairs[(c, t)].compute(jd), float).reshape(3, -1)

    def barycentric(self, body, jd_tdb):
        jd = np.atleast_1d(np.asarray(jd_tdb, float))
        if body in ("earth", "moon"):
            emb = self._seg(0, 3, jd)
            tgt = 399 if body == "earth" else 301
            km = emb + self._seg(3, tgt, jd)
        else:
            km = self._seg(0, self._SEG[body], jd)
        return km / AU_KM


# --------------------------------------------------------------------------
class LegacyDeEphemeris(Ephemeris):
    """DE421 in the legacy jplephem ``.npy`` layout (Chebyshev coefficients per body)."""
    reference_quality = True
    _FILE = {"mercury": "mercury", "venus": "venus", "mars": "mars", "jupiter": "jupiter",
             "saturn": "saturn", "uranus": "uranus", "neptune": "neptune", "pluto": "pluto",
             "sun": "sun", "emb": "earthmoon", "moon_geo": "moon"}

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        c = dict((n.decode() if isinstance(n, bytes) else n, v)
                 for n, v in np.load(self.dir / "constants.npy", allow_pickle=True))
        self.jalpha, self.jomega = float(c["jalpha"]), float(c["jomega"])
        self.emrat = float(c["EMRAT"])
        self.au_km = float(c["AU"])
        self.name = f"JPL DE{int(c['DENUM'])}"
        self._coef: dict[str, np.ndarray] = {}

    def span_text(self) -> str:
        from astropy.time import Time
        return f"{Time(self.jalpha, format='jd').datetime.year}–{Time(self.jomega, format='jd').datetime.year}"

    def _cheb(self, key, jd):
        if key not in self._coef:
            self._coef[key] = np.load(self.dir / f"jpl-{self._FILE[key]}.npy")
        cs = self._coef[key]
        nsets, _, ncoef = cs.shape
        span = (self.jomega - self.jalpha) / nsets
        if np.any(jd < self.jalpha) or np.any(jd > self.jomega):
            raise ValueError("Čas mimo rozsah efemeridy")
        idx, off = np.divmod(jd - self.jalpha, span)
        idx = np.minimum(idx.astype(int), nsets - 1)
        x = 2.0 * off / span - 1.0
        T = np.empty((ncoef, len(jd)))
        T[0] = 1.0
        T[1] = x
        for i in range(2, ncoef):
            T[i] = 2 * x * T[i - 1] - T[i - 2]
        c = cs[idx]                              # (T, 3, ncoef)
        return np.einsum("tak,kt->at", c, T)     # km

    def barycentric(self, body, jd_tdb):
        jd = np.atleast_1d(np.asarray(jd_tdb, float))
        if body in ("earth", "moon"):
            emb = self._cheb("emb", jd)
            moon = self._cheb("moon_geo", jd)
            if body == "earth":
                km = emb - moon / (1.0 + self.emrat)
            else:
                km = emb + moon * self.emrat / (1.0 + self.emrat)
        else:
            km = self._cheb(body, jd)
        return km / self.au_km


# --------------------------------------------------------------------------
class BuiltinEphemeris(Ephemeris):
    """Astropy / ERFA analytic ephemeris – needs no files (no Pluto; its effect is negligible)."""
    name = "astropy builtin (ERFA)"
    bodies = tuple(b for b in BODIES if b != "pluto")

    def barycentric(self, body, jd_tdb):
        from astropy import units as u
        from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
        from astropy.time import Time

        jd = np.atleast_1d(np.asarray(jd_tdb, float))
        t = Time(jd, format="jd", scale="tdb")
        with solar_system_ephemeris.set("builtin"):
            p = get_body_barycentric(body, t)
        return p.get_xyz().to_value(u.au).reshape(3, -1)


class FallbackEphemeris(Ephemeris):
    """``primary`` (JPL) inside its time span, ``secondary`` (ERFA) outside – element by element.

    Pluto is not offered: ERFA does not have it, and a body missing in part of
    a time table must never be replaced by a dummy position (a mass at the
    barycentre would create a large spurious indirect acceleration)."""

    def __init__(self, primary: "LegacyDeEphemeris", secondary: Ephemeris):
        self.primary, self.secondary = primary, secondary
        self.name = f"{primary.name} {primary.span_text()} (mimo: {secondary.name})"
        self.reference_quality = True
        self.bodies = tuple(b for b in primary.bodies if b in secondary.bodies)

    def barycentric(self, body, jd_tdb):
        jd = np.atleast_1d(np.asarray(jd_tdb, float))
        inside = (jd >= self.primary.jalpha) & (jd <= self.primary.jomega)
        if inside.all():
            return self.primary.barycentric(body, jd)
        out = np.empty((3, len(jd)))
        if inside.any():
            out[:, inside] = self.primary.barycentric(body, jd[inside])
        out[:, ~inside] = self.secondary.barycentric(body, jd[~inside])
        return out


PACKAGED_DE421 = Path(__file__).parent / "data" / "de421_2020_2035"

_cache: dict[str, Ephemeris] = {}


def load_ephemeris(spec: str = "auto", search_dirs: list[Path] | None = None) -> Ephemeris:
    """``spec``: "auto", "builtin", path to a ``.bsp`` file or to a legacy ``de4xx`` folder."""
    key = f"{spec}|{search_dirs}"
    if key in _cache:
        return _cache[key]
    eph: Ephemeris | None = None
    if spec not in ("auto", "builtin", ""):
        p = Path(spec)
        eph = SpkEphemeris(p) if p.suffix.lower() == ".bsp" else LegacyDeEphemeris(p)
    elif spec in ("auto", ""):
        for d in search_dirs or []:
            d = Path(d)
            for name in ("de440s.bsp", "de440.bsp", "de441.bsp", "de430.bsp", "de421.bsp"):
                if (d / name).exists():
                    try:
                        eph = SpkEphemeris(d / name)
                        break
                    except Exception:
                        pass
            if eph is None:
                for sub in sorted(d.glob("de4*")):
                    if (sub / "constants.npy").exists():
                        eph = LegacyDeEphemeris(sub)
                        break
            if eph is not None:
                break
    if eph is None and spec != "builtin" and (PACKAGED_DE421 / "constants.npy").exists():
        eph = FallbackEphemeris(LegacyDeEphemeris(PACKAGED_DE421), BuiltinEphemeris())
    if eph is None:
        eph = BuiltinEphemeris()
    _cache[key] = eph
    return eph


# --------------------------------------------------------------------------
@dataclass
class BodyTable:
    """Positions of Sun and perturbers on a time grid with cubic-spline interpolation.

    Used by the N-body integrator so that the ephemeris is evaluated only a
    few hundred times per propagation instead of at every integrator stage.
    All bodies share one spline, so one evaluation per integrator stage gives
    every heliocentric position at once (``helio_all``). Spline error with
    0.25-day spacing is < 100 m for the Moon and Mercury.
    """
    t0: float
    t1: float
    splines: dict
    names: tuple = ()
    combined: object = None          # CubicSpline over (B, 3) heliocentric positions

    @classmethod
    def build(cls, eph: Ephemeris, t0: float, t1: float, step: float = 0.25,
              bodies=None) -> "BodyTable":
        from scipy.interpolate import CubicSpline

        lo, hi = min(t0, t1) - 2.0, max(t0, t1) + 2.0
        if hi - lo > 400:          # long propagations (old element epochs): 0.5 d is still < 1 km
            step = max(step, 0.5)
        n = max(8, int(np.ceil((hi - lo) / step)) + 1)
        grid = np.linspace(lo, hi, n)
        sp, pos = {}, {}
        for b in bodies or eph.bodies:
            pos[b] = eph.barycentric(b, grid)
            sp[b] = CubicSpline(grid, pos[b], axis=1)
        names = tuple(b for b in pos if b != "sun")
        combined = None
        if names and "sun" in pos:
            helio = np.stack([pos[b] - pos["sun"] for b in names], axis=0)     # (B, 3, T)
            combined = CubicSpline(grid, helio, axis=2)
        return cls(lo, hi, sp, names, combined)

    def helio_all(self, t) -> np.ndarray:
        """Heliocentric positions (B, 3) of all perturbers in ``names`` order."""
        return self.combined(t)

    def bary(self, body: str, t) -> np.ndarray:
        return self.splines[body](t)

    def helio(self, body: str, t) -> np.ndarray:
        return self.splines[body](t) - self.splines["sun"](t)
