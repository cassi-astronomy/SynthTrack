"""Variable stars around a position: AAVSO VSX online, VizieR B/vsx mirrors, ASTAP offline.

Used by the GUI's survey (DSS) view: crosses with links to the VSX star page. The order of
sources is VSX itself (current, has the OID for a direct link) → VizieR B/vsx on several
mirrors (the same catalogue, updated weekly; CDS is sometimes down) → the local ASTAP
``variable_stars_13.csv`` (VSX to ~18 mag, no OID, link = search by name)."""
from __future__ import annotations

import math
import re
import urllib.parse

VSX_API = "https://vsx.aavso.org/index.php"
VIZIER_MIRRORS = (
    ("VizieR CDS", "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"),
    ("VizieR CfA", "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"),
    ("VizieR IUCAA", "https://vizier.iucaa.in/viz-bin/asu-tsv"),
    ("VizieR NAOJ", "https://vizier.nao.ac.jp/viz-bin/asu-tsv"),
    ("VizieR SAAO", "http://vizieridia.saao.ac.za/viz-bin/asu-tsv"),
)


def detail_url(oid=None, name: str | None = None) -> str:
    if oid:
        return f"{VSX_API}?view=detail.top&oid={int(oid)}"
    return f"{VSX_API}?view=results.get&ident=" + urllib.parse.quote(name or "")


def _f(v):
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _star(name, ra, dec, vtype=None, vmax=None, vmin=None, period=None, oid=None) -> dict | None:
    if name is None or ra is None or dec is None:
        return None
    rng = " – ".join(str(v).strip() for v in (vmax, vmin) if v not in (None, ""))
    return {"name": str(name).strip(), "ra": float(ra), "dec": float(dec), "type": (vtype or "").strip() or None,
            "range": rng, "period": (str(period).strip() or None) if period not in (None, "") else None,
            "max_mag": _f(vmax), "oid": int(oid) if oid not in (None, "") and str(oid).strip().isdigit() else None,
            "url": detail_url(oid if oid not in (None, "") else None, str(name).strip())}


def parse_vsx_json(obj) -> list[dict]:
    """``view=api.list&format=json``: {"VSXObjects": {"VSXObject": [ … ] | { … }}} (or [] when empty)."""
    if not isinstance(obj, dict):
        return []
    lst = (obj.get("VSXObjects") or {})
    lst = lst.get("VSXObject", []) if isinstance(lst, dict) else []
    if isinstance(lst, dict):
        lst = [lst]
    out = []
    for o in lst:
        s = _star(o.get("Name"), _f(o.get("RA2000")), _f(o.get("Declination2000")), o.get("VariabilityType"),
                  o.get("MaxMag"), o.get("MinMag"), o.get("Period"), o.get("OID"))
        if s:
            out.append(s)
    return out


def parse_vizier_vsx(text: str) -> list[dict]:
    """VizieR asu-tsv of B/vsx/vsx with -out.all and _RAJ/_DEJ (degrees)."""
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not rows:
        return []
    hdr = [h.strip() for h in rows[0].split("\t")]

    def col(*names):
        for nm in names:
            if nm in hdr:
                return hdr.index(nm)
        return None
    ix = {k: col(*v) for k, v in {"ra": ("_RAJ", "_RAJ2000", "RAJ2000"), "de": ("_DEJ", "_DEJ2000", "DEJ2000"),
                                  "oid": ("OID",), "name": ("Name",), "type": ("Type",), "max": ("max",),
                                  "nmax": ("n_max",), "min": ("min",), "nmin": ("n_min",), "fmin": ("f_min",),
                                  "per": ("Period",)}.items()}
    if ix["ra"] is None or ix["de"] is None or ix["name"] is None:
        raise ValueError("VizieR B/vsx: chybí sloupce souřadnic / jména")

    def g(f, k):
        i = ix[k]
        return f[i].strip() if i is not None and i < len(f) else ""
    out = []
    for ln in rows[1:]:
        f = ln.split("\t")
        ra, de = _f(g(f, "ra")), _f(g(f, "de"))
        if ra is None or de is None:
            continue                           # units / dashes lines
        vmax = (g(f, "max") + " " + g(f, "nmax")).strip()
        vmin = g(f, "min")
        if vmin:
            vmin = (f"({vmin})" if g(f, "fmin") == "(" else vmin) + (" " + g(f, "nmin") if g(f, "nmin") else "")
        s = _star(g(f, "name"), ra, de, g(f, "type"), vmax, vmin, g(f, "per"), g(f, "oid"))
        if s:
            out.append(s)
    return out


def query(ra: float, dec: float, radius_deg: float, tomag: float = 20.0, session=None,
          timeout: float = 8.0) -> tuple[list[dict], str]:
    """(stars, source name) from VSX, else the VizieR mirrors; RuntimeError listing what failed."""
    import requests
    s = session or requests
    tried = []
    try:
        r = s.get(VSX_API, params={"view": "api.list", "ra": f"{ra:.6f}", "dec": f"{dec:+.6f}",
                                   "radius": f"{radius_deg:.5f}", "tomag": f"{tomag:g}", "format": "json"},
                  timeout=(3.0, timeout))
        r.raise_for_status()
        return parse_vsx_json(r.json() if r.text.strip() not in ("", "[]") else {}), "AAVSO VSX"
    except Exception as e:             # noqa: BLE001 – try the next source
        tried.append(f"AAVSO VSX: {type(e).__name__}")
    for name, url in VIZIER_MIRRORS:
        try:
            r = s.get(url, params={"-source": "B/vsx/vsx", "-c": f"{ra:.6f} {dec:+.6f}", "-c.rd": f"{radius_deg:.5f}",
                                   "-out.all": "", "-out.add": "_RAJ,_DEJ", "-out.max": "2000"},
                      timeout=(3.0, timeout))
            r.raise_for_status()
            if "VizieR" in r.text[:2000] and "<html" in r.text[:500].lower():
                raise ValueError("HTML místo tabulky")
            return parse_vizier_vsx(r.text), name
        except Exception as e:         # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("; ".join(tried))


def from_astap(var: dict, ra0: float, de0: float, half: float, limit: int = 2000) -> list[dict]:
    """Stars of ``twilight.load_variables`` (ASTAP labels "NAME range_Period_P") in a box ±half°."""
    import numpy as np
    if not var:
        return []
    cd = max(math.cos(math.radians(de0)), 1e-3)
    sel = np.nonzero(np.abs(var["dec"] - de0) <= half)[0]
    dE = ((var["ra"][sel] - ra0 + 540.0) % 360.0 - 180.0) * cd
    sel = sel[np.abs(dE) <= half]
    out = []
    for i in sel[:limit]:
        lab = str(var["label"][i])
        name, _, rest = lab.partition(" ")
        name = name.rstrip("#").replace("_", " ")
        rng, _, per = rest.rstrip("#").partition("_Period_")
        m = re.match(r"\s*(-?[\d.]+)", rng)
        out.append({"name": name, "ra": float(var["ra"][i]), "dec": float(var["dec"][i]), "type": None,
                    "range": re.sub(r"(?<=[^\s-])-", " – ", rng, count=1) if rng else "", "period": per or None,
                    "max_mag": float(m.group(1)) if m else None, "oid": None, "url": detail_url(None, name)})
    return out


def offsets(stars: list[dict], ra0: float, de0: float) -> list[dict]:
    """Adds east/north offsets (deg) from the centre."""
    cd = max(math.cos(math.radians(de0)), 1e-3)
    for s in stars:
        s["de"] = ((s["ra"] - ra0 + 540.0) % 360.0 - 180.0) * cd
        s["dn"] = s["dec"] - de0
    return stars
