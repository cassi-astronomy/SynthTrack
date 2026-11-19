"""Local UCAC4 catalogue (offline) – star magnitudes for approximate photometry.

UCAC4 (Zacharias et al. 2013) as distributed: ``u4b/z001`` … ``z900`` (900 zones of
0.2° in declination from the south pole, records sorted by RA), 78-byte little-endian
binary records. Only the fields needed here are decoded: position (epoch J2000.0),
proper motion and the APASS B, V, g, r, i magnitudes (mmag, 20000 = none).
No index file is needed: within a zone the RA column is binary searched (memmap).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REC = np.dtype([
    ("ra", "<i4"), ("spd", "<i4"), ("magm", "<i2"), ("maga", "<i2"), ("sigmag", "i1"), ("objt", "i1"),
    ("cdf", "i1"), ("sigra", "i1"), ("sigdc", "i1"), ("na1", "i1"), ("nu1", "i1"), ("cu1", "i1"),
    ("cepra", "<i2"), ("cepdc", "<i2"), ("pmrac", "<i2"), ("pmdc", "<i2"), ("sigpmr", "i1"), ("sigpmd", "i1"),
    ("pts_key", "<i4"), ("j_m", "<i2"), ("h_m", "<i2"), ("k_m", "<i2"), ("icqflg", "i1", 3), ("e2mpho", "i1", 3),
    ("apasm", "<i2", 5), ("apase", "i1", 5), ("gcflg", "i1"), ("icf", "<i4"), ("leda", "i1"), ("x2cat", "i1"),
    ("rnm", "<i4"), ("zn2", "<i2"), ("rn2", "<i4"),
])
assert REC.itemsize == 78
BANDS = {"B": 0, "V": 1, "g": 2, "r": 3, "i": 4}
MAS = 3600.0 * 1000.0


def find_root(path: str | Path | None) -> Path | None:
    """The ``u4b`` folder (given it or its parent), or None when it has no zone files."""
    if not path:
        return None
    p = Path(path)
    for cand in (p, p / "u4b", p / "U4B"):
        if cand.is_dir() and any(f.is_file() and len(f.name) == 4 and f.name[0] in "zZ" and f.name[1:].isdigit()
                                 for f in cand.iterdir()):
            return cand
    return None


def _zone_file(root: Path, z: int) -> Path | None:
    for name in (f"z{z:03d}", f"Z{z:03d}"):
        f = root / name
        if f.is_file():
            return f
    return None


def band_for_filter(filt: str | None) -> str:
    """APASS band closest to the frames' filter (clear / unfiltered CMOS ≈ r)."""
    f = (filt or "").strip()
    if f in ("B", "b") or f.upper().startswith("B"):
        return "B"
    if f.upper() in ("V", "G") or f == "Green":
        return "V"
    if f == "g" or f.lower() == "sg":
        return "g"
    if f.upper() in ("I", "IC", "SI") or f == "i":
        return "i"
    return "r"


def query(root: Path, ra_min: float, ra_max: float, dec_min: float, dec_max: float,
          band: str = "r", epoch_jyear: float = 2000.0, mag_max: float = 17.0) -> dict:
    """Stars in a box (degrees; ``ra_min`` > ``ra_max`` wraps through 0h) with an APASS magnitude.

    Returns arrays ra, dec (at ``epoch_jyear``), mag (``band``) and mag_all (UCAC4 model mag
    for the isolation test – also stars without APASS)."""
    bi = BANDS[band]
    z0 = max(1, int(math.floor((dec_min + 90.0) / 0.2)) + 1)
    z1 = min(900, int(math.floor((dec_max + 90.0) / 0.2)) + 1)
    if ra_min <= ra_max:
        ranges = [(ra_min, ra_max)]
    else:
        ranges = [(ra_min, 360.0), (0.0, ra_max)]
    out = {"ra": [], "dec": [], "mag": [], "ra_all": [], "dec_all": [], "mag_all": [],
           "j_all": [], "leda_all": []}
    dt = epoch_jyear - 2000.0
    # ra_all/dec_all/mag_all/j_all/leda_all: every star; ra/dec/mag: with an APASS magnitude
    for z in range(z0, z1 + 1):
        f = _zone_file(root, z)
        if f is None or f.stat().st_size < REC.itemsize:
            continue
        if f.stat().st_size % REC.itemsize:
            raise ValueError(f"{f.name} nemá formát UCAC4 (velikost není násobkem 78 B)")
        mm = np.memmap(f, dtype=REC, mode="r")
        ras = mm["ra"]
        for a, b in ranges:
            i0 = int(np.searchsorted(ras, int(a * MAS), "left"))
            i1 = int(np.searchsorted(ras, int(b * MAS), "right"))
            if i1 <= i0:
                continue
            r = np.array(mm[i0:i1])
            dec = r["spd"] / MAS - 90.0
            ok = (dec >= dec_min) & (dec <= dec_max)
            r, dec = r[ok], dec[ok]
            if not len(r):
                continue
            cosd = np.maximum(np.cos(np.radians(dec)), 1e-6)
            ra = r["ra"] / MAS + (r["pmrac"] * 0.1 * dt / MAS) / cosd
            dec = dec + r["pmdc"] * 0.1 * dt / MAS
            mall = np.where(r["magm"] < 20000, r["magm"], r["maga"]) / 1000.0
            out["ra_all"].append(ra % 360.0)
            out["dec_all"].append(dec)
            out["mag_all"].append(mall)
            jm = r["j_m"].astype(float)
            # 2MASS J; 20.000 is UCAC4's "no 2MASS magnitude" (10 % of stars in a Sgr field 15. 11.:
            # a 12.4 mag star got colour UCAC4 − J = −7.6 and "expected 14.1 mag" → false brightening)
            out["j_all"].append(np.where((jm > 0) & (jm < 19990), jm / 1000.0, np.nan))
            out["leda_all"].append(r["leda"].astype(np.int8))                              # LEDA galaxy match
            m = r["apasm"][:, bi] / 1000.0
            good = (r["apasm"][:, bi] > 0) & (r["apasm"][:, bi] < 20000) & (m <= mag_max)
            out["ra"].append(ra[good] % 360.0)
            out["dec"].append(dec[good])
            out["mag"].append(m[good])
        del mm
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
