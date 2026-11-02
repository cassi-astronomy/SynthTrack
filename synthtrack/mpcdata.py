"""MPCORB.DAT / CometEls.txt – manual download, parsing and fast local cache.

The user presses "Update" in the GUI (only if online access for MPC is
enabled). The file is downloaded to a temporary name, verified, parsed into a
compact ``.npz`` and only then swapped in; the previous version is kept as
``*.prev.npz`` so a failed update never breaks identification.

Column layout follows the MPC documentation (1-based, inclusive).
"""
from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .jobs import _retry_io

MPCORB_NAME = "mpcorb"
COMETS_NAME = "comets"


# --------------------------------------------------------------------------
# Packed dates
# --------------------------------------------------------------------------
_CENT = {"I": 1800, "J": 1900, "K": 2000, "L": 2100}


def _packed_digit(c: str) -> int:
    if c.isdigit():
        return int(c)
    return ord(c) - ord("A") + 10


def unpack_epoch(s: str) -> tuple[int, int, int]:
    s = s.strip()
    year = _CENT[s[0]] + int(s[1:3])
    return year, _packed_digit(s[3]), _packed_digit(s[4])


def jd_from_ymd(y: int, m: int, d: float) -> float:
    """Julian Date (TT, 0h) from a Gregorian calendar date with fractional day."""
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _f(line: str, a: int, b: int) -> float:
    s = line[a - 1:b].strip()
    return float(s) if s else np.nan


def parse_mpcorb(lines) -> dict[str, np.ndarray]:
    """Parse MPCORB lines (header lines are skipped automatically)."""
    names, H, G, epoch, M, peri, node, inc, e, n, a, nobs, flags = ([] for _ in range(13))
    in_body = False
    for raw in lines:
        line = raw.rstrip("\n\r")
        if not in_body:
            if line.startswith("-----"):
                in_body = True
                continue
            # files without header (e.g. NEA.txt, unit tests) start directly with data
            if len(line) >= 103 and line[0:1].strip() and _looks_like_orbit(line):
                in_body = True
            else:
                continue
        if len(line) < 103 or not line.strip():
            continue
        try:
            y, mo, d = unpack_epoch(line[20:25])
            ep = jd_from_ymd(y, mo, d)
            vals = (_f(line, 27, 35), _f(line, 38, 46), _f(line, 49, 57), _f(line, 60, 68),
                    _f(line, 71, 79), _f(line, 81, 91), _f(line, 93, 103))
        except (ValueError, KeyError, IndexError):
            continue
        h = _f(line, 9, 13)
        g = _f(line, 15, 19)
        readable = line[166:194].strip() if len(line) > 166 else ""
        names.append(readable or line[0:7].strip())
        H.append(h)
        G.append(0.15 if np.isnan(g) else g)
        epoch.append(ep)
        M.append(vals[0]); peri.append(vals[1]); node.append(vals[2]); inc.append(vals[3])
        e.append(vals[4]); n.append(vals[5]); a.append(vals[6])
        try:
            nobs.append(int(line[117:122]))
        except ValueError:
            nobs.append(0)
        try:
            flags.append(int(line[161:165], 16))
        except ValueError:
            flags.append(0)
    return {
        "name": np.array(names, dtype=object),
        "H": np.array(H, np.float32), "G": np.array(G, np.float32),
        "epoch_jd": np.array(epoch, np.float64),
        "M": np.array(M, np.float64), "peri": np.array(peri, np.float64),
        "node": np.array(node, np.float64), "inc": np.array(inc, np.float64),
        "e": np.array(e, np.float64), "n": np.array(n, np.float64), "a": np.array(a, np.float64),
        "nobs": np.array(nobs, np.int32), "flags": np.array(flags, np.int32),
    }


def _looks_like_orbit(line: str) -> bool:
    try:
        unpack_epoch(line[20:25])
        float(line[92:103])
        return True
    except Exception:
        return False


def parse_comets(lines) -> dict[str, np.ndarray]:
    """Parse CometEls.txt (MPC one-line comet elements)."""
    names, Tp, q, e, peri, node, inc, M1, K1, ep = ([] for _ in range(10))
    for raw in lines:
        line = raw.rstrip("\n\r")
        if len(line) < 102:
            continue
        try:
            y = int(line[14:18]); mo = int(line[19:21]); d = float(line[22:29])
            vals = (_f(line, 31, 39), _f(line, 42, 49), _f(line, 52, 59), _f(line, 62, 69), _f(line, 72, 79))
        except ValueError:
            continue
        if any(np.isnan(v) for v in vals):
            continue
        names.append(line[102:158].strip())
        Tp.append(jd_from_ymd(y, mo, d))
        q.append(vals[0]); e.append(vals[1]); peri.append(vals[2]); node.append(vals[3]); inc.append(vals[4])
        M1.append(_f(line, 92, 95)); K1.append(_f(line, 97, 100))
        try:   # epoch (cols 82-89) only for perturbed solutions
            ep.append(jd_from_ymd(int(line[81:85]), int(line[85:87]), float(line[87:89])))
        except ValueError:
            ep.append(np.nan)
    return {
        "name": np.array(names, dtype=object), "Tp_jd": np.array(Tp), "q": np.array(q), "e": np.array(e),
        "peri": np.array(peri), "node": np.array(node), "inc": np.array(inc),
        "M1": np.array(M1, np.float32), "K1": np.array(K1, np.float32),
        "epoch_jd": np.array(ep, np.float64),
    }


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

@dataclass
class CatalogStatus:
    kind: str
    present: bool
    count: int = 0
    downloaded_at: float | None = None
    age_days: float | None = None
    source: str = ""

    def to_dict(self):
        return self.__dict__.copy()


class MpcStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}

    def npz(self, kind: str) -> Path:
        return self.dir / f"{kind}.npz"

    def meta_path(self, kind: str) -> Path:
        return self.dir / f"{kind}.json"

    def status(self, kind: str) -> CatalogStatus:
        p, m = self.npz(kind), self.meta_path(kind)
        if not p.exists():
            return CatalogStatus(kind, False)
        meta = json.loads(m.read_text()) if m.exists() else {}
        ts = meta.get("downloaded_at", p.stat().st_mtime)
        return CatalogStatus(kind, True, meta.get("count", 0), ts, (time.time() - ts) / 86400, meta.get("source", ""))

    def load(self, kind: str) -> dict[str, np.ndarray] | None:
        p = self.npz(kind)
        if not p.exists():
            return None
        key = f"{kind}:{p.stat().st_mtime}"
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(kind + ":")}
            with np.load(p, allow_pickle=True) as z:
                self._cache[key] = {k: z[k] for k in z.files}
        return self._cache[key]

    def save(self, kind: str, data: dict[str, np.ndarray], source: str) -> CatalogStatus:
        tmp = self.dir / f"{kind}.tmp.npz"
        np.savez(tmp, **data)
        if self.npz(kind).exists():
            shutil.copy2(self.npz(kind), self.dir / f"{kind}.prev.npz")
        _retry_io(lambda: tmp.replace(self.npz(kind)))
        self.meta_path(kind).write_text(json.dumps({"downloaded_at": time.time(), "count": int(len(data["name"])),
                                                    "source": source}))
        return self.status(kind)

    def import_file(self, kind: str, path: str | Path) -> CatalogStatus:
        """Import a locally available MPCORB.DAT(.gz) / CometEls.txt (fully offline)."""
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="latin-1", errors="replace") as fh:
            data = parse_mpcorb(fh) if kind == MPCORB_NAME else parse_comets(fh)
        if len(data["name"]) == 0:
            raise ValueError(f"Soubor {path} neobsahuje žádné platné elementy")
        return self.save(kind, data, str(path))

    def download(self, kind: str, url: str, progress=None, timeout: int = 60) -> CatalogStatus:
        import requests

        raw_path = self.dir / (Path(url).name + ".part")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            done = 0
            with open(raw_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        final = self.dir / Path(url).name
        raw_path.replace(final)
        if final.suffix == ".gz":   # integrity check: full decompression
            with gzip.open(final, "rb") as g:
                while g.read(1 << 22):
                    pass
        self.import_file(kind, final)
        meta = json.loads(self.meta_path(kind).read_text())
        meta["source"] = url
        self.meta_path(kind).write_text(json.dumps(meta))
        return self.status(kind)
