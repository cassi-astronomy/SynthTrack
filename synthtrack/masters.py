"""Master calibration library and automatic selection of dark / flat / bias.

Each instrument owns a folder (recursively scanned) with master frames.
Type is taken from IMAGETYP, falling back to the file name.
Selection rules
  dark: same binning, same gain/offset (when both known), temperature within
        tolerance (when both known), exposure within tolerance. Among matches the
        closest temperature, then the closest date wins. If nothing matches and
        ``allow_dark_scaling`` is on and a bias exists, the nearest longer
        (else nearest) dark is scaled: bias + (dark - bias) * t_light / t_dark.
  flat: same canonical filter and binning; closest date wins.
  bias: same binning, gain/offset; closest date.
Every decision is returned with a human readable reason and warnings.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from astropy.io import fits

from .fitsmeta import FrameMeta, is_fits, read_meta
from .instruments import Instrument

TYPE_WORDS = {
    "dark": ("dark",),
    "flat": ("flat",),
    "bias": ("bias", "offset", "zero"),
}


@dataclass
class MasterEntry:
    type: str          # dark | flat | bias | unknown
    meta: FrameMeta
    mtime: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Selection:
    kind: str
    path: str | None
    reason: str
    warnings: list[str] = field(default_factory=list)
    scale: float | None = None          # dark scaling factor
    bias_path: str | None = None        # bias used for scaling

    @property
    def ok(self) -> bool:
        return self.path is not None

    def to_dict(self) -> dict:
        return asdict(self)


def classify(meta: FrameMeta) -> str:
    t = meta.imagetyp.lower()
    for kind, words in TYPE_WORDS.items():
        if any(w in t for w in words):
            return kind
    name = Path(meta.path).name.lower()
    for kind, words in TYPE_WORDS.items():
        if any(w in name for w in words):
            return kind
    return "unknown"


class MasterLibrary:
    """Index of master frames for one instrument (cached by mtime)."""

    def __init__(self, inst: Instrument, cache_file: Path | None = None):
        self.inst = inst
        self.cache_file = cache_file
        self.entries: list[MasterEntry] = []

    def scan(self) -> list[MasterEntry]:
        root = Path(self.inst.masters_dir) if self.inst.masters_dir else None
        cache = self._load_cache()
        entries: list[MasterEntry] = []
        if root and root.is_dir():
            for p in sorted(root.rglob("*")):
                if not is_fits(p):
                    continue
                mt = p.stat().st_mtime
                c = cache.get(str(p))
                if c and abs(c["mtime"] - mt) < 1e-6:
                    entries.append(MasterEntry(c["type"], FrameMeta(**c["meta"]), mt))
                    continue
                try:
                    meta = read_meta(p, self.inst, fits.getheader(p))
                except Exception:
                    continue
                entries.append(MasterEntry(classify(meta), meta, mt))
        self.entries = entries
        self._save_cache()
        return entries

    # ---- cache -------------------------------------------------------
    def _load_cache(self) -> dict:
        if not self.cache_file or not self.cache_file.exists():
            return {}
        try:
            return {e["meta"]["path"]: e for e in json.loads(self.cache_file.read_text())}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps([e.to_dict() for e in self.entries]))

    # ---- selection ---------------------------------------------------
    def of_type(self, kind: str) -> list[MasterEntry]:
        return [e for e in self.entries if e.type == kind]

    def select_all(self, light: FrameMeta) -> dict[str, Selection]:
        return {
            "dark": self.select_dark(light),
            "flat": self.select_flat(light),
        }

    def select_bias(self, light: FrameMeta) -> Selection:
        cands = [e for e in self.of_type("bias")
                 if _same_geometry(e.meta, light) and _same_gain(e.meta, light)]
        if not cands:
            return Selection("bias", None, "Žádný master bias se shodným rozměrem, binningem a gainem.")
        best = min(cands, key=lambda e: _date_dist(e.meta, light))
        return Selection("bias", best.meta.path, "Shodný binning a gain; nejbližší datum.")

    def select_dark(self, light: FrameMeta) -> Selection:
        inst = self.inst
        base = [e for e in self.of_type("dark") if _same_geometry(e.meta, light)]
        if not base:
            return Selection("dark", None, f"Žádný master dark s rozměrem {light.width}×{light.height} "
                             f"a binningem {light.xbin}x{light.ybin}.")
        warnings: list[str] = []
        g = [e for e in base if _same_gain(e.meta, light)]
        if not g:
            return Selection("dark", None, f"Žádný master dark se shodným gain/offset ({light.gain}/{light.offset}).")
        temp_ok = [e for e in g if _temp_ok(e.meta, light, inst.dark_temp_tolerance_c)]
        if not temp_ok:
            warnings.append(f"Žádný dark v toleranci teploty ±{inst.dark_temp_tolerance_c} °C – použita nejbližší teplota.")
            temp_ok = g
        if light.exptime is None:
            return Selection("dark", None, "Snímek nemá v hlavičce délku expozice.")
        exact = [e for e in temp_ok if e.meta.exptime is not None
                 and abs(e.meta.exptime - light.exptime) <= inst.dark_exptime_tolerance_s]
        if exact:
            best = min(exact, key=lambda e: (_temp_dist(e.meta, light), _date_dist(e.meta, light)))
            return Selection("dark", best.meta.path,
                             f"Expozice {best.meta.exptime:g} s = {light.exptime:g} s, "
                             f"teplota {_fmt(best.meta.temperature)} °C (snímek {_fmt(light.temperature)} °C), "
                             f"binning {light.xbin}x{light.ybin}.", warnings)
        # --- scaling fallback ------------------------------------------
        if not inst.allow_dark_scaling:
            avail = sorted({e.meta.exptime for e in temp_ok if e.meta.exptime})
            return Selection("dark", None,
                             f"Chybí dark {light.exptime:g} s (k dispozici: {', '.join(f'{a:g}' for a in avail) or '–'} s). "
                             "Škálování darku je u přístroje vypnuté.", warnings)
        bias = self.select_bias(light)
        if not bias.ok:
            return Selection("dark", None, f"Chybí dark {light.exptime:g} s a pro škálování chybí master bias.", warnings)
        with_t = [e for e in temp_ok if e.meta.exptime]
        longer = [e for e in with_t if e.meta.exptime >= light.exptime]
        pool = longer or with_t
        best = min(pool, key=lambda e: (abs(e.meta.exptime - light.exptime), _temp_dist(e.meta, light)))
        scale = light.exptime / best.meta.exptime
        warnings.append("Dark je škálovaný – u CMOS s amp glow může zůstat gradient.")
        return Selection("dark", best.meta.path,
                         f"Škálovaný dark {best.meta.exptime:g} s × {scale:.3f} (přes bias).",
                         warnings, scale=scale, bias_path=bias.path)

    def select_flat(self, light: FrameMeta) -> Selection:
        cands = [e for e in self.of_type("flat") if _same_geometry(e.meta, light)]
        same_f = [e for e in cands if e.meta.filter == light.filter]
        if not same_f:
            have = sorted({e.meta.filter for e in cands})
            msg = (f"Žádný master flat pro filtr '{light.filter}' (z '{light.raw_filter}') "
                   f"a binning {light.xbin}x{light.ybin}. K dispozici: {', '.join(have) or '–'}.")
            return Selection("flat", None, msg)
        best = min(same_f, key=lambda e: _date_dist(e.meta, light))
        days = _date_dist(best.meta, light)
        warnings = []
        if days != float("inf") and days > 30:
            warnings.append(f"Flat je {days:.0f} dní od data snímků – prach se mohl změnit.")
        return Selection("flat", best.meta.path,
                         f"Filtr {light.filter}, binning {light.xbin}x{light.ybin}, "
                         f"rozdíl data {'?' if days == float('inf') else f'{days:.1f} d'}.", warnings)


def _same_bin(a: FrameMeta, b: FrameMeta) -> bool:
    return a.xbin == b.xbin and a.ybin == b.ybin


def _same_geometry(a: FrameMeta, b: FrameMeta) -> bool:
    """Masters must cover the same raw detector area; zero means unknown."""
    if not _same_bin(a, b):
        return False
    return ((not a.width or not b.width or a.width == b.width)
            and (not a.height or not b.height or a.height == b.height))


def _same_gain(a: FrameMeta, b: FrameMeta) -> bool:
    for x, y in ((a.gain, b.gain), (a.offset, b.offset)):
        if x is not None and y is not None and abs(x - y) > 1e-6:
            return False
    return True


def _temp_ok(a: FrameMeta, b: FrameMeta, tol: float) -> bool:
    if a.temperature is None or b.temperature is None:
        return True
    return abs(a.temperature - b.temperature) <= tol


def _temp_dist(a: FrameMeta, b: FrameMeta) -> float:
    if a.temperature is None or b.temperature is None:
        return 0.0
    return abs(a.temperature - b.temperature)


def _date_dist(a: FrameMeta, b: FrameMeta) -> float:
    if a.mjd_mid is None or b.mjd_mid is None:
        return float("inf")
    return abs(a.mjd_mid - b.mjd_mid)


def _fmt(v) -> str:
    return "?" if v is None else f"{v:g}"
