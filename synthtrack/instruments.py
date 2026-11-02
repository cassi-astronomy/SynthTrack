"""Instrument profiles (telescope/lens + camera).

Each instrument is one TOML file in ``<data_dir>/instruments/<id>.toml`` and
owns its own folder of master darks / flats / biases.
"""
from __future__ import annotations

import re
try:                      # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

# Default FITS keyword candidates – first one found in the header wins.
DEFAULT_KEYMAP: dict[str, list[str]] = {
    "exptime": ["EXPTIME", "EXPOSURE", "EXP_TIME"],
    "date_obs": ["DATE-OBS", "DATE_OBS", "DATE-BEG"],
    "filter": ["FILTER", "FILTNAME", "FILT"],
    "temperature": ["CCD-TEMP", "CCD_TEMP", "CCDTEMP", "SET-TEMP", "CCD_SET", "TEMPERAT", "DET-TEMP"],
    "gain": ["GAIN", "EGAIN_SET", "CAM-GAIN"],
    "offset": ["OFFSET", "BLKLEVEL", "PEDESTAL"],
    "xbinning": ["XBINNING", "BINX", "XBIN"],
    "ybinning": ["YBINNING", "BINY", "YBIN"],
    "imagetyp": ["IMAGETYP", "FRAMETYP", "FRAME", "OBSTYPE"],
    "ra": ["RA", "OBJCTRA", "RA_OBJ", "CRVAL1"],
    "dec": ["DEC", "OBJCTDEC", "DEC_OBJ", "CRVAL2"],
    "object": ["OBJECT"],
    "instrume": ["INSTRUME"],
    "telescop": ["TELESCOP"],
    "focallen": ["FOCALLEN", "FOCAL"],
    "site_lat": ["SITELAT", "LATITUDE", "OBSLAT", "LAT-OBS"],
    "site_lon": ["SITELONG", "LONGITUD", "OBSLONG", "LONG-OBS"],
    "site_alt": ["SITEELEV", "ALTITUDE", "OBSALT", "ALT-OBS"],
    "datasec": ["DATASEC", "TRIMSEC"],
}

# Filter aliases shared by all instruments (lower-case key -> canonical).
DEFAULT_FILTER_ALIASES: dict[str, str] = {
    "l": "L", "lum": "L", "luminance": "L", "clear": "L", "c": "L", "cl": "L", "none": "L",
    "r": "R", "red": "R", "g": "G", "green": "G", "b": "B", "blue": "B",
    "v": "V", "ha": "Ha", "h-alpha": "Ha", "halpha": "Ha",
    "sr": "r'", "r'": "r'", "sg": "g'", "g'": "g'", "si": "i'", "i'": "i'",
}


@dataclass
class Instrument:
    id: str
    name: str = ""
    focal_length_mm: float = 1000.0
    pixel_size_um: float = 3.76
    binning: int = 1                    # native binning the data is usually taken at
    width_px: int = 0                   # informational, filled from first frame
    height_px: int = 0
    saturation_adu: float = 60000.0
    gain_e_adu: float = 1.0
    read_noise_e: float = 3.0
    masters_dir: str = ""               # folder with master darks/flats/biases
    software_binning: int = 1           # extra binning before search (oversampled optics)
    # Master selection
    dark_exptime_tolerance_s: float = 0.5
    dark_temp_tolerance_c: float = 2.0
    allow_dark_scaling: bool = False    # needs a master bias; not recommended for CMOS with amp glow
    require_flat: bool = True
    # Header handling
    date_obs_reference: str = "start"   # start | mid  (what DATE-OBS means for this software)
    crop: str = "auto"                  # auto = DATASEC/TRIMSEC from header | none | "[x1:x2,y1:y2]" (FITS, 1-based)
    site_from_header: bool = True       # observatory location from FITS header if present
    registration: str = "stars"         # stars (polynomial, handles distortion) | wcs | phase (translation)
    registration_order: int = 3         # polynomial order of the star-based registration
    cosmics: bool = False               # L.A.Cosmic per frame (slow; per-frame clipping already limits cosmics)
    time_offset_s: float = 0.0          # known clock / shutter latency correction
    # Observatory defaults for this instrument/site. FITS coordinates may override
    # lat/lon/alt when site_from_header is enabled; the MPC code never comes from FITS.
    observatory_code: str = ""
    observer_lat_deg: float | None = None
    observer_lon_deg: float | None = None
    observer_alt_m: float | None = None
    keymap: dict[str, list[str]] = field(default_factory=dict)       # overrides DEFAULT_KEYMAP
    filter_aliases: dict[str, str] = field(default_factory=dict)     # overrides DEFAULT_FILTER_ALIASES
    match: dict[str, str] = field(default_factory=dict)              # header rules for auto-mode
    notes: str = ""

    # ---- derived -----------------------------------------------------
    @property
    def pixel_scale_arcsec(self) -> float:
        """Plate scale at the instrument's native binning (″/px)."""
        return 206.264806 * self.pixel_size_um * self.binning / self.focal_length_mm

    def fov_deg(self) -> tuple[float, float] | None:
        if not (self.width_px and self.height_px):
            return None
        s = self.pixel_scale_arcsec / 3600.0
        return (self.width_px * s, self.height_px * s)

    def keys_for(self, quantity: str) -> list[str]:
        return list(self.keymap.get(quantity, [])) + DEFAULT_KEYMAP.get(quantity, [])

    def canonical_filter(self, name: str | None) -> str:
        if name is None:
            return "L"
        raw = str(name).strip()
        key = raw.lower()
        aliases = {**DEFAULT_FILTER_ALIASES, **{k.lower(): v for k, v in self.filter_aliases.items()}}
        return aliases.get(key, raw)

    def matches_header(self, header) -> bool:
        """True if all ``match`` rules are satisfied (case-insensitive substring)."""
        if not self.match:
            return False
        for key, expected in self.match.items():
            val = header.get(key)
            if val is None:
                return False
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    if abs(float(val) - float(expected)) > max(1.0, 0.02 * abs(float(expected))):
                        return False
                except (TypeError, ValueError):
                    return False
            elif str(expected).lower() not in str(val).lower():
                return False
        return True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def summary(self) -> dict:
        d = self.to_dict()
        for k in ("observer_lat_deg", "observer_lon_deg", "observer_alt_m"):
            d.setdefault(k, None)
        d["pixel_scale_arcsec"] = round(self.pixel_scale_arcsec, 3)
        fov = self.fov_deg()
        d["fov_deg"] = [round(fov[0], 3), round(fov[1], 3)] if fov else None
        return d


_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(text: str) -> str:
    s = _ID_RE.sub("-", text.strip()).strip("-").lower()
    return s or "instrument"


class InstrumentStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, inst_id: str) -> Path:
        return self.dir / f"{slugify(inst_id)}.toml"

    def list(self) -> list[Instrument]:
        out = []
        for p in sorted(self.dir.glob("*.toml")):
            try:
                out.append(self._load(p))
            except Exception:  # broken file must not kill the GUI
                continue
        return out

    def get(self, inst_id: str) -> Instrument:
        p = self.path(inst_id)
        if not p.exists():
            raise KeyError(f"Instrument '{inst_id}' not found")
        return self._load(p)

    def save(self, inst: Instrument) -> Instrument:
        inst.id = slugify(inst.id or inst.name)
        self.path(inst.id).write_bytes(tomli_w.dumps(inst.to_dict()).encode())
        return inst

    def delete(self, inst_id: str) -> None:
        self.path(inst_id).unlink(missing_ok=True)

    def find_for_header(self, header) -> Instrument | None:
        for inst in self.list():
            if inst.matches_header(header):
                return inst
        return None

    @staticmethod
    def _load(p: Path) -> Instrument:
        d = tomllib.loads(p.read_text(encoding="utf-8"))
        known = Instrument.__dataclass_fields__
        d = {k: v for k, v in d.items() if k in known}
        d.setdefault("id", p.stem)
        return Instrument(**d)


def instrument_from_dict(d: dict) -> Instrument:
    known = Instrument.__dataclass_fields__
    inst = Instrument(**{k: v for k, v in d.items() if k in known})
    inst.observatory_code = inst.observatory_code.strip().upper()
    if inst.observatory_code and len(inst.observatory_code) != 3:
        raise ValueError("Kód observatoře MPC musí mít tři znaky.")
    if inst.observer_lat_deg is not None and not -90 <= float(inst.observer_lat_deg) <= 90:
        raise ValueError("Zeměpisná šířka musí být mezi −90° a +90°.")
    if inst.observer_lon_deg is not None and not -180 <= float(inst.observer_lon_deg) <= 180:
        raise ValueError("Zeměpisná délka musí být mezi −180° a +180°.")
    if inst.observer_alt_m is not None and not -500 <= float(inst.observer_alt_m) <= 10000:
        raise ValueError("Nadmořská výška musí být mezi −500 m a 10 000 m.")
    return inst
