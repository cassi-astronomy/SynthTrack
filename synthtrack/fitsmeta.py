"""Reading FITS frames and normalising their header metadata."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .instruments import Instrument

FITS_EXT = (".fits", ".fit", ".fts", ".fits.fz")


def is_fits(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and n.endswith(FITS_EXT)


def list_fits(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if is_fits(p))


@dataclass
class FrameMeta:
    path: str
    exptime: float | None
    filter: str
    raw_filter: str | None
    temperature: float | None
    gain: float | None
    offset: float | None
    xbin: int
    ybin: int
    imagetyp: str
    date_obs: str | None          # ISO, as written
    mjd_mid: float | None         # UTC MJD of mid-exposure
    ra_deg: float | None
    dec_deg: float | None
    width: int
    height: int
    object: str | None = None
    datasec: str | None = None
    site_lat: float | None = None
    site_lon: float | None = None
    site_alt: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _first(header, keys):
    for k in keys:
        if k in header and header[k] not in (None, ""):
            return header[k]
    return None


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_SEXA = re.compile(r"^\s*([+-]?)(\d+)[\s:hd°]+(\d+)[\s:m']+([\d.]+)")


def parse_angle(v, hours: bool) -> float | None:
    """Accept degrees (float) or sexagesimal strings ('05 31 59.6', '+33:22:35')."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _SEXA.match(s)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    val = float(m.group(2)) + float(m.group(3)) / 60 + float(m.group(4)) / 3600
    return sign * val * (15.0 if hours else 1.0)


def read_meta(path: str | Path, inst: Instrument, header=None) -> FrameMeta:
    path = Path(path)
    if header is None:
        header = fits.getheader(path)
    exptime = _num(_first(header, inst.keys_for("exptime")))
    raw_filter = _first(header, inst.keys_for("filter"))
    date_obs = _first(header, inst.keys_for("date_obs"))
    mjd_mid = None
    off = inst.time_offset_s
    if inst.date_obs_reference == "start" and exptime:
        off += exptime / 2.0
    if date_obs:
        try:
            t = Time(str(date_obs), format="isot", scale="utc")
            mjd_mid = t.mjd + off / 86400.0
        except ValueError:
            mjd_mid = None
    if mjd_mid is None and "MJD-OBS" in header:
        try:
            mjd_mid = float(header["MJD-OBS"]) + off / 86400.0
        except (TypeError, ValueError):
            mjd_mid = None

    ra_raw = _first(header, inst.keys_for("ra"))
    ra_is_hours = isinstance(ra_raw, str) and not _looks_like_deg(ra_raw)
    return FrameMeta(
        path=str(path),
        exptime=exptime,
        filter=inst.canonical_filter(raw_filter),
        raw_filter=None if raw_filter is None else str(raw_filter),
        temperature=_sane_temp(_num(_first(header, inst.keys_for("temperature")))),
        gain=_num(_first(header, inst.keys_for("gain"))),
        offset=_num(_first(header, inst.keys_for("offset"))),
        xbin=int(_num(_first(header, inst.keys_for("xbinning"))) or 1),
        ybin=int(_num(_first(header, inst.keys_for("ybinning"))) or 1),
        imagetyp=str(_first(header, inst.keys_for("imagetyp")) or "").strip(),
        date_obs=None if date_obs is None else str(date_obs),
        mjd_mid=mjd_mid,
        ra_deg=parse_angle(ra_raw, hours=ra_is_hours),
        dec_deg=parse_angle(_first(header, inst.keys_for("dec")), hours=False),
        width=int(header.get("NAXIS1", 0)),
        height=int(header.get("NAXIS2", 0)),
        object=None if header.get("OBJECT") is None else str(header.get("OBJECT")),
        datasec=_str(_first(header, inst.keys_for("datasec"))),
        site_lat=parse_angle(_first(header, inst.keys_for("site_lat")), hours=False),
        site_lon=parse_angle(_first(header, inst.keys_for("site_lon")), hours=False),
        site_alt=_num(_first(header, inst.keys_for("site_alt"))),
    )


def _str(v):
    return None if v is None else str(v).strip()


def _sane_temp(t):
    """Capture software often writes placeholders (-274, -999, 999) when no sensor exists."""
    if t is None or t < -150 or t > 80:
        return None
    return t


_SEC = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")


def parse_section(sec: str | None, shape: tuple[int, int]) -> tuple[slice, slice] | None:
    """FITS section '[x1:x2,y1:y2]' (1-based, inclusive) -> numpy (rows, cols) slices."""
    if not sec:
        return None
    m = _SEC.match(sec)
    if not m:
        return None
    x1, x2, y1, y2 = (int(g) for g in m.groups())
    h, w = shape
    x1, x2 = max(1, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(1, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


def _looks_like_deg(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_image(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load primary (or first image) HDU as float32. Handles 16-bit BZERO/BSCALE."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
                data = np.asarray(hdu.data, dtype=np.float32)
                if data.ndim > 2:   # e.g. (1, H, W)
                    data = data.reshape(data.shape[-2:])
                return data, hdu.header
    raise ValueError(f"No image data in {path}")
