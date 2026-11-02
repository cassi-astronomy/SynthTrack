"""Export of results: MPC 80-column observations and CSV.

Positions are measured by PSF fits in sub-stacks along the track (one
position per sub-stack at its mean time) and converted with the WCS of the
reference frame. Before submitting to the MPC the planned Gaia DR3 refit of
that WCS should be used; the exported file says so in its header.
"""
from __future__ import annotations

import csv
import io
import math

from astropy.time import Time


def _ra_str(ra_deg: float) -> str:
    h = (ra_deg / 15.0) % 24
    hh = int(h)
    m = (h - hh) * 60
    mm = int(m)
    ss = (m - mm) * 60
    if ss >= 59.995:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, hh = 0, (hh + 1) % 24
    return f"{hh:02d} {mm:02d} {ss:05.2f}"


def _dec_str(dec_deg: float) -> str:
    sgn = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    dd = int(d)
    m = (d - dd) * 60
    mm = int(m)
    ss = (m - mm) * 60
    if ss >= 59.95:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, dd = 0, dd + 1
    return f"{sgn}{dd:02d} {mm:02d} {ss:04.1f}"


def _date_str(mjd: float) -> str:
    t = Time(mjd, format="mjd", scale="utc")
    y, mo, d = t.ymdhms.year, t.ymdhms.month, t.ymdhms.day
    frac = (mjd % 1.0)
    return f"{y:04d} {mo:02d} {d + frac:08.5f}"


def mpc80_line(desig: str, mjd: float, ra: float, dec: float, mag: float | None, band: str,
               obs_code: str, discovery: bool = False) -> str:
    desig = desig[:7]
    line = (f"{'':5s}{desig:<7s}{'*' if discovery else ' '} C{_date_str(mjd)} "
            f"{_ra_str(ra)} {_dec_str(dec)}          "
            f"{'' if mag is None else f'{mag:4.1f}':>4s} {band[:1]:1s}      {obs_code[:3]:>3s}")
    return line


def positions_along_track(c: dict, mjd_ref: float, mjd_list: list[float], n_points: int = 3) -> list[tuple]:
    """Linear-motion positions at ``n_points`` epochs spread over the sequence."""
    if c.get("ra") is None or c.get("rate_arcsec_min") is None:
        return []
    out = []
    idx = [round(i * (len(mjd_list) - 1) / max(1, n_points - 1)) for i in range(n_points)]
    pa = math.radians(c["pa_deg"])
    for i in idx:
        dt_min = (mjd_list[i] - mjd_ref) * 1440.0
        d = c["rate_arcsec_min"] * dt_min / 3600.0
        dec = c["dec"] + d * math.cos(pa)
        ra = c["ra"] + d * math.sin(pa) / max(math.cos(math.radians(c["dec"])), 1e-6)
        out.append((mjd_list[i], ra % 360.0, dec))
    return out


def export_mpc80(cands: list[dict], mjd_ref: float, mjd_list: list[float], obs_code: str,
                 band: str = "G") -> str:
    buf = io.StringIO()
    buf.write("COM SynthTrack 0.1 – POZOR: astrometrie zatím bez Gaia refitu, neodesílat bez kontroly\n")
    for c in cands:
        if c.get("status") != "accepted":
            continue
        desig = c.get("trksub") or (f"ST{c['id']:05d}")
        measured = [(p["mjd"], p["ra"], p["dec"]) for p in c.get("positions", []) if p.get("ok") and "ra" in p]
        for mjd, ra, dec in (measured or positions_along_track(c, mjd_ref, mjd_list)):
            buf.write(mpc80_line(desig, mjd, ra, dec, c.get("mag"), band, obs_code,
                                 discovery=c.get("known") is None) + "\n")
    return buf.getvalue()


def export_csv(cands: list[dict]) -> str:
    buf = io.StringIO()
    cols = ["id", "status", "auto_status", "snr", "x", "y", "ra", "dec", "rate_arcsec_min", "pa_deg",
            "mag", "known_name", "o_c_arcsec", "seg_snr", "flags"]
    w = csv.writer(buf)
    w.writerow(cols)
    for c in cands:
        k = c.get("known") or {}
        w.writerow([c.get("id"), c.get("status"), c.get("auto_status"), c.get("snr"), c.get("x"), c.get("y"),
                    c.get("ra"), c.get("dec"), c.get("rate_arcsec_min"), c.get("pa_deg"), c.get("mag"),
                    k.get("name", ""), k.get("o_c_arcsec", ""), "/".join(map(str, c.get("seg_snr", []))),
                    ";".join(c.get("flags", []))])
    return buf.getvalue()
