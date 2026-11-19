"""Twilight survey with very few frames: a source *in addition* to the catalogue.

A separate, side algorithm for special cases – 3–6 frames of a field taken in
bright twilight (Sun about −16° … −8°), low above the horizon, over 5–15 minutes.
Motion is then too small to rely on (2″/min × 10 min ≈ 1 FWHM at 6″/px) and the
synthetic-tracking search has too few frames; it is not used and not changed.
Instead:

1. median of the registered frames (single-frame events, chip-fixed pixels moved
   by the dither, satellites drop out), sources above ``nsigma``;
2. UCAC4 projected with the frame WCS (SIP), then the residual field distortion
   is fitted on the matched stars (polynomial of order 3–4 in x, y) – wide-field
   lenses have arcminute distortion at the corners that a bare WCS misses;
3. photometric zero point from the matched stars (2-D polynomial: extinction
   gradient at airmass 3–5);
4. candidates: a source with no catalogue star within the match radius and
   brighter than the catalogue completeness (``mag_limit``, UCAC4 ≈ 16), or a
   catalogue star much brighter than catalogued ("zjasnění" – an object on a star,
   a variable, a nova);
5. vetting: present in (almost) every frame, width vs. the stars (hot pixel,
   cosmic, comet coma), the halo of bright stars, NGC/IC/M objects (OpenNGC),
   planets, known asteroids and comets; motion from the per-frame centroids is
   reported, never required.

Pure NumPy/SciPy/astropy; the catalogue lookups reuse ``ucac4.query``.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import registration as RG


@dataclass
class TwilightParams:
    nsigma: float = 5.0             # matched-filter threshold in the median stack (σ of the filtered image)
    mag_limit: float = 15.5         # unmatched sources fainter than this are not reported (catalogue completeness)
    match_fwhm: float = 1.5         # match radius in FWHM (at least 3 × the astrometric rms)
    distortion_order: int = 3       # residual distortion polynomial after the WCS
    brighten_mag: float = 1.5       # a catalogue star measured this much brighter → "zjasnění" (clear filter vs. r: colour terms ~1 mag)
    halo_mag: float = 10.0          # stars brighter than this have a halo / ghosts around them
    min_frames_frac: float = 0.75   # present (SNR ≥ 3) in at least this share of the frames
    sharp_ratio: float = 0.6        # narrower than this × star FWHM → hot pixel / cosmic
    wide_ratio: float = 1.4         # wider than this → extended (coma? galaxy?)
    catdiff: bool = True            # difference image against the catalogue (catdiff.py): new sources beside / on stars
    catdiff_nsigma: float = 5.0     # … threshold in its normalised matched-filter map
    catdiff_bright_mag: float = 9.0 # … stars brighter than this are masked (no model is good enough)
    coma_scale_ratio: float = 1.5   # two-scale width ≥ this × stars → labelled "rozmazaný – kometa?" (label only)
    border_px: int = 24
    known_radius_arcsec: float = 30.0
    max_candidates: int = 300
    diffuse: bool = True            # search for diffuse sources (comet comae) on the star-masked median
    diffuse_scales: tuple = (3.0, 6.0, 12.0)  # Gaussian σ of the smoothing, px (σ 2 px picked faint uncatalogued stars)
    diffuse_nsigma: float = 6.0
    diffuse_star_sum_mag: float = 0.3      # small glow ≤ this fainter than the stars around it → their wings
    few_frame_snr_min: float = 4.0  # 2–4 frames: a point source ≥ this in every frame
    few_glow_ratio_min: float = 0.35  # 2–4 frames: a glow's per-frame sum ≥ this × its stack significance
    sn_z_min: float = 6.0          # supernova in a galaxy: residual peak ≥ this × the galaxy's own asymmetry
    sn_sharp_min: float = 1.6      # … and point-like: filtered at 1 / 2 PSF widths ≥ this (point: ~2.5)
    sn_min_offset_fwhm: float = 0.5  # … and off the nucleus by ≥ this (closer the symmetry cannot see it)
    sn_light_max: float = 1.8      # galaxy light under the peak / residual: above → structure of the galaxy
    sn_light_warn: float = 1.35    # … above this a warning "ověřit v DSS"
    sn_light_strong_z: float = 15.0  # … unless the excess is this strong: kept, "ověřit v DSS"
    sn_sharp_max: float = 4.5      # sharper than a point source (Gaussian ≈ 2.5, a single pixel 4): hot pixel / residual
    frame_ratio_min: float = 0.25  # 2–4 frames: weakest frame ≥ this × the strongest (point sources ≥ 10σ)
    diffuse_frame_snr: float = 2.5  # per-frame SNR of a diffuse source at its scale (only "count" test / display)
    # noise of the normalised convolution next to masks: with a fraction of the kernel unmasked the
    # smoothed value averages fewer pixels – its σ grows by sqrt(Σg²·m)/(Σg·m) (CTA-N 322 #394:
    # 6 % of the kernel unmasked, noise × 2.9, a "7σ glow" was 2.9σ). False = old behaviour
    diffuse_noise_norm: bool = True
    # per-frame test: "thirds" = the value at the glow's position (following a fitted motion) summed
    # in three time thirds, each must carry ≥ diffuse_third_frac of its expected share (≥ 1σ);
    # "count" = old: max in a ±dither window ≥ diffuse_frame_snr in ≥ min_frames_frac of frames
    # (on empty sky that passed 35 % of random positions; with an unbiased value a real 7σ glow of
    # 15 frames has 1.8σ per frame and would fail it)
    diffuse_frame_test: str = "thirds"
    diffuse_third_frac: float = 0.35
    faint_extended_mag: float = 1.0  # extended, still "extra" sources within this of the limit → rejected (galaxies)
    dither_alpha_max: float = 0.35  # position following the dither (ghost, reflection, chip-fixed): |α| above → rejected


# ----------------------------------------------------------------------------- DSO (OpenNGC)

def _sexa(s: str, hours: bool) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    sign = -1.0 if s.startswith("-") else 1.0
    p = [float(x) for x in s.lstrip("+-").split(":")]
    v = p[0] + (p[1] if len(p) > 1 else 0) / 60 + (p[2] if len(p) > 2 else 0) / 3600
    return sign * v * (15.0 if hours else 1.0)


def load_dso(paths) -> dict:
    """OpenNGC NGC.csv (+ addendum.csv): name, type, ra, dec (deg), axes (′), PA, mag, Messier."""
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                ra, dec = _sexa(r.get("RA"), True), _sexa(r.get("Dec"), False)
                if ra is None or dec is None or r.get("Type") in ("*", "**", "Dup", "NonEx"):
                    continue
                f = lambda k: float(r[k]) if r.get(k) not in (None, "") else np.nan
                mag = f("V-Mag")
                if not np.isfinite(mag):
                    mag = f("B-Mag")
                m = (r.get("M") or "").strip()
                rows.append((r["Name"], r.get("Type", ""), ra, dec, f("MajAx"), f("MinAx"), f("PosAng"), mag,
                             f"M{int(m)}" if m.isdigit() else ""))
    if not rows:
        return {}
    cols = list(zip(*rows))
    return {"name": np.array(cols[0]), "type": np.array(cols[1]), "ra": np.array(cols[2], float),
            "dec": np.array(cols[3], float), "maj": np.array(cols[4], float), "min": np.array(cols[5], float),
            "pa": np.array(cols[6], float), "mag": np.array(cols[7], float), "messier": np.array(cols[8])}


def _astap_rows(path: Path, skip: int = 2):
    """ASTAP/HNSKY csv: RA [0.1 s of time, 0..864000], Dec [″], then fields."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            p = line.rstrip("\n").split(",")
            try:
                yield int(p[0]) / 864000 * 360.0, int(p[1]) / 3600.0, p[2:]
            except (ValueError, IndexError):
                continue


def load_astap_dso(path) -> dict:
    """ASTAP ``deep_sky.csv`` (30 000 objects: NGC/IC, LEDA/PGC galaxies ≥ 1′, Sh2, vdB, PN, …)."""
    rows = []
    for ra, dec, f in _astap_rows(Path(path)):
        if not f or f[0].endswith("_2000") or f[0].endswith("_2025"):
            continue
        g = lambda i: float(f[i]) / 10 if len(f) > i and f[i].strip() not in ("", "0") else np.nan
        rows.append((f[0].replace("/", " = "), "", ra, dec, g(1), g(2), float(f[3]) if len(f) > 3 and f[3].strip() else np.nan,
                     np.nan, ""))
    if not rows:
        return {}
    cols = list(zip(*rows))
    return {"name": np.array(cols[0]), "type": np.array(cols[1]), "ra": np.array(cols[2], float),
            "dec": np.array(cols[3], float), "maj": np.array(cols[4], float), "min": np.array(cols[5], float),
            "pa": np.array(cols[6], float), "mag": np.array(cols[7], float), "messier": np.array(cols[8])}


def merge_dso(*cats) -> dict:
    cats = [c for c in cats if c]
    if not cats:
        return {}
    return {k: np.concatenate([c[k] for c in cats]) for k in cats[0]}


def load_variables(path, max_mag: float | None = 16.0) -> dict:
    """ASTAP ``variable_stars_13.csv`` (local file, AAVSO VSX to ~18 mag): ra, dec, label.

    ``max_mag``: only variables that get at least this bright at maximum – in twilight the frames
    reach ~14–15 mag, fainter variables only add chance coincidences."""
    ra, dec, lab = [], [], []
    for r, d, f in _astap_rows(Path(path)):
        # the file also holds AAVSO VSP comparison stars ("000-BKM-370 V=12.808(0.049)_B=…",
        # 64 879 of 313 059 rows) – constant stars, not variables
        if "=" in ",".join(f):
            continue
        if max_mag is not None:
            vm = _var_max(" " + ",".join(f).split(" ", 1)[-1]) if " " in ",".join(f) else None
            if vm is not None and vm > max_mag:
                continue
        ra.append(r)
        dec.append(d)
        lab.append(",".join(f))
    if not ra:
        return {}
    return {"ra": np.array(ra), "dec": np.array(dec), "label": np.array(lab)}


def variable_at(var: dict, ra: float, dec: float, rad_arcsec: float) -> str | None:
    if not var:
        return None
    sel = np.abs(var["dec"] - dec) < rad_arcsec / 3600 * 2
    if not sel.any():
        return None
    cd = math.cos(math.radians(dec))
    d = np.hypot(((var["ra"][sel] - ra + 540) % 360 - 180) * cd, var["dec"][sel] - dec) * 3600
    k = int(np.argmin(d))
    return str(var["label"][sel][k]) if d[k] <= rad_arcsec else None


def _var_max(label: str) -> float | None:
    """Brightest magnitude of a VSX label 'NAME 11.0V-<14.8V_Period_219' → 11.0."""
    import re
    m = re.search(r"\s(-?\d+(?:\.\d+)?)[A-Za-z]", label or "")
    return float(m.group(1)) if m else None


def _var_amplitude(label: str) -> float | None:
    """Amplitude: 'max-min' range or '(0.005)'; None when not given."""
    import re
    m = re.search(r"\s(-?\d+(?:\.\d+)?)[A-Za-z]*-<?(\d+(?:\.\d+)?)", label or "")
    if m:
        return abs(float(m.group(2)) - float(m.group(1)))
    m = re.search(r"\((\d+(?:\.\d+)?)\)", label or "")
    return float(m.group(1)) if m else None


def _var_min(label: str) -> float | None:
    """Faintest magnitude of a VSX label 'NAME 7.7V-12.4V_Period_0.275' → 12.4 (None for '(amp)' / '?')."""
    import re
    m = re.search(r"\s-?\d+(?:\.\d+)?[A-Za-z]*-<?(\d+(?:\.\d+)?)", label or "")
    return float(m.group(1)) if m else None


def _var_period(label: str) -> float | None:
    import re
    m = re.search(r"_Period_(\d+(?:\.\d+)?)", label or "")
    return float(m.group(1)) if m else None


def outburst_note(label: str, mag: float, quiescent: float | None, min_excess: float = 1.5) -> str | None:
    """A known large-amplitude variable seen far brighter than its quiet level: possibly an outburst.

    VSX via ASTAP has no type, only the range and the period: amplitude ≥ 2 mag with no period or
    a period < 3 d (orbital periods of cataclysmic variables: SS Cyg 0.275, U Gem 0.177, WZ Sge
    0.057, GK Per 2) → "dwarf / recurrent nova"; a long period (Miras 100–500 d, but also the
    symbiotic recurrent novae T CrB 228 d, RS Oph 454 d) → "bright phase or outburst".
    ``quiescent``: the catalogue (expected) magnitude, else the faint end of the VSX range."""
    amp = _var_amplitude(label)
    if amp is None or amp < 2.0:
        return None
    q = quiescent if quiescent is not None else _var_min(label)
    if q is None or not math.isfinite(mag) or mag > q - min_excess:
        return None
    name, _, rest = label.partition(" ")
    rng, _, per = rest.partition("_Period_")
    p = _var_period(label)
    desc = f"VSX {name.rstrip('#').replace('_', ' ')} ({rng.rstrip('#')}{', P ' + per.rstrip('#') + ' d' if per else ''})"
    if p is None or p < 3.0:
        return f"vzplanutí? {desc} – o {q - mag:.1f} mag jasnější než v klidu (trpasličí / rekurentní nova?)"
    return f"{desc} – o {q - mag:.1f} mag jasnější než v klidu: jasná fáze (Mira?) nebo vzplanutí"


_DARK_NEB = None


def _is_dark_nebula(name: str) -> bool:
    """Lynds / Barnard dark clouds: an absence of light – a star, nova or comet seen in front of
    one is no part of it (CTA-N 324/4: an injected comet rejected as "DSO LDN1251")."""
    import re
    global _DARK_NEB
    if _DARK_NEB is None:
        _DARK_NEB = re.compile(r"^(LDN|LDN_|B|Barnard|Bernes)\s*_?\d", re.I)
    return bool(_DARK_NEB.match(name or ""))


def dso_at(dso: dict, ra: float, dec: float, pad_arcsec: float, skip_dark: bool = False) -> str | None:
    """Name of an NGC/IC/M object whose ellipse (+ pad) contains the position."""
    if not dso:
        return None
    cd = math.cos(math.radians(dec))
    dx = ((dso["ra"] - ra + 540) % 360 - 180) * cd * 60          # ′, east
    dy = (dso["dec"] - dec) * 60
    near = np.nonzero(np.hypot(dx, dy) < 60)[0]
    best = None
    for k in near:
        a = (dso["maj"][k] if np.isfinite(dso["maj"][k]) else 0.5) / 2 + pad_arcsec / 60
        b = (dso["min"][k] if np.isfinite(dso["min"][k]) else dso["maj"][k] if np.isfinite(dso["maj"][k]) else 0.5) / 2 + pad_arcsec / 60
        pa = math.radians(dso["pa"][k]) if np.isfinite(dso["pa"][k]) else 0.0
        # rotate the offset of the target from the object into the ellipse frame (PA from north to east)
        u = -dx[k] * math.sin(pa) - dy[k] * math.cos(pa)
        v = -dx[k] * math.cos(pa) + dy[k] * math.sin(pa)
        if (u / a) ** 2 + (v / b) ** 2 <= 1.0:
            if skip_dark and _is_dark_nebula(str(dso["name"][k])):
                continue
            nm = dso["name"][k] + (f" ({dso['messier'][k]})" if dso["messier"][k] else "")
            mag = dso["mag"][k]
            nm += (f", {dso['type'][k]}" if dso["type"][k] else "") + (f" {mag:.1f} mag" if np.isfinite(mag) else "")
            if best is None or (np.isfinite(mag) and mag < best[0]):
                best = (mag if np.isfinite(mag) else 99, nm)
    return best[1] if best else None


# ----------------------------------------------------------------------------- measurements

def robust_sigma(img: np.ndarray) -> float:
    v = img[np.isfinite(img)]
    if v.size > 400_000:
        v = v[:: v.size // 400_000]
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med))) or 1.0


def detect(img: np.ndarray, fwhm: float, nsigma: float, border: int = 24, max_n: int = 80000) -> np.ndarray:
    """Point sources (N, 3: x, y, flux) with a matched filter normalised to its own noise.

    ``registration.detect_stars`` thresholds the image smoothed by σ = 1 px against the
    *unsmoothed* noise – fine for registration stars, but a 6σ source at FWHM 2.6 px then
    stays below 4 (317/5: half of the injected 11.5–12.5 mag sources lost)."""
    s = max(fwhm / 2.3548, 0.6)
    sm = ndimage.gaussian_filter(np.nan_to_num(img), s)
    sm = sm / robust_sigma(sm)
    mx = ndimage.maximum_filter(sm, size=5)
    pk = (sm == mx) & (sm > nsigma)
    pk[:border] = pk[-border:] = False
    pk[:, :border] = pk[:, -border:] = False
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return np.zeros((0, 3))
    order = np.argsort(sm[ys, xs])[::-1][:max_n]
    ys, xs = ys[order], xs[order]
    r = 2
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    cut = np.nan_to_num(img[ys[:, None, None] + oy[None], xs[:, None, None] + ox[None]])
    cut = np.clip(cut - np.median(cut.reshape(len(ys), -1), axis=1)[:, None, None], 0, None)
    tot = cut.sum(axis=(1, 2))
    good = tot > 0
    cx = xs[good] + (cut[good] * ox).sum(axis=(1, 2)) / tot[good]
    cy = ys[good] + (cut[good] * oy).sum(axis=(1, 2)) / tot[good]
    return np.stack([cx, cy, tot[good]], axis=1)


class ScaleWidth:
    """Equivalent FWHM of a source from the ratio of its peaks filtered at σ and 2σ of the star profile
    (a Gaussian of width s: Q = (s² + 4σ²)/(s² + σ²)), relative to the catalogue stars around it.
    Unlike second moments in a small box (whose median "background" is the coma itself) it sees a coma of
    a few pixels, and it is local and robust to noise and neighbours."""

    def __init__(self, img, fw_star, cxy, cat_mag, fw_loc, k: int = 30):
        self.sg = max(fw_star / 2.3548, 0.6)
        v = np.nan_to_num(np.asarray(img, np.float32))
        h, w = v.shape
        f = 4
        bg = ndimage.median_filter(v[::f, ::f], size=9)
        bg = ndimage.zoom(bg, f, order=1)[:h, :w]
        if bg.shape != v.shape:
            bg = np.pad(bg, ((0, h - bg.shape[0]), (0, w - bg.shape[1])), mode="edge")
        v = v - bg
        self.S1 = ndimage.gaussian_filter(v, self.sg)
        self.S2 = ndimage.gaussian_filter(v, 2 * self.sg)
        self.fw_loc = fw_loc
        ok = np.isfinite(cxy).all(axis=1) & (cat_mag > 10.5) & (cat_mag < 13.5)
        ok &= (cxy[:, 0] > 10) & (cxy[:, 0] < w - 10) & (cxy[:, 1] > 10) & (cxy[:, 1] < h - 10)
        allc = cxy[np.isfinite(cxy).all(axis=1)]
        T = cKDTree(allc)
        pts, qs = [], []
        for i in np.nonzero(ok)[0][:4000]:
            if len(T.query_ball_point(cxy[i], 3 * fw_star)) > 1:
                continue
            q = self.q(*cxy[i])
            if q is not None and 1.2 < q < 3.9:
                pts.append(cxy[i]); qs.append(q)
        if len(qs) < 30:
            raise ValueError("málo hvězd")
        self.tree, self.qs, self.k = cKDTree(np.array(pts)), np.array(qs), min(k, len(qs))

    def q(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        h, w = self.S1.shape
        if not (1 <= xi < w - 1 and 1 <= yi < h - 1):
            return None
        win = self.S1[yi - 1:yi + 2, xi - 1:xi + 2]
        j = np.unravel_index(int(np.argmax(win)), win.shape)
        a, b = float(self.S1[yi - 1 + j[0], xi - 1 + j[1]]), float(self.S2[yi - 1 + j[0], xi - 1 + j[1]])
        return a / b if b > 0 and a > 0 else None

    @staticmethod
    def _s2(q):
        return (4 - q) / (q - 1) if 1 < q < 4 else (float("inf") if q <= 1 else 0.0)

    def __call__(self, x, y):
        q = self.q(x, y)
        if q is None:
            return None
        _, j = self.tree.query([x, y], k=self.k)
        qst = float(np.median(self.qs[np.atleast_1d(j)]))
        s_o, s_s = self._s2(q), self._s2(qst)
        if not math.isfinite(s_o):
            return 99.0
        return float(self.fw_loc(x, y) * math.sqrt(max(s_o, 0.0) / max(s_s, 1e-6)))


def moments_fwhm(img: np.ndarray, x: float, y: float, r: int = 4) -> float | None:
    """FWHM from second moments in a (2r+1)² box (background = box median)."""
    xi, yi = int(round(x)), int(round(y))
    h, w = img.shape
    if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
        return None
    cut = np.nan_to_num(img[yi - r:yi + r + 1, xi - r:xi + r + 1]).astype(float)
    cut = np.clip(cut - np.median(cut), 0, None)
    s = cut.sum()
    if s <= 0:
        return None
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    mx, my = (cut * ox).sum() / s, (cut * oy).sum() / s
    var = ((cut * ((ox - mx) ** 2 + (oy - my) ** 2)).sum() / s) / 2
    return float(2.3548 * math.sqrt(max(var, 1e-6)))


def forced(frame: np.ndarray, x: float, y: float, fwhm: float, sig: float):
    """Peak SNR (Gaussian-smoothed) and centroid near (x, y) in one registered frame."""
    r = int(max(4, round(1.5 * fwhm)))
    xi, yi = int(round(x)), int(round(y))
    h, w = frame.shape
    if xi - r - 2 < 0 or yi - r - 2 < 0 or xi + r + 2 >= w or yi + r + 2 >= h:
        return None
    cut = np.nan_to_num(frame[yi - r - 2:yi + r + 3, xi - r - 2:xi + r + 3]).astype(float)
    cut = cut - np.median(cut)
    sm = ndimage.gaussian_filter(cut, max(fwhm / 2.3548, 0.6))
    c = r + 2
    win = sm[c - r:c + r + 1, c - r:c + r + 1]
    j = int(np.argmax(win))
    py, px = divmod(j, win.shape[1])
    # noise of the smoothed frame: σ / (2√π s) for a Gaussian kernel of width s
    s = max(fwhm / 2.3548, 0.6)
    snr = float(win[py, px] / (sig / (2 * math.sqrt(math.pi) * s)))
    yy, xx = py + c - r, px + c - r
    q = np.clip(cut[yy - 2:yy + 3, xx - 2:xx + 3], 0, None)
    tot = q.sum()
    oy, ox = np.mgrid[-2:3, -2:3]
    cx = xi - c + xx + ((q * ox).sum() / tot if tot > 0 else 0.0)
    cy = yi - c + yy + ((q * oy).sum() / tot if tot > 0 else 0.0)
    return snr, float(cx), float(cy)


# ----------------------------------------------------------------------------- diffuse sources

def _normconv(img: np.ndarray, good: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores masked pixels (normalised convolution)."""
    num = ndimage.gaussian_filter(np.where(good, img, 0.0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(good.astype(np.float32), sigma)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)


def _normconv_noise(good: np.ndarray, sigma: float) -> np.ndarray:
    """σ of ``_normconv`` of unit white noise relative to an unmasked kernel: sqrt(G₂/G₁²) with
    G₁ = g∗m, G₂ = g'∗m, g' the Gaussian of σ/√2 (g² ∝ g'); ≥ 1, NaN where ``_normconv`` is."""
    m = good.astype(np.float32)
    g1 = ndimage.gaussian_filter(m, sigma)
    g2 = ndimage.gaussian_filter(m, sigma / math.sqrt(2.0))
    return np.where(g1 > 0.05, np.sqrt(np.maximum(g2, 1e-12)) / np.maximum(g1, 1e-6), np.nan)


def _lowpass(img: np.ndarray, good: np.ndarray, sigma: float, block: int = 8) -> np.ndarray:
    """Large-scale background (σ of tens of px) via a block-averaged image."""
    h, w = img.shape
    hb, wb = -(-h // block), -(-w // block)
    pad = lambda a, v: np.pad(a, ((0, hb * block - h), (0, wb * block - w)), constant_values=v)
    I = pad(np.where(good, img, 0.0), 0).reshape(hb, block, wb, block).sum(axis=(1, 3))
    G = pad(good.astype(np.float32), 0).reshape(hb, block, wb, block).sum(axis=(1, 3))
    bg = _normconv(I / np.maximum(G, 1e-6), G > 0, sigma / block)
    bg = np.nan_to_num(bg, nan=float(np.nanmedian(bg)) if np.isfinite(bg).any() else 0.0)
    return ndimage.zoom(bg, block, order=1)[:h, :w]


def wing_radii(img: np.ndarray, xy: np.ndarray, mag: np.ndarray, fwhm: float, level: float = 0.2,
               rmax: int = 60) -> tuple[np.ndarray, np.ndarray] | None:
    """Measured extent of the stars' wings: per 1-mag bin the radius where the mean radial profile
    of isolated catalogue stars drops below ``level`` (σ of the image). Lenses differ a lot: CTA-N
    135 mm (FWHM 2.2 px) has wings of 10–13 mag stars far beyond the FWHM-based mask (user report
    13. 11.: ~60 false "diffuse objects" per field 20–60″ from 10–13 mag stars)."""
    h, w = img.shape
    tree = cKDTree(xy)
    R = rmax + 6
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy).astype(int)
    edges, radii = [], []
    for m0 in np.arange(6.0, 15.0, 1.0):
        sel = np.nonzero((mag >= m0) & (mag < m0 + 1))[0]
        prof, cnt = np.zeros(R + 1), np.zeros(R + 1)
        used = 0
        for i in sel:
            x, y = xy[i]
            xi, yi = int(round(x)), int(round(y))
            if xi - R < 0 or yi - R < 0 or xi + R >= w or yi + R >= h:
                continue
            nb = tree.query_ball_point([x, y], 30.0)
            if any(mag[j] < mag[i] + 1.5 for j in nb if j != i):
                continue                              # a comparable neighbour would bias the wing
            cut = np.nan_to_num(img[yi - R:yi + R + 1, xi - R:xi + R + 1])
            bg = float(np.median(cut[rr >= R - 4]))
            v = np.bincount(rr.ravel(), weights=(cut - bg).ravel(), minlength=R + 1)[:R + 1]
            c = np.bincount(rr.ravel(), minlength=R + 1)[:R + 1]
            prof += v
            cnt += c
            used += 1
            if used >= 120:
                break
        if used < 5:
            continue
        p = prof / np.maximum(cnt, 1)
        below = np.nonzero(p[1:] < level)[0]
        r = float(below[0] + 1) if len(below) else float(rmax)
        edges.append(m0 + 0.5)
        radii.append(min(r, rmax))
    if len(edges) < 2:
        return None
    radii = np.maximum.accumulate(np.array(radii)[::-1])[::-1]   # brighter stars never smaller
    return np.array(edges), radii


def star_mask(shape, xy: np.ndarray, mag: np.ndarray, fwhm: float, extra_xy: np.ndarray | None = None,
              wings=None) -> np.ndarray:
    """True = sky. Disks around catalogue stars (radius grows with brightness, at least the measured
    wing extent ``wings`` = (mag, radius)) and all point detections."""
    h, w = shape
    good = np.ones(shape, bool)
    rad = np.maximum(1.3 * fwhm, fwhm * (1.0 + 0.55 * (13.0 - np.asarray(mag, float))))
    if wings is not None:
        # only within the measured magnitudes: fainter stars keep the FWHM-based radius
        rad = np.maximum(rad, np.interp(mag, wings[0], wings[1], left=wings[1][0], right=0.0) + 1.0)
    pts = [(x, y, r) for (x, y), r in zip(xy, rad)]
    if extra_xy is not None:
        pts += [(x, y, 1.3 * fwhm) for x, y in extra_xy]
    for x, y, r in pts:
        r = min(r, 60.0)
        xi, yi, ri = int(round(x)), int(round(y)), int(math.ceil(r))
        y0, y1, x0, x1 = max(0, yi - ri), min(h, yi + ri + 1), max(0, xi - ri), min(w, xi + ri + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        good[y0:y1, x0:x1] &= (xx - x) ** 2 + (yy - y) ** 2 > r * r
    return good


def _diffuse_fixed_values(zmaps, dts, cx0, cy0, sc, cxs, cys, used, x0, y0) -> list:
    """Per-frame significance at the glow's own position (max within ±max(1, σ/4) px), not the
    maximum of a wide window. A clearly fitted motion of the per-frame centroids (a moving
    comet) is followed; otherwise the position is fixed."""
    n = len(zmaps)
    t = np.asarray(dts, float)
    tm = float(t.mean()) if n else 0.0
    vx = vy = 0.0
    if len(used) >= 4:
        tu = t[used]
        A = np.stack([np.ones_like(tu), tu - tm], 1)
        cx = np.asarray(cxs) - x0
        cy = np.asarray(cys) - y0
        bx, *_ = np.linalg.lstsq(A, cx, rcond=None)
        by, *_ = np.linalg.lstsq(A, cy, rcond=None)
        rms = float(np.sqrt(np.mean((cx - A @ bx) ** 2 + (cy - A @ by) ** 2)))
        span = float(t.max() - t.min())
        if (rms <= max(1.5, 0.5 * sc) and math.hypot(bx[1], by[1]) * span >= 2.0
                and math.hypot(bx[0] - cx0, by[0] - cy0) <= max(2.0, sc)):
            vx, vy = float(bx[1]), float(by[1])
    rf = max(1, int(round(0.25 * sc)))
    out = []
    for i, Z in enumerate(zmaps):
        px = int(round(cx0 + vx * (t[i] - tm)))
        py = int(round(cy0 + vy * (t[i] - tm)))
        win = Z[max(0, py - rf):py + rf + 1, max(0, px - rf):px + rf + 1]
        win = win[np.isfinite(win)] if win.size else win
        out.append(float(win.max()) if win.size else 0.0)
    return out


def _thirds(vals, dts, frac: float) -> tuple[bool, list, float]:
    """Signal in each time third of the series: sums in units of their noise vs the share expected
    from the whole series. A transient (satellite, cosmic, a few frames) carries one third only.
    Returns (passed, [T1, T2, T3], expected per third)."""
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return True, [float(v.sum())], float(v.sum())
    parts = np.array_split(np.argsort(np.asarray(dts, float)), min(3, n))     # 2 frames: each on its own
    tot = float(v.sum())
    T, ok, E = [], True, []
    for p in parts:
        m = len(p)
        tk = float(v[p].sum() / math.sqrt(m))
        ek = tot / n * math.sqrt(m)                 # this third's share of the series' signal
        T.append(tk)
        E.append(ek)
        if tk < max(1.0, frac * ek):
            ok = False
    return ok, T, float(np.mean(E))


def diffuse_centroid(res: np.ndarray, good: np.ndarray, x: float, y: float, sc: float,
                     iters: int = 4) -> tuple[float, float]:
    """Centre of a glow: light-weighted centroid of the σ-smoothed residual (``good`` = sky
    mask without point sources), iterated within 1.5σ. Falls back to (x, y) when the result is
    not finite or wanders more than 3σ + 2 px away (another source took over)."""
    if res is None or good is None:
        return x, y
    h, w = res.shape
    R = int(math.ceil(6 * sc + 4))
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, yi - R), min(h, yi + R + 1), max(0, xi - R), min(w, xi + R + 1)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return x, y
    S = _normconv(np.nan_to_num(res[y0:y1, x0:x1]), good[y0:y1, x0:x1], sc)
    S = np.nan_to_num(S, nan=0.0)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    cx, cy = float(x), float(y)
    for _ in range(iters):
        sel = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (1.5 * sc + 1) ** 2) & (S > 0)
        wsum = float(S[sel].sum())
        if wsum <= 0:
            return x, y
        nx, ny = float((S[sel] * xx[sel]).sum() / wsum), float((S[sel] * yy[sel]).sum() / wsum)
        if math.hypot(nx - cx, ny - cy) < 0.05:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    if not (math.isfinite(cx) and math.isfinite(cy)) or math.hypot(cx - x, cy - y) > 3 * sc + 2:
        return x, y
    return cx, cy


def detect_diffuse(img: np.ndarray, good: np.ndarray, scales, nsigma: float, border: int, neg_stats=None,
                   noise_norm: bool = True):
    """Peaks of the star-masked, background-subtracted image smoothed at several scales.
    Returns rows (x, y, σ_scale, significance, flux within 2σ)."""
    res = img - _lowpass(img, good, 40.0)
    rows = []
    for sc in scales:
        S = _normconv(res, good, sc)
        Sn = S / _normconv_noise(good, sc) if noise_norm else S      # flux stays from S
        nz = robust_sigma(Sn)
        Z = np.nan_to_num(Sn / nz, nan=0.0)
        mx = ndimage.maximum_filter(Z, size=int(2 * sc) * 2 + 1)
        b = border + int(3 * sc)
        # false-alarm calibration on the negative image: large-scale sky structure (twilight gradient,
        # flat residuals, unresolved star wings) is correlated noise that robust σ underestimates.
        # Whatever the negative map reaches by chance, the positive one can too (CTA-N 318/5:
        # 15 "glows" of 12.3–13.9 mag at 6–10σ, nothing there).
        mn = ndimage.maximum_filter(-Z, size=int(2 * sc) * 2 + 1)
        npk = (-Z == mn) & (-Z > 0.5 * nsigma) & good
        npk[:b] = npk[-b:] = False
        npk[:, :b] = npk[:, -b:] = False
        negz = np.sort(-Z[npk])[::-1]
        thr = nsigma
        # a handful of isolated dark structures (over-subtracted halos of bright stars, frame edges)
        # are not the noise level: the 5th strongest negative peak sets it
        if negz.size >= 5:
            thr = min(max(nsigma, float(negz[4]) * 1.05), 2.0 * nsigma)
        if neg_stats is not None:
            neg_stats[f"{sc:g}"] = {"neg_max": round(float(negz[0]), 1) if negz.size else None,
                                    "neg_top": [round(float(v), 1) for v in negz[:8]],
                                    "neg_over": int((negz > nsigma).sum()), "threshold": round(thr, 1)}
        pk = (Z == mx) & (Z > thr)
        pk[:b] = pk[-b:] = False
        pk[:, :b] = pk[:, -b:] = False
        for y, x in zip(*np.nonzero(pk)):
            rows.append((float(x), float(y), float(sc), float(Z[y, x]), float(S[y, x] * 2 * math.pi * sc * sc)))
    # merge the same source found at several scales: keep the most significant; the 6th element
    # is the smallest scale at which the source passed – its size for the halo rule (with the noise
    # of masked regions accounted for a larger kernel may win the significance next to stars;
    # its +3σ halo radius then reached an 8.7 mag star 35 px away from a 20″ comet)
    rows.sort(key=lambda r: -r[3])
    out = []
    for r in rows:
        hit = next((q for q in out if math.hypot(r[0] - q[0], r[1] - q[1]) <= 1.5 * max(r[2], q[2])), None)
        if hit is None:
            out.append(list(r) + [r[2]])
        else:
            hit[5] = min(hit[5], r[2])
    return [tuple(q) for q in out]


def dither_test(t, xs, ys, dith) -> tuple[float | None, float | None]:
    """Fit pos_i = p0 + v·t_i + α·d_i (d_i = registration offset of frame i, px).
    A source on the sky has α ≈ 0; a chip-fixed pixel ±1; an internal reflection / ghost of a
    bright star moves with the pointing by some other α. Returns (α, its σ) or (None, None)."""
    d = np.asarray(dith, float)
    t = np.asarray(t, float)
    if len(t) < 3 or d.shape != (len(t), 2) or np.ptp(d[:, 0]) + np.ptp(d[:, 1]) < 2.0:
        return None, None
    # stacked x and y: unknowns x0, y0, vx, vy, α
    n = len(t)
    A = np.zeros((2 * n, 5))
    A[:n, 0] = 1
    A[n:, 1] = 1
    A[:n, 2] = t - t.mean()
    A[n:, 3] = t - t.mean()
    A[:n, 4] = d[:, 0] - d[:, 0].mean()
    A[n:, 4] = d[:, 1] - d[:, 1].mean()
    yv = np.concatenate([xs, ys])
    sol, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r = yv - A @ sol
    dof = max(1, 2 * n - 5)
    s2 = max(float(r @ r) / dof, 0.15 ** 2)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return None, None
    return float(sol[4]), float(math.sqrt(max(cov[4, 4], 1e-12)))


# ----------------------------------------------------------------------------- astrometry

def _world2pix_sip(wcs, ra, dec, shape, margin: float = 200.0) -> np.ndarray:
    """Pixel positions with the full WCS (SIP included, as the catalogue stars), NaN far outside the
    frame (where SIP may not converge): a linear projection selects first, then ``all_world2pix``."""
    h, w = shape
    ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
    out = np.full((len(ra), 2), np.nan)
    x0, y0 = wcs.wcs_world2pix(ra, dec, 0)
    m = np.isfinite(x0) & np.isfinite(y0) & (x0 > -margin) & (x0 < w + margin) & (y0 > -margin) & (y0 < h + margin)
    if m.any():
        try:
            x1, y1 = wcs.all_world2pix(ra[m], dec[m], 0, quiet=True)
        except Exception:                  # noqa: BLE001 – keep the linear positions
            x1, y1 = x0[m], y0[m]
        out[m, 0], out[m, 1] = x1, y1
    return out


def project_catalog(cat: dict, wcs, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UCAC4 (all stars) → pixel coordinates with the full WCS (incl. SIP); inside the frame."""
    ra, dec, mag = (np.asarray(cat[k], float) for k in ("ra_all", "dec_all", "mag_all"))
    x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)   # far outside the frame SIP may not converge
    h, w = shape
    ok = np.isfinite(x) & np.isfinite(y) & (x > -20) & (y > -20) & (x < w + 20) & (y < h + 20) & np.isfinite(mag)
    # a point where the iteration diverged could land inside the frame: keep only round trips < 1″
    if ok.any():
        bk = wcs.all_pix2world(x[ok], y[ok], 0)
        sep = np.hypot(((bk[0] - ra[ok] + 540) % 360 - 180) * np.cos(np.radians(dec[ok])), bk[1] - dec[ok]) * 3600
        ok[np.nonzero(ok)[0][~(sep < 1.0)]] = False
    return np.stack([x[ok], y[ok]], 1), mag[ok], np.nonzero(ok)[0]


def fit_distortion(det_xy, cat_xy, cat_mag, shape, fwhm, order=3):
    """Residual distortion (catalogue → measured pixels) from bright, isolated matches."""
    h, w = shape
    center, scale = (w / 2, h / 2), max(w, h) / 2
    tree = cKDTree(det_xy)
    # isolated catalogue stars (no neighbour within 3 FWHM up to 2 mag fainter)
    ct = cKDTree(cat_xy)
    pairs = ct.query_pairs(3 * fwhm, output_type="ndarray")
    crowded = np.zeros(len(cat_xy), bool)
    if len(pairs):
        dm = cat_mag[pairs[:, 0]] - cat_mag[pairs[:, 1]]
        crowded[pairs[dm > -2, 0]] = True
        crowded[pairs[dm < 2, 1]] = True
    sel = np.nonzero(~crowded & (cat_mag < 14.0) & (cat_mag > 7.0))[0]
    transform = None
    rms = None
    # first passes with few bright stars and a wide radius: at the corners of a wide field the
    # WCS can be off by many pixels (Coihueco 300 mm: unmatched 11–13 mag stars in all corners)
    for rad0, mlim in ((20.0, 11.0), (10.0, 12.5)):
        s0 = sel[cat_mag[sel] < mlim]
        src = cat_xy[s0] if transform is None else np.stack(transform(cat_xy[s0, 0], cat_xy[s0, 1]), 1)
        d, j = tree.query(src, distance_upper_bound=rad0)
        ok = np.isfinite(d)
        if ok.sum() >= 30:
            transform = RG.PolyTransform.fit(cat_xy[s0][ok], det_xy[j[ok]], order, center, scale)
    rad = max(4.0, 1.5 * fwhm)
    for it in range(4):
        src = cat_xy[sel] if transform is None else np.stack(transform(cat_xy[sel, 0], cat_xy[sel, 1]), 1)
        d, j = tree.query(src, distance_upper_bound=rad)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        transform = RG.PolyTransform.fit(cat_xy[sel][ok], det_xy[j[ok]], order, center, scale)
        px, py = transform(cat_xy[sel][ok, 0], cat_xy[sel][ok, 1])
        res = np.hypot(px - det_xy[j[ok], 0], py - det_xy[j[ok], 1])
        rms = float(1.4826 * np.median(res))
        rad = max(3 * rms, 0.8 * fwhm)
    return transform, rms


class LocalCorrection:
    """Remaining distortion as a smooth local offset: median residual of the k nearest matches."""

    def __init__(self, cat_xy, det_xy, k: int = 12):
        self.tree = cKDTree(cat_xy)
        self.res = det_xy - cat_xy
        self.k = min(k, len(cat_xy))

    def __call__(self, xy):
        _, j = self.tree.query(xy, k=self.k)
        return xy + np.median(self.res[j], axis=1)


def zero_point(det_flux, det_xy, cat_mag, shape, order: int = 2):
    """m_cat = ZP(x, y) − 2.5 log F; returns a callable and the scatter (mag)."""
    h, w = shape
    ok = (det_flux > 0) & np.isfinite(cat_mag) & (cat_mag > 8.5) & (cat_mag < 14.5)
    if ok.sum() < 20:
        return None, None
    u, v = (det_xy[ok, 0] - w / 2) / (w / 2), (det_xy[ok, 1] - h / 2) / (h / 2)
    T = RG._terms(u, v, order)
    yv = cat_mag[ok] + 2.5 * np.log10(det_flux[ok])
    keep = np.ones(len(yv), bool)
    for _ in range(5):
        c, *_ = np.linalg.lstsq(T[keep], yv[keep], rcond=None)
        res = yv - T @ c
        s = 1.4826 * np.median(np.abs(res[keep])) + 1e-3
        keep = np.abs(res) < 3 * s
    def zp(x, y):
        return RG._terms((np.asarray(x) - w / 2) / (w / 2), (np.asarray(y) - h / 2) / (h / 2), order) @ c
    return zp, float(s)


# ----------------------------------------------------------------------------- main

@dataclass
class TwilightResult:
    candidates: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    zp: object = None                 # callable ZP(x, y): m = ZP − 2.5 log F (F in frame σ units)
    aux: dict = field(default_factory=dict)   # for the side searches (sngal): median image, catalogue in px …


def run(cube, dts_min, wcs, fwhm: float, catalog: dict, dso: dict | None = None,
        known: list | None = None, planets: list | None = None, params: TwilightParams | None = None,
        sky_matrix=None, variables: dict | None = None, dither=None) -> TwilightResult:
    """``cube`` (n, h, w) registered frames in σ units; ``catalog`` from ``ucac4.query`` (field box);
    ``known`` [(name, x, y)] of known asteroids/comets; ``planets`` [(name, x, y)];
    ``sky_matrix`` 2×2 deg/px (ΔRA·cosδ, ΔDec) for rates; ``dither`` (n, 2) registration offsets of
    the frames (px) for the reflection / chip-fixed test."""
    P = params or TwilightParams()
    out = TwilightResult()
    n, h, w = cube.shape
    dts = np.asarray(dts_min, float)
    med = np.nanmedian(np.asarray(cube, np.float32), axis=0)
    sig = robust_sigma(med)
    img = med / sig
    det = detect(img, fwhm, P.nsigma, border=P.border_px)
    if len(det) < 30:
        out.log.append(f"Málo zdrojů v mediánu ({len(det)}) – porovnání s katalogem nelze udělat.")
        return out
    det_xy, det_flux = det[:, :2], det[:, 2] * sig            # flux back to the frame σ units
    cat_xy, cat_mag, cat_idx = project_catalog(catalog, wcs, (h, w))
    cat_j = (np.asarray(catalog["j_all"], float)[cat_idx] if "j_all" in catalog
             else np.full(len(cat_idx), np.nan))
    # a star is never > 1.5 mag fainter in J than in the optical: such a J is a placeholder / mismatch
    cat_j = np.where(cat_j - cat_mag > 1.5, np.nan, cat_j)
    cat_leda = (np.asarray(catalog["leda_all"])[cat_idx] if "leda_all" in catalog
                else np.zeros(len(cat_idx), np.int8))
    tr, rms = fit_distortion(det_xy, cat_xy, cat_mag, (h, w), fwhm, P.distortion_order)
    if tr is None:
        out.log.append("Nepodařilo se spárovat hvězdy s UCAC4 (WCS?).")
        return out
    cxy = np.stack(tr(cat_xy[:, 0], cat_xy[:, 1]), 1)
    # local correction from the matched stars (what the polynomial leaves, e.g. at the corners)
    dtree = cKDTree(det_xy)
    d0, j0 = dtree.query(cxy, distance_upper_bound=max(3.0, 1.2 * fwhm))
    m0 = np.isfinite(d0) & (cat_mag < 14.0)
    if m0.sum() >= 50:
        lc = LocalCorrection(cxy[m0], det_xy[j0[m0]])
        cxy = lc(cxy)
        d0, j0 = dtree.query(cxy, distance_upper_bound=max(3.0, 1.2 * fwhm))
        m0 = np.isfinite(d0) & (cat_mag < 14.0)
        rms = float(1.4826 * np.median(d0[m0])) if m0.any() else rms
    r_match = max(P.match_fwhm * fwhm, 3 * (rms or 0))
    ctree = cKDTree(cxy)
    d1, j1 = ctree.query(det_xy, k=1)
    matched = d1 <= r_match
    zp, zp_s = zero_point(det_flux[matched], det_xy[matched], cat_mag[j1[matched]], (h, w))
    if zp is None:
        out.log.append("Málo hvězd pro nulový bod.")
        return out
    out.zp = zp
    mag = zp(det_xy[:, 0], det_xy[:, 1]) - 2.5 * np.log10(np.clip(det_flux, 1e-6, None))
    # colour term of the unfiltered camera vs. the UCAC4 magnitude: m − m_cat = a + b (m_cat − J);
    # red giants are 2–4 mag brighter without a filter (317/5: ten false "brightenings" of 9–15 mag stars)
    col = cat_mag[j1] - cat_j[j1]
    sel_c = matched & np.isfinite(col) & (cat_mag[j1] > 8.5) & (cat_mag[j1] < 14.0)
    cterm = (0.0, 0.0)
    if sel_c.sum() >= 50:
        A = np.stack([np.ones(int(sel_c.sum())), col[sel_c]], 1)
        yv = mag[sel_c] - cat_mag[j1[sel_c]]
        keep = np.ones(len(yv), bool)
        for _ in range(5):
            cf, *_ = np.linalg.lstsq(A[keep], yv[keep], rcond=None)
            rr = yv - A @ cf
            keep = np.abs(rr) < 3 * (1.4826 * np.median(np.abs(rr[keep])) + 0.02)
        cterm = (float(cf[0]), float(cf[1]))

    def expected_mag(i):
        """Catalogue magnitude transformed to the frames' band (colour term where J is known)."""
        c_ = cat_mag[i] - cat_j[i]
        return cat_mag[i] + (cterm[0] + cterm[1] * c_ if np.isfinite(c_) else 0.0)
    # completeness of our own detections: faintest magnitude reached by 90 % of catalogue stars
    star_fw = np.array([moments_fwhm(img, x, y) or np.nan for x, y in det_xy[matched][:3000]])
    fw_star = float(np.nanmedian(star_fw)) if np.isfinite(star_fw).any() else fwhm
    # the star image changes over a wide field (coma / astigmatism towards the edges and corners):
    # local FWHM = median of the nearest well-exposed matched stars spread over the whole field
    mi_all = np.nonzero(matched & (cat_mag[j1] > 8.5) & (cat_mag[j1] < 13.0))[0]
    if len(mi_all) > 6000:
        mi_all = np.random.default_rng(1).choice(mi_all, 6000, replace=False)
    fw_pts = np.array([moments_fwhm(img, *det_xy[i]) or np.nan for i in mi_all]) if len(mi_all) else np.zeros(0)
    okf = np.isfinite(fw_pts)
    fw_tree = cKDTree(det_xy[mi_all[okf]]) if okf.sum() >= 50 else None
    fw_vals = fw_pts[okf]

    def fw_loc(x, y):
        """Local star FWHM (px) around (x, y); never sharper than the field median."""
        if fw_tree is None:
            return fw_star
        _, nn = fw_tree.query([x, y], k=min(25, len(fw_vals)))
        return max(fw_star, float(np.median(fw_vals[np.atleast_1d(nn)])))
    out.stats = {"n_frames": int(n), "span_min": float(dts.max() - dts.min()), "detections": int(len(det)),
                 "matched": int(matched.sum()), "catalog_in_field": int(len(cxy)),
                 "astrometry_rms_px": round(rms or 0, 3), "match_radius_px": round(r_match, 2),
                 "zp_scatter_mag": round(zp_s, 3), "star_fwhm_px": round(fw_star, 2),
                 "colour_term": [round(cterm[0], 3), round(cterm[1], 3)],
                 "limit_mag_detected": round(float(np.percentile(mag[matched], 95)), 2)}
    # how well the catalogue fits across the field: bright catalogue stars with a detection and the
    # star FWHM in the centre / middle / edge zone (radius normalised to the corner)
    try:
        inside = (cxy[:, 0] > P.border_px + 5) & (cxy[:, 1] > P.border_px + 5) & \
                 (cxy[:, 0] < w - P.border_px - 5) & (cxy[:, 1] < h - P.border_px - 5) & (cat_mag < 12.0)
        dd, _ = cKDTree(det_xy).query(cxy[inside])
        rz = np.hypot((cxy[inside, 0] - w / 2) / (w / 2), (cxy[inside, 1] - h / 2) / (h / 2)) / math.sqrt(2)
        zones = {}
        for nm, lo, hi in (("střed", 0, 0.5), ("mezikruží", 0.5, 0.8), ("okraj", 0.8, 1.01)):
            z = (rz >= lo) & (rz < hi)
            if z.sum() >= 10:
                pts = cxy[inside][z][:: max(1, int(z.sum() // 40))]
                zones[nm] = {"hvězd_do_12mag": int(z.sum()),
                             "spárováno_%": round(100.0 * float((dd[z] <= r_match).mean()), 1),
                             "fwhm_px": round(float(np.median([fw_loc(*q) for q in pts])), 2)}
        out.stats["field_zones"] = zones
    except Exception:                     # noqa: BLE001 – diagnostics only
        pass
    lim_det = out.stats["limit_mag_detected"]
    # bright stars for the halo test
    bright = np.nonzero(cat_mag <= P.halo_mag)[0]
    btree = cKDTree(cxy[bright]) if len(bright) else None
    known = known or []
    planets = planets or []
    sig_f = [robust_sigma(np.asarray(cube[i], np.float32)) for i in range(n)]
    cands = []
    try:
        fw_scale = ScaleWidth(img, fw_star, cxy, cat_mag, fw_loc)
    except Exception:                      # noqa: BLE001 – display only
        fw_scale = None

    def make(k, kind, m_cat=None, frame_vals=None, raw_frames_valid=False):
        x, y = float(det_xy[k, 0]), float(det_xy[k, 1])
        flags, status = [], "candidate"
        # per-frame presence and centroids
        meas = [forced(np.asarray(cube[i]), x, y, fw_star, sig_f[i]) for i in range(n)]
        okm = [(i, m) for i, m in enumerate(meas) if m is not None and m[0] >= 3.0
               and math.hypot(m[1] - x, m[2] - y) <= 1.5 * fw_star]
        frames_with = len(okm)
        if frame_vals is not None:
            # a source from the difference image: its presence is judged on the residual of every frame
            # (the raw frame at this place is mostly the neighbouring star)
            g_ = [v for v in frame_vals if v is not None]
            frames_with = sum(v >= 1.5 for v in g_)
            pos_ = sum(max(v, 0.0) for v in g_)
            if len(g_) < 2:
                flags.append("nelze ověřit po snímcích")
                status = "auto_rejected"
            elif n >= 5 and frames_with < math.ceil(0.7 * len(g_)):
                flags.append(f"jen v {frames_with}/{len(g_)} snímcích (zbytek po odečtení katalogu)")
                status = "auto_rejected"
            elif n >= 3 and pos_ > 0 and max(g_) > 0.5 * pos_:
                flags.append(f"hlavně v jednom snímku ({' / '.join(f'{v:.0f}' for v in g_)}σ) – kosmika / družice?")
                status = "auto_rejected"
            elif n <= 4 and max(g_) >= 10 and min(g_) < P.frame_ratio_min * max(g_):
                flags.append(f"jasnost nesouhlasí mezi snímky ({' / '.join(f'{v:.0f}' for v in g_)}σ) – "
                             f"kosmika / družice v jednom snímku?")
                status = "auto_rejected"
            if raw_frames_valid and status == "candidate":
                # the source outshines the catalogue star under it: the raw frames' centroids are its own
                # (325/1: a 2.0× wide "brightening" 11.5 mag on a 15.2 mag star, in 1 of 2 frames)
                if len(okm) < max(2, math.ceil(P.min_frames_frac * n)):
                    flags.append(f"jen v {len(okm)}/{n} snímcích")
                    status = "auto_rejected"
                elif n <= 4:
                    fs_raw = [m[0] for m in meas if m is not None]
                    if len(fs_raw) >= 2 and max(fs_raw) >= 10 and min(fs_raw) < P.frame_ratio_min * max(fs_raw):
                        flags.append(f"jasnost nesouhlasí mezi snímky ({' / '.join(f'{v:.0f}' for v in fs_raw)}σ) – "
                                     f"kosmika / družice v jednom snímku?")
                        status = "auto_rejected"
        elif frames_with < max(2, math.ceil(P.min_frames_frac * n)):
            flags.append(f"jen v {frames_with}/{n} snímcích")
            status = "auto_rejected"
        elif n <= 4:
            # with 2–4 frames "present (≥ 3σ) in every frame" is weak: a cosmic / satellite of 40σ in
            # one frame and 3σ of noise or a faint star in the other passed (325/1, 2 frames: 16 of 26
            # unknown point sources, ratios 0.07–0.2; injections ≥ 0.41). A real source shines in
            # every frame alike within the transparency changes
            fs = [m[0] for m in meas if m is not None]
            if len(fs) >= 2 and max(fs) >= 10 and min(fs) < P.frame_ratio_min * max(fs):
                flags.append(f"jasnost nesouhlasí mezi snímky ({' / '.join(f'{v:.0f}' for v in fs)}σ) – "
                             f"kosmika / družice v jednom snímku?")
                status = "auto_rejected"
        vx = vy = 0.0
        rate_sig = 0.0
        if len(okm) >= 3:
            t = np.array([dts[i] for i, _ in okm])
            xs = np.array([m[1] for _, m in okm])
            ys = np.array([m[2] for _, m in okm])
            if np.ptp(t) > 0:
                A = np.stack([np.ones_like(t), t - t.mean()], 1)
                bx, rx, *_ = np.linalg.lstsq(A, xs, rcond=None)
                by, ry, *_ = np.linalg.lstsq(A, ys, rcond=None)
                vx, vy = float(bx[1]), float(by[1])
                resid = np.concatenate([xs - A @ bx, ys - A @ by])
                s_pos = max(float(np.sqrt((resid ** 2).sum() / max(1, 2 * len(t) - 4))), 0.15)
                s_v = s_pos / math.sqrt(((t - t.mean()) ** 2).sum())
                rate_sig = math.hypot(vx, vy) / s_v
        if dither is not None and len(okm) >= 3:
            al, als = dither_test([dts[i] for i, _ in okm], [m[1] for _, m in okm], [m[2] for _, m in okm],
                                  [dither[i] for i, _ in okm])
            if al is not None and abs(al) > P.dither_alpha_max and abs(al) > 3 * als:
                flags.append(f"poloha sleduje dithering (α = {al:.2f}) – odlesk / vada čipu")
                status = "auto_rejected"
        fw_obj = moments_fwhm(img, x, y)
        fwl = fw_loc(x, y)
        fw_sc = fw_scale(x, y) if fw_scale is not None else None
        if frame_vals is not None:
            pass                                   # shape in the median image = mostly the neighbouring star
        elif fw_obj is not None and fw_obj < P.sharp_ratio * fw_star:
            flags.append("ostřejší než hvězdy")
            status = "auto_rejected"
        elif fw_obj is not None and fw_obj > P.wide_ratio * fwl:
            flags.append("rozmazaný (koma? galaxie?)")
        if btree is not None:
            for b in btree.query_ball_point([x, y], fw_star * (2 + 1.5 * (P.halo_mag - 4))):
                mb = cat_mag[bright[b]]
                rb = fw_star * (2 + 1.5 * (P.halo_mag - mb))
                if math.hypot(cxy[bright[b], 0] - x, cxy[bright[b], 1] - y) <= rb:
                    flags.append(f"u jasné hvězdy ({mb:.1f} mag)")
                    status = "auto_rejected"
                    break
        if x < P.border_px * 2 or y < P.border_px * 2 or x > w - 2 * P.border_px or y > h - 2 * P.border_px:
            flags.append("u okraje")
        ra, dec = (float(v) for v in wcs.all_pix2world([[x, y]], 0)[0])
        nm = dso_at(dso or {}, ra, dec, pad_arcsec=1.5 * fw_star * abs(_scale(wcs)) * 3600, skip_dark=True)
        if nm:
            flags.append(f"DSO {nm}")
            status = "auto_rejected"
        vn = variable_at(variables or {}, ra, dec, 1.5 * fw_star * abs(_scale(wcs)) * 3600)
        vmax = _var_max(vn) if vn else None
        vamp = _var_amplitude(vn) if vn else None
        if vn and vamp is not None and vamp < 0.3:
            vn = None                                  # micro-variable (Kepler/TESS): cannot make a new source
        # a known eruptive / large-amplitude variable far above its quiet level stays a candidate:
        # dwarf and recurrent novae are in VSX, their outbursts are what is worth seeing
        burst = outburst_note(vn, float(mag[k]), float(m_cat) if m_cat is not None else None) if vn else None
        if burst:
            flags.append(burst)
        elif vn and (vmax is None or mag[k] >= vmax - 1.0):
            flags.append(f"proměnná hvězda (VSX {vn.split(',')[0]})")
            status = "auto_rejected"
        elif vn:
            # a variable that never gets this bright: coincidence (VSX to 18 mag is dense) – 317/1:
            # an injected 8.4 mag comet on an 11.3 mag Kepler variable
            flags.append(f"u proměnné hvězdy {vn.split(' ')[0]} (max {vmax:.1f} mag)")
        kn = known_match(known, x, y, abs(_scale(wcs)) * 3600, P.known_radius_arcsec, float(mag[k]))
        for name, px, py in planets:
            if math.hypot(px - x, py - y) <= 10 * fw_star:
                flags.append(f"planeta {name}")
                status = "auto_rejected"
        rate = pa = None
        if sky_matrix is not None:
            M = np.asarray(sky_matrix(x, y) if callable(sky_matrix) else sky_matrix, float)
            de = M @ np.array([vx, vy])
            rate = float(math.hypot(*de) * 3600)
            pa = float(math.degrees(math.atan2(de[0], de[1])) % 360)
        if rate_sig >= 3:
            flags.append(f"pohyb {rate:.2f}″/min ({rate_sig:.0f}σ)" if rate is not None else f"pohyb ({rate_sig:.0f}σ)")
        c = {"x": x, "y": y, "vx": vx, "vy": vy, "ra": ra, "dec": dec, "mag": round(float(mag[k]), 2),
             "snr": round(float(det[k, 2] / (1.0 * math.sqrt(math.pi) * max(fw_star / 2.3548, 0.6) * 2)), 1),
             "kind": kind, "catalog_mag": None if m_cat is None else round(float(m_cat), 2),
             "nearest_catalog_px": round(float(d1[k]), 2), "frames_used": frames_with,
             "obj_fwhm": None if fw_obj is None else round(fw_obj, 2), "fwhm": round(fwl, 2),
             "obj_fwhm_scale": None if fw_sc is None else round(fw_sc, 2),
             "rate_arcsec_min": rate, "pa_deg": pa, "motion_sigma": round(rate_sig, 1),
             "frame_snr": ([None if v is None else round(v, 1) for v in frame_vals] if frame_vals is not None
                           else [None if m is None else round(m[0], 1) for m in meas]),
             "flags": flags, "auto_status": status, "known": {"name": kn} if kn else None}
        return c

    def _blend_mag(p, idx, fwl):
        """What the detection's matched filter collects from catalogue stars ``idx`` around it:
        each weighted by the overlap of two PSFs at its distance, exp(−d²/4σ²). The plain sum of
        every star within 2.5 FWHM counted ~17 Milky Way stars (UCAC4 0.09 per px² at 14″/px,
        CTA-N 324/4) as ~12.5 mag of light at any position – injected 12–13 mag novae and
        comets were all "explained" (2 of 60 found)."""
        sg2 = max(fwl / 2.3548, 0.6) ** 2
        f = 0.0
        for i in idx:
            d2 = float((cxy[i, 0] - p[0]) ** 2 + (cxy[i, 1] - p[1]) ** 2)
            f += math.exp(-d2 / (4.0 * sg2)) * 10 ** (-0.4 * expected_mag(i))
        return -2.5 * math.log10(f) if f > 0 else 99.0

    # a "match" with a catalogue star much fainter than the detection and off its position is no
    # match: at 14″/px the match radius is ~50″ and (43) Ariadne 13.4 mag 2.5 px from a 16.5 mag star
    # came out as a "brightening" (CTA-N 318, user report). A brightening stays centred on the star.
    off_faint = matched & (d1 > 0.5 * fw_star) & (cat_mag[j1] - mag > P.brighten_mag)
    matched = matched & ~off_faint
    # 1) sources without a catalogue star
    blend = np.array([len(b) for b in ctree.query_ball_point(det_xy, r_match)])
    # catalogue stars that already have their detection
    cat_taken = np.zeros(len(cxy), bool)
    cat_taken[j1[matched]] = True
    # local astrometric scatter: residuals of the 12 nearest matches (corners: coma, distortion)
    mi = np.nonzero(matched)[0]
    mtree = cKDTree(det_xy[mi])
    for k in np.nonzero(~matched & ((blend == 0) | off_faint) & (mag <= P.mag_limit))[0]:
        _, nn = mtree.query(det_xy[k], k=min(12, len(mi)))
        loc = float(1.4826 * np.median(d1[mi[np.atleast_1d(nn)]]))
        if not off_faint[k] and d1[k] <= max(r_match, 4 * loc):
            continue                                # matched within the local scatter
        # a catalogue star of the same brightness within 2.5 FWHM that has no detection of its own:
        # the same star, displaced (PSF at the field corners, local distortion)
        # Near the edges the star is a comet-like "arrow": centroid further off and the matched filter
        # (built for the round central PSF) measures it fainter – both scale with the local FWHM.
        jj = j1[k]
        fwl = fw_loc(*det_xy[k])
        dm = mag[k] - expected_mag(jj)
        faint_tol = 0.75 + min(1.5, 2.5 * max(0.0, fwl / fw_star - 1.0))
        if (not cat_taken[jj] and d1[k] <= 2.5 * fwl and -0.75 < dm < faint_tol):
            continue
        # an unresolved group of fainter catalogue stars (Milky Way): their summed light explains it
        grp = [i for i in ctree.query_ball_point(det_xy[k], 2.5 * fwl) if not cat_taken[i]]
        if grp:
            m_sum = _blend_mag(det_xy[k], grp, fwl)
            if m_sum <= mag[k] + 0.75:
                continue
        cn = make(k, "navíc")
        if cn["obj_fwhm"] is not None and cn["obj_fwhm"] <= P.wide_ratio * fwl:
            sc_ = cn.get("obj_fwhm_scale")
            if sc_ is not None and sc_ >= P.coma_scale_ratio * fwl:
                # the small-box moments miss a coma (its light is the box "background"); the two-scale
                # width sees it (325/3: injected comets 25–110″ → 1.6–2.4 × the stars, point sources ≤ 1.35 ×)
                cn["flags"].insert(0, f"nový rozmazaný zdroj – kometa? (šířka {sc_ / fwl:.1f}× hvězdy; "
                                      f"nebo galaxie mimo katalog)")
            else:
                cn["flags"].insert(0, "nový bodový zdroj (planetka? kometa v outburstu? nova?)")
        elif (cn["obj_fwhm"] is not None and mag[k] > lim_det - P.faint_extended_mag
              and cn["motion_sigma"] < 5):
            # faint, extended and still: at high galactic latitude mostly galaxies smaller than the 1′
            # of the DSO catalogues (CTA-N 318: 33 per field at 14.4–15.3 mag, limit 14.9)
            cn["flags"].append(f"slabý rozmazaný u meze ({lim_det:.1f} mag) – nejspíš galaxie mimo katalog")
            cn["auto_status"] = "auto_rejected"
        cands.append(cn)
    # 2) catalogue stars measured much brighter
    for k in np.nonzero(matched)[0]:
        nb = ctree.query_ball_point(det_xy[k], max(r_match, 2.5 * fw_star))
        # summed expected flux of every catalogue star in the aperture (blends)
        mc = _blend_mag(det_xy[k], nb, fw_loc(*det_xy[k])) if nb else expected_mag(j1[k])
        if cat_mag[j1[k]] > 9.0 and mag[k] < mc - P.brighten_mag:
            c = make(k, "zjasnění", mc)
            if c["motion_sigma"] >= 5:
                # it moves: an object passing a faint star, not the star brightening
                c["kind"] = "navíc"
                c["flags"].insert(0, f"pohybuje se u slabé hvězdy UCAC4 {cat_mag[j1[k]]:.1f} mag")
                cands.append(c)
                continue
            c["flags"].insert(0, f"zjasnění: katalog {mc:.1f} (UCAC4 {cat_mag[j1[k]]:.1f}"
                                 + (f", J {cat_j[j1[k]]:.1f}" if np.isfinite(cat_j[j1[k]]) else "")
                                 + f"), změřeno {mag[k]:.1f} mag")
            jk = cat_j[j1[k]]
            if np.isfinite(jk) and mag[k] > jk - 0.5:
                # no brighter than the star in 2MASS J: a red star seen in the near-IR by the
                # unfiltered camera, not a new light (317/4 and 317/5, user-checked: red stars,
                # centred on the catalogue position 0.09–0.15 FWHM). An object *on* the star pulls
                # the centroid away (injection on a 15.6 mag star, 317/1: 0.43 FWHM) – kept.
                if d1[k] <= 0.35 * fw_star:
                    c["flags"].append(f"jasnost jako v 2MASS J ({jk:.1f} mag) – červená hvězda")
                    c["auto_status"] = "auto_rejected"
                else:
                    c["flags"].append(f"posun {d1[k]:.1f} px od hvězdy – objekt na hvězdě?")
            rj = cat_mag[j1[k]] - cat_j[j1[k]]
            if np.isfinite(rj) and rj >= 3.0:
                # extremely red (carbon stars, Miras): the unfiltered camera sees their near-IR light,
                # 0.3–0.55 × (UCAC4 − J) brighter (317/5: seven such stars, UCAC4 − J 5.4–8.5)
                c["flags"].append(f"velmi červená hvězda (UCAC4 − J = {rj:.1f})")
                c["auto_status"] = "auto_rejected"
            if any(cat_leda[i] for i in nb):
                c["flags"].append("galaxie (LEDA v UCAC4)")
                c["auto_status"] = "auto_rejected"
            if (c["obj_fwhm"] is not None and c["obj_fwhm"] > P.wide_ratio * fw_loc(*det_xy[k])
                    and mag[k] > lim_det - P.faint_extended_mag and c["auto_status"] == "candidate"):
                c["flags"].append(f"slabý rozmazaný u meze ({lim_det:.1f} mag) – nejspíš galaxie mimo katalog")
                c["auto_status"] = "auto_rejected"
            cands.append(c)
    # 3) difference image against the catalogue: sources that merged with a catalogue star or hid in a
    #    clump of faint ones (327-2/3, Milky Way: 3.4 UCAC4 stars within the match radius of any point;
    #    injections of SNR 16–25 found 16/25 by the matching, 25/25 in the difference image)
    if P.catdiff and len(cxy) >= 200:
        try:
            from . import catdiff as CDF
            dcol = cat_mag - cat_j
            em_all = cat_mag + np.where(np.isfinite(dcol), cterm[0] + cterm[1] * dcol, 0.0)
            ef_all = 10 ** (0.4 * (np.asarray(zp(cxy[:, 0], cxy[:, 1]), float) - em_all))
            cdf = CDF.CatalogueDiff(img, cxy, ef_all, cat_mag, fw_star, bright_mask_mag=P.catdiff_bright_mag)
            scales = cdf.frame_scales(cube)
            pk = cdf.peaks(P.catdiff_nsigma, P.border_px)
            flux_full = np.zeros(len(cxy))
            flux_full[cdf.idx] = cdf.flux
            sgs = max(fw_star / 2.3548, 0.6)
            ac = cdf.psf_ac
            acc_ = ac.shape[0] // 2
            n_br = n_small = n_new = n_skip = 0
            stars_done = set()
            for x_, y_, z_ in pk:
                near_c = [c_ for c_ in cands if math.hypot(c_["x"] - x_, c_["y"] - y_) <= 1.5 * fw_star]
                if any(c_["auto_status"] == "candidate" for c_ in near_c):
                    n_skip += 1
                    continue                       # the matching already has it
                nb_ = ctree.query_ball_point([x_, y_], 2.0 * fw_star)
                jd, dd = None, None
                if nb_:
                    wgt = [flux_full[j_] * math.exp(-((cxy[j_, 0] - x_) ** 2 + (cxy[j_, 1] - y_) ** 2) / (4 * sgs * sgs))
                           for j_ in nb_]
                    jd = nb_[int(np.argmax(wgt))]
                    dd = math.hypot(cxy[jd, 0] - x_, cxy[jd, 1] - y_)
                    if flux_full[jd] <= 0:
                        jd = None
                zi = lambda a_, b_: float(cdf.z[int(min(max(round(b_), 0), h - 1)), int(min(max(round(a_), 0), w - 1))])
                is_star = False
                if jd is not None:
                    sx_, sy_ = cxy[jd]
                    mir, cen = zi(2 * sx_ - x_, 2 * sy_ - y_), zi(sx_, sy_)
                    # the excess in flux (not in σ: the model error is largest right on a bright star, so the
                    # σ-map peaks on its flank – 327-2/1 #896, a variable 2.7× brighter than modelled, came
                    # out as "a new source" 4 px beside it). A new source beside a star leaves at the star's
                    # centre only its own wing (≈ 0.1–0.4 of its peak at 0.5–1.2 FWHM).
                    Sp = float(cdf.S[int(round(y_)), int(round(x_))])
                    Sc = float(cdf.S[int(round(sy_)), int(round(sx_))])
                    is_star = (dd < 0.3 * fw_star or (mir > 0.35 * z_ and cen > 0.5 * z_)
                               or (Sp > 0 and Sc >= 0.6 * Sp and dd <= 2.0 * fw_star))
                    if is_star:
                        if jd in stars_done:
                            continue               # this star already has its candidate
                        stars_done.add(jd)
                        x_, y_ = float(sx_), float(sy_)   # the brightening is the star's
                if is_star:
                    z_ = max(z_, zi(x_, y_))
                    near_c = [c_ for c_ in cands if math.hypot(c_["x"] - x_, c_["y"] - y_) <= 1.5 * fw_star]
                    if any(c_["auto_status"] == "candidate" for c_ in near_c):
                        n_skip += 1
                        continue
                fv = [cdf.frame_snr(cube[i], scales[i], x_, y_) for i in range(n)]
                if is_star:
                    # the catalogue star is brighter: by how much (excess at its centre / its own light there)
                    sxi, syi = int(round(cxy[jd, 0])), int(round(cxy[jd, 1]))
                    own = flux_full[jd] * float(cdf.kmap[syi, sxi]) * float(ac[acc_, acc_])
                    exc = float(cdf.S[syi, sxi]) / max(own, 1e-12)
                    dm = 2.5 * math.log10(max(1.0 + exc, 1e-6))
                    if dm < P.brighten_mag:
                        n_small += 1               # colour terms / catalogue errors (clear filter vs r)
                        continue
                    m_c = float(em_all[jd])
                    m_meas = m_c - dm
                    kind_, d_ = "zjasnění", dd
                else:
                    fx = cdf.flux_at(x_, y_)
                    if fx <= 0:
                        continue
                    m_meas = float(zp(x_, y_)) - 2.5 * math.log10(fx)
                    if m_meas > P.mag_limit:
                        continue
                    kind_, d_, m_c = "navíc", (dd if dd is not None else 99.0), None
                det_xy = np.vstack([det_xy, [[x_, y_]]])
                det = np.vstack([det, [[x_, y_, z_ * 2.0 * math.sqrt(math.pi) * sgs]]])
                mag = np.append(mag, m_meas)
                d1 = np.append(d1, d_)
                k_ = len(mag) - 1
                m_nb = float(em_all[jd]) if jd is not None else 99.0
                c = make(k_, kind_, m_c, frame_vals=fv, raw_frames_valid=(n <= 4 and m_meas < m_nb - 2.0))
                if kind_ == "zjasnění":
                    c["flags"].insert(0, f"zjasnění: katalog {m_c:.1f} (UCAC4 {cat_mag[jd]:.1f}"
                                         + (f", J {cat_j[jd]:.1f}" if np.isfinite(cat_j[jd]) else "")
                                         + f"), změřeno {m_meas:.1f} mag – v rozdílu proti katalogu")
                    jk = cat_j[jd]
                    if np.isfinite(jk) and m_meas > jk - 0.5 and dd <= 0.35 * fw_star:
                        c["flags"].append(f"jasnost jako v 2MASS J ({jk:.1f} mag) – červená hvězda")
                        c["auto_status"] = "auto_rejected"
                    rj = cat_mag[jd] - cat_j[jd]
                    if np.isfinite(rj) and rj >= 3.0:
                        c["flags"].append(f"velmi červená hvězda (UCAC4 − J = {rj:.1f})")
                        c["auto_status"] = "auto_rejected"
                    if cat_leda[jd]:
                        c["flags"].append("galaxie (LEDA v UCAC4)")
                        c["auto_status"] = "auto_rejected"
                    n_br += 1
                else:
                    nm_ = (f" vedle hvězdy UCAC4 {cat_mag[jd]:.1f} mag ({dd:.1f} px)" if jd is not None else "")
                    c["flags"].insert(0, f"nový bodový zdroj{nm_} – v rozdílu proti katalogu (planetka? kometa? nova?)")
                    n_new += 1
                c["catdiff"] = True
                if c["auto_status"] == "candidate":
                    for c_ in near_c:              # a rejected match there is this source, explained away
                        if c_ in cands:
                            cands.remove(c_)
                cands.append(c)
            out.stats["catdiff"] = {"peaks": int(len(pk)), "eps": round(cdf.eps, 3), "confusion": round(cdf.conf, 2),
                                    "brightenings": n_br, "small_brightenings": n_small, "new": n_new,
                                    "already_found": n_skip}
            out.log.append(f"Rozdíl proti katalogu: {len(pk)} přebytků nad {P.catdiff_nsigma:g}σ (chyba modelu hvězd "
                           f"{100 * cdf.eps:.0f} %, šum zahuštění {cdf.conf:.2f}×); {n_skip} už nalezeno, {n_small} hvězd "
                           f"jen o < {P.brighten_mag:g} mag jasnějších (barva / katalog), kandidátů: {n_new} nových zdrojů, "
                           f"{n_br} zjasnění.")
            del cdf
        except Exception as e:             # noqa: BLE001 – a side search must not break the comparison
            import traceback as _tb
            out.log.append(f"Rozdíl proti katalogu selhal ({e}): " + " | ".join(_tb.format_exc().splitlines()[-3:]))
    if P.diffuse:
        cands += _diffuse_candidates(cube, img, sig, det_xy, cxy, cat_mag, fw_star, dts, wcs, zp, P, dso, variables,
                                     known, planets, sky_matrix, dither, btree, bright, cands, out)
    if n <= 4:
        _few_frames_checks(cands, n, P)
    try:
        _lc = locals().get("lc")

        def _to_pix(ra_, de_):
            xy = _world2pix_sip(wcs, ra_, de_, (h, w))
            xy = np.stack(tr(xy[:, 0], xy[:, 1]), 1)
            if _lc is not None:
                ok_ = np.isfinite(xy).all(axis=1)
                if ok_.any():
                    xy[ok_] = _lc(xy[ok_])
            return xy[:, 0], xy[:, 1]
        out.aux = {"img": img, "sig": sig, "cxy": cxy, "cat_mag": cat_mag, "sig_f": sig_f, "fw_star": fw_star,
                   "exp_flux": (10 ** (0.4 * (np.asarray(zp(cxy[:, 0], cxy[:, 1]), float)
                                           - (cat_mag + np.where(np.isfinite(cat_mag - cat_j),
                                                                 cterm[0] + cterm[1] * (cat_mag - cat_j), 0.0)))))
                               if len(cxy) else np.zeros(0),
                   "to_pix": _to_pix, "lim": lim_det}
    except Exception:                      # noqa: BLE001 – side searches only
        out.aux = {}
    cands.sort(key=lambda c: (c["auto_status"] != "candidate", c["mag"]))
    out.candidates = cands[:P.max_candidates]
    out.stats["candidates"] = int(sum(c["auto_status"] == "candidate" for c in cands))
    out.stats["rejected"] = int(sum(c["auto_status"] != "candidate" for c in cands))
    return out


def _few_frames_checks(cands: list, n: int, P) -> None:
    """2–4 frames (no median rejection, no time thirds): the per-frame evidence must carry it.

    325 (CTA-N, 2 frames per field, 16. 11.), passed unknowns vs passed injections:
      point sources – the weakest frame: injections ≥ 4.1σ, unknowns mostly 0.7–3.9σ (coincidences
      of noise peaks and faint stars at ~15 000 detections per frame);
      glows – Σ per-frame value at the position / √n vs the stack significance: the injection 0.82,
      unknowns 0.02–0.4 (large structure of the background seen only in the median stack: the halo
      of Spica, not in UCAC4)."""
    for c in cands:
        if c.get("auto_status") != "candidate" or c.get("known"):
            continue
        fl = c.setdefault("flags", [])
        if c.get("kind") == "mlhavý":
            fx = c.get("frame_snr_fixed")
            if fx:
                r = (sum(fx) / math.sqrt(len(fx))) / max(float(c.get("snr") or 0), 1e-6)
                if r < P.few_glow_ratio_min:
                    fl.append(f"mlhavý jen ve stacku (v jednotlivých snímcích {r:.2f} × očekávaného) – struktura pozadí?")
                    c["auto_status"] = "auto_rejected"
            continue
        fs = [v for v in (c.get("frame_snr") or []) if v is not None]
        if len(fs) >= 2 and min(fs) < P.few_frame_snr_min:
            fl.append(f"slabý v některém snímku ({' / '.join(f'{v:.1f}' for v in fs)}σ; při {n} snímcích "
                      f"nutno ≥ {P.few_frame_snr_min:g}σ v každém)")
            c["auto_status"] = "auto_rejected"


HARD_PREFIX = ("u jasné hvězdy", "DSO ", "proměnná hvězda", "ostřejší", "poloha sleduje", "planeta",
               "jasnost jako v 2MASS", "velmi červená", "galaxie (LEDA", "jen v ", "galaxie PGC")


def _diffuse_candidates(cube, img, sig, det_xy, cxy, cat_mag, fw_star, dts, wcs, zp, P, dso, variables, known,
                        planets, sky_matrix, dither, btree, bright, point_cands, out):
    """Diffuse sources (comet comae) that remain after masking every star and point source."""
    n, h, w = cube.shape
    wings = wing_radii(img, cxy, cat_mag, fw_star)
    if wings is not None:
        out.stats["star_wing_radius_px"] = {f"{m:.1f}": round(float(r), 1) for m, r in zip(*wings)}
    good = star_mask((h, w), cxy, cat_mag, fw_star, extra_xy=det_xy, wings=wings) & np.isfinite(img)
    # near the data edge the background (low-pass σ 40 px) is extrapolated and the dithered frames
    # end at different columns: 8σ "glows" of empty sky there (319-1/2 #612, x = 56, dither 29 px)
    cov = np.all(np.isfinite(cube), axis=0) if cube.shape[0] <= 64 else np.isfinite(img)
    edge_px = 40 + 3 * max(P.diffuse_scales)
    good &= ndimage.binary_erosion(cov, iterations=1, border_value=0) & \
        (ndimage.distance_transform_edt(np.pad(cov, 1, constant_values=False))[1:-1, 1:-1] > edge_px)
    out.stats["diffuse_masked_frac"] = round(float(1 - good.mean()), 3)
    ctree_d = cKDTree(cxy)
    neg = {}
    rows = detect_diffuse(img, good, P.diffuse_scales, P.diffuse_nsigma, P.border_px, neg_stats=neg,
                          noise_norm=P.diffuse_noise_norm)
    out.stats["diffuse_threshold"] = neg
    out.stats["diffuse_peaks"] = len(rows)
    scale_as = abs(_scale(wcs)) * 3600
    # for the reported position: only catalogue stars masked. The point-source mask also takes the
    # brightest part of a coma (its centre is detected as a point) and the peak of the smoothed
    # map then lands on the coma's edge (CTA-N 322 #394: injected comet, 12.5 px off)
    good_cat = (star_mask((h, w), cxy, cat_mag, fw_star, wings=wings) & np.isfinite(img)) if len(rows) else None
    bgsub = (img - _lowpass(img, good, 40.0)) if len(rows) else None
    res = []
    for x, y, sc, zsig, flux, sc_min in rows[:2000]:
        xr, yr = diffuse_centroid(bgsub, good_cat, x, y, sc)
        near = [c for c in point_cands if math.hypot(c["x"] - x, c["y"] - y) < max(2 * sc, 2 * fw_star)]
        flags, status = [], "candidate"
        # per frame at the same scale (local cut-outs, the same star mask)
        dam0 = float(np.max(np.abs(np.asarray(dither, float) - np.mean(dither, axis=0)))) if dither is not None else 0.0
        R = int(4 * sc + 12 + dam0)
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - R), min(h, yi + R + 1), max(0, xi - R), min(w, xi + R + 1)
        gm = good[y0:y1, x0:x1]
        vals, cxs, cys, used = [], [], [], []
        zmaps = []
        cy0, cx0 = yi - y0, xi - x0
        for i in range(n):
            cut = np.asarray(cube[i, y0:y1, x0:x1], np.float32)
            ok = gm & np.isfinite(cut)
            cut = cut - float(np.nanmedian(np.where(ok, cut, np.nan))) if ok.any() else cut
            Sm = _normconv(np.nan_to_num(cut), ok, sc)
            if P.diffuse_noise_norm:
                Sm = Sm / _normconv_noise(ok, sc)
            nz = robust_sigma(Sm)
            zmaps.append(Sm / nz)
            # window wide enough to follow a reflection that moves with the dither
            dam = float(np.max(np.abs(np.asarray(dither, float) - np.mean(dither, axis=0)))) if dither is not None else 0.0
            r2 = int(max(2, 1.5 * sc, min(dam + 2, R - 2)))
            win = np.nan_to_num(Sm[max(0, cy0 - r2):cy0 + r2 + 1, max(0, cx0 - r2):cx0 + r2 + 1], nan=-1e9)
            if not win.size:
                vals.append(0.0)
                continue
            j = int(np.argmax(win))
            py, px = divmod(j, win.shape[1])
            vals.append(float(win[py, px] / nz))
            # centroid in the smoothed map (sub-pixel by a parabola)
            ay, ax = max(0, cy0 - r2) + py, max(0, cx0 - r2) + px
            def par(a, b, c):
                d = a - 2 * b + c
                return 0.5 * (a - c) / d if d < 0 else 0.0
            if 0 < ay < Sm.shape[0] - 1 and 0 < ax < Sm.shape[1] - 1 and np.isfinite(Sm[ay - 1:ay + 2, ax - 1:ax + 2]).all():
                cxs.append(x0 + ax + par(Sm[ay, ax - 1], Sm[ay, ax], Sm[ay, ax + 1]))
                cys.append(y0 + ay + par(Sm[ay - 1, ax], Sm[ay, ax], Sm[ay + 1, ax]))
                used.append(i)
        thirds = None
        fixed = None
        if P.diffuse_frame_test == "thirds":
            fixed = _diffuse_fixed_values(zmaps, dts, cx0, cy0, sc, cxs, cys, used, x0, y0)
            frames_with = int(sum(v >= P.diffuse_frame_snr for v in fixed))
            thirds = _thirds(fixed, dts, P.diffuse_third_frac)
            frames_ok = thirds[0]
        else:
            frames_with = int(sum(v >= P.diffuse_frame_snr for v in vals))
            frames_ok = frames_with >= max(2, math.ceil(P.min_frames_frac * n))
        if near:
            # already reported as a point source (nucleus) – the coma joins it. A diffuse comet has a
            # weak nucleus: per frame it may fail the point test but pass at the coma's scale
            c0 = near[0]
            c0["flags"].append(f"mlhavé okolí σ {sc:.0f} px ({zsig:.0f}σ)")
            # a diffuse glow 1–3′ from a bright star is its halo / ghost (317/1, 317/3, user-checked:
            # nothing on DSS, 2.1–2.5′ from 6.5–7.6 mag stars) – the wider diffuse halo radius applies
            if btree is not None:
                hit = None
                for b in btree.query_ball_point([x, y], 3 * fw_star * (2 + 1.5 * (P.halo_mag - 4)) + 3 * sc_min):
                    mb = cat_mag[bright[b]]
                    if math.hypot(cxy[bright[b], 0] - x, cxy[bright[b], 1] - y) <= 2 * fw_star * (2 + 1.5 * (P.halo_mag - mb)) + 3 * sc_min:
                        hit = mb
                        break
                if hit is not None:
                    c0["flags"].append(f"mlhavé okolí u jasné hvězdy ({hit:.1f} mag) – halo / odlesk")
                    c0["auto_status"] = "auto_rejected"
                    continue
            if dither is not None and len(used) >= 3:
                # the coma's centroid is the better probe for a large reflection than the "nucleus"
                al, als = dither_test([dts[i] for i in used], cxs, cys, [dither[i] for i in used])
                if al is not None and abs(al) > P.dither_alpha_max and abs(al) > 3 * als:
                    c0["flags"].append(f"poloha sleduje dithering (α = {al:.2f}) – odlesk")
                    c0["auto_status"] = "auto_rejected"
                    continue
            if frames_ok:
                fl0 = [f for f in c0["flags"] if not f.startswith("jen v ")]
                if len(fl0) < len(c0["flags"]):
                    c0["flags"] = fl0 + [f"v {frames_with}/{n} snímcích jako mlhavý"]
                    if not any(f.startswith(HARD_PREFIX) for f in fl0):
                        c0["auto_status"] = "candidate"
            continue
        if not frames_ok:
            if thirds is not None:
                flags.append("signál jen v části série (třetiny " + " / ".join(f"{t:.1f}" for t in thirds[1])
                             + f"σ, čekáno ~{thirds[2]:.1f}σ)")
            else:
                flags.append(f"jen v {frames_with}/{n} snímcích")
            status = "auto_rejected"
        # a group of catalogue stars much brighter than the glow: their summed wings (CTA-N 318: glows
        # of 13.5–14 mag among 12–13 mag stars 2–8 px away)
        mag_d = float(zp(x, y)) - 2.5 * math.log10(max(flux * sig, 1e-6))
        nb = ctree_d.query_ball_point([x, y], 1.5 * sc + 3)
        if nb:
            m_sum = -2.5 * math.log10(sum(10 ** (-0.4 * cat_mag[i]) for i in nb))
            # the masks cut the cores, but the σ-scale smoothing still collects their wings: a glow no
            # brighter than the stars around it is their leftover light (CTA-N 318/5: 13.2–13.9 mag
            # glows with 2–4 stars of 13–14.5 mag at 3–7 px, nothing on the image itself)
            if m_sum < mag_d - 1.0 or (m_sum < mag_d + P.diffuse_star_sum_mag and sc <= 2 * fw_star + 1):
                flags.append(f"mezi hvězdami (součet {m_sum:.1f} mag) – křídla hvězd")
                status = "auto_rejected"
        if dither is not None and len(used) >= 3:
            al, als = dither_test([dts[i] for i in used], cxs, cys, [dither[i] for i in used])
            if al is not None and abs(al) > P.dither_alpha_max and abs(al) > 3 * als:
                flags.append(f"poloha sleduje dithering (α = {al:.2f}) – odlesk")
                status = "auto_rejected"
        if btree is not None:
            for b in btree.query_ball_point([x, y], 3 * fw_star * (2 + 1.5 * (P.halo_mag - 4)) + 3 * sc_min):
                mb = cat_mag[bright[b]]
                rb = 2 * fw_star * (2 + 1.5 * (P.halo_mag - mb)) + 3 * sc_min
                if math.hypot(cxy[bright[b], 0] - x, cxy[bright[b], 1] - y) <= rb:
                    flags.append(f"u jasné hvězdy ({mb:.1f} mag)")
                    status = "auto_rejected"
                    break
        shift = math.hypot(xr - x, yr - y)
        x, y = xr, yr
        ra, dec = (float(v) for v in wcs.all_pix2world([[x, y]], 0)[0])
        nm = dso_at(dso or {}, ra, dec, pad_arcsec=2.5 * sc * scale_as)
        if nm:
            flags.append(f"DSO {nm}")
            status = "auto_rejected"
        for name, px_, py_ in (planets or []):
            if math.hypot(px_ - x, py_ - y) <= 10 * fw_star + 3 * sc:
                flags.append(f"planeta {name}")
                status = "auto_rejected"
        mag = float(zp(x, y)) - 2.5 * math.log10(max(flux * sig, 1e-6))
        # a glow can be a comet's coma (wide radius); an asteroid is a point – the usual radius
        kn = known_match(known, x, y, scale_as, P.known_radius_arcsec, mag,
                         wide_arcsec=3 * sc * scale_as)
        flags.insert(0, f"mlhavý objekt (σ {sc:.0f} px ≈ {2.35 * sc * scale_as:.0f}″, {zsig:.0f}σ)")
        res.append({"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "ra": ra, "dec": dec, "mag": round(mag, 2),
                    "snr": round(zsig, 1), "kind": "mlhavý", "catalog_mag": None, "nearest_catalog_px": None,
                    "frames_used": frames_with, "obj_fwhm": round(2.3548 * sc, 1), "fwhm": round(fw_star, 2),
                    "rate_arcsec_min": None, "pa_deg": None, "motion_sigma": 0.0,
                    "frame_snr": [round(v, 1) for v in vals], "flags": flags, "auto_status": status,
                    "frame_snr_fixed": None if fixed is None else [round(v, 2) for v in fixed],
                    "known": {"name": kn} if kn else None, "diffuse_scale_px": sc,
                    "diffuse_shift_px": round(shift, 1)})
    return res


def _scale(wcs) -> float:
    """Pixel scale in degrees (mean of the CD matrix)."""
    try:
        return float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))))
    except Exception:          # noqa: BLE001
        return 1 / 3600


# ----------------------------------------------------------------------------- pipeline glue

def dss_source_at(gray: np.ndarray, radius_px: float, nsigma: float = 5.0, smooth_px: float = 1.0) -> tuple[bool, float]:
    """Is there a source at the centre of a survey cut-out (2-D array, any monotonic scale)?
    Peak of the lightly smoothed image within ``radius_px`` of the centre vs. the median and MAD of
    the outer part. Returns (found, significance)."""
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g.mean(axis=2)
    h, w = g.shape
    sm = ndimage.gaussian_filter(g, max(1.0, smooth_px))
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    outer = sm[rr > 0.35 * min(h, w)]
    med = float(np.median(outer))
    mad = float(1.4826 * np.median(np.abs(outer - med))) or 1.0
    peak = float(sm[rr <= max(radius_px, 1.5)].max())
    z = (peak - med) / mad
    return z >= nsigma, round(z, 1)


def dss_mag_at_centre(gray: np.ndarray, ra0: float, dec0: float, fov_arcmin: float, cat: dict | None,
                      radius_px: float) -> tuple[bool, float, float | None]:
    """Source at the centre of a DSS cut-out and its rough magnitude from the UCAC4 stars in the
    same cut-out (north up, east left): (found, significance, estimated mag or None)."""
    found, z = dss_source_at(gray, radius_px)
    if not found or not cat:
        return found, z, None
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g.mean(axis=2)
    h, w = g.shape
    px = fov_arcmin * 60.0 / w
    cd = math.cos(math.radians(dec0))
    ra, dec, mg = (np.asarray(cat[k], float) for k in ("ra_all", "dec_all", "mag_all"))
    dx = -((ra - ra0 + 540) % 360 - 180) * cd * 3600 / px          # east left
    dy = -(dec - dec0) * 3600 / px                                  # north up (row 0 = north)
    cx, cy = (w - 1) / 2 + dx, (h - 1) / 2 + dy
    sel = (cx > 6) & (cy > 6) & (cx < w - 7) & (cy < h - 7) & (np.hypot(dx, dy) > 3 * radius_px) & (mg > 10) & (mg < 18)
    sm = ndimage.gaussian_filter(g, 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    outer = sm[rr > 0.35 * min(h, w)]
    med = float(np.median(outer))
    mad = float(1.4826 * np.median(np.abs(outer - med))) or 1.0
    zs, ms = [], []
    for x, y, m in zip(cx[sel], cy[sel], mg[sel]):
        xi, yi = int(round(x)), int(round(y))
        zs.append((float(sm[yi - 2:yi + 3, xi - 2:xi + 3].max()) - med) / mad)
        ms.append(m)
    if len(zs) < 3:
        return found, z, None
    zs, ms = np.array(zs), np.array(ms)
    # the catalogue star whose DSS peak is closest to the central one (saturating plates: use rank order)
    k = np.argsort(np.abs(zs - z))[:3]
    return found, z, float(np.median(ms[k]))


def dss_check(cands: list, fwhm_arcsec: float, fov_arcmin: float = 6.0, max_n: int = 120, log=None,
              cat: dict | None = None) -> int:
    """Online: compare the candidates with DSS2 red (the plates are decades old – an asteroid, a comet
    or a nova is not there, a galaxy or a faint star is). A source on DSS at the position → rejected;
    nothing there → a rejection only for "faint extended" is lifted. Returns the number checked."""
    from . import sky as SKY
    from PIL import Image
    import io
    done = 0
    todo = [c for c in cands if not any(f.startswith("galaxie PGC") for f in c.get("flags", [])) and
            (c.get("auto_status") == "candidate" or
             any(f.startswith("slabý rozmazaný u meze") for f in c.get("flags", [])))]
    # passed ones first (the difference image adds its candidates at the end of the list)
    todo.sort(key=lambda c: c.get("auto_status") != "candidate")
    for c in todo[:max_n]:
        try:
            data, ct, src = SKY.fetch(c["ra"], c["dec"], fov_arcmin / 60.0, "DSS2r", pixels=240, timeout=6.0,
                                     gamma=1.0)
            g = np.asarray(Image.open(io.BytesIO(data)).convert("L"), float)
        except Exception as e:     # noqa: BLE001 – offline / service down: no change
            if log:
                log(f"Zdroj navíc: kontrola proti DSS nedostupná ({e}) – přeskočena.")
            return done
        px_as = fov_arcmin * 60.0 / g.shape[1]
        done += 1
        sc = float(c.get("diffuse_scale_px") or 0.0)
        if sc > 0 and c.get("fwhm"):
            # a glow: look for it at its own scale (σ 1 px of a 1.5″/px cut-out cannot see a 45″
            # glow – "na DSS nic není" was meaningless for "mlhavý"); no star calibration
            sc_as = sc * fwhm_arcsec / float(c["fwhm"])
            found, z = dss_source_at(g, max(8.0, sc_as) / px_as, smooth_px=0.7 * sc_as / px_as)
            if found:
                c["flags"].append(f"na DSS je mlhavý zdroj ({z:.0f}σ) – galaxie / mlhovina")
                c["auto_status"] = "auto_rejected"
            else:
                c["flags"].append("na DSS nic není")
            continue
        found, z, m_dss = dss_mag_at_centre(g, c["ra"], c["dec"], fov_arcmin, cat, max(8.0, 0.5 * fwhm_arcsec) / px_as)
        # only a DSS source bright enough to be the detection counts: a 19–20 mag star or galaxy
        # within the (large) radius of a 14″/px camera must not reject a 14 mag asteroid
        if found and m_dss is not None and m_dss > c["mag"] + 1.5:
            c["flags"].append(f"na DSS jen slabý zdroj (~{m_dss:.0f} mag)")
            found = False
        elif found and m_dss is None and z < 30:
            # too few catalogue stars to calibrate the cut-out: only an obvious source counts
            c["flags"].append(f"na DSS slabý zdroj ({z:.0f}σ, nekalibrováno)")
            found = False
        if found:
            c["flags"].append(f"na DSS je zdroj ({z:.0f}σ{'' if m_dss is None else f', ~{m_dss:.0f} mag'}) – statický objekt")
            c["auto_status"] = "auto_rejected"
        else:
            c["flags"].append("na DSS nic není")
            fl = [f for f in c["flags"] if not f.startswith("slabý rozmazaný u meze")]
            if len(fl) < len(c["flags"]):
                c["flags"] = fl
                if not any(f.startswith(HARD_PREFIX) for f in fl):
                    c["auto_status"] = "candidate"
    return done



# ----------------------------------------------------------------------------- PGC galaxies (HyperLEDA)

PGC_URLS = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv", "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
            "https://vizier.iucaa.in/viz-bin/asu-tsv")
# whole PGC2003 (VII/237 pgc.dat.gz, 19 MB, 983 261 galaxies); CDS Strasbourg is often unreachable
PGC_DAT_URLS = ("https://vizier.cfa.harvard.edu/ftp/cats/7/237/pgc.dat.gz",
                "https://cdsarc.cds.unistra.fr/ftp/cats/VII/237/pgc.dat.gz",
                "https://cdsarc.u-strasbg.fr/ftp/cats/VII/237/pgc.dat.gz",
                "https://vizier.iucaa.in/ftp/cats/VII/237/pgc.dat.gz")


def parse_pgc_dat(lines) -> dict:
    """VII/237 pgc.dat (fixed width, ReadMe): PGC 4–10, RA 13–20, Dec 21–27, logD25 37–41 (9.99 = none)."""
    pg, ra, de, d25 = [], [], [], []
    for ln in lines:
        if isinstance(ln, bytes):
            ln = ln.decode("latin-1")
        if len(ln) < 41 or not ln.startswith("PGC"):
            continue
        try:
            a = (int(ln[12:14]) + int(ln[14:16]) / 60.0 + float(ln[16:20]) / 3600.0) * 15.0
            d = int(ln[21:23]) + int(ln[23:25]) / 60.0 + int(ln[25:27]) / 3600.0
            if ln[20] == "-":
                d = -d
            n = int(ln[3:10])
        except ValueError:
            continue
        try:
            ld = float(ln[36:41])
        except ValueError:
            ld = 9.99
        pg.append(n)
        ra.append(a)
        de.append(d)
        d25.append(0.1 * 10 ** ld if ld < 9.9 else float("nan"))
    return {"ra": np.array(ra, float), "dec": np.array(de, float), "pgc": np.array(pg, float),
            "d25": np.array(d25, float)}


_PGC_ALL = None


def pgc_all(catalogs_dir, online: bool = False, log=None, timeout: float = 120.0) -> dict | None:
    """Whole-sky PGC2003 kept as catalogs/pgc/pgc.npz (~16 MB); built from catalogs/pgc/pgc.dat(.gz),
    which is downloaded once (mirrors: Harvard, CDS, IUCAA) when online use is allowed."""
    global _PGC_ALL
    log = log or (lambda s: None)
    if _PGC_ALL is not None:
        return _PGC_ALL
    if not catalogs_dir:
        return None
    import gzip
    d = Path(catalogs_dir) / "pgc"
    npz = d / "pgc.npz"
    if npz.exists():
        try:
            z = np.load(npz)
            _PGC_ALL = {k: np.asarray(z[k], float) for k in ("ra", "dec", "pgc", "d25")}
            return _PGC_ALL
        except Exception as e:            # noqa: BLE001
            log(f"Zdroj navíc: pgc.npz nečitelný ({e}) – vytvořím znovu.")
    src = next((p for p in (d / "pgc.dat.gz", d / "pgc.dat") if p.exists() and p.stat().st_size > 1000), None)
    if src is None and online:
        import requests
        d.mkdir(parents=True, exist_ok=True)
        last = None
        for url in PGC_DAT_URLS:
            try:
                log(f"Zdroj navíc: stahuji katalog galaxií PGC2003 (19 MB) z {url.split('/')[2]} …")
                tmp = d / "pgc.dat.gz.part"
                with requests.get(url, stream=True, timeout=(10.0, timeout)) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                if tmp.stat().st_size < 10_000_000:
                    raise IOError(f"neúplný soubor ({tmp.stat().st_size} B)")
                tmp.replace(d / "pgc.dat.gz")
                src = d / "pgc.dat.gz"
                break
            except Exception as e:        # noqa: BLE001
                last = e
        if src is None:
            log(f"Zdroj navíc: katalog PGC2003 se nepodařilo stáhnout ({last}).")
    if src is None:
        return None
    try:
        op = gzip.open if src.suffix == ".gz" else open
        with op(src, "rb") as f:
            g = parse_pgc_dat(f)
        if len(g["ra"]) < 100_000 and src.stat().st_size > 10_000_000:
            raise ValueError(f"jen {len(g['ra'])} galaxií")
        np.savez(npz, ra=g["ra"].astype(np.float64), dec=g["dec"].astype(np.float64),
                 pgc=g["pgc"].astype(np.float64), d25=g["d25"].astype(np.float32))
        log(f"Zdroj navíc: katalog galaxií PGC2003 připraven ({len(g['ra'])} galaxií, {npz}).")
        _PGC_ALL = g
        return g
    except Exception as e:                # noqa: BLE001
        log(f"Zdroj navíc: katalog PGC2003 nečitelný ({e}).")
        return None


def parse_vizier_tsv(text: str) -> dict:
    """VizieR asu-tsv of VII/237/pgc with _RAJ2000, _DEJ2000 (deg), PGC, logD25 → arrays."""
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    out = {"ra": [], "dec": [], "pgc": [], "d25": []}
    if not rows:
        return {k: np.array(v, float) for k, v in out.items()}
    hdr = [h.strip() for h in rows[0].split("\t")]
    def col(*names):
        for nm in names:
            for i, h in enumerate(hdr):
                if h == nm:
                    return i
        return None
    ira, ide = col("_RAJ2000", "RAJ2000", "_RA"), col("_DEJ2000", "DEJ2000", "_DE")
    ipg, ild = col("PGC"), col("logD25")
    if ira is None or ide is None:
        raise ValueError("VizieR: chybí sloupce souřadnic")
    for ln in rows[1:]:
        f = ln.split("\t")
        try:
            ra, de = float(f[ira]), float(f[ide])
        except (ValueError, IndexError):
            continue                      # units / dashes lines
        try:
            ld = float(f[ild]) if ild is not None and f[ild].strip() else float("nan")
        except ValueError:
            ld = float("nan")
        try:
            pg = float(f[ipg]) if ipg is not None else float("nan")
        except ValueError:
            pg = float("nan")
        out["ra"].append(ra)
        out["dec"].append(de)
        out["pgc"].append(pg)
        out["d25"].append(0.1 * 10 ** ld if math.isfinite(ld) else float("nan"))   # D25 in 0.1′ → ′
    return {k: np.array(v, float) for k, v in out.items()}


def load_pgc_field(catalogs_dir, ra0: float, dec0: float, radius_deg: float, online: bool = False,
                   log=None, timeout: float = 30.0) -> dict | None:
    """Galaxies of HyperLEDA / PGC2003 (VII/237, ~1 M galaxies to ~18 B) around the field centre.
    First the whole-sky offline copy (``pgc_all``, downloaded once); otherwise per field from VizieR,
    cached in catalogs/pgc_cache, so a repeated field needs no network. A whole-sky
    file catalogs/pgc/pgc.tsv (VizieR export with the same columns) is used if present."""
    log = log or (lambda s: None)
    base = Path(catalogs_dir) if catalogs_dir else None
    g = pgc_all(catalogs_dir, online=online, log=log)
    if g is not None:
        cd = math.cos(math.radians(dec0))
        sel = np.hypot(((g["ra"] - ra0 + 540) % 360 - 180) * cd, g["dec"] - dec0) <= radius_deg + 0.5
        return {k: v[sel] for k, v in g.items()}
    if base is not None:
        full = base / "pgc" / "pgc.tsv"
        if full.exists():
            try:
                g = parse_vizier_tsv(full.read_text(encoding="utf-8", errors="replace"))
                cd = math.cos(math.radians(dec0))
                sel = np.hypot(((g["ra"] - ra0 + 540) % 360 - 180) * cd, g["dec"] - dec0) <= radius_deg
                return {k: v[sel] for k, v in g.items()}
            except Exception as e:        # noqa: BLE001
                log(f"Zdroj navíc: soubor PGC nečitelný ({e}).")
        cdir = base / "pgc_cache"
        # cached fields on a 0.5° grid; any cached field that covers this one is reused
        if cdir.exists():
            for fp in sorted(cdir.glob("pgc_*.tsv")):
                try:
                    _, a, d, r = fp.stem.split("_")
                    a, d, r = float(a), float(d), float(r)
                except ValueError:
                    continue
                cd = math.cos(math.radians(dec0))
                sep = math.hypot(((a - ra0 + 540) % 360 - 180) * cd, d - dec0)
                if sep + radius_deg <= r + 1e-6:
                    try:
                        return parse_vizier_tsv(fp.read_text(encoding="utf-8", errors="replace"))
                    except Exception:     # noqa: BLE001
                        pass
    if not online:
        return None
    import requests
    ra_c, de_c = round(ra0 * 2) / 2, round(dec0 * 2) / 2
    rad = round(radius_deg + 0.75, 1)
    params = {"-source": "VII/237/pgc", "-c": f"{ra_c:.4f} {de_c:+.4f}", "-c.rd": f"{rad:.2f}",
              "-out": ["PGC", "logD25"], "-out.add": "_RAJ,_DEJ", "-oc.form": "d", "-out.max": "500000"}
    last = None
    for url in PGC_URLS:
        try:
            r = requests.get(url, params=params, timeout=(5.0, timeout))
            r.raise_for_status()
            g = parse_vizier_tsv(r.text)
            if base is not None:
                try:
                    (base / "pgc_cache").mkdir(parents=True, exist_ok=True)
                    (base / "pgc_cache" / f"pgc_{ra_c:.1f}_{de_c:.1f}_{rad:.1f}.tsv").write_text(r.text, encoding="utf-8")
                except OSError:
                    pass
            log(f"Zdroj navíc: galaxie PGC z VizieR ({len(g['ra'])} v okruhu {rad:.1f}°, uloženo do catalogs/pgc_cache).")
            return g
        except Exception as e:            # noqa: BLE001
            last = e
    log(f"Zdroj navíc: katalog galaxií PGC (VizieR) nedostupný ({last}) – přeskočeno.")
    return None


def pgc_galaxy_mag(d25_arcmin: float) -> float:
    """Rough total magnitude of a galaxy from its D25 (PGC has no photometry here):
    ~14.5 mag at 1′, 5 mag per decade of size (surface brightness ~ constant at the isophote)."""
    d = d25_arcmin if math.isfinite(d25_arcmin) and d25_arcmin > 0 else 0.3
    return 14.5 - 5.0 * math.log10(max(d, 0.05))


def pgc_check(cands: list, pgc: dict | None, fwhm_arcsec: float, scale_arcsec: float, slack_mag: float = 2.0) -> int:
    """Reject "zdroj navíc" / glows sitting on a PGC galaxy that could be that bright. Returns count."""
    if not pgc or not len(pgc.get("ra", [])):
        return 0
    n = 0
    for c in cands:
        if c.get("kind") == "supernova":      # on a galaxy by definition (sngal)
            continue
        if c.get("known") or not (c.get("auto_status") == "candidate" or
                                  any(f.startswith("slabý rozmazaný u meze") for f in c.get("flags", []))):
            continue
        if any(f.startswith("galaxie PGC") for f in c.get("flags", [])):
            continue
        cd = math.cos(math.radians(c["dec"]))
        dx = ((pgc["ra"] - c["ra"] + 540) % 360 - 180) * cd * 3600
        dy = (pgc["dec"] - c["dec"]) * 3600
        dist = np.hypot(dx, dy)
        sc = float(c.get("diffuse_scale_px") or 0.0) * scale_arcsec
        d25 = np.nan_to_num(pgc["d25"], nan=0.3)
        rad = np.maximum(1.5 * fwhm_arcsec, 2.0 * sc) + 30.0 * d25          # + D25/2 in ″
        hit = np.nonzero(dist <= rad)[0]
        best = None
        for k in hit[np.argsort(dist[hit])]:
            if pgc_galaxy_mag(float(pgc["d25"][k])) <= float(c["mag"]) + slack_mag:
                best = k
                break
        if best is None:
            continue
        nm = f"PGC{int(pgc['pgc'][best])}" if math.isfinite(pgc["pgc"][best]) else "PGC"
        dd = pgc["d25"][best]
        c["flags"].append(f"galaxie {nm}" + (f" (D25 {dd:.1f}′, {dist[best]:.0f}″)" if math.isfinite(dd) else f" ({dist[best]:.0f}″)"))
        c["auto_status"] = "auto_rejected"
        n += 1
    return n




def local_sky_matrix(wcs, x: float, y: float) -> list[list[float]]:
    """[[dξ/dx, dξ/dy], [dη/dx, dη/dy]] in degrees per pixel at (x, y) (ξ = ΔRA·cosδ east)."""
    p = wcs.all_pix2world([[x, y], [x + 1.0, y], [x, y + 1.0]], 0)
    cd = math.cos(math.radians(p[0, 1]))
    e = lambda q: ((q[0] - p[0, 0] + 180.0) % 360.0 - 180.0) * cd
    return [[float(e(p[1])), float(e(p[2]))], [float(p[1, 1] - p[0, 1]), float(p[2, 1] - p[0, 1])]]

def field_box(wcs, w: int, h: int, pad: float = 0.3):
    """RA/Dec box of the whole frame for a catalogue query: (r0, r1, d0, d1, edge points).
    The whole border, not just the corners: on a wide field far from the equator the edge middle
    reaches ~0.5° beyond the corners in Dec (CTA-N 135 mm, Dec +51°: 100 px strip at the top edge
    without catalogue stars → 238 false "extra sources", user report 13. 11.). r0 > r1 never;
    a box across RA 0h is returned with r0 > r1 modulo 360 handled by ``ucac4.query``."""
    t = np.linspace(0, 1, 17)
    bx = np.r_[t * (w - 1), np.full(17, w - 1), t * (w - 1), np.zeros(17)]
    by = np.r_[np.zeros(17), t * (h - 1), np.full(17, h - 1), t * (h - 1)]
    edge = wcs.all_pix2world(np.stack([bx, by], 1), 0)
    ras, decs = edge[:, 0] % 360, edge[:, 1]
    d0, d1 = float(np.min(decs)) - pad, float(np.max(decs)) + pad
    # a celestial pole inside the frame? The linear part is enough (SIP diverges 30–90° away:
    # "all_world2pix failed to converge", CTA-N 318 all fields, 13. 11.)
    try:
        cpole = wcs.wcs_world2pix([[0.0, 90.0], [0.0, -90.0]], 0)
    except Exception:                     # noqa: BLE001
        cpole = np.full((2, 2), np.nan)
    pole_in = any(np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h for p in cpole)
    if pole_in or d1 > 89.0 or d0 < -89.0:
        r0, r1 = 0.0, 360.0
        d0, d1 = max(d0, -90.0), min(d1, 90.0)
        if pole_in:
            d1 = 90.0 if np.mean(decs) > 0 else d1
            d0 = -90.0 if np.mean(decs) < 0 else d0
        return r0, r1, d0, d1, edge
    pr = pad / max(math.cos(math.radians(max(abs(d0), abs(d1)))), 0.05)
    if np.ptp(ras) > 180:                        # field across RA 0h
        r0, r1 = float(np.min(ras[ras > 180])) - pr, float(np.max(ras[ras < 180])) + pr
    else:
        r0, r1 = float(np.min(ras)) - pr, float(np.max(ras)) + pr
    return r0 % 360, r1 % 360, d0, d1, edge



def _refit_sip(wcs, px, py, ra, de, order: int = 5):
    """New SIP distortion for the header's linear TAN part (CRVAL, CD, CRPIX kept).
    Forward A, B (with first-order terms, so a small CD/CRPIX error is absorbed) and inverse
    AP, BP fitted by robust least squares on the star pairs. ``fit_wcs_from_points`` over a 16°
    field converged to 6–11 px residuals (CTA-N 320/2); refitting only the polynomial cannot."""
    from astropy.wcs import Sip
    lin = wcs.deepcopy()
    lin.sip = None
    for attr in ("cpdis1", "cpdis2", "det2im1", "det2im2"):
        try:
            setattr(lin, attr, None)
        except Exception:                  # noqa: BLE001
            pass
    crpix = lin.wcs.crpix.copy()
    # ideal (undistorted) pixel position of each star, relative to CRPIX (FITS 1-based → 0-based)
    ux, uy = lin.wcs_world2pix(ra, de, 0)
    u, v = ux - (crpix[0] - 1), uy - (crpix[1] - 1)
    x, y = px - (crpix[0] - 1), py - (crpix[1] - 1)
    ok = np.isfinite(u) & np.isfinite(v)
    S = 2000.0

    def terms(a, b):
        cols, idx = [], []
        for pq in range(1, order + 1):
            for q in range(pq + 1):
                p_ = pq - q
                cols.append((a / S) ** p_ * (b / S) ** q)
                idx.append((p_, q))
        return np.stack(cols, 1), idx

    def fit(a, b, ta, tb):
        T, idx = terms(a, b)
        keep = ok.copy()
        for _ in range(4):
            ca, *_ = np.linalg.lstsq(T[keep], ta[keep], rcond=None)
            cb, *_ = np.linalg.lstsq(T[keep], tb[keep], rcond=None)
            r = np.hypot(T @ ca - ta, T @ cb - tb)
            sig = 1.4826 * np.median(r[keep]) + 1e-3
            keep = ok & (r < 4 * sig)
        A = np.zeros((order + 1, order + 1)); B = np.zeros((order + 1, order + 1))
        for (p_, q), va, vb in zip(idx, ca, cb):
            A[p_, q] = va / S ** (p_ + q)
            B[p_, q] = vb / S ** (p_ + q)
        return A, B
    A, B = fit(x, y, u - x, v - y)            # forward: u = x + A(x, y)
    AP, BP = fit(u, v, x - u, y - v)          # inverse: x = u + AP(u, v)
    new = wcs.deepcopy()
    new.sip = Sip(A, B, AP, BP, crpix)
    new.wcs.set()
    return new

def refine_wcs(img: np.ndarray, wcs, fwhm: float, catalog: dict, log=None, sip_degree: int = 5):
    """Refit the frame's WCS on UCAC4 stars (TAN-SIP).

    The header WCS of a wide field can be off by several pixels in the corners (CTA-N 320/2,
    Dec −72°, header TAN-SIP order 4: +2.5/+5 px at 300 px from the corner, 7 px at the corner –
    a 9.6 mag star became a "brightening" of a 16 mag one 5 px away, the DSS cut-out of the
    candidate showed another place). Matching: header WCS → 5th-order polynomial on isolated
    bright stars (wide radius first) → local median offsets (two passes) → unique pairs within
    1 FWHM → ``fit_wcs_from_points``. Accepted only when it fits the stars better everywhere.
    Returns (new_wcs or None, info dict)."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    log = log or (lambda s_: None)
    h, w = img.shape
    info = {}
    det = detect(img, fwhm, 8.0, border=8)
    if len(det) < 100:
        info["note"] = f"málo hvězd ({len(det)})"
        return None, info
    det = det[np.argsort(-det[:, 2])][:20000]
    det_xy = det[:, :2]
    sel_c = np.asarray(catalog["mag_all"], float) < 13.5
    cat = {k: np.asarray(v)[sel_c] for k, v in catalog.items() if hasattr(v, "__len__") and len(v) == len(sel_c)}
    cat_xy, cat_mag, cat_idx = project_catalog(cat, wcs, (h, w))
    if len(cat_xy) < 100:
        info["note"] = "málo katalogových hvězd v poli"
        return None, info
    tr, rms = fit_distortion(det_xy, cat_xy, cat_mag, (h, w), fwhm, order=5)
    if tr is None:
        info["note"] = "hvězdy se nepodařilo spárovat"
        return None, info
    cxy = np.stack(tr(cat_xy[:, 0], cat_xy[:, 1]), 1)
    dtree = cKDTree(det_xy)
    for rad in (max(4.0, 2.0 * fwhm), max(3.0, 1.2 * fwhm)):
        d0, j0 = dtree.query(cxy, distance_upper_bound=rad)
        m0 = np.isfinite(d0) & (cat_mag < 13.0)
        if m0.sum() < 50:
            break
        cxy = LocalCorrection(cxy[m0], det_xy[j0[m0]], k=12)(cxy)
    d, j = dtree.query(cxy, distance_upper_bound=max(1.5, 1.0 * fwhm))
    ok = np.isfinite(d)
    # one detection – one star (crowded fields)
    order = np.argsort(d[ok])
    ii = np.nonzero(ok)[0][order]
    used, keep = set(), []
    for i in ii:
        if j[i] not in used:
            used.add(j[i])
            keep.append(i)
    keep = np.array(keep, int)
    if len(keep) < 50:
        info["note"] = f"málo párů ({len(keep)})"
        return None, info
    px, py = det_xy[j[keep], 0], det_xy[j[keep], 1]
    ra = np.asarray(cat["ra_all"], float)[cat_idx[keep]]
    de = np.asarray(cat["dec_all"], float)[cat_idx[keep]]

    def zones(wcs_):
        x, y = wcs_.all_world2pix(ra, de, 0, quiet=True)
        r = np.hypot(x - px, y - py)
        rz = np.hypot((px - w / 2) / (w / 2), (py - h / 2) / (h / 2)) / math.sqrt(2)
        out = {}
        for nm, lo, hi in (("střed", 0, 0.5), ("mezikruží", 0.5, 0.8), ("okraj", 0.8, 0.92), ("roh", 0.92, 1.01)):
            z = (rz >= lo) & (rz < hi) & np.isfinite(r)
            if z.sum() >= 10:
                out[nm] = round(float(np.median(r[z])), 2)
        return out, float(np.nanmedian(r))
    old_z, old_med = zones(wcs)
    try:
        new = _refit_sip(wcs, px, py, ra, de, sip_degree)
    except Exception as e:                # noqa: BLE001
        info["note"] = f"fit selhal ({e})"
        return None, info
    new_z, new_med = zones(new)
    info.update({"pairs": int(len(keep)), "header_px": old_z, "refined_px": new_z,
                 "header_median_px": round(old_med, 2), "refined_median_px": round(new_med, 2)})
    worst_old = max(old_z.values()) if old_z else 0.0
    worst_new = max(new_z.values()) if new_z else 99.0
    if new_med <= old_med * 1.05 and worst_new <= worst_old * 1.05 and worst_new < max(1.0, 0.5 * fwhm):
        info["accepted"] = True
        return new, info
    info["accepted"] = False
    return None, info

def choose_mode(requested: str, n: int, span_min: float, twilight_frames, max_frames: int = 6,
                max_span_min: float = 0.0) -> tuple[str, str]:
    """Search mode of a job: tracking | twilight | both, and a Czech reason.

    auto: few frames (≤ ``max_frames``; optionally a short series < ``max_span_min`` with ≤ 2× as many) → twilight (the
    synthetic-tracking search has almost nothing to work with); a longer series with frames taken
    with the Sun above −18° → both (tracking finds the movers, the catalogue comparison adds
    sources that hardly move: a distant comet, a comet in outburst, a nova); otherwise tracking."""
    req = (requested or "auto").strip().lower()
    tw = None if twilight_frames is None else int(np.sum(twilight_frames))
    if n < 5:
        # synthetic tracking needs ≥ 5 frames; 2–4 frames: the catalogue comparison alone
        # (a source must be in every frame – cosmics, satellites and dithered warm pixels are not)
        return "twilight", f"jen {n} snímky – synthetic tracking potřebuje aspoň 5, jen porovnání s katalogem"
    if req in ("tracking", "twilight", "both"):
        return req, "zvoleno ručně"
    # few frames only (a short series with many frames is fine for tracking: tests, fast NEOs)
    if n <= max_frames or (max_span_min > 0 and span_min < max_span_min and n <= 2 * max_frames):
        return "twilight", f"{n} snímků za {span_min:.0f} min – málo na synthetic tracking"
    if tw:
        return "both", f"{n} snímků za {span_min:.0f} min, {tw} za soumraku (Slunce nad −18°)"
    return "tracking", f"{n} snímků za {span_min:.0f} min v noci"


PLANET_MAG = {"Merkur": 0.0, "Venuše": -4.2, "Mars": 0.5, "Jupiter": -2.3, "Saturn": 0.8, "Uran": 5.7, "Neptun": 7.8}
_BSC = None


def bright_stars() -> dict:
    """Yale Bright Star Catalogue (9 096 stars to ~6.5 V, J2000, synthtrack/data/bsc5.csv) – UCAC4 has
    no stars brighter than ~7–8 mag (325/2: Spica 1 mag reported as a "new source" of 8.3 mag and
    four of its reflections as more)."""
    global _BSC
    if _BSC is None:
        p = Path(__file__).parent / "data" / "bsc5.csv"
        rows = []
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln or ln.startswith("#"):
                    continue
                f = ln.split(",")
                try:
                    rows.append((float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]), f[7]))
                except (ValueError, IndexError):
                    continue
        a = np.array([r[:5] for r in rows], float) if rows else np.zeros((0, 5))
        _BSC = {"ra": a[:, 0], "dec": a[:, 1], "mag": a[:, 2], "pmra": a[:, 3], "pmdec": a[:, 4],
                "name": [r[5] for r in rows]}
    return _BSC


def add_bright_stars(cat: dict, box, epoch_yr: float, planets_radec=None) -> tuple[dict, int]:
    """Catalogue + bright stars (and planets, as very bright stars) that UCAC4 lacks: a BSC star with
    no UCAC4 star within 20″ that is at most 3 mag fainter. Proper motion applied from J2000.
    Returns the new catalogue dict and how many were added."""
    b = bright_stars()
    r0, r1, d0, d1 = box[:4]
    dt = epoch_yr - 2000.0
    cd = np.cos(np.radians(np.clip(b["dec"], -89.9, 89.9))) if len(b["ra"]) else np.zeros(0)
    ra = (b["ra"] + b["pmra"] * dt / 3600.0 / np.maximum(cd, 1e-3)) % 360.0
    de = b["dec"] + b["pmdec"] * dt / 3600.0
    in_ra = (ra >= r0) & (ra <= r1) if r0 <= r1 else (ra >= r0) | (ra <= r1)
    sel = np.nonzero(in_ra & (de >= d0) & (de <= d1))[0]
    add_ra, add_de, add_m = [], [], []
    ua, ud, um = (np.asarray(cat.get(k, []), float) for k in ("ra_all", "dec_all", "mag_all"))
    for i in sel:
        if len(ua):
            near = np.abs(ud - de[i]) < 20 / 3600
            if near.any():
                dd = np.hypot(((ua[near] - ra[i] + 540) % 360 - 180) * math.cos(math.radians(de[i])), ud[near] - de[i]) * 3600
                if np.any((dd < 20) & (um[near] < b["mag"][i] + 3)):
                    continue
        add_ra.append(ra[i]); add_de.append(de[i]); add_m.append(b["mag"][i])
    for nm, pra, pde in (planets_radec or []):
        if nm in PLANET_MAG:
            add_ra.append(pra); add_de.append(pde); add_m.append(PLANET_MAG[nm])
    if not add_ra:
        return cat, 0
    out = dict(cat)
    k = len(add_ra)
    for key, vals in (("ra_all", add_ra), ("dec_all", add_de), ("mag_all", add_m)):
        out[key] = np.concatenate([np.asarray(cat.get(key, []), float), np.asarray(vals, float)])
    out["j_all"] = np.concatenate([np.asarray(cat.get("j_all", []), float), np.full(k, np.nan)])
    out["leda_all"] = np.concatenate([np.asarray(cat.get("leda_all", []), np.int8), np.zeros(k, np.int8)])
    return out, k


def planets_in_field(wcs, shape, mjd: float, lat: float, lon: float, alt: float) -> list:
    """(name, x, y) of the Moon and planets inside (or just outside) the frame."""
    try:
        from astropy.coordinates import EarthLocation, get_body
        from astropy.time import Time
        import astropy.units as u
        t = Time(mjd, format="mjd")
        loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, (alt or 0.0) * u.m)
        out = []
        h, w = shape
        names = {"moon": "Měsíc", "mercury": "Merkur", "venus": "Venuše", "mars": "Mars", "jupiter": "Jupiter",
                 "saturn": "Saturn", "uranus": "Uran", "neptune": "Neptun"}
        for b, cz in names.items():
            # the direction seen from the site (GCRS with the observer's position); ``.icrs`` moved the
            # origin to the solar-system barycentre: Venus 136° off (325/2, 16. 11.: Venus in the field
            # at −4 mag, not flagged)
            c = get_body(b, t, loc)
            ra_b, de_b = float(c.ra.deg), float(c.dec.deg)
            try:
                x, y = wcs.wcs_world2pix([[ra_b, de_b]], 0)[0]
            except Exception:              # noqa: BLE001
                continue
            if np.isfinite(x) and -200 < x < w + 200 and -200 < y < h + 200:
                out.append((cz, float(x), float(y)))
        return out
    except Exception:          # noqa: BLE001 – an optional check
        return []


def known_match(known, x: float, y: float, scale_as: float, radius_arcsec: float, mag: float | None,
                wide_arcsec: float | None = None, mag_tol: float = 3.0) -> str | None:
    """Name of the known object at (x, y), or None.

    ``known`` items are (name, x, y[, kind, predicted mag]). ``wide_arcsec`` widens the radius
    for comets only (a coma of a glow); an asteroid is a point. An asteroid predicted more
    than ``mag_tol`` fainter than the source is not it – CTA-N 322/1: a 13.1 mag glow
    "identified" as 19.3 mag (29822) 1999 DS2 99″ away. Nearest acceptable match wins."""
    best, bd = None, 1e18
    for item in (known or []):
        name, kx, ky = item[0], float(item[1]), float(item[2])
        kind = item[3] if len(item) > 3 else None
        kmag = item[4] if len(item) > 4 else None
        r = radius_arcsec
        if wide_arcsec is not None and kind != "asteroid":
            r = max(r, wide_arcsec)
        d = math.hypot(kx - x, ky - y) * scale_as
        if d > r:
            continue
        if (kind == "asteroid" and kmag is not None and mag is not None and math.isfinite(mag)
                and float(kmag) - mag > mag_tol):
            continue
        if d < bd:
            best, bd = name, d
    return best


def _known_info(kn, c, known_list, wcs) -> dict | None:
    """The GUI's known-object badge: name, O−C (″), predicted magnitude and motion."""
    if not kn:
        return None
    k = next((o for o in (known_list or []) if o.name == kn.get("name")), None)
    if k is None or k.x is None:
        return kn
    oc = math.hypot(k.x - c["x"], k.y - c["y"]) * abs(_scale(wcs)) * 3600
    return {"name": k.name, "kind": k.kind, "o_c_arcsec": round(oc, 2),
            "pred_mag": None if k.mag is None else round(float(k.mag), 2),
            "pred_rate": round(float(k.rate_arcsec_min), 3), "pred_pa": round(float(k.pa_deg), 1)}


def pipeline_run(reg, dts, wcs, fwhm: float, u4root, mjd_ref: float, site, catalogs_dir, astap_dir,
                 known_list, injections, regs, img_dir, band: str = "r", params: TwilightParams | None = None,
                 log=None, dss_online: bool = False, pgc_online: bool = False,
                 sn_inject_n: int = 0) -> tuple[list[dict], dict]:
    """Run the catalogue comparison on the job's registered frames; dicts ready for candidates.json.
    Then the side search for supernovae in catalogued galaxies (``sngal``); with ``sn_inject_n`` that
    many injected supernovae are put into galaxies first (appended to ``injections``)."""
    from astropy.time import Time
    from . import ucac4 as U4
    from .imaging import track_crops, write_candidate_images, write_motion_frames
    log = log or (lambda s: None)
    n, h, w = reg.shape
    r0, r1, d0, d1, edge = field_box(wcs, w, h)
    cat = U4.query(u4root, r0, r1, d0, d1, band, float(Time(mjd_ref, format="mjd").jyear), mag_max=17.0)
    dso_files = [Path(catalogs_dir) / "openngc" / "NGC.csv", Path(catalogs_dir) / "openngc" / "addendum.csv"]
    dso = load_dso(dso_files)
    var = {}
    if astap_dir and (Path(astap_dir) / "deep_sky.csv").exists():
        dso = merge_dso(dso, load_astap_dso(Path(astap_dir) / "deep_sky.csv"))
    if astap_dir and (Path(astap_dir) / "variable_stars_13.csv").exists():
        var = load_variables(Path(astap_dir) / "variable_stars_13.csv")
    log(f"Zdroj navíc: UCAC4 {len(cat.get('ra_all', []))} hvězd, DSO {len(dso.get('ra', [])) if dso else 0} "
        f"(OpenNGC{' + ASTAP deep_sky' if astap_dir else ''}), proměnné VSX {len(var.get('ra', [])) if var else 0}.")
    known = [(k.name, float(k.x), float(k.y), getattr(k, "kind", None), getattr(k, "mag", None))
             for k in (known_list or []) if k.x is not None and k.y is not None]
    planets = planets_in_field(wcs, (h, w), mjd_ref, *site)
    # UCAC4 has no stars brighter than ~7–8 mag: the Bright Star Catalogue and the planets join it
    # (matching, halo rule, star masks)
    pl_radec = []
    for nm, px_, py_ in planets:
        try:
            pr, pd = wcs.wcs_pix2world([[px_, py_]], 0)[0]
            pl_radec.append((nm, float(pr), float(pd)))
        except Exception:                  # noqa: BLE001
            pass
    cat, n_bright = add_bright_stars(cat, (r0, r1, d0, d1), float(Time(mjd_ref, format="mjd").jyear), pl_radec)
    if n_bright:
        log(f"Zdroj navíc: doplněno {n_bright} jasných hvězd a planet, které UCAC4 nemá (Bright Star Catalogue"
            + (", " + ", ".join(nm for nm, _, _ in pl_radec) if pl_radec else "") + ").")
    dither = np.array([[r.offset[0], r.offset[1]] for r in regs], float) if regs else None
    # local sky matrix at the centre for the motion rates
    c0 = wcs.all_pix2world([[w / 2, h / 2], [w / 2 + 1, h / 2], [w / 2, h / 2 + 1]], 0)
    cd = math.cos(math.radians(c0[0, 1]))
    M = np.array([[((c0[1, 0] - c0[0, 0] + 540) % 360 - 180) * cd, ((c0[2, 0] - c0[0, 0] + 540) % 360 - 180) * cd],
                  [c0[1, 1] - c0[0, 1], c0[2, 1] - c0[0, 1]]])
    # the local matrix, not the field centre's: on a wide field near a pole north turns by
    # Δα·sin δ across the frame (CTA-N 320/2, Dec −72°: ~20° at the corner – DSS comparison rotated)
    # galaxies (PGC): the faint ones below the DSO catalogues' 1′ limit (CTA-N 318/5, user-checked #444, #522,
    # #532) and the hosts searched for supernovae
    try:
        cc = wcs.all_pix2world([[w / 2, h / 2]], 0)[0]
        rr = float(np.max(np.hypot(((edge[:, 0] - cc[0] + 540) % 360 - 180) * math.cos(math.radians(cc[1])),
                                    edge[:, 1] - cc[1])))
        pgc = load_pgc_field(catalogs_dir, float(cc[0]), float(cc[1]), rr, online=pgc_online, log=log)
    except Exception as e:                # noqa: BLE001
        pgc = None
        log(f"Zdroj navíc: katalog galaxií PGC selhal ({e}).")
    if sn_inject_n > 0 and (pgc is not None or dso) and injections is not None:
        try:
            from . import sngal as SNG0
            from . import inject as INJ0

            def _pix0(ra_, de_):
                xy = _world2pix_sip(wcs, ra_, de_, (h, w))
                return xy[:, 0], xy[:, 1]
            g0 = SNG0.galaxies(pgc, dso, _pix0, (h, w), abs(_scale(wcs)) * 3600, fwhm, 14.0, pgc_galaxy_mag,
                               border=60)
            sn_inj = SNG0.plan_injections(g0, int(sn_inject_n), n, fwhm, abs(_scale(wcs)) * 3600, 8.0, 60.0,
                                          INJ0.psf_gain(None, fwhm), seed=int(abs(mjd_ref) * 1e4) % (2 ** 31) + 11)
            base = max([int(o["id"]) for o in injections], default=-1) + 1
            for k_, o in enumerate(sn_inj):
                o["id"] = base + k_
            INJ0.add_to_frames(reg, sn_inj, dts, fwhm)
            if hasattr(reg, "flush"):
                reg.flush()
            injections.extend(sn_inj)
            log(f"Injekce: {len(sn_inj)} umělých supernov do {len(set(o['galaxy'] for o in sn_inj))} galaxií "
                f"(odstup 0,3–1 × poloměr galaxie, nominální SNR 8–60).")
        except Exception as e:            # noqa: BLE001
            log(f"Injekce supernov se nepodařily ({e}).")
    res = run(reg, dts, wcs, fwhm, cat, dso=dso, known=known, planets=planets, params=params,
              sky_matrix=lambda x_, y_: local_sky_matrix(wcs, x_, y_),
              variables=var, dither=dither)
    for line in res.log:
        log(line)
    if pgc is not None:
        ng = pgc_check(res.candidates, pgc, fwhm * abs(_scale(wcs)) * 3600, abs(_scale(wcs)) * 3600)
        res.stats["pgc_galaxies_in_field"] = int(len(pgc["ra"]))
        res.stats["pgc_matched"] = ng
        log(f"Zdroj navíc: {len(pgc['ra'])} galaxií PGC v poli, {ng} kandidátů leží na galaxii.")
        # "faint and extended near the limit" was a stand-in for small galaxies missing from the DSO
        # lists (CTA-N 318). With PGC checked, such a source on no PGC galaxy may be a comet: kept,
        # with a warning (CTA-N 324/4: injected comets of 13.4 mag rejected by it)
        back = 0
        for c in res.candidates:
            fl = c.get("flags", [])
            if (c.get("auto_status") != "candidate" and any(f.startswith("slabý rozmazaný u meze") for f in fl)
                    and not any(f.startswith(HARD_PREFIX) for f in fl)):
                c["flags"] = [("rozmazaný u meze, v PGC není – kometa? (nebo galaxie mimo katalog)"
                               if f.startswith("slabý rozmazaný u meze") else f) for f in fl]
                c["auto_status"] = "candidate"
                back += 1
        if back:
            log(f"Zdroj navíc: {back} slabých rozmazaných zdrojů u meze neleží na galaxii PGC – ponechány "
                f"(možné komety).")
    elif not pgc_online:
        log("Zdroj navíc: katalog galaxií PGC není k dispozici (zapni Online: náhledy oblohy pro stažení z VizieR).")
    from . import sngal as SNG
    sn_cands, sn_stats = [], None
    if res.aux and (pgc is not None or dso):
        try:
            A = res.aux
            # a supernova can be bright in a faint galaxy: galaxies at least to 15 mag (and the injection plan's set)
            gals = SNG.galaxies(pgc, dso, A["to_pix"], (h, w), abs(_scale(wcs)) * 3600, A["fw_star"],
                                max(A["lim"], 14.0),
                                pgc_galaxy_mag)
            if os.environ.get("ST_SN_DUMP"):
                import pickle as _pk
                _zg = np.array([[float(res.zp(x_, y_)) for x_ in np.linspace(0, w, 9)] for y_ in np.linspace(0, h, 9)])
                with open(os.environ["ST_SN_DUMP"], "wb") as _f:
                    _pk.dump({"cube": np.asarray(reg[:]), "img": A["img"], "sig": A["sig"], "gals": gals, "cxy": A["cxy"],
                              "cat_mag": A["cat_mag"], "exp_flux": A["exp_flux"], "zgrid": _zg, "fw": A["fw_star"],
                              "scale": abs(_scale(wcs)) * 3600, "wcs": wcs, "known": known, "sig_f": A["sig_f"],
                              "inj": [o for o in injections if o.get("kind") == "sn"]}, _f)
            sn_cands, sn_stats = SNG.search(reg, A["img"], A["sig"], gals, A["to_pix"], A["cxy"], A["cat_mag"],
                                            A["exp_flux"], res.zp, A["fw_star"], abs(_scale(wcs)) * 3600, wcs,
                                            params or TwilightParams(), known=known, sig_f=A["sig_f"])
            # one place, one candidate: a passed "zdroj navíc" there already is that source
            # a rejected one there is replaced by a passed supernova candidate (the galaxy explained it away);
            # a rejected supernova candidate where the general search has anything is dropped (no duplicates)
            rr_ = 2 * A["fw_star"]
            near_ = lambda c_, d_: math.hypot(c_["x"] - d_["x"], c_["y"] - d_["y"]) <= rr_
            keep_sn, drop_gen = [], set()
            for c in sn_cands:
                here = [k_ for k_, d in enumerate(res.candidates) if near_(c, d)]
                if any(res.candidates[k_].get("auto_status") == "candidate" for k_ in here):
                    continue
                if c["auto_status"] == "candidate":
                    drop_gen.update(here)
                    keep_sn.append(c)
                elif not here:
                    keep_sn.append(c)
            if drop_gen:
                res.candidates = [d for k_, d in enumerate(res.candidates) if k_ not in drop_gen]
            sn_cands = keep_sn
            res.candidates += sn_cands
            log(f"Supernovy v galaxiích: prohlédnuto {sn_stats['examined']} z {sn_stats['galaxies']} galaxií "
                f"(PGC + OpenNGC, viditelných do ~{A['lim'] + 1:.1f} mag), nesouměrnost nad prahem u "
                f"{sn_stats['peaks']}, prošlo {sum(1 for c in sn_cands if c['auto_status'] == 'candidate')} (včetně umělých).")
        except Exception as e:            # noqa: BLE001 – a side search must not break the job
            import traceback as _tb
            log(f"Supernovy v galaxiích: hledání selhalo ({e}).")
            log("  " + " | ".join(l.strip() for l in _tb.format_exc().splitlines()[-6:]))
    if dss_online:
        # supernova candidates sit on galaxies that DSS shows: the DSS test says nothing about them
        nn = dss_check([c for c in res.candidates if not c.get("known") and c.get("kind") != "supernova"],
                       fwhm * abs(_scale(wcs)) * 3600, log=log, cat=cat)
        if nn:
            log(f"Zdroj navíc: {nn} kandidátů porovnáno s DSS2 (zdroj na DSS → zamítnut; nic na DSS → ponechán).")
        res.stats["dss_checked"] = nn
    tm = float(np.mean(dts))
    ipos = (np.array([[o["x"] + o["vx"] * tm, o["y"] + o["vy"] * tm] for o in injections], float)
            if injections else None)
    out = []
    for k, c in enumerate(res.candidates):
        tag = f"tw{k:04d}"
        size = int(max(48, min(128, max(14 * fwhm, 6 * (c.get("diffuse_scale_px") or 0) + 24))))
        try:
            images = write_candidate_images(img_dir, tag, track_crops(reg, dts, c["x"], c["y"], c["vx"], c["vy"], size),
                                            3, fwhm=fwhm)
            images["frames"] = write_motion_frames(img_dir, tag, reg, dts, c["x"], c["y"], c["vx"], c["vy"], size)
        except Exception as e:     # noqa: BLE001 – a missing preview must not lose the candidate
            images = {}
            log(f"Zdroj navíc: náhled #{k} se nepodařil ({e}).")
        inj = None
        if ipos is not None and len(ipos):
            d = np.hypot(ipos[:, 0] - c["x"], ipos[:, 1] - c["y"])
            j = int(np.argmin(d))
            # a glow's peak sits off the coma centre when stars around it are masked (CTA-N 322
            # #394: injected comet, coma 7 px, glow found 12.5 px away – shown as "unknown")
            r_inj = max(8.0, 2 * fwhm, 2.0 * (c.get("diffuse_scale_px") or 0),
                        2.0 * float(injections[j].get("coma_px") or 0))
            if d[j] < r_inj:
                inj = int(injections[j]["id"])
        label = {"navíc": "zdroj navíc (není v UCAC4)", "zjasnění": "zjasnění hvězdy", "mlhavý": "mlhavý objekt",
                 "supernova": "supernova v galaxii?"}[c["kind"]]
        flags = [f"Twilight: {label}"] + list(c["flags"])
        out.append({
            "x": c["x"], "y": c["y"], "vx": c["vx"], "vy": c["vy"], "snr": float(c["snr"]),
            "seg_snr": [], "median_ratio": 1.0, "frames_used": int(c["frames_used"]), "star_contaminated": False,
            "snr_raw": float(c["snr"]), "fwhm": c["fwhm"], "chi2": 0.0, "snr_reverse": 0.0,
            "positions": [{"dt_min": 0.0, "x": c["x"], "y": c["y"], "ok": True, "ra": c["ra"], "dec": c["dec"],
                           "mjd": float(mjd_ref)}],
            "obj_fwhm": c["obj_fwhm"], "obj_fwhm_scale": c.get("obj_fwhm_scale"), "injected": inj, "flags": flags, "auto_status": c["auto_status"],
            "ra": c["ra"], "dec": c["dec"], "rate_arcsec_min": c["rate_arcsec_min"], "pa_deg": c["pa_deg"],
            "mag": c["mag"], "known": _known_info(c["known"], c, known_list, wcs), "sky_matrix": local_sky_matrix(wcs, c["x"], c["y"]), "images": images,
            "source": "twilight", "twilight_kind": c["kind"], "catalog_mag": c.get("catalog_mag"),
            "motion_sigma": c.get("motion_sigma"), "frame_snr": c.get("frame_snr"),
        })
    # a known variable star right at a static source: its name on the card and a cross in the map
    # (dense fields: most "new" sources and brightenings are variables – the user wants to see it at once)
    try:
        n_v = annotate_vsx_near(out, fwhm * abs(_scale(wcs)) * 3600, var, online=bool(dss_online or pgc_online), log=log,
                                img_dir=img_dir, scale_arcsec=abs(_scale(wcs)) * 3600)
        if n_v:
            log(f"Zdroj navíc: {n_v} kandidátů má v těsném okolí známou proměnnou hvězdu (VSX) – označeno.")
    except Exception as e:                # noqa: BLE001 – annotation only
        log(f"Zdroj navíc: kontrola proměnných VSX selhala ({e}).")
    stats = dict(res.stats)
    stats["planets"] = [p[0] for p in planets]
    return out, stats


def annotate_vsx_near(out: list, fwhm_arcsec: float, var: dict | None, online: bool, log=None,
                      max_n: int = 200, img_dir=None, scale_arcsec: float | None = None) -> int:
    """For passed unknown static sources: VSX variables within ~1 FWHM (10–40″). Online AAVSO VSX /
    VizieR (complete, Gaia variables included), offline the ASTAP table. Adds ``vsx_near`` (name, type,
    range, period, url, separation, east/north offsets) and a flag with the nearest one."""
    from . import vsx as VSX
    r_as = float(min(max(fwhm_arcsec, 10.0), 40.0))
    todo = [d for d in out if d.get("auto_status") == "candidate" and d.get("injected") is None
            and not d.get("known") and d.get("ra") is not None][:max_n]
    if not todo:
        return 0

    def near(d):
        ra0, de0 = float(d["ra"]), float(d["dec"])
        stars = None
        if online:
            try:
                stars, _ = VSX.query(ra0, de0, 1.2 * r_as / 3600.0, timeout=6.0)
            except Exception:              # noqa: BLE001 – fall back to the offline table
                stars = None
        if stars is None:
            stars = VSX.from_astap(var or {}, ra0, de0, 1.2 * r_as / 3600.0) if var else []
        VSX.offsets(stars, ra0, de0)
        res_ = []
        for v in stars:
            sep = math.hypot(v["de"], v["dn"]) * 3600.0
            if sep <= r_as:
                res_.append({k: v.get(k) for k in ("name", "type", "range", "period", "url", "max_mag", "de", "dn")}
                            | {"sep_arcsec": round(sep, 1)})
        res_.sort(key=lambda v: v["sep_arcsec"])
        res_ = res_[:5]
        if online and img_dir is not None:          # the survey map for every passed static source
            # the survey map for the card, fetched once here (STScI red plate directly – the colour CDS
            # service is often down and each card asking the server at once timed out)
            try:
                from . import sky as SKY
                # the card's own cut-out (S px) and a margin for the field rotation: the map is turned into
                # the cards' orientation in the browser
                S_ = float(((d.get("images") or {}).get("frames") or {}).get("size") or 64)
                fov = min(max(S_ * scale_arcsec * 1.45 if scale_arcsec else 8.0 * fwhm_arcsec, 90.0), 900.0) / 3600.0
                data, _ct, src = SKY.fetch(ra0, de0, fov, "DSS2r", pixels=360, timeout=8.0)
                name = f"vsxmap_{abs(hash((round(ra0, 6), round(de0, 6)))) % 10 ** 10:010d}.png"
                (Path(img_dir) / name).write_bytes(data)
                d.setdefault("images", {})["vsx_map"] = name
                d["vsx_map_fov"] = fov
                d["vsx_map_src"] = src
            except Exception:              # noqa: BLE001 – the card shows the names without the map
                pass
        return res_

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6 if online else 1) as ex:
        found = list(ex.map(near, todo))
    n = 0
    for d, vs in zip(todo, found):
        if not vs:
            continue
        d["vsx_near"] = vs
        v = vs[0]
        info = ", ".join(x for x in (v.get("type"), v.get("range")) if x)
        d.setdefault("flags", []).insert(1 if d["flags"] else 0,
                                         f"VSX {v['name']}" + (f" ({info})" if info else "") + f" – {v['sep_arcsec']:.0f}″"
                                         + (f" (+{len(vs) - 1} další)" if len(vs) > 1 else ""))
        n += 1
    return n
