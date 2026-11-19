"""Splitting a directory of frames into separate fields of view.

One night folder often holds several fields (a mosaic or a list of targets
visited in turn, e.g. 90 frames of 5 fields, interleaved in time). Each field
must be processed as its own job – registration and the static template only
make sense for frames of the same sky.

The pointing of every frame is taken from the header WCS (centre of the
image) or, without a WCS, from the RA/DEC keywords of the instrument key map.
Frames of one field differ only by dithering and drift, which is small
compared to the field of view, while different fields are separated by a
large fraction of it (≈ 0.8–0.9 FOV for a mosaic with overlap). Frames whose
pointings are closer than ``tolerance_fov`` × (smaller side of the FOV)
belong to the same field.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from astropy.wcs.utils import proj_plane_pixel_scales

from .astrometry import header_wcs
from .fitsmeta import list_fits, load_image, read_meta
from .instruments import Instrument

MIN_FRAMES_PER_FIELD = 5          # the pipeline needs at least this many frames


@dataclass
class FramePointing:
    path: str
    ra_deg: float | None
    dec_deg: float | None
    mjd: float | None
    fov_deg: tuple[float, float] | None     # (width, height)
    scale_deg: float | None                 # deg / px
    source: str                             # wcs | header | none
    object: str | None = None


@dataclass
class FieldGroup:
    index: int                    # 1-based, in order of the first frame
    label: str                    # OBJECT if unambiguous, else "Pole k"
    ra_deg: float | None
    dec_deg: float | None
    files: list[str]
    object: str | None = None
    spread_arcmin: float = 0.0    # max distance of a frame from the field centre (dithering)
    spread_px: float = 0.0
    mjd_first: float | None = None
    mjd_last: float | None = None
    source: str = "wcs"
    n_without_position: int = 0   # frames attached by folder / OBJECT / time, not by coordinates

    @property
    def n(self) -> int:
        return len(self.files)

    def describe(self) -> str:
        if self.ra_deg is None:
            return f"{self.label}: souřadnice neznámé, {self.n} snímků"
        return (f"{self.label}: RA {fmt_ra(self.ra_deg)}, Dec {fmt_dec(self.dec_deg)}, {self.n} snímků, "
                f"dithering do {self.spread_arcmin:.1f}′ ({self.spread_px:.0f} px)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n"] = self.n
        d["text"] = self.describe()
        return d


@dataclass
class FieldSplit:
    fields: list[FieldGroup]
    unassigned: list[str] = field(default_factory=list)   # frames without usable coordinates
    fov_deg: tuple[float, float] | None = None
    tolerance_deg: float | None = None
    notes: list[str] = field(default_factory=list)        # Czech, for logs / GUI
    n_frames: int = 0

    @property
    def multiple(self) -> bool:
        return len(self.fields) > 1

    def to_dict(self) -> dict:
        return {"fields": [f.to_dict() for f in self.fields], "unassigned": self.unassigned,
                "fov_deg": self.fov_deg, "tolerance_deg": self.tolerance_deg, "notes": self.notes,
                "n_frames": self.n_frames}


# ------------------------------------------------------------------ helpers
def fmt_ra(ra: float) -> str:
    h = (ra % 360.0) / 15.0
    hh = int(h)
    m = (h - hh) * 60.0
    if round(m, 1) >= 60.0:
        hh, m = (hh + 1) % 24, 0.0
    return f"{hh:02d}h{m:04.1f}m"


def fmt_dec(dec: float) -> str:
    s = "-" if dec < 0 else "+"
    d = abs(dec)
    dd = int(d)
    m = round((d - dd) * 60.0)
    if m == 60:
        dd, m = dd + 1, 0
    return f"{s}{dd:02d}°{m:02d}′"


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def _ang_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit vectors (robust for tiny angles)."""
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.sum(u * v, axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def _radec(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def frame_pointing(path: str | Path, inst: Instrument, header=None) -> FramePointing:
    """Sky position of the image centre and the field of view of one frame."""
    if header is None:
        header = fits.getheader(path)
    meta = read_meta(path, inst, header)
    w_px, h_px = meta.width, meta.height
    w = header_wcs(header)
    if w is not None and w_px and h_px:
        try:
            ra, dec = w.pixel_to_world_values((w_px - 1) / 2.0, (h_px - 1) / 2.0)
            scale = float(np.mean(np.abs(proj_plane_pixel_scales(w.celestial))))
            if not (np.isfinite(ra) and np.isfinite(dec)):
                raise ValueError
            if not scale or not np.isfinite(scale) or scale <= 0:
                scale = _inst_scale(inst, meta.xbin)
            return FramePointing(str(path), float(ra) % 360.0, float(dec), meta.mjd_mid,
                                 (w_px * scale, h_px * scale), scale, "wcs", meta.object)
        except Exception:
            pass
    scale = _inst_scale(inst, meta.xbin)
    fov = (w_px * scale, h_px * scale) if (w_px and h_px and scale) else None
    if meta.ra_deg is None or meta.dec_deg is None or not (-90.0 <= meta.dec_deg <= 90.0):
        return FramePointing(str(path), None, None, meta.mjd_mid, fov, scale, "none", meta.object)
    return FramePointing(str(path), meta.ra_deg % 360.0, meta.dec_deg, meta.mjd_mid, fov, scale, "header",
                         meta.object)


def _inst_scale(inst: Instrument, xbin: int) -> float | None:
    """deg/px from the instrument profile, corrected for the binning in the header."""
    s = inst.pixel_scale_arcsec
    if not s or s <= 0:
        return None
    return s / 3600.0 * max(1, int(xbin or 1)) / max(1, int(inst.binning or 1))


# ------------------------------------------------------------------ clustering
def cluster_pointings(pts: list[FramePointing], tolerance_deg: float) -> list[list[int]]:
    """Group frames whose pointings lie within ``tolerance_deg`` of a common centre.

    Greedy assignment in time order (a field is followed while it dithers or
    drifts), then fields closer than the tolerance are merged and every frame
    is re-assigned to the nearest field centre (two iterations), so the result
    does not depend on the order of the frames.
    """
    idx = [i for i, p in enumerate(pts) if p.ra_deg is not None]
    if not idx:
        return []
    order = sorted(idx, key=lambda i: (pts[i].mjd if pts[i].mjd is not None else math.inf, pts[i].path))
    vec = {i: _unit(pts[i].ra_deg, pts[i].dec_deg) for i in idx}
    groups: list[list[int]] = []
    centres: list[np.ndarray] = []
    for i in order:
        if centres:
            d = _ang_deg(np.array(centres), vec[i][None, :])
            j = int(np.argmin(d))
            if d[j] <= tolerance_deg:
                groups[j].append(i)
                c = np.sum([vec[k] for k in groups[j]], axis=0)
                centres[j] = c / np.linalg.norm(c)
                continue
        groups.append([i])
        centres.append(vec[i])
    # merge fields whose centres are within the tolerance
    merged = True
    while merged and len(groups) > 1:
        merged = False
        C = np.array(centres)
        for a in range(len(groups)):
            d = _ang_deg(C, C[a][None, :])
            d[a] = np.inf
            b = int(np.argmin(d))
            if d[b] <= tolerance_deg:
                lo, hi = min(a, b), max(a, b)
                groups[lo] += groups.pop(hi)
                centres.pop(hi)
                c = np.sum([vec[k] for k in groups[lo]], axis=0)
                centres[lo] = c / np.linalg.norm(c)
                merged = True
                break
    # re-assign to the nearest centre
    for _ in range(2):
        C = np.array(centres)
        new: list[list[int]] = [[] for _ in centres]
        for i in idx:
            new[int(np.argmin(_ang_deg(C, vec[i][None, :])))].append(i)
        keep = [k for k, g in enumerate(new) if g]
        groups = [new[k] for k in keep]
        centres = []
        for g in groups:
            c = np.sum([vec[k] for k in g], axis=0)
            centres.append(c / np.linalg.norm(c))
    return groups


def split_fields(paths: list[str | Path], inst: Instrument, tolerance_fov: float = 0.25) -> FieldSplit:
    """Read the headers of ``paths`` and split them into fields of view."""
    pts: list[FramePointing] = []
    unreadable: list[str] = []
    for p in paths:
        try:
            pts.append(frame_pointing(p, inst))
        except Exception:      # noqa: BLE001 – a broken file must not stop the split
            unreadable.append(str(p))
    split = FieldSplit(fields=[], n_frames=len(paths))
    if unreadable:
        split.notes.append(f"{len(unreadable)} souborů nelze přečíst (poškozená hlavička) – vynechány.")
    with_pos = [p for p in pts if p.ra_deg is not None]
    no_pos = [p.path for p in pts if p.ra_deg is None]
    fovs = [p.fov_deg for p in pts if p.fov_deg]
    if fovs:
        split.fov_deg = (float(np.median([f[0] for f in fovs])), float(np.median([f[1] for f in fovs])))
    if not with_pos:
        # no coordinates at all: split by the star patterns in the images
        if not pts:
            return split
        groups, unusable = split_by_stars([p.path for p in pts], tolerance_fov, [p.mjd for p in pts])
        if groups is None:
            split.notes.append("Hlavičky neobsahují souřadnice (WCS ani RA/DEC) a na snímcích nejsou měřitelné "
                               "hvězdy – snímky nelze rozdělit podle zorného pole, zpracují se jako jedno pole.")
            split.fields.append(_make_group(1, list(pts), "none"))
            return split
        made = [_make_group(0, [pts[i] for i in g], "stars") for g in groups]
        if unusable:
            split.unassigned = [pts[i].path for i in unusable]
            split.notes.append(f"{len(unusable)} snímků bez souřadnic má málo hvězd (mraky?) a nelze je podle "
                               f"hvězd přiřadit k poli – vynechány.")
        split.notes.append("Hlavičky neobsahují souřadnice – pole rozpoznána podle shody hvězd na snímcích"
                           + (f": {len(made)} zorná pole." if len(made) > 1 else ": jedno zorné pole."))
        _finish(made, split)
        return split
    if split.fov_deg is None:
        split.tolerance_deg = 0.5
        split.notes.append("Velikost zorného pole nelze určit (chybí měřítko) – tolerance 0,5°.")
    else:
        split.tolerance_deg = float(tolerance_fov) * min(split.fov_deg)
    groups = cluster_pointings(pts, split.tolerance_deg)
    made = []
    for g in groups:
        made.append(_make_group(0, [pts[i] for i in g], pts[g[0]].source))
    if no_pos:
        _assign_without_position(made, [p for p in pts if p.ra_deg is None], pts, split)
    _finish(made, split)
    if len(made) > 1:
        tol_arcmin = split.tolerance_deg * 60.0
        split.notes.append(f"Nalezeno {len(made)} zorných polí (snímky s rozdílem pointace pod {tol_arcmin:.1f}′ "
                           f"= {tolerance_fov:.0%} menší strany pole patří k jednomu poli).")
    for f in made:
        if f.spread_arcmin > 0.8 * split.tolerance_deg * 60.0:
            split.notes.append(f"{f.label}: velký rozptyl pointace ({f.spread_arcmin:.1f}′) blízko tolerance – "
                               f"zkontrolujte, zda nejde o dvě blízká pole.")
    return split


def _finish(made: list[FieldGroup], split: FieldSplit) -> None:
    made.sort(key=lambda f: (f.mjd_first if f.mjd_first is not None else math.inf, f.files[0]))
    for k, f in enumerate(made, 1):
        f.index = k
    _label(made)
    split.fields = made


# ------------------------------------------------------------------ star patterns
_STAR_CACHE: dict = {}
STAR_GRID_PX = 1024          # frames are binned to about this size for matching
MIN_STARS = 15


def frame_stars(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Star list (x, y, flux) of a raw frame, binned to ~``STAR_GRID_PX``.

    A 2×2 median bin (small frames: 3×3 median) removes hot pixels and cosmics
    (fixed on the chip, they would "match" between different fields), the background is removed on a coarse
    mesh and the image scaled to σ units. Cached per file (path, size, mtime)."""
    from .registration import detect_stars
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key in _STAR_CACHE:
        return _STAR_CACHE[key]
    from scipy import ndimage
    img, _ = load_image(p)
    img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    f = max(1, int(round(max(img.shape) / STAR_GRID_PX)))
    if f >= 2:
        # 2×2 median bin: a single hot pixel / cosmic never survives (8× faster than a 3×3 median filter)
        hh, ww = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
        q = img[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).transpose(0, 2, 1, 3).reshape(hh // 2, ww // 2, 4)
        q.sort(axis=2)
        img = 0.5 * (q[..., 1] + q[..., 2])
        f //= 2
    else:
        img = ndimage.median_filter(img, size=3)
    if f > 1:
        hh, ww = (img.shape[0] // f) * f, (img.shape[1] // f) * f
        img = img[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    m = 32
    hh, ww = (img.shape[0] // m) * m, (img.shape[1] // m) * m
    cells = np.median(img[:hh, :ww].reshape(hh // m, m, ww // m, m).transpose(0, 2, 1, 3).reshape(hh // m, ww // m, -1),
                      axis=2)
    bkg = ndimage.zoom(cells, (img.shape[0] / cells.shape[0], img.shape[1] / cells.shape[1]), order=1)
    res = img - bkg[:img.shape[0], :img.shape[1]]
    sig = 1.4826 * float(np.median(np.abs(res - np.median(res)))) or 1.0
    stars = detect_stars(res / sig, nsigma=8.0, max_n=800)
    out = (stars, img.shape)
    if len(_STAR_CACHE) > 4000:
        _STAR_CACHE.clear()
    _STAR_CACHE[key] = out
    return out


def same_field(a, b, tolerance_fov: float = 0.25) -> bool:
    """Do two star lists (from ``frame_stars``) show the same field of view?"""
    from .registration import match_frame, match_quality
    (sa, shape), (sb, _) = a, b
    if len(sa) < MIN_STARS or len(sb) < MIN_STARS:
        return False
    try:
        reg = match_frame(sa, sb, shape, order=1)
    except RuntimeError:
        return False
    ok, _ = match_quality(reg, sa, sb, shape, max_offset_px=tolerance_fov * min(shape))
    return ok


def split_by_stars(paths: list[str], tolerance_fov: float = 0.25, mjds: list | None = None):
    """Group frames without coordinates by matching their star patterns.

    Frames are taken in time order; each joins the first field whose first or
    most recent frame it matches, otherwise it starts a new field. Fields whose
    first frames match are merged afterwards. Returns ``(groups, unusable)``
    (lists of indices into ``paths``) or ``(None, [])`` when no frame has
    enough stars."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    def _one(p):
        try:
            return frame_stars(p)
        except Exception:  # noqa: BLE001 – unreadable image = unusable frame
            return (np.zeros((0, 3)), (1, 1))

    with ThreadPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 2) - 1))) as ex:
        data = list(ex.map(_one, paths))
    usable = [i for i, d in enumerate(data) if len(d[0]) >= MIN_STARS]
    unusable = [i for i in range(len(paths)) if i not in set(usable)]
    if not usable:
        return None, []
    mjd = {i: (mjds[i] if mjds is not None else None) for i in usable}
    order = sorted(usable, key=lambda i: (mjd[i] if mjd[i] is not None else math.inf, paths[i]))
    groups: list[list[int]] = []
    recent: list[int] = []              # group indices, most recently extended first
    for i in order:
        for gk in recent:
            g = groups[gk]
            if same_field(data[g[-1]], data[i], tolerance_fov) or (
                    len(g) > 1 and same_field(data[g[0]], data[i], tolerance_fov)):
                g.append(i)
                recent.remove(gk)
                recent.insert(0, gk)
                break
        else:
            groups.append([i])
            recent.insert(0, len(groups) - 1)
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if same_field(data[groups[a][0]], data[groups[b][0]], tolerance_fov):
                    groups[a] += groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    return groups, unusable


def _assign_without_position(fields: list[FieldGroup], orphans: list[FramePointing],
                             pts: list[FramePointing], split: FieldSplit) -> None:
    """Attach frames without coordinates (no WCS, no RA/DEC) to a field.

    * one field in the directory → all of them belong to it (the folder is the field);
    * several fields → the OBJECT keyword if it names exactly one field, otherwise
      the time sequence: a frame taken between two frames of the same field
      belongs to that field (at the start/end of the night the two nearest
      frames must be of the same field).
    Frames that stay ambiguous are left out and reported."""
    if not fields:
        split.unassigned = [p.path for p in orphans]
        return
    by_path = {p.path: p for p in pts}
    owner = {fp: k for k, f in enumerate(fields) for fp in f.files}
    added: dict[int, list[FramePointing]] = {}
    how = {"jediné pole": 0, "OBJECT": 0, "shoda hvězd": 0, "časová posloupnost": 0}
    left: list[str] = []
    if len(fields) == 1:
        added[0] = list(orphans)
        how["jediné pole"] = len(orphans)
    else:
        objects = {}
        for k, f in enumerate(fields):
            for fp in f.files:
                o = (by_path[fp].object or "").strip()
                if o:
                    objects.setdefault(o, set()).add(k)
        timeline = sorted(((by_path[fp].mjd, owner[fp]) for fp in owner if by_path[fp].mjd is not None))
        t_arr = np.array([t for t, _ in timeline]) if timeline else np.array([])
        reps = {}
        for p in orphans:
            o = (p.object or "").strip()
            if o and len(objects.get(o, ())) == 1:
                added.setdefault(next(iter(objects[o])), []).append(p)
                how["OBJECT"] += 1
                continue
            k = _match_by_stars(p.path, fields, reps)
            if k is not None:
                added.setdefault(k, []).append(p)
                how["shoda hvězd"] += 1
                continue
            if p.mjd is not None and t_arr.size:
                j = int(np.searchsorted(t_arr, p.mjd))
                if 0 < j < len(timeline):
                    # between two frames: both neighbours from the same field
                    if timeline[j - 1][1] == timeline[j][1]:
                        k = timeline[j][1]
                else:
                    # before the first / after the last frame: the two nearest
                    # frames must agree (interleaved fields stay ambiguous)
                    near = timeline[:2] if j == 0 else timeline[-2:]
                    if len(near) == 2 and near[0][1] == near[1][1]:
                        k = near[0][1]
            if k is None:
                left.append(p.path)
            else:
                added.setdefault(k, []).append(p)
                how["časová posloupnost"] += 1
    for k, extra in added.items():
        f = fields[k]
        members = [by_path[fp] for fp in f.files] + extra
        g = _make_group(f.index, members, f.source)
        g.n_without_position = len(extra)
        fields[k] = g
    done = sum(how.values())
    if done:
        why = ", ".join(f"{v}× {k}" for k, v in how.items() if v)
        split.notes.append(f"{done} snímků bez souřadnic v hlavičce přiřazeno k poli ({why}); "
                           f"astrometrie se pro ně převezme ze sesazení s ostatními snímky.")
    if left:
        split.unassigned = left
        split.notes.append(f"{len(left)} snímků bez souřadnic nelze jednoznačně přiřadit k žádnému poli "
                           f"(shoda hvězd ani časová posloupnost nerozhodla) – vynechány.")


def _match_by_stars(path: str, fields: list[FieldGroup], reps: dict) -> int | None:
    """Index of the only field whose representative frame shows the same stars."""
    try:
        s = frame_stars(path)
    except Exception:  # noqa: BLE001
        return None
    if len(s[0]) < MIN_STARS:
        return None
    hits = []
    for k, f in enumerate(fields):
        if k not in reps:
            try:
                reps[k] = frame_stars(f.files[len(f.files) // 2])
            except Exception:  # noqa: BLE001
                reps[k] = None
        if reps[k] is not None and same_field(reps[k], s):
            hits.append(k)
    return hits[0] if len(hits) == 1 else None


def _make_group(index: int, pts: list[FramePointing], source: str) -> FieldGroup:
    pts = sorted(pts, key=lambda p: (p.mjd if p.mjd is not None else math.inf, p.path))
    pos = [p for p in pts if p.ra_deg is not None]
    if pos:
        V = np.array([_unit(p.ra_deg, p.dec_deg) for p in pos])
        c = V.sum(axis=0)
        ra, dec = _radec(c)
        d = _ang_deg(V, (c / np.linalg.norm(c))[None, :])
        spread = float(d.max()) if len(d) else 0.0
        scales = [p.scale_deg for p in pos if p.scale_deg]
        spread_px = spread / float(np.median(scales)) if scales else 0.0
    else:
        ra = dec = None
        spread = spread_px = 0.0
    objs = {(p.object or "").strip() for p in pts}
    obj = objs.pop() if len(objs) == 1 else None
    mjds = [p.mjd for p in pts if p.mjd is not None]
    return FieldGroup(index=index, label="", ra_deg=ra, dec_deg=dec, files=[p.path for p in pts],
                      object=obj or None, spread_arcmin=spread * 60.0, spread_px=spread_px,
                      mjd_first=min(mjds) if mjds else None, mjd_last=max(mjds) if mjds else None, source=source)


def _label(fields: list[FieldGroup]) -> None:
    names = [f.object for f in fields]
    for f in fields:
        if f.object and names.count(f.object) == 1:
            f.label = f.object
        else:
            f.label = f"Pole {f.index}"


# ------------------------------------------------------------------ jobs
def field_job_name(base: str, f: FieldGroup, total: int) -> str:
    return f"{base} · {f.label} ({f.index}/{total})" if total > 1 else base


def create_field_jobs(store, inst: Instrument, lights_dir: str | Path, base_name: str, source: str = "manual",
                      params: dict | None = None, split: bool = True, tolerance_fov: float = 0.25,
                      select: list[int] | None = None):
    """Create one job per field of view found in ``lights_dir``.

    Returns ``(jobs, FieldSplit, skipped_notes)``. With a single field (or
    ``split=False``) exactly one job for the whole directory is created, as
    before. Jobs are queued in the order of the fields' first frames and the
    single worker processes them one after another.
    """
    lights_dir = str(lights_dir)
    paths = list_fits(lights_dir)
    if not split:
        fs = FieldSplit(fields=[], n_frames=len(paths))
        return [store.create(base_name, inst.id, lights_dir, source, params or {})], fs, []
    fs = split_fields(paths, inst, tolerance_fov)
    if not fs.multiple and not fs.unassigned:
        job = store.create(base_name, inst.id, lights_dir, source, params or {})
        if fs.fields:
            job.field = _field_record(fs.fields[0], 1, fs)
            store.save(job)
        return [job], fs, []
    jobs, skipped = [], []
    total = len(fs.fields)
    for f in fs.fields:
        if select is not None and f.index not in select:
            continue
        if f.n < MIN_FRAMES_PER_FIELD:
            skipped.append(f"{f.label}: jen {f.n} snímků (min. {MIN_FRAMES_PER_FIELD}) – nezpracováno.")
            continue
        job = store.create(field_job_name(base_name, f, total), inst.id, lights_dir, source, params or {})
        job.files = list(f.files)
        job.field = _field_record(f, total, fs)
        job.n_frames = f.n
        store.save(job)
        jobs.append(job)
    return jobs, fs, skipped


def _field_record(f: FieldGroup, total: int, fs: FieldSplit) -> dict:
    return {"index": f.index, "total": total, "label": f.label, "object": f.object,
            "ra_deg": f.ra_deg, "dec_deg": f.dec_deg, "n": f.n, "spread_arcmin": round(f.spread_arcmin, 2),
            "spread_px": round(f.spread_px, 1), "source": f.source, "text": f.describe(),
            "tolerance_arcmin": None if fs.tolerance_deg is None else round(fs.tolerance_deg * 60.0, 2)}


# ------------------------------------------------------------------ folder trees
_CALIB_WORDS = ("dark", "flat", "bias", "offset", "zero")


def _is_calibration_dir(files: list[Path]) -> bool:
    """A folder of darks/flats/biases (judged from the middle frame's IMAGETYP)."""
    try:
        h = fits.getheader(files[len(files) // 2])
    except Exception:  # noqa: BLE001
        return False
    typ = str(h.get("IMAGETYP") or h.get("FRAMETYP") or h.get("FRAME") or h.get("OBSTYPE") or "").lower()
    if not typ:
        return False
    return "light" not in typ and "object" not in typ and any(w in typ for w in _CALIB_WORDS)


def find_light_dirs(root: str | Path, exclude: list[str | Path] = ()) -> list[Path]:
    """``root`` and all its sub-directories (any depth) that contain light frames.

    Hidden folders, the instruments' master folders (``exclude``) and folders
    of darks/flats/biases are skipped. Sorted naturally by relative path, so
    ``Pole2`` comes before ``Pole10``.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    ex = []
    for e in exclude:
        if e:
            try:
                ex.append(Path(e).resolve())
            except OSError:
                pass
    cands = [root]
    try:
        cands += [p for p in root.rglob("*") if p.is_dir()
                  and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    except OSError:
        pass
    out = []
    for d in cands:
        r = d.resolve()
        if any(r == m or m in r.parents for m in ex):
            continue
        files = list_fits(d)
        if not files or _is_calibration_dir(files):
            continue
        out.append(d)
    return sorted(out, key=lambda p: _natural(p.relative_to(root).as_posix()))


def _natural(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def rel_name(root: str | Path, d: str | Path) -> str:
    rel = Path(d).relative_to(Path(root)).as_posix()
    return "" if rel == "." else rel


def create_tree_jobs(store, inst: Instrument, root: str | Path, base_name: str, source: str = "manual",
                     params: dict | None = None, split: bool = True, tolerance_fov: float = 0.25,
                     dirs: list[str] | None = None, select: list[int] | None = None,
                     exclude: list[str | Path] = ()):
    """Jobs for a folder that may hold pre-sorted fields in sub-folders.

    Every folder with light frames (``root`` itself and/or its sub-folders)
    becomes its own job(s); inside each folder the frames are additionally
    split by pointing (``create_field_jobs``), so a sub-folder that still
    mixes fields is handled too. ``dirs`` limits the folders (paths), ``select``
    the fields when only one folder is processed. Returns
    ``(jobs, [(dir, FieldSplit)], skipped_notes)`` in processing order.
    """
    root = Path(root)
    found = find_light_dirs(root, exclude)
    if dirs is not None:
        want = {str(Path(d).resolve()) for d in dirs}
        found = [d for d in found if str(d.resolve()) in want]
    single = len(found) == 1
    jobs, splits, skipped = [], [], []
    for d in found:
        rel = rel_name(root, d)
        name = f"{base_name}/{rel}" if rel else base_name
        js, fs, sk = create_field_jobs(store, inst, d, name, source, params, split=split,
                                       tolerance_fov=tolerance_fov, select=select if single else None)
        if rel:
            for j in js:
                f = {**(j.field or {}), "folder": rel}
                if not f.get("label") or (f.get("total", 1) == 1 and str(f["label"]).startswith("Pole ")):
                    old = f.get("label")
                    f["label"] = rel
                    if old and f.get("text"):
                        f["text"] = f["text"].replace(old, rel, 1)
                j.field = f
                store.save(j)
        jobs += js
        splits.append((d, fs))
        skipped += [f"{rel}: {s}" if rel else s for s in sk]
    return jobs, splits, skipped
