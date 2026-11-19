"""Splitting a folder with several fields of view into separate jobs."""
import math
import shutil

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta

from synthtrack.fields import create_field_jobs, split_fields
from synthtrack.fitsmeta import list_fits
from synthtrack.instruments import Instrument, InstrumentStore
from synthtrack.jobs import JobStore, RunContext
from synthtrack.pipeline import run_job
from synthtrack.settings import load_settings
from synthtrack.simulate import make_sequence

W, H = 640, 480
SCALE = 10.0 / 3600.0            # 10″/px -> FOV 1.78° × 1.33°


def _hdr(t, ra=None, dec=None, wcs=True, obj=None, dx=0.0, dy=0.0, keys=("RA", "DEC")):
    h = fits.Header()
    h["IMAGETYP"] = "Light Frame"
    h["EXPTIME"] = 30.0
    h["FILTER"] = "L"
    h["DATE-OBS"] = t.isot
    if obj:
        h["OBJECT"] = obj
    if ra is not None and wcs:
        h.update({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": ra, "CRVAL2": dec,
                  "CRPIX1": W / 2 + dx, "CRPIX2": H / 2 + dy,
                  "CD1_1": -SCALE, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": SCALE})
    elif ra is not None:
        h[keys[0]] = ra
        h[keys[1]] = dec
    return h


def _write(path, hdr):
    hdu = fits.PrimaryHDU(np.zeros((H, W), np.uint16), hdr)
    hdu.writeto(path, overwrite=True)


def _mosaic(d, centres, n_per_field=18, wcs=True, objects=None, dither_px=40, seed=3, keys=("RA", "DEC"),
            interleave=True):
    """Fields visited in turn (A B C D E A B C …) with random dithering;
    ``interleave=False``: one field after another (A A A … B B B …)."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    t0 = Time("2026-10-20T20:00:00", scale="utc")
    truth = {}
    k = 0
    order = [(rep, f) for rep in range(n_per_field) for f in range(len(centres))]
    if not interleave:
        order.sort(key=lambda rf: (rf[1], rf[0]))
    for rep, f in order:
        ra, dec = centres[f]
        t = t0 + TimeDelta(k * 35.0, format="sec")
        dx, dy = rng.uniform(-dither_px, dither_px, 2)
        if wcs:
            hdr = _hdr(t, ra, dec, True, objects[f] if objects else None, dx, dy)
        else:   # mount coordinates with the dither applied on the sky
            r = ra + dx * SCALE / math.cos(math.radians(dec))
            hdr = _hdr(t, r, dec + dy * SCALE, False, objects[f] if objects else None, keys=keys)
        p = d / f"img_{k:04d}.fits"
        _write(p, hdr)
        truth[str(p)] = f
        k += 1
    return truth


def _inst(**kw):
    # 10″/px: 4.85 µm at 100 mm
    return Instrument(id="m", focal_length_mm=100.0, pixel_size_um=4.848, **kw)


def _check(split, truth, n_fields, n_per_field):
    assert len(split.fields) == n_fields
    for f in split.fields:
        assert f.n == n_per_field
        assert len({truth[p] for p in f.files}) == 1          # no mixing
    assert not split.unassigned


def test_five_field_mosaic_with_dither(tmp_path):
    fov_w, fov_h = W * SCALE, H * SCALE
    # 3 + 2 mosaic, 15 % overlap, at Dec 40°
    cs = []
    for j in range(2):
        for i in range(3 if j == 0 else 2):
            cs.append((120.0 + i * 0.85 * fov_w / math.cos(math.radians(40)), 40.0 + j * 0.85 * fov_h))
    truth = _mosaic(tmp_path / "night", cs, objects=["M1", "M2", "M3", "M4", "M5"])
    split = split_fields(list_fits(tmp_path / "night"), _inst())
    _check(split, truth, 5, 18)
    assert [f.label for f in split.fields] == ["M1", "M2", "M3", "M4", "M5"]   # order of first frame
    assert all(f.spread_px < 60 for f in split.fields)
    assert split.tolerance_deg < 0.4 * fov_h


def test_header_coordinates_without_wcs_and_ra_wrap(tmp_path):
    # fields straddling RA 0h, sexagesimal mount coordinates, no WCS, same OBJECT for all
    cs = [(359.4, 10.0), (0.6, 10.0), (0.0, 11.1)]
    rng_truth = _mosaic(tmp_path / "n", cs, n_per_field=7, wcs=False, objects=["Mosaic"] * 3)
    for p in list_fits(tmp_path / "n"):          # convert to strings "hh mm ss" / "+dd mm ss"
        with fits.open(p, mode="update") as hd:
            h = hd[0].header
            ra, dec = h["RA"] % 360.0, h["DEC"]
            del h["RA"], h["DEC"]
            hh = ra / 15
            h["OBJCTRA"] = f"{int(hh):02d} {int(hh * 60 % 60):02d} {hh * 3600 % 60:05.2f}"
            h["OBJCTDEC"] = f"+{int(dec):02d} {int(dec * 60 % 60):02d} {dec * 3600 % 60:04.1f}"
    split = split_fields(list_fits(tmp_path / "n"), _inst())
    _check(split, rng_truth, 3, 7)
    assert [f.label for f in split.fields] == ["Pole 1", "Pole 2", "Pole 3"]   # OBJECT not unique
    assert split.fields[0].ra_deg > 359 and split.fields[1].ra_deg < 1


def test_single_field_and_missing_coordinates(tmp_path):
    truth = _mosaic(tmp_path / "a", [(80.0, -20.0)], n_per_field=10)
    s = split_fields(list_fits(tmp_path / "a"), _inst())
    _check(s, truth, 1, 10)
    assert not s.multiple
    d = tmp_path / "b"
    d.mkdir()
    for i in range(6):
        _write(d / f"x{i}.fits", _hdr(Time("2026-10-20T20:00:00") + TimeDelta(i * 30, format="sec")))
    s = split_fields(list_fits(d), _inst())
    assert len(s.fields) == 1 and s.fields[0].n == 6 and "nelze rozdělit" in s.notes[0]


def test_frames_without_coordinates_are_attached(tmp_path):
    """A frame without WCS/RA/DEC: one field -> the folder is the field;
    several fields -> OBJECT, or the time sequence when unambiguous."""
    _mosaic(tmp_path / "a", [(80.0, -20.0)], n_per_field=8)
    d = tmp_path / "a"
    t_mid = Time(fits.getheader(sorted(list_fits(d))[3])["DATE-OBS"]) + TimeDelta(5, format="sec")
    for i in range(3):
        _write(d / f"zz_nowcs{i}.fits", _hdr(t_mid + TimeDelta(i, format="sec")))
    s = split_fields(list_fits(d), _inst())
    assert len(s.fields) == 1 and s.fields[0].n == 11 and not s.unassigned
    assert s.fields[0].n_without_position == 3 and "jediné pole" in s.notes[0]
    # two fields in blocks: an orphan inside a block of field 2 belongs to it
    blocks = tmp_path / "b"
    _mosaic(blocks, [(80.0, -20.0), (82.0, -20.0)], n_per_field=6, interleave=False)
    files = sorted(list_fits(blocks), key=lambda p: Time(fits.getheader(p)["DATE-OBS"]).mjd)
    t_in = Time(fits.getheader(files[8])["DATE-OBS"]) + TimeDelta(1, format="sec")
    _write(blocks / "zz_orphan.fits", _hdr(t_in))
    s = split_fields(list_fits(blocks), _inst())
    assert [f.n for f in s.fields] == [6, 7] and not s.unassigned
    assert any("časová posloupnost" in n for n in s.notes)


def test_create_field_jobs_and_process_one_field(tmp_path):
    settings = load_settings(tmp_path / "data")
    settings.search.rate_min_arcsec_min = 0.5
    settings.search.rate_max_arcsec_min = 3.0
    lights = tmp_path / "mixed"
    lights.mkdir()
    # two fields, 256 px at 1″/px (FOV 4.3′), 3.6′ apart, interleaved in time
    for f, (dec, start) in enumerate([(20.0, "2026-10-20T22:00:00"), (20.06, "2026-10-20T22:00:20")]):
        make_sequence(tmp_path / f"f{f}", n=8, shape=(256, 256), drift_px_per_frame=(0.4, -0.3),
                      wcs_center=(45.0, dec), start=start, seed=5 + f)
        for p in list_fits(tmp_path / f"f{f}" / "lights"):
            shutil.copy(p, lights / f"F{f}_{p.name}")
    _write(lights / "zz_nocoords.fits", _hdr(Time("2026-10-20T22:10:00")))
    inst = Instrument(id="sim", masters_dir=str(tmp_path / "f0" / "masters"), pixel_size_um=4.85,
                      focal_length_mm=1000)
    InstrumentStore(settings.instruments_dir).save(inst)
    store = JobStore(settings.jobs_dir)
    jobs, fs, skipped = create_field_jobs(store, inst, lights, "noc", tolerance_fov=settings.field_tolerance_fov)
    assert len(jobs) == 2 and not skipped
    # after interleaved fields the frame is ambiguous -> left out
    assert fs.unassigned and fs.unassigned[0].endswith("zz_nocoords.fits")
    assert [j.n_frames for j in jobs] == [8, 8]
    assert all(all(f"F{k}_" in p for p in j.files) for k, j in enumerate(jobs))
    assert jobs[0].name == "noc · Pole 1 (1/2)" and jobs[1].field["index"] == 2
    job = store.get(jobs[1].id)                 # persisted
    run_job(job, RunContext(None, job, store), settings)
    assert job.n_frames == 8
    assert job.summary["field"]["index"] == 2
    # selecting one field only
    only, _, _ = create_field_jobs(store, inst, lights, "noc", select=[1])
    assert len(only) == 1 and only[0].field["index"] == 1


def test_pipeline_safety_net_takes_largest_field(tmp_path):
    settings = load_settings(tmp_path / "data")
    lights = tmp_path / "mixed"
    lights.mkdir()
    for f, (dec, n) in enumerate([(20.0, 8), (20.06, 5)]):
        make_sequence(tmp_path / f"f{f}", n=n, shape=(128, 128), wcs_center=(45.0, dec), seed=9 + f)
        for p in list_fits(tmp_path / f"f{f}" / "lights"):
            shutil.copy(p, lights / f"F{f}_{p.name}")
    inst = Instrument(id="sim", masters_dir=str(tmp_path / "f0" / "masters"), pixel_size_um=4.85,
                      focal_length_mm=1000)
    InstrumentStore(settings.instruments_dir).save(inst)
    store = JobStore(settings.jobs_dir)
    job = store.create("old-style", inst.id, str(lights))      # no explicit files
    run_job(job, RunContext(None, job, store), settings)
    assert job.n_frames == 8
    assert any("2 zorných polí" in w for w in job.warnings)


def test_watcher_creates_job_per_field(tmp_path):
    from synthtrack.watcher import Watcher

    settings = load_settings(tmp_path / "data")
    root = tmp_path / "night"
    truth = _mosaic(root / "20261020", [(10.0, 50.0), (13.0, 50.0), (16.0, 50.0)], n_per_field=6)
    inst = _inst(match={"IMAGETYP": "Light"})
    InstrumentStore(settings.instruments_dir).save(inst)
    settings.watch.root = str(root)
    settings.watch.frames_per_set = 10
    settings.watch.settle_seconds = 0

    class FakeRunner:
        submitted = []

        def submit(self, job):
            self.submitted.append(job.id)

    store = JobStore(settings.jobs_dir)
    w = Watcher(settings, store, FakeRunner())
    started = w.scan_once()
    assert len(started) == 3
    for jid in started:
        j = store.get(jid)
        assert len(j.files) == 6 and len({truth[p] for p in j.files}) == 1
    assert w.scan_once() == []


def _presorted(root, n=6):
    """root/<field>/ with one field each, plus a darks folder and a hidden folder."""
    cs = {"Pole2": (40.0, 10.0), "Pole10": (60.0, 10.0), "M31/noc1": (10.7, 41.3)}
    truth = {}
    for name, c in cs.items():
        truth.update({p: name for p in _mosaic(root / name, [c], n_per_field=n, objects=None)})
    (root / "darks").mkdir()
    for i in range(3):
        h = fits.Header({"IMAGETYP": "Dark Frame", "EXPTIME": 30.0})
        fits.PrimaryHDU(np.zeros((8, 8), np.uint16), h).writeto(root / "darks" / f"d{i}.fits")
    (root / ".thumbs").mkdir()
    _write(root / ".thumbs" / "x.fits", _hdr(Time("2026-10-20T20:00:00")))
    return truth


def test_presorted_subfolders_one_job_each(tmp_path):
    from synthtrack.fields import create_tree_jobs, find_light_dirs

    root = tmp_path / "noc"
    truth = _presorted(root)
    dirs = find_light_dirs(root)
    assert [d.relative_to(root).as_posix() for d in dirs] == ["M31/noc1", "Pole2", "Pole10"]   # natural order
    store = JobStore(tmp_path / "jobs")
    jobs, splits, skipped = create_tree_jobs(store, _inst(), root, "noc")
    assert [j.name for j in jobs] == ["noc/M31/noc1", "noc/Pole2", "noc/Pole10"] and not skipped
    for j in jobs:
        files = j.files or [str(p) for p in list_fits(j.lights_dir)]
        assert len(files) == 6 and len({truth[p] for p in files}) == 1
        assert j.field["folder"] == j.name.split("/", 1)[1] and j.field["label"] == j.field["folder"]
    # choose only some folders
    jobs2, _, _ = create_tree_jobs(store, _inst(), root, "noc", dirs=[str(root / "Pole10")])
    assert [j.name for j in jobs2] == ["noc/Pole10"]


def test_presorted_subfolder_that_still_mixes_fields(tmp_path):
    from synthtrack.fields import create_tree_jobs

    root = tmp_path / "noc"
    _mosaic(root / "A", [(40.0, 10.0)], n_per_field=6)
    _mosaic(root / "B", [(80.0, 10.0), (83.0, 10.0)], n_per_field=6)      # not sorted after all
    jobs, splits, _ = create_tree_jobs(JobStore(tmp_path / "jobs"), _inst(), root, "noc")
    assert [j.name for j in jobs] == ["noc/A", "noc/B · Pole 1 (1/2)", "noc/B · Pole 2 (2/2)"]


def test_api_presorted(tmp_path):
    from fastapi.testclient import TestClient
    from synthtrack.web.app import create_app

    settings = load_settings(tmp_path / "data")
    InstrumentStore(settings.instruments_dir).save(_inst())
    root = tmp_path / "noc"
    _presorted(root)
    c = TestClient(create_app(settings, start_worker=False))
    r = c.post("/api/instruments/m/preview", json={"lights_dir": str(root)}).json()
    assert r["frames"] == 18 and [d["rel"] for d in r["dirs"]] == ["M31/noc1", "Pole2", "Pole10"]
    r = c.post("/api/jobs", json={"instrument_id": "m", "lights_dir": str(root), "name": "noc",
                                  "dirs": [str(root / "Pole2"), str(root / "Pole10")]}).json()
    assert [j["name"] for j in r["jobs"]] == ["noc/Pole2", "noc/Pole10"]


def test_watcher_presorted_subfolders(tmp_path):
    from synthtrack.watcher import Watcher

    settings = load_settings(tmp_path / "data")
    root = tmp_path / "noc"
    _presorted(root)
    InstrumentStore(settings.instruments_dir).save(_inst(match={"IMAGETYP": "Light"}))
    settings.watch.root, settings.watch.frames_per_set, settings.watch.settle_seconds = str(root), 6, 0

    class FakeRunner:
        def submit(self, job):
            pass

    store = JobStore(settings.jobs_dir)
    started = Watcher(settings, store, FakeRunner()).scan_once()
    assert sorted(store.get(j).name for j in started) == ["noc/M31/noc1", "noc/Pole10", "noc/Pole2"]


def _strip_coords(path):
    with fits.open(path, mode="update") as hd:
        h = hd[0].header
        for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
                  "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "RA", "DEC", "OBJCTRA", "OBJCTDEC"):
            h.remove(k, ignore_missing=True, remove_all=True)
        h["OBJECT"] = "survey"          # one script for all fields -> the same OBJECT


def _fields_without_coords(tmp_path, n_fields=3, n_per_field=6, hot=150):
    """``n_fields`` star fields observed in turn, no WCS / RA / DEC, the same
    hot pixels on every frame (one chip)."""
    lights = tmp_path / "nocoords"
    lights.mkdir()
    truth = {}
    t0 = Time("2026-10-20T22:00:00")
    for f in range(n_fields):
        start = (t0 + TimeDelta(f * 30, format="sec")).isot
        make_sequence(tmp_path / f"s{f}", n=n_per_field, shape=(384, 384), n_stars=250, cadence_s=30.0 * n_fields,
                      drift_px_per_frame=(0.8, -0.6), start=start, seed=40 + f)
        for p in list_fits(tmp_path / f"s{f}" / "lights"):
            q = lights / f"S{f}_{p.name}"
            shutil.copy(p, q)
            truth[str(q)] = f
    rng = np.random.default_rng(0)
    hy, hx = rng.integers(0, 384, (2, hot))
    for p in truth:
        _strip_coords(p)
        with fits.open(p, mode="update") as hd:
            d = hd[0].data.astype(np.int32)
            d[hy, hx] = 60000
            hd[0].data = np.clip(d, 0, 65535).astype(np.uint16)
    return lights, truth


def test_split_by_stars_without_coordinates(tmp_path):
    lights, truth = _fields_without_coords(tmp_path)
    s = split_fields(list_fits(lights), _inst())
    assert len(s.fields) == 3 and not s.unassigned, s.notes
    for f in s.fields:
        assert f.n == 6 and f.source == "stars"
        assert len({truth[p] for p in f.files}) == 1          # never mixed
    assert any("shoda hvězd" in n or "shody hvězd" in n for n in s.notes)


def test_pipeline_drops_frames_of_another_field(tmp_path):
    """Safety net: frames that do not match the reference by stars are dropped
    (here: 2 frames of another field slipped into an explicit file list)."""
    settings = load_settings(tmp_path / "data")
    settings.search.rate_min_arcsec_min = 0.5
    settings.search.rate_max_arcsec_min = 3.0
    a = make_sequence(tmp_path / "a", n=8, shape=(256, 256), n_stars=150, drift_px_per_frame=(0.4, -0.3), seed=21)
    b = make_sequence(tmp_path / "b", n=2, shape=(256, 256), n_stars=150, seed=22,
                      start="2026-10-20T22:02:10")
    inst = Instrument(id="sim", masters_dir=str(tmp_path / "a" / "masters"), pixel_size_um=4.85,
                      focal_length_mm=1000)
    InstrumentStore(settings.instruments_dir).save(inst)
    store = JobStore(settings.jobs_dir)
    files = [str(p) for p in list(a) + list(b)]
    # 1) job without field info: the split by stars before the pipeline takes the larger field
    job = store.create("mix", inst.id, str(tmp_path / "a" / "lights"))
    job.files = files
    store.save(job)
    run_job(job, RunContext(None, job, store), settings)
    assert job.n_frames == 8 and any("2 zorných polí" in w for w in job.warnings)
    # 2) job already assigned to one field (as created by the GUI): the registration drops them
    job = store.create("mix2", inst.id, str(tmp_path / "a" / "lights"))
    job.files = files
    job.field = {"index": 1, "total": 1, "label": "Pole 1", "text": "Pole 1"}
    store.save(job)
    run_job(job, RunContext(None, job, store), settings)
    assert job.n_frames == 8
    assert any("nepodařilo sesadit" in w for w in job.warnings)
