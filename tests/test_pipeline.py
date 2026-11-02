import math
import threading
import time

import pytest
from astropy.io import fits
from astropy.time import Time

from synthtrack.export import mpc80_line
from synthtrack.fitsmeta import list_fits, read_meta
from synthtrack.gpu import Backend
from synthtrack.identify import predict_field
from synthtrack.instruments import Instrument, InstrumentStore, instrument_from_dict
from synthtrack.jobs import JobStore, RunContext
from synthtrack.masters import MasterLibrary
from synthtrack.mpcdata import MpcStore, parse_mpcorb
from synthtrack.pipeline import run_job
from synthtrack.settings import load_settings
from synthtrack.simulate import Mover, make_sequence
from test_identify import _mpcorb_line


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(tmp_path / "data")
    s.search.rate_min_arcsec_min = 0.3
    s.search.rate_max_arcsec_min = 3.0
    s.save()
    return s


def _run(settings, inst, lights):
    InstrumentStore(settings.instruments_dir).save(inst)
    store = JobStore(settings.jobs_dir)
    job = store.create("t", inst.id, str(lights))
    run_job(job, RunContext(None, job, store), settings)
    return job, store.read_json(job.id, "candidates.json"), store.read_json(job.id, "known.json")


def _near(cands, x, y, tol=3.0):
    return [c for c in cands if math.hypot(c["x"] - x, c["y"] - y) < tol and c["auto_status"] == "candidate"]


def test_end_to_end_detects_movers(settings, tmp_path):
    movers = [Mover(60, 80, 1.5, 0.8, 450), Mover(180, 200, -0.6, -1.2, 600)]
    make_sequence(tmp_path / "seq", n=20, shape=(256, 256), movers=movers, drift_px_per_frame=(0.3, -0.2),
                  gradient=0.3, vignetting=0.3, hot_pixels=20)
    inst = Instrument(id="sim", masters_dir=str(tmp_path / "seq" / "masters"), pixel_size_um=4.85,
                      focal_length_mm=1000, observatory_code="B12", observer_lat_deg=49.2,
                      observer_lon_deg=16.6, observer_alt_m=320, site_from_header=False)
    job, cands, _ = _run(settings, inst, tmp_path / "seq" / "lights")
    assert job.summary["observatory_code"] == "B12"
    assert job.summary["site"] == [49.2, 16.6, 320]
    assert job.summary["site_source"] == "profil přístroje"
    tref = 10 * 40 / 60 + 0.25          # minutes from first frame (ref = middle frame)
    for m in movers:
        x = m.x0 + m.vx * tref + 0.3 * 10
        y = m.y0 + m.vy * tref - 0.2 * 10
        hit = _near(cands, x, y)
        assert hit, f"mover at {x:.1f},{y:.1f} not found"
        assert abs(hit[0]["vx"] - m.vx) < 0.3 and abs(hit[0]["vy"] - m.vy) < 0.3
    # the real movers must be the strongest candidates; few false ones on a dense synthetic field
    passed = sorted([c for c in cands if c["auto_status"] == "candidate"], key=lambda c: -c["snr"])
    top = passed[:len(movers)]
    for m in movers:
        x = m.x0 + m.vx * tref + 0.3 * 10
        y = m.y0 + m.vy * tref - 0.2 * 10
        assert any(math.hypot(c["x"] - x, c["y"] - y) < 3 for c in top)
    assert len(passed) <= len(movers) + 5


def test_known_object_identified(settings, tmp_path):
    line = _mpcorb_line(name="(99999) Testik", H=12.0, epoch="K26AK", M=100.0, a=2.6, e=0.12, inc=6.0,
                        n=math.degrees(0.01720209895 / 2.6 ** 1.5))
    el = parse_mpcorb([line])
    MpcStore(settings.catalogs_dir).save("mpcorb", el, "test")
    start = Time("2026-10-20T22:00:00", scale="utc")
    t_mid = start + (10 * 40 + 15) / 86400.0
    # predicted position: search a wide field around a first guess
    k = predict_field(el, None, t_mid, 0, 0, 180, settings.observer_lat_deg, settings.observer_lon_deg,
                      settings.observer_alt_m, mag_limit=99)[0]
    scale = 1.5
    # pixel velocity for north up / east left
    pa = math.radians(k.pa_deg)
    vx = -k.rate_arcsec_min * math.sin(pa) / scale
    vy = k.rate_arcsec_min * math.cos(pa) / scale
    tref = 10 * 40 / 60 + 0.25
    xc = yc = 127.5
    mover = Mover(xc - vx * tref, yc - vy * tref, vx, vy, 700)
    make_sequence(tmp_path / "seq", n=20, shape=(256, 256), movers=[mover], wcs_center=(k.ra, k.dec),
                  scale_arcsec=scale, seed=3)
    inst = Instrument(id="simw", masters_dir=str(tmp_path / "seq" / "masters"), pixel_size_um=7.27, focal_length_mm=1000)
    job, cands, known = _run(settings, inst, tmp_path / "seq" / "lights")
    assert job.summary["wcs"]
    hit = _near(cands, xc, yc)
    assert hit and hit[0]["known"] and hit[0]["known"]["name"] == "(99999) Testik"
    assert hit[0]["known"]["o_c_arcsec"] < 5
    assert any(kk["matched_candidate"] == hit[0]["id"] for kk in known)
    assert abs(hit[0]["rate_arcsec_min"] - k.rate_arcsec_min) < 0.2


def test_master_selection(tmp_path):
    make_sequence(tmp_path / "a", n=5, shape=(64, 64), exptime=30, filter_name="Luminance")
    inst = Instrument(id="x", masters_dir=str(tmp_path / "a" / "masters"))
    lib = MasterLibrary(inst)
    lib.scan()
    light = read_meta(list_fits(tmp_path / "a" / "lights")[0], inst)
    assert light.filter == "L"            # alias Luminance -> L
    assert lib.select_dark(light).ok
    assert lib.select_flat(light).ok      # flat FILTER=Luminance also canonical L
    light.exptime = 60
    s = lib.select_dark(light)
    assert not s.ok and "60" in s.reason
    light.filter = "R"
    assert not lib.select_flat(light).ok


def test_master_selection_rejects_wrong_detector_size(tmp_path):
    make_sequence(tmp_path / "a", n=5, shape=(64, 64), exptime=30, filter_name="L")
    dark = next((tmp_path / "a" / "masters").glob("*dark*"))
    h = fits.getheader(dark)
    h["IMAGETYP"] = "Dark"
    fits.writeto(tmp_path / "a" / "masters" / "wrong_dark.fits", data=[[0.0] * 32] * 32,
                 header=h, overwrite=True)
    inst = Instrument(id="geometry", masters_dir=str(tmp_path / "a" / "masters"))
    lib = MasterLibrary(inst)
    lib.scan()
    light = read_meta(list_fits(tmp_path / "a" / "lights")[0], inst)
    assert lib.select_dark(light).path != str(tmp_path / "a" / "masters" / "wrong_dark.fits")


def test_instrument_site_roundtrip_and_validation(tmp_path):
    store = InstrumentStore(tmp_path)
    inst = instrument_from_dict({"id": "site", "observatory_code": "b12", "observer_lat_deg": 49.2,
                                 "observer_lon_deg": 16.6, "observer_alt_m": 320})
    store.save(inst)
    loaded = store.get("site")
    assert (loaded.observatory_code, loaded.observer_lat_deg, loaded.observer_lon_deg,
            loaded.observer_alt_m) == ("B12", 49.2, 16.6, 320)
    with pytest.raises(ValueError, match="šířka"):
        instrument_from_dict({"id": "bad", "observer_lat_deg": 91})


def test_mpc80_format():
    l = mpc80_line("K26U01A", 61333.912345, 83.0, 33.5, 18.4, "G", "B12", discovery=True)
    assert len(l) == 80
    assert l[5:12] == "K26U01A" and l[12] == "*" and l[14] == "C"
    assert l[15:32].startswith("2026 10 20.91234")
    assert l[32:44].strip() == "05 32 00.00" and l[44:56].strip() == "+33 30 00.0"
    assert l[65:70].strip() == "18.4" and l[70] == "G" and l[77:80] == "B12"


def test_atomic_json_updates_preserve_concurrent_changes(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create("atomic", "x", str(tmp_path))
    store.write_json(job.id, "candidate.json", {"a": 0, "b": 0})
    barrier = threading.Barrier(2)

    def update(key):
        barrier.wait()
        store.update_json(job.id, "candidate.json", lambda d: {**d, key: 1}, {})

    threads = [threading.Thread(target=update, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.read_json(job.id, "candidate.json") == {"a": 1, "b": 1}


def test_gpu_cache_is_scoped_to_device_and_memory_limit(monkeypatch):
    import synthtrack.gpu as gpu

    calls = []

    def fake(device, memory_fraction):
        calls.append((device, memory_fraction))
        return Backend("fake", object(), device=device)

    monkeypatch.setattr(gpu, "_try_cupy", fake)
    monkeypatch.setattr(gpu, "_cached", None)
    monkeypatch.setattr(gpu, "_cached_config", None)
    assert gpu.get_backend("auto", 0, 0.5).device == 0
    assert gpu.get_backend("auto", 0, 0.5).device == 0
    assert gpu.get_backend("auto", 1, 0.7).device == 1
    assert calls == [(0, 0.5), (1, 0.7)]


def test_api_smoke(settings, tmp_path):
    from fastapi.testclient import TestClient

    from synthtrack.web.app import create_app
    make_sequence(tmp_path / "seq", n=8, shape=(128, 128), movers=[Mover(40, 40, 2.0, 1.0, 1500)])
    client = TestClient(create_app(settings))
    r = client.post("/api/instruments", json={"id": "api", "name": "API test", "masters_dir": str(tmp_path / "seq" / "masters"),
                                             "pixel_size_um": 4.85, "focal_length_mm": 1000})
    assert r.status_code == 200 and r.json()["pixel_scale_arcsec"] == pytest.approx(1.0, abs=0.01)
    r = client.post("/api/instruments/api/preview", json={"lights_dir": str(tmp_path / "seq" / "lights")})
    assert r.json()["selection"]["dark"]["path"]
    r = client.post("/api/jobs", json={"instrument_id": "api", "lights_dir": str(tmp_path / "seq" / "lights")})
    jid = r.json()["jobs"][0]["id"]
    for _ in range(120):
        st = client.get(f"/api/jobs/{jid}").json()["status"]
        if st in ("done", "failed"):
            break
        time.sleep(0.5)
    assert st == "done", client.get(f"/api/jobs/{jid}/log").text
    cands = client.get(f"/api/jobs/{jid}/candidates").json()
    assert cands
    c0 = cands[0]
    assert client.patch(f"/api/jobs/{jid}/candidates/{c0['id']}", json={"status": "accepted"}).json()["status"] == "accepted"
    assert client.get(f"/api/jobs/{jid}/img/{c0['images']['gif']}").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/catalogs/update", json={"kind": "mpcorb"}).status_code == 403   # offline by default


def test_watcher_starts_job_for_complete_set(settings, tmp_path):
    from synthtrack.jobs import JobStore
    from synthtrack.watcher import Watcher

    root = tmp_path / "night"
    make_sequence(root / "20261020" / "T1", n=6, shape=(64, 64))
    make_sequence(root / "20261020" / "T2", n=3, shape=(64, 64))      # incomplete set
    inst = Instrument(id="w", masters_dir=str(root / "20261020" / "T1" / "masters"), match={"IMAGETYP": "Light"})
    InstrumentStore(settings.instruments_dir).save(inst)
    settings.watch.root = str(root)
    settings.watch.frames_per_set = 6
    settings.watch.settle_seconds = 0
    store = JobStore(settings.jobs_dir)

    class FakeRunner:
        submitted = []

        def submit(self, job):
            self.submitted.append(job.id)

    w = Watcher(settings, store, FakeRunner())
    started = w.scan_once()
    assert len(started) == 1
    job = store.get(started[0])
    assert job.lights_dir.endswith("T1" + __import__("os").sep + "lights") and job.instrument_id == "w"
    assert any("T2" in d for d in w.pending)
    assert w.scan_once() == []            # not processed twice
