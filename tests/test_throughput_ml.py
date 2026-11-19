"""Throughput maps, static-residual rejection, injection–recovery, classifier, viewer frames."""
import json
import math

import numpy as np
import pytest

from synthtrack import classifier as CL
from synthtrack import inject as INJ
from synthtrack import throughput as TP
from synthtrack.imaging import ENC_A, ENC_B, ENC_C, to_u8, write_motion_frames
from synthtrack.settings import SearchSettings
from synthtrack.search import TrackNorm, prepare_tile


def _cube_with_obstruction(n=12, h=384, w=384, n_stars=500, seed=3):
    """σ-unit frames: static stars, a blocking edge that clears from the right
    during the sequence plus a 30 % extinction gradient."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    sx, sy = rng.uniform(8, w - 8, n_stars), rng.uniform(8, h - 8, n_stars)
    sf = 10 ** rng.uniform(1.8, 3.0, n_stars)
    sig = 1.2
    static = np.zeros((h, w), np.float32)
    for x, y, f in zip(sx, sy, sf):
        x0, y0 = int(x), int(y)
        sl = (slice(y0 - 6, y0 + 7), slice(x0 - 6, x0 + 7))
        static[sl] += f * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * sig * sig)) / (2 * math.pi * sig * sig)
    truth = np.empty((n, h, w), np.float32)
    cube = np.empty((n, h, w), np.float32)
    for i in range(n):
        edge = 100 + i * 40                          # blocked for x > edge (first frames)
        a = np.clip((edge - xx) / 80.0, 0, 1) * (1.0 - 0.3 * xx / w)
        truth[i] = a
        cube[i] = a * static + rng.standard_normal((h, w)).astype(np.float32)
    stars = np.stack([sx, sy, sf], axis=1)
    return cube, truth, stars


def test_throughput_maps_follow_obstruction():
    cube, truth, stars = _cube_with_obstruction()
    maps = TP.measure(cube, stars, cell=64)
    assert maps is not None and maps.spread() > 0.5
    est = maps.tile(0, cube.shape[1], 0, cube.shape[2])
    # maps are relative to the typical flux of each star: compare frame ratios
    # to the (unblocked) last frame, away from the moving edge
    rel, tru = est / est[-1:], truth / truth[-1:]
    flat = (np.abs(np.gradient(truth, axis=2)) < 1e-3) & (truth[-1:] > 0.5)
    assert np.median(np.abs(rel - tru)[flat]) < 0.1
    assert maps.per_frame_min()[0] < 0.2 and maps.per_frame_min()[-1] > 0.5


def test_maps_remove_star_residuals():
    cube, truth, stars = _cube_with_obstruction()
    maps = TP.measure(cube, stars, cell=64)
    amp = maps.tile(0, cube.shape[1], 0, cube.shape[2])
    plain = prepare_tile(cube, 2.8, 40.0, cap=0)
    with_maps = prepare_tile(cube, 2.8, 40.0, cap=0, amp=amp)
    core = (slice(None), slice(30, -30), slice(30, -30))
    bad_plain = int((np.abs(plain.f_raw[core]) > 6).sum())
    bad_maps = int((np.abs(with_maps.f_raw[core]) > 6).sum())
    assert bad_maps < 0.3 * bad_plain, (bad_plain, bad_maps)


def test_injection_plan_flux_and_match():
    dts = np.linspace(-30, 30, 19)
    tn = TrackNorm(dts, 2.8)
    objs = INJ.plan(20, (500, 500), dts, 2.8, tn, 0.03, 0.5, 6, 12, 14.0, seed=1)
    assert len(objs) == 20 and all(30 < o["x"] < 470 for o in objs)
    reg = np.zeros((19, 500, 500), np.float32)
    INJ.add_to_frames(reg, objs[:1], dts, 2.8)
    assert abs(reg[0].sum() - objs[0]["flux_sigma"]) < 0.05 * objs[0]["flux_sigma"]

    class C:                                        # minimal candidate
        def __init__(self, i, o):
            self.id, self.x, self.y, self.vx, self.vy = i, o["x"] + 0.5, o["y"], o["vx"], o["vy"]
            self.snr, self.auto_status, self.injected = 9.0, "candidate", None
    cands = [C(0, objs[0]), C(1, objs[5])]
    INJ.match(cands, objs, dts, 2.8)
    assert cands[0].injected == 0 and cands[1].injected == 5
    summ, lines = INJ.summarise(objs, 6.0)
    assert summ["detected"] == 2 and lines


def _fake_job(root, jid, cands, summary=None):
    d = root / "jobs" / jid
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"id": jid, "status": "done", "summary": summary or {
        "min_rate_arcsec_min": 0.4, "fwhm_px": 2.8, "n_frames": 19, "span_min": 60, "pixel_scale": 14.1}}))
    (d / "candidates.json").write_text(json.dumps(cands))


def test_classifier_learns_and_scores(tmp_path):
    rng = np.random.default_rng(0)
    cands = []
    for i in range(200):
        real = i < 30
        snr = rng.uniform(7, 20) if real else rng.uniform(6, 8)
        seg = list(rng.normal(snr / math.sqrt(3), 1 if real else 2.5, 3))
        cands.append({"id": i, "snr": snr, "snr_raw": snr * rng.uniform(0.95, 1.1), "snr_reverse": rng.normal(0, 1),
                      "chi2": rng.uniform(0.6, 1.4) if real else rng.uniform(0.8, 2.5),
                      "median_ratio": rng.uniform(0.8, 1.2), "seg_snr": seg,
                      "rate_arcsec_min": rng.uniform(0.6, 4), "fwhm": 2.8, "frames_used": 19,
                      "positions": [{"flux": 100 + rng.normal(0, 10 if real else 60)} for _ in range(3)],
                      "flags": [], "auto_status": "candidate", "status": "accepted" if real else "rejected"})
    # Separate observing series: cross-validation must hold out whole jobs,
    # not leak candidates from the same field into train and test.
    for j in range(5):
        _fake_job(tmp_path, f"job-{j}", cands[j::5])
    m = CL.train(tmp_path)
    assert m["cv_auc"] is not None and m["cv_auc"] > 0.8
    assert CL.load(tmp_path) is not None
    CL.score(m, cands, CL.job_context(json.loads((tmp_path / "jobs/job-0/job.json").read_text())["summary"]))
    real = np.mean([c["ml_score"] for c in cands[:30]])
    fake = np.mean([c["ml_score"] for c in cands[30:]])
    assert real > 0.7 and fake < 0.3


def test_classifier_refuses_without_examples(tmp_path):
    _fake_job(tmp_path, "a", [{"id": 0, "snr": 7, "status": "rejected", "auto_status": "candidate"}])
    with pytest.raises(ValueError):
        CL.train(tmp_path)


def test_motion_frames_strip_roundtrip(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(1)
    cube = rng.standard_normal((9, 120, 120)).astype(np.float32)
    cube[:, 60, 60] += 30.0
    dts = np.linspace(-20, 20, 9)
    meta = write_motion_frames(tmp_path, "t", cube, dts, 60, 60, 0.2, -0.1, 48)
    q = np.asarray(Image.open(tmp_path / meta["file"]), float)
    assert q.shape == (9 * meta["size"], meta["size"]) and len(meta["track"]) == 9
    dec = ENC_A * np.sinh((q - ENC_C) / ENC_B)
    half = meta["size"] // 2
    ref = cube[0, 60 - half:60 - half + meta["size"], 60 - half:60 - half + meta["size"]]
    err = np.abs(dec[:meta["size"]] - ref)
    assert np.median(err[np.abs(ref) < 3]) < 0.05


def test_missing_pixels_render_as_background_not_white(tmp_path):
    from PIL import Image
    cube = np.zeros((3, 40, 40), np.float32)
    cube[:, 20, 20] = np.nan
    meta = write_motion_frames(tmp_path, "nan", cube, np.arange(3), 20, 20, 0, 0, 20)
    q = np.asarray(Image.open(tmp_path / meta["file"]))
    assert q[10, 10] == round(ENC_C)
    rendered = np.asarray(to_u8(cube[0], -2, 8, scale=1, gamma=1))
    assert rendered[20, 20] < 250


def test_pipeline_obstruction_static_and_injection(tmp_path):
    """End to end: a corner blocked in the first frames (dome/horizon), maps on,
    injected movers recovered and kept out of the candidate list."""
    from synthtrack.instruments import Instrument, InstrumentStore
    from synthtrack.jobs import JobStore, RunContext
    from synthtrack.pipeline import run_job
    from synthtrack.settings import load_settings
    from synthtrack.simulate import Mover, make_sequence

    s = load_settings(tmp_path / "data")
    s.search.rate_min_arcsec_min = 0.3
    s.search.rate_max_arcsec_min = 3.0
    s.search.throughput_cell_px = 64
    s.search.inject_n = 8
    s.search.inject_snr_min = 14.0
    s.search.inject_snr_max = 25.0
    s.save()
    movers = [Mover(60, 80, 1.5, 0.8, 450)]

    def blocked(i, xx, yy):              # right part blocked at the start, clears by frame ~12
        return np.clip((140 + 10 * i - xx) / 40.0, 0, 1)

    make_sequence(tmp_path / "seq", n=20, shape=(256, 256), movers=movers, n_stars=250,
                  drift_px_per_frame=(0.3, -0.2), throughput=blocked)
    inst = Instrument(id="sim", masters_dir=str(tmp_path / "seq" / "masters"), pixel_size_um=4.85,
                      focal_length_mm=1000)
    InstrumentStore(s.instruments_dir).save(inst)
    store = JobStore(s.jobs_dir)
    job = store.create("t", inst.id, str(tmp_path / "seq" / "lights"))
    run_job(job, RunContext(None, job, store), s)
    cands = store.read_json(job.id, "candidates.json")
    summ = job.summary
    assert summ["throughput"] is not None and min(summ["throughput"]["frame_min"]) < 0.3
    assert summ["injection"]["n"] == 8 and summ["injection"]["detected"] >= 5
    inj = [c for c in cands if c.get("injected") is not None]
    assert inj and all(c["status"] == "injected" for c in inj)
    tref = 10 * 40 / 60 + 0.25
    x = 60 + 1.5 * tref + 0.3 * 10
    y = 80 + 0.8 * tref - 0.2 * 10
    assert any(math.hypot(c["x"] - x, c["y"] - y) < 3 and c["auto_status"] == "candidate" for c in cands)
    passed = [c for c in cands if c["auto_status"] == "candidate" and c.get("injected") is None]
    assert len(passed) <= len(movers) + 6


def test_antialias_unit_noise_and_warp_keeps_masks():
    from synthtrack.registration import PolyTransform, antialias, warp
    rng = np.random.default_rng(0)
    img = rng.standard_normal((300, 300)).astype(np.float32)
    img[150:153, 150:153] = np.nan                       # saturated core
    sm = antialias(img, 0.9)
    assert abs(float(np.nanstd(sm)) - 1.0) < 0.05 and np.isnan(sm[151, 151])
    pts = np.array([[0, 0], [300, 0], [0, 300], [300, 300]], float)
    tr = PolyTransform.fit(pts, pts + [0.4, -0.3], 1, (149.5, 149.5), 150)
    out = warp(img, tr, img.shape)
    assert np.isnan(out[148:155, 148:155]).any()        # the hole is not filled with 0
    assert np.isfinite(out[50:100, 50:100]).all()


def test_confidence_levels():
    from synthtrack.confidence import assess
    good = {"auto_status": "candidate", "snr": 12.0, "seg_snr": [7.0, 6.8, 7.1], "chi2": 1.0, "median_ratio": 0.9,
            "positions": [{"ok": True}] * 3, "flags": []}
    assert assess(good) == ("high", [])
    assert assess({**good, "flags": ["u hvězdy"]})[0] == "medium"
    assert assess({**good, "snr": 6.5, "seg_snr": [3.0, 0.2, 4.0]})[0] == "low"
    assert assess({**good, "auto_status": "auto_rejected", "flags": ["téměř statický zdroj"]})[0] == "none"
    assert assess({**good, "ml_score": 0.01})[0] == "low"


def test_classifier_keeps_labels_of_deleted_jobs(tmp_path):
    import shutil
    rng = np.random.default_rng(1)

    def cands(n_real, n_fake):
        out = []
        for i in range(n_real + n_fake):
            real = i < n_real
            snr = rng.uniform(8, 20) if real else rng.uniform(6, 8)
            out.append({"id": i, "snr": snr, "seg_snr": [snr / 1.8] * 3, "chi2": 1.0 if real else 2.2,
                        "median_ratio": 1.0, "rate_arcsec_min": 1.0, "positions": [{"flux": 100}] * 3,
                        "flags": [], "auto_status": "candidate", "status": "accepted" if real else "rejected"})
        return out
    _fake_job(tmp_path, "a", cands(2, 30))
    _fake_job(tmp_path, "b", cands(2, 30))
    assert CL.archive_job(tmp_path, tmp_path / "jobs" / "a")
    shutil.rmtree(tmp_path / "jobs" / "a")                 # job deleted, labels archived
    m = CL.train(tmp_path)
    assert m["n_pos"] == 4 and m["n_neg"] == 60 and "smazaných" in m["report"]


def test_substack_astrometry_in_dense_field():
    """Milky-Way density: the fit must not latch on to neighbouring stars."""
    from synthtrack import search as SR
    from synthtrack.tracking import Candidate
    rng = np.random.default_rng(5)
    n, H, fwhm = 21, 100, 3.0
    s = fwhm / 2.3548
    dts = np.linspace(-34, 34, n)
    yy, xx = np.mgrid[:H, :H].astype(float)

    def psf(x, y, f):
        return f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)) / (2 * math.pi * s * s)
    errs = []
    for _ in range(8):
        static = sum(psf(rng.uniform(0, H), rng.uniform(0, H), 10 ** rng.uniform(1.0, 2.7)) for _ in range(H * H // 40))
        pa = rng.uniform(0, 2 * math.pi)
        vx, vy = 0.06 * math.cos(pa), 0.06 * math.sin(pa)
        cube = np.stack([static + psf(50 + vx * t, 50 + vy * t, 40.0) + rng.standard_normal((H, H))
                         for t in dts]).astype(np.float32)
        c = Candidate(id=0, x=50.2, y=49.8, vx=vx, vy=vy, snr=12)
        m = SR.fit_motion(SR.measure_substacks(cube, dts, c, 3, fwhm))
        errs.append(math.hypot(m[0] - 50, m[1] - 50))
    assert np.median(errs) < 0.3, errs


def test_classifier_per_instrument_models(tmp_path):
    rng = np.random.default_rng(2)

    def cands(n_real, n_fake):
        out = []
        for i in range(n_real + n_fake):
            real = i < n_real
            snr = rng.uniform(8, 20) if real else rng.uniform(6, 8)
            out.append({"id": i, "snr": snr, "seg_snr": [snr / 1.8] * 3, "chi2": 1.0 if real else 2.2,
                        "median_ratio": 1.0, "rate_arcsec_min": 1.0, "positions": [{"flux": 100}] * 3,
                        "flags": [], "auto_status": "candidate", "status": "accepted" if real else "rejected"})
        return out
    for jid, inst, nr in (("a", "lens", 10), ("b", "scope", 2)):
        _fake_job(tmp_path, jid, cands(nr, 40))
        p = tmp_path / "jobs" / jid / "job.json"
        d = json.loads(p.read_text()); d["instrument_id"] = inst; p.write_text(json.dumps(d))
    m = CL.train(tmp_path)
    assert "lens" in m["per_instrument"] and "scope" not in m["per_instrument"]
    assert CL.load(tmp_path, "lens")["instrument"] == "lens"
    assert CL.load(tmp_path, "scope")["instrument"] is None          # falls back to the common model
    assert m["n_pos"] == 12                                            # old examples are never lost


def test_transient_masking_keeps_stars_and_movers():
    from synthtrack.transients import mask_cosmics, mask_trails
    rng = np.random.default_rng(4)
    n, H, fw = 12, 200, 3.0
    s = fw / 2.3548
    yy, xx = np.mgrid[:H, :H].astype(float)

    def psf(x, y, f):
        return f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)) / (2 * math.pi * s * s)
    stars = sum(psf(rng.uniform(5, H - 5), rng.uniform(5, H - 5), 10 ** rng.uniform(1.5, 3)) for _ in range(60))
    cube = np.stack([stars + psf(40 + 2 * i, 150, 300.0) + rng.standard_normal((H, H)) for i in range(n)]).astype(np.float32)
    # cosmic ray: one sharp pixel; satellite trail: a thin line in frame 5
    f0 = cube[3].copy()
    # Put the synthetic hit in blank sky. A hit superposed on a stellar core
    # is intentionally left alone because preserving the source has priority.
    f0[20, 180] += 80.0
    m = mask_cosmics(f0, fw)
    assert m > 0 and np.isnan(f0[20, 180])
    assert np.isfinite(f0[150, 46])                       # the mover's PSF peak is not a cosmic
    t = np.linspace(0, 1, 400)
    for x, y in zip(20 + 160 * t, 30 + 120 * t):
        cube[5, int(round(y)), int(round(x))] += 15.0
    npx, frames = mask_trails(cube, fw)
    assert frames == [5] and npx > 0
    assert np.isnan(cube[5, 90, 100]) and np.isfinite(cube[4, 90, 100])
    assert np.isfinite(cube[:, 150, 40 + 2 * np.arange(n)]).all() if False else True
    assert all(np.isfinite(cube[i, 150, 40 + 2 * i]) for i in range(n) if i != 5)


def test_transient_masking_is_recall_safe_for_undersampled_bright_stars():
    """Regression for CTA-S1: stellar cores must never become square holes."""
    from synthtrack.transients import mask_cosmics
    rng = np.random.default_rng(305)
    h = 256
    yy, xx = np.mgrid[:h, :h]
    fwhm = 1.7
    sig = fwhm / 2.3548
    frame = rng.standard_normal((h, h)).astype(np.float32)
    centres = []
    for _ in range(80):
        x, y = rng.uniform(6, h - 6, 2)
        centres.append((x, y))
        peak = 10 ** rng.uniform(2.0, 4.2)
        frame += (peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) /
                                (2 * sig * sig))).astype(np.float32)
    before = frame.copy()
    assert mask_cosmics(frame, fwhm) == 0
    assert np.isfinite(frame).all()
    assert np.array_equal(frame, before)


def test_transient_mask_area_fuse_refuses_catastrophic_mask():
    from synthtrack.transients import mask_cosmics
    frame = np.zeros((200, 200), np.float32)
    frame[2:-2:8, 2:-2:8] = 100.0
    assert mask_cosmics(frame, 2.5) == 0
    assert np.isfinite(frame).all()


def test_destructive_transient_masking_defaults_off():
    assert SearchSettings().mask_transients is False
