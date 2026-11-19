"""Automatic recognition of the instrument (CTA-S1 / CTA-N / Coihueco …) from FITS headers."""
import numpy as np
from astropy.io import fits

from synthtrack.instruments import Instrument, InstrumentStore

# header values taken from real frames (CTA-S1 WF0, FRAM2 Coihueco WF1)
H_CTA_S1 = {"TELESCOP": "CTA-S1", "INSTRUME": "CTA-S1 FRAM", "CCD_NAME": "WF0", "LATITUDE": -24.63368}
H_COIHUECO = {"TELESCOP": "FRAM2", "INSTRUME": "FRAM2 300/2.8", "CCD_NAME": "WF1", "LATITUDE": -35.1167}
H_CTA_N = {"TELESCOP": "CTA-N", "INSTRUME": "CTA-N FRAM", "CCD_NAME": "WF0", "LATITUDE": 28.76}


def _store(tmp_path):
    st = InstrumentStore(tmp_path / "instruments")
    for i, m in (("cta-s1", {"TELESCOP": "CTA-S1", "CCD_NAME": "WF0"}),
                 ("cta-n", {"TELESCOP": "CTA-N", "CCD_NAME": "WF0"}),
                 ("coihueco", {"TELESCOP": "FRAM2", "CCD_NAME": "WF1"})):
        st.save(Instrument(id=i, name=i.upper(), focal_length_mm=135, pixel_size_um=9, match=m))
    return st


def _write(d, n, hdr):
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        h = fits.Header()
        for key, v in hdr.items():
            h[key] = v
        fits.writeto(d / f"f{k:02d}.fits", np.zeros((4, 4), np.uint16), h)
    return sorted(d.glob("*.fits"))


def test_three_telescopes_told_apart(tmp_path):
    st = _store(tmp_path)
    for hdr, want in ((H_CTA_S1, "cta-s1"), (H_CTA_N, "cta-n"), (H_COIHUECO, "coihueco")):
        inst, why = st.identify(hdr)
        assert inst is not None and inst.id == want, why
        assert "TELESCOP=" in why


def test_unknown_header_reports_values(tmp_path):
    inst, why = _store(tmp_path).identify({"TELESCOP": "Neznámý", "CCD_NAME": "X"})
    assert inst is None and "TELESCOP=Neznámý" in why


def test_most_specific_profile_wins_and_tie_is_refused(tmp_path):
    st = _store(tmp_path)
    st.save(Instrument(id="any-cta", name="CTA obecně", focal_length_mm=135, pixel_size_um=9,
                       match={"TELESCOP": "CTA"}))
    inst, _ = st.identify(H_CTA_S1)                 # 2 rules beat 1 rule
    assert inst.id == "cta-s1"
    st.save(Instrument(id="cta-s1-copy", name="kopie", focal_length_mm=135, pixel_size_um=9,
                       match={"TELESCOP": "CTA-S1", "INSTRUME": "FRAM"}))
    inst, why = st.identify(H_CTA_S1)               # equally specific -> no guessing
    assert inst is None and "více profilů" in why


def test_numeric_rule_tolerance(tmp_path):
    st = _store(tmp_path)
    st.save(Instrument(id="smph", name="SMPH", focal_length_mm=600, pixel_size_um=7.52,
                       match={"INSTRUME": "C2-9000", "FOCALLEN": 600}))
    assert st.identify({"INSTRUME": "MI C2-9000", "FOCALLEN": 600.0})[0].id == "smph"
    assert st.identify({"INSTRUME": "MI C2-9000", "FOCALLEN": 1000.0})[0] is None


def test_identify_files_checks_several_frames(tmp_path):
    st = _store(tmp_path)
    files = _write(tmp_path / "s1", 6, H_CTA_S1)
    inst, _ = st.identify_files(files)
    assert inst.id == "cta-s1"
    mixed = _write(tmp_path / "mix", 3, H_CTA_S1) + _write(tmp_path / "mix2", 3, H_COIHUECO)
    inst, why = st.identify_files(mixed)
    assert inst is None and "různých přístrojů" in why


def test_watcher_waits_for_rules_instead_of_skipping(tmp_path):
    from synthtrack.jobs import JobStore
    from synthtrack.settings import Settings
    from synthtrack.watcher import Watcher

    class FakeRunner:
        def __init__(self): self.jobs = []
        def submit(self, job): self.jobs.append(job)

    settings = Settings(data_dir=str(tmp_path / "data"))
    st = InstrumentStore(settings.instruments_dir)
    root = tmp_path / "upload"
    _write(root / "night" / "WF0", 4, H_CTA_S1)
    settings.watch.root, settings.watch.frames_per_set, settings.watch.settle_seconds = str(root), 4, 0
    store = JobStore(settings.jobs_dir)
    w = Watcher(settings, store, FakeRunner())
    assert w.scan_once() == []                       # no profile yet
    key = str(root / "night" / "WF0")
    assert "nerozpoznán" in " ".join(w.messages) and "error" in w.pending[key]
    assert key not in w.state["processed"]
    st.save(Instrument(id="cta-s1", name="CTA-S1", focal_length_mm=135, pixel_size_um=9,
                       match={"TELESCOP": "CTA-S1", "CCD_NAME": "WF0"}))
    started = w.scan_once()                          # rules added -> starts by itself
    assert started and store.get(started[0]).instrument_id == "cta-s1"


def _watch_env(tmp_path, root):
    from synthtrack.jobs import JobStore
    from synthtrack.settings import Settings
    from synthtrack.watcher import Watcher

    class FakeRunner:
        def submit(self, job): pass

    settings = Settings(data_dir=str(tmp_path / "data"))
    InstrumentStore(settings.instruments_dir).save(Instrument(
        id="cta-s1", name="CTA-S1", focal_length_mm=135, pixel_size_um=9,
        match={"TELESCOP": "CTA-S1", "CCD_NAME": "WF0"}))
    InstrumentStore(settings.instruments_dir).save(Instrument(
        id="coihueco", name="Coihueco", focal_length_mm=300, pixel_size_um=9,
        match={"TELESCOP": "FRAM2", "CCD_NAME": "WF1"}))
    settings.watch.root, settings.watch.frames_per_set, settings.watch.settle_seconds = str(root), 4, 0
    return Watcher(settings, JobStore(settings.jobs_dir), FakeRunner()), settings


def test_root_with_field_folders_from_several_telescopes(tmp_path):
    root = tmp_path / "stazene"
    _write(root / "01047" / "WF0", 4, H_CTA_S1)
    _write(root / "03935" / "WF1", 2, H_COIHUECO)           # still uploading
    w, _ = _watch_env(tmp_path, root)
    started = w.scan_once()
    assert len(started) == 1
    p = w.pending[str(root / "03935" / "WF1")]
    assert p["instrument"] == "coihueco" and p["count"] == 2    # recognised before complete


def test_renamed_root_is_not_processed_again(tmp_path):
    root = tmp_path / "stazene"
    _write(root / "01047" / "WF0", 4, H_CTA_S1)
    w, settings = _watch_env(tmp_path, root)
    assert len(w.scan_once()) == 1
    new_root = tmp_path / "stazene_2026"
    root.rename(new_root)
    settings.watch.root = str(new_root)
    assert w.scan_once() == []                                  # same set, new place
    _write(new_root / "01047_b" / "WF0", 4, H_CTA_S1)            # genuinely new field
    assert len(w.scan_once()) == 1
