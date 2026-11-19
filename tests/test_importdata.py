"""Moving the data folder to another computer (export on Windows, import on Linux)."""
import numpy as np
from astropy.io import fits

from synthtrack.importdata import export_data, import_data
from synthtrack.instruments import Instrument, InstrumentStore
from synthtrack.settings import Settings, load_settings


def test_export_import_rewrites_paths(tmp_path):
    old = Settings(data_dir=str(tmp_path / "old"))
    old.ensure_dirs()
    missing = tmp_path / "paths-from-another-machine"
    old.astap.executable = str(missing / "astap.exe")
    old.ucac4_dir = str(missing / "ucac4")
    old.watch.root, old.watch.enabled = str(missing / "watch"), True
    old.save()
    masters = tmp_path / "Kalibrace_CTA-S1"
    masters.mkdir()
    fits.writeto(masters / "masterdark_WF0-30s.fits", np.zeros((4, 4), np.float32))
    st = InstrumentStore(old.instruments_dir)
    st.save(Instrument(id="cta-s1", name="CTA-S1", focal_length_mm=135, pixel_size_um=9,
                       masters_dir=r"C:\Astro\Kalibrace_CTA-S1", match={"TELESCOP": "CTA-S1"}))
    st.save(Instrument(id="bez", name="bez masterů", focal_length_mm=300, pixel_size_um=9,
                       masters_dir=r"C:\neexistuje"))
    (old.root / "catalogs" / "mpcorb.json").write_text("{}")

    exp = export_data(old.root, tmp_path / "export",
                      path_map={r"C:\Astro\Kalibrace_CTA-S1": str(masters)})
    assert (exp / "masters" / "cta-s1" / "masterdark_WF0-30s.fits").exists()

    new_root = tmp_path / "new"
    notes = import_data(exp, new_root)
    new = load_settings(new_root)
    inst = InstrumentStore(new.instruments_dir).get("cta-s1")
    assert inst.masters_dir == str(new_root / "masters" / "cta-s1")
    assert inst.match == {"TELESCOP": "CTA-S1"}
    assert new.astap.executable == "" and new.ucac4_dir == "" and new.watch.root == ""
    assert new.watch.enabled is False
    assert (new_root / "catalogs" / "mpcorb.json").exists()
    assert any("bez" in n for n in notes) and any("ASTAP" in n for n in notes)
