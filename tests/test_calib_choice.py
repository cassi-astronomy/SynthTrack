"""Master selection without EXPTIME, manual masters and processing without calibration (2. 11. 2026)."""
import numpy as np
from astropy.io import fits

from synthtrack.fitsmeta import FrameMeta
from synthtrack.instruments import Instrument
from synthtrack.masters import MasterEntry, MasterLibrary, exptime_from_name, manual_selection


def _meta(path, exptime, filt="N", t=-20.0, w=100, h=100):
    return FrameMeta(path=str(path), exptime=exptime, filter=filt, raw_filter=filt, temperature=t, gain=None,
                     offset=None, xbin=1, ybin=1, imagetyp="", date_obs=None, mjd_mid=60000.0, ra_deg=None,
                     dec_deg=None, width=w, height=h)


def _lib(*metas):
    lib = MasterLibrary(Instrument(id="t"))
    lib.entries = [MasterEntry("dark", m, 0.0) for m in metas]
    return lib


def test_exptime_from_name():
    assert exptime_from_name("MasterDark_60s.fits") == 60
    assert exptime_from_name("dark-30sec-bin1.fit") == 30
    assert exptime_from_name("dark 0,5s.fits") == 0.5
    assert exptime_from_name("master_dark.fits") is None
    assert exptime_from_name("dark_10s_30s.fit") is None       # ambiguous


def test_dark_without_exptime_is_used_with_warning():
    s = _lib(_meta("C:/m/MasterDark.fits", None)).select_dark(_meta("l.fits", 60.0))
    assert s.ok and s.path.endswith("MasterDark.fits")
    assert any("délku expozice" in w for w in s.warnings)


def test_dark_exptime_from_name_beats_unknown():
    s = _lib(_meta("C:/m/dark_30s.fits", None), _meta("C:/m/dark_60s.fits", None)).select_dark(_meta("l.fits", 60.0))
    assert s.path.endswith("dark_60s.fits")


def test_header_match_still_preferred_and_wrong_exptime_still_fails():
    lib = _lib(_meta("C:/m/x.fits", None), _meta("C:/m/d60.fits", 60.0))
    assert lib.select_dark(_meta("l.fits", 60.0)).path.endswith("d60.fits")
    assert not _lib(_meta("C:/m/d30.fits", 30.0)).select_dark(_meta("l.fits", 60.0)).ok


def test_manual_selection(tmp_path):
    inst = Instrument(id="t")
    p = tmp_path / "dark_any.fits"
    fits.writeto(p, np.zeros((100, 100), np.float32), header=fits.Header({"IMAGETYP": "Dark Frame", "EXPTIME": 30.0}))
    s = manual_selection("dark", str(p), _meta("l.fits", 60.0), inst)
    assert s.ok and any("30 s" in w for w in s.warnings)
    bad = manual_selection("dark", str(p), _meta("l.fits", 60.0, w=200, h=200), inst)
    assert not bad.ok and "rozměr" in bad.reason
    assert not manual_selection("flat", str(tmp_path / "none.fits"), _meta("l.fits", 60.0), inst).ok
