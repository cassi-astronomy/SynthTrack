"""Twilight 'extra source' mode on a synthetic field with uncorrected distortion."""
import math

import numpy as np
from astropy.wcs import WCS

from synthtrack import twilight as TW


def _field(seed=1):
    rng = np.random.default_rng(seed)
    n, H = 5, 700
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [180.0, -5.0]
    w.wcs.crpix = [H / 2, H / 2]
    s = 6.4 / 3600
    w.wcs.cd = [[-s, 0], [0, s]]
    N = 900
    x = rng.uniform(5, H - 5, N)
    y = rng.uniform(5, H - 5, N)
    mag = np.clip(rng.exponential(1.6, N) * -1 + 15.5, 8.5, 15.5)
    ra, dec = w.all_pix2world(x, y, 0)
    cat = {"ra_all": ra, "dec_all": dec, "mag_all": mag, "j_all": mag - 1.0, "leda_all": np.zeros(N, np.int8)}
    # distortion the WCS does not know: up to ~3 px at the corners
    u, v = (x - H / 2) / (H / 2), (y - H / 2) / (H / 2)
    xd, yd = x + 2.0 * u * (u * u + v * v), y + 2.0 * v * (u * u + v * v)
    fw = 2.8
    sg = fw / 2.3548
    zp = 18.0
    cube = rng.normal(0, 1, (n, H, H)).astype(np.float32)
    yy, xx = np.mgrid[-8:9, -8:9]

    def add(px, py, m):
        F = 10 ** (-0.4 * (m - zp))
        xi, yi = int(px), int(py)
        if 8 <= xi < H - 9 and 8 <= yi < H - 9:
            g = np.exp(-((xx + xi - px) ** 2 + (yy + yi - py) ** 2) / (2 * sg * sg)) / (2 * math.pi * sg * sg)
            cube[:, yi - 8:yi + 9, xi - 8:xi + 9] += (F * g).astype(np.float32)
    for a, b, m in zip(xd, yd, mag):
        add(a, b, m)
    # one new source far from stars, one star brightened by 2.5 mag
    d = np.hypot(xd - 350, yd - 260)
    ex, ey = 350.0, 260.0
    if d.min() < 15:
        ex, ey = 200.0, 520.0
    add(ex, ey, 11.0)
    k = int(np.argmin(np.abs(mag - 12.5) + 100 * ((xd < 60) | (xd > H - 60) | (yd < 60) | (yd > H - 60))))
    add(xd[k], yd[k], mag[k] - 1.0)          # total ≈ 2.5 mag brighter… (two sources: 12.5 + 11.5)
    add(xd[k], yd[k], mag[k] - 2.5)
    return cube, w, cat, fw, (ex, ey), (xd[k], yd[k])


def test_extra_source_and_brightening_found_nothing_else():
    cube, w, cat, fw, extra, bright = _field()
    res = TW.run(cube, np.arange(5) * 2.0, w, fw, cat, params=TW.TwilightParams(border_px=10))
    ok = [c for c in res.candidates if c["auto_status"] == "candidate"]
    assert res.stats["astrometry_rms_px"] < 0.5
    kinds = {c["kind"]: c for c in ok}
    assert "navíc" in kinds and math.hypot(kinds["navíc"]["x"] - extra[0], kinds["navíc"]["y"] - extra[1]) < 2
    assert "zjasnění" in kinds and math.hypot(kinds["zjasnění"]["x"] - bright[0], kinds["zjasnění"]["y"] - bright[1]) < 2
    assert len(ok) == 2, [(c["x"], c["y"], c["kind"], c["flags"]) for c in ok]


def test_dso_ellipse():
    dso = {"name": np.array(["NGC0001"]), "type": np.array(["G"]), "ra": np.array([10.0]), "dec": np.array([20.0]),
           "maj": np.array([4.0]), "min": np.array([1.0]), "pa": np.array([90.0]), "mag": np.array([12.0]),
           "messier": np.array([""])}
    cd = math.cos(math.radians(20))
    assert TW.dso_at(dso, 10.0 + 1.8 / 60 / cd, 20.0, 0) is not None       # along the major axis (east)
    assert TW.dso_at(dso, 10.0, 20.0 + 1.8 / 60, 0) is None                 # beyond the minor axis


def test_dither_test_separates_reflection_from_sky():
    rng = np.random.default_rng(0)
    t = np.array([0.0, 2, 4, 6, 8])
    d = np.array([[14.6, -4.3], [8.8, -5.2], [-2.7, 12.6], [-10.9, -2.4], [-9.8, -0.6]])
    sky = np.array([100.0, 200.0]) + rng.normal(0, 0.2, (5, 2))
    al, s = TW.dither_test(t, sky[:, 0], sky[:, 1], d)
    assert abs(al) < 3 * s + 0.1
    ghost = np.array([100.0, 200.0]) - 0.6 * (d - d.mean(0)) + rng.normal(0, 0.2, (5, 2))
    al, s = TW.dither_test(t, ghost[:, 0], ghost[:, 1], d)
    assert abs(al + 0.6) < 0.1 and abs(al) > 3 * s


def test_diffuse_comet_found():
    from synthtrack import inject as INJ
    cube, w, cat, fw, extra, bright = _field(seed=4)
    prof = INJ.comet_profile(10.0, 0.0, fw)          # ~1′ coma, no nucleus
    r = prof.shape[0] // 2
    x, y = 520, 180
    cube[:, y - r:y + r + 1, x - r:x + r + 1] += (10 ** (-0.4 * (10.5 - 18.0)) * prof).astype(np.float32)
    res = TW.run(cube, np.arange(5) * 2.0, w, fw, cat, params=TW.TwilightParams(border_px=10))
    hit = [c for c in res.candidates if c["auto_status"] == "candidate" and math.hypot(c["x"] - x, c["y"] - y) < 6]
    assert hit, [(round(c["x"]), round(c["y"]), c["kind"], c["flags"]) for c in res.candidates]


def test_astap_catalogues(tmp_path):
    ds = tmp_path / "deep_sky.csv"
    ds.write_text("header\nfields\n57,324000,NP_2000\n" + f"{int(184.74618 / 360 * 864000)},{int(-20.08215 * 3600)},PGC39602,13,10,2\n",
                  encoding="utf-8")
    vs = tmp_path / "var.csv"
    vs.write_text("header\nfields\n" + f"{int(188.7 / 360 * 864000)},{int(-24.5 * 3600)},X_Crv 10.0V-12V\n", encoding="utf-8")
    d = TW.load_astap_dso(ds)
    assert list(d["name"]) == ["PGC39602"] and abs(d["maj"][0] - 1.3) < 1e-9
    assert TW.dso_at(d, 184.74618, -20.08215, 5).startswith("PGC39602")
    v = TW.load_variables(vs)
    assert TW.variable_at(v, 188.7, -24.5, 5).startswith("X_Crv")
    assert TW._var_amplitude("EPIC_201233864 11.26Kp-(0.005)Kp#") == 0.005
    assert abs(TW._var_amplitude("VX_Crv 11.0V-<14.8V_Period_219") - 3.8) < 1e-9


def test_choose_mode():
    tw = np.array([False, True])
    assert TW.choose_mode("auto", 5, 8.0, None)[0] == "twilight"
    assert TW.choose_mode("auto", 19, 32.0, np.ones(19, bool))[0] == "both"
    assert TW.choose_mode("auto", 19, 32.0, np.zeros(19, bool))[0] == "tracking"
    assert TW.choose_mode("auto", 12, 10.0, None)[0] == "tracking"      # short series with many frames
    assert TW.choose_mode("auto", 2, 4.0, None)[0] == "twilight"          # 2–4 frames: catalogue only
    assert TW.choose_mode("tracking", 3, 4.0, None)[0] == "twilight"
    assert TW.choose_mode("twilight", 19, 32.0, tw)[0] == "twilight"


def test_dss_measurement_with_catalogue_calibration():
    rng = np.random.default_rng(2)
    W = 240
    g = rng.normal(50, 3, (W, W))
    yy, xx = np.mgrid[0:W, 0:W]
    px = 6.0 * 60 / W                                   # ″/px of a 6′ cut-out
    ra0, dec0 = 200.0, 50.0
    cd = math.cos(math.radians(dec0))
    stars = [(30, 40, 12.0), (200, 60, 13.0), (60, 200, 14.0), (180, 190, 15.0), (100, 30, 16.0)]
    ras, decs, mags = [], [], []
    for x, y, m in stars:
        g += 10 ** (-0.4 * (m - 20)) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.0)
        ras.append(ra0 - (x - (W - 1) / 2) * px / 3600 / cd)
        decs.append(dec0 - (y - (W - 1) / 2) * px / 3600)
        mags.append(m)
    cat = {"ra_all": np.array(ras), "dec_all": np.array(decs), "mag_all": np.array(mags)}
    empty = TW.dss_mag_at_centre(g, ra0, dec0, 6.0, cat, 5.0)
    assert not empty[0]
    c = (W - 1) / 2
    g2 = g + 10 ** (-0.4 * (13.0 - 20)) * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / 4.0)
    found, z, m = TW.dss_mag_at_centre(g2, ra0, dec0, 6.0, cat, 5.0)
    assert found and m is not None and abs(m - 13.0) <= 1.0


def test_pgc_tsv_and_check(tmp_path):
    from synthtrack import twilight as TW
    txt = ("#\n# VizieR\n#\nPGC\tlogD25\t_RAJ2000\t_DEJ2000\n\t[0.1arcmin]\tdeg\tdeg\n"
           "-------\t-----\t--------\t--------\n"
           "1234567\t0.75\t177.64300\t-0.57250\n"     # 0.56′ galaxy at the glow
           "7654321\t0.10\t176.41840\t-7.49290\n"     # 0.13′ – too small/faint for 13.5 mag
           "1111111\t1.10\t180.00000\t+1.00000\n")
    g = TW.parse_vizier_tsv(txt)
    assert len(g["ra"]) == 3 and abs(g["d25"][0] - 0.562) < 0.01
    (tmp_path / "pgc_cache").mkdir()
    (tmp_path / "pgc_cache" / "pgc_177.5_-0.5_3.0.tsv").write_text(txt)
    g2 = TW.load_pgc_field(tmp_path, 177.6, -0.6, 1.0, online=False)
    assert g2 is not None and len(g2["ra"]) == 3
    assert TW.load_pgc_field(tmp_path, 190.0, 10.0, 1.0, online=False) is None
    cands = [{"ra": 177.6433, "dec": -0.57237, "mag": 14.26, "diffuse_scale_px": 3.0, "flags": [], "auto_status": "candidate"},
             {"ra": 176.41834, "dec": -7.49292, "mag": 13.5, "flags": [], "auto_status": "candidate"},
             {"ra": 175.0, "dec": 2.0, "mag": 13.0, "flags": [], "auto_status": "candidate"}]
    assert TW.pgc_check(cands, g, 32.0, 14.3) == 1
    assert cands[0]["auto_status"] == "auto_rejected" and cands[0]["flags"][0].startswith("galaxie PGC1234567")
    assert cands[1]["auto_status"] == "candidate" and cands[2]["auto_status"] == "candidate"


def test_pgc_dat_offline(tmp_path):
    import gzip
    from synthtrack import twilight as TW
    lines = ["PGC2402729 J000000.1+515515 G        0.38+/-0.11  0.10+/-0.03  126.+/- 21.  0",
             "PGC0520795 J000000.1-455715 G        0.53+/-0.07  0.18+/-0.05   22.+/-  4.  0",
             "PGC1234567 J114634.4-003421 G        0.75+/-0.07  0.18+/-0.05   22.+/-  4.  0",
             "PGC0000009 J114634.4-003421 G        9.99+/-9.99  9.99+/-9.99  999.+/-999.  0"]
    g = TW.parse_pgc_dat(lines)
    assert len(g["ra"]) == 4
    assert abs(g["dec"][0] - (51 + 55 / 60 + 15 / 3600)) < 1e-6 and abs(g["dec"][1] + 45.954166) < 1e-4
    assert abs(g["ra"][2] - 176.643333) < 1e-4 and abs(g["d25"][0] - 0.1 * 10 ** 0.38) < 1e-6
    assert g["pgc"][2] == 1234567 and not np.isfinite(g["d25"][3])
    d = tmp_path / "pgc"
    d.mkdir()
    body = "\n".join(lines * 3000) + "\n"
    with gzip.open(d / "pgc.dat.gz", "wt") as f:
        f.write(body)
    TW._PGC_ALL = None
    try:
        f = TW.load_pgc_field(tmp_path, 176.6, -0.6, 1.0, online=False)
        assert f is not None and set(f["pgc"].astype(int)) == {1234567, 9}
        assert (d / "pgc.npz").exists()
    finally:
        TW._PGC_ALL = None


def test_normconv_noise_matches_empirical():
    import numpy as np
    from synthtrack.twilight import _normconv, _normconv_noise
    rng = np.random.default_rng(1)
    h = w = 120
    yy, xx = np.mgrid[0:h, 0:w]
    good = ((xx - 60) ** 2 + (yy - 60) ** 2 > 100) & ((xx - 75) ** 2 + (yy - 60) ** 2 > 49)
    f = _normconv_noise(good, 3.0)
    pts = [(20, 20), (60, 71), (68, 52)]
    acc = np.array([[_normconv(rng.normal(0, 1, (h, w)), good, 3.0)[y, x] for x, y in pts] for _ in range(800)])
    emp = acc.std(0) / acc.std(0)[0]
    for (x, y), e in zip(pts, emp):
        assert abs(e - f[y, x] / f[20, 20]) < 0.12 * e


def test_thirds_transient_vs_constant():
    from synthtrack.twilight import _thirds
    assert _thirds([1.8] * 15, list(range(15)), 0.35)[0]          # a real 7σ glow, 15 frames
    assert not _thirds([0.0] * 10 + [6.0] * 5, list(range(15)), 0.35)[0]   # last third only
    assert _thirds([3.0] * 6, list(range(6)), 0.35)[0]            # twilight: 6 frames


def test_static_injections_plan_and_summary():
    from synthtrack import inject as INJ
    objs = INJ.plan_static(20, (400, 400), 6, 2.5, 5, 30, 6.4, seed=3, border=60, comet_fraction=0.5)
    assert len(objs) == 20 and all(o["vx"] == 0 and o["vy"] == 0 for o in objs)
    assert all(60 <= o["x"] <= 340 and 5 <= o["snr_nominal"] <= 30 for o in objs)
    assert {o["kind"] for o in objs} == {"point", "comet"}
    cube = np.zeros((6, 400, 400), np.float32)
    INJ.add_to_frames(cube, objs[:1], np.arange(6.0), 2.5)
    assert abs(float(cube[0].sum()) - objs[0]["flux_sigma"]) < 0.05 * objs[0]["flux_sigma"] + 1e-3
    dets = [{"injected": objs[0]["id"], "auto_status": "candidate", "snr": 9.0, "mag": 13.2, "flags": ["Twilight: x"]},
            {"injected": objs[1]["id"], "auto_status": "auto_rejected", "snr": 7.0, "mag": 14.0,
             "flags": ["Twilight: x", "jen v 2/6 snímcích"]}]
    summ, lines = INJ.summarise_static(objs, dets)
    assert summ["detected"] == 2 and summ["passed"] == 1 and summ["rejected_by"] == {"jen v 2/6 snímcích": 1}
    assert any("statických zdrojů" in ln for ln in lines)


def test_outburst_note_dwarf_nova_vs_mira():
    o = TW.outburst_note
    assert o("SS_Cyg 7.7V-12.4V_Period_0.275", 8.5, 12.1).startswith("vzplanutí? VSX SS Cyg")
    assert "Mira?" in o("R_Cam 6.97V-14.4V_Period_270", 9.9, 13.0)
    assert o("SS_Cyg 7.7V-12.4V_Period_0.275", 11.9, 12.1) is None          # quiet
    assert o("X 11.26Kp-(0.005)Kp#", 9.0, 12.0) is None                    # micro-variable
    assert o("NSV_1 13.5V-16.0V", 13.8, None).startswith("vzplanutí?")      # faint end as quiet level


def test_bright_stars_fill_ucac4_gap():
    b = TW.bright_stars()
    assert len(b["ra"]) > 9000 and "Spica" in b["name"]
    cat = {"ra_all": np.array([201.0]), "dec_all": np.array([-11.0]), "mag_all": np.array([12.0]),
           "j_all": np.array([np.nan]), "leda_all": np.zeros(1, np.int8)}
    out, k = TW.add_bright_stars(cat, (200.5, 202.0, -12.0, -10.5), 2026.9, [("Venuše", 201.5, -11.5)])
    assert k >= 2 and len(out["ra_all"]) == 1 + k
    i = int(np.argmin(np.abs(out["ra_all"] - 201.298)))
    assert abs(out["dec_all"][i] + 11.161) < 0.01 and out["mag_all"][i] < 1.5        # Spica
    assert -4.5 < float(np.min(out["mag_all"])) < -4.0                               # Venus


def test_planets_seen_from_the_site():
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [200.46, -9.86]          # Venus seen from La Palma, 2026-11-16 06:37 UTC
    w.wcs.crpix = [500, 500]
    w.wcs.cd = [[-0.004, 0], [0, 0.004]]
    mjd = 61360.2758
    pl = TW.planets_in_field(w, (1000, 1000), mjd, 28.77, -17.89, 2184.0)
    assert any(nm == "Venuše" and abs(x - 499) < 20 and abs(y - 499) < 20 for nm, x, y in pl), pl


def test_annotate_vsx_near_offline():
    import numpy as np
    from synthtrack import twilight as TW
    var = {"ra": np.array([10.0010, 10.5]), "dec": np.array([20.0, 20.0]),
           "label": np.array(["V0001_Sco 12.1-15.3_Period_0.52", "FAR_Var 11-12"])}
    out = [{"ra": 10.0, "dec": 20.0, "auto_status": "candidate", "injected": None, "known": None,
            "flags": ["Twilight: zdroj navíc (není v UCAC4)"]},
           {"ra": 10.0, "dec": 20.0, "auto_status": "candidate", "injected": 5, "known": None, "flags": ["x"]},
           {"ra": 30.0, "dec": -5.0, "auto_status": "candidate", "injected": None, "known": None, "flags": ["y"]}]
    n = TW.annotate_vsx_near(out, 20.0, var, online=False)
    assert n == 1
    v = out[0]["vsx_near"][0]
    assert v["name"] == "V0001 Sco" and 3.0 < v["sep_arcsec"] < 4.0
    assert out[0]["flags"][1].startswith("VSX V0001 Sco")
    assert "vsx_near" not in out[1] and "vsx_near" not in out[2]       # injections / far away untouched
