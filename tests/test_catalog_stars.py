

def test_no_star_rescue_slow_segments_and_static_peak():
    import numpy as np
    from types import SimpleNamespace
    from synthtrack.pipeline import _catalog_no_star_rescue
    from synthtrack.tracking import Candidate
    cat = {"x": np.array([100.0, 150.0, 300.0]), "y": np.array([100.0, 150.0, 300.0]), "mag": np.array([12.0, 14.0, 13.0])}
    sp = SimpleNamespace(ucac4_complete_mag=15.5)
    def cand(x, y, flags, **kw):
        c = Candidate(id=0, x=x, y=y, vx=0.02, vy=0.0, snr=7.5, seg_snr=[4, 1, 4], median_ratio=0.9, frames_used=17)
        c.flags = list(flags); c.auto_status = "auto_rejected"; c.chi2 = 0.8
        for k, v in kw.items():
            setattr(c, k, v)
        return c
    free_seg = cand(200, 200, ["nekonzistentní segmenty", "snížená propustnost"])
    star_seg = cand(101, 100, ["nekonzistentní segmenty"])
    free_peak = cand(200, 220, ["u hvězdy", "reziduum hvězdy"], mag=14.5, star_ratio=2.0,
                     static_star_x=201.0, static_star_y=221.0)
    faint_peak = cand(220, 200, ["u hvězdy"], mag=16.5, star_ratio=1.5, static_star_x=221.0, static_star_y=200.0)
    other_hard = cand(200, 240, ["nekonzistentní segmenty", "jednosnímkový artefakt"])
    cs = [free_seg, star_seg, free_peak, faint_peak, other_hard]
    assert _catalog_no_star_rescue(cs, cat, 3.0, sp, span_min=30.0) == 2
    assert free_seg.auto_status == "candidate" and free_peak.auto_status == "candidate"
    assert all(c.auto_status == "auto_rejected" for c in (star_seg, faint_peak, other_hard))


def test_fast_star_residual_on_catalogue_star_rejected():
    import numpy as np
    from synthtrack.pipeline import _catalog_stars_reject
    from synthtrack.tracking import Candidate
    cat = {"x": np.array([100.0, 400.0]), "y": np.array([100.0, 400.0]), "mag": np.array([9.5, 10.0])}
    def cand(x, y, flags, ratio):
        c = Candidate(id=0, x=x, y=y, vx=0.4, vy=0.0, snr=7.0)   # 12 px over 30 min = 4 FWHM
        c.flags = list(flags); c.auto_status = "candidate"; c.star_ratio = ratio; c.mag = 14.0
        return c
    resid = cand(100.5, 100.2, ["u hvězdy", "spíš proměnná hvězda"], 120.0)
    real_near = cand(100.5, 106.0, ["u hvězdy"], 2.0)          # 2 FWHM beside, no residual signs
    weak_ratio = cand(400.3, 400.1, ["u hvězdy", "spíš proměnná hvězda"], 11.0)
    assert _catalog_stars_reject([resid, real_near, weak_ratio], cat, 3.0, 30.0) == 1
    assert resid.auto_status == "auto_rejected" and real_near.auto_status == "candidate"
    assert weak_ratio.auto_status == "candidate"


def test_fast_residual_near_bright_star_wider_zone():
    import numpy as np
    from synthtrack.pipeline import _catalog_stars_reject
    from synthtrack.tracking import Candidate
    cat = {"x": np.array([100.0]), "y": np.array([100.0]), "mag": np.array([9.4])}
    def cand(flags, ratio):
        c = Candidate(id=0, x=100.0, y=105.4, vx=0.4, vy=0.0, snr=6.5)    # 1.8 FWHM beside
        c.flags = list(flags); c.auto_status = "candidate"; c.star_ratio = ratio
        return c
    resid = cand(["u hvězdy", "reziduum hvězdy", "vráceno klasifikátorem"], None)
    var_only = cand(["spíš proměnná hvězda"], 11.0)
    assert _catalog_stars_reject([resid, var_only], cat, 3.0, 30.0) == 1
    assert resid.auto_status == "auto_rejected" and var_only.auto_status == "candidate"


def test_catalog_rule_probes_known_and_injected():
    import numpy as np
    from synthtrack.pipeline import _catalog_stars_reject
    from synthtrack.tracking import Candidate
    cat = {"x": np.array([100.0]), "y": np.array([100.0]), "mag": np.array([9.5])}
    c = Candidate(id=0, x=100.2, y=100.1, vx=0.0, vy=0.0, snr=7.0)
    c.flags = ["u hvězdy"]; c.auto_status = "candidate"; c.mag = 14.0; c.injected = 7
    probe = []
    assert _catalog_stars_reject([c], cat, 3.0, 30.0, probe=probe) == 0
    assert c.auto_status == "candidate" and probe == ["injekce #7"]
