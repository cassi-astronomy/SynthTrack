import json

import numpy as np

from synthtrack import vsx as VSX

SAMPLE = {"VSXObjects": {"VSXObject": [
    {"Name": "OGLE-BLG-LPV-245895", "RA2000": "264.91720", "Declination2000": "-24.73473", "VariabilityType": "M",
     "Period": "402.6", "MaxMag": "12.411 Ic", "MinMag": "(2.451) Ic", "Category": "Variable", "OID": "2356059"},
    {"Name": "Gaia DR3 4068237832273940992", "RA2000": "264.92534", "Declination2000": "-24.71177",
     "VariabilityType": "L", "MaxMag": "17.61 G", "MinMag": "17.77 G", "OID": "9362561"}]}}


def test_parse_vsx_json():
    s = VSX.parse_vsx_json(SAMPLE)
    assert len(s) == 2 and s[0]["oid"] == 2356059 and s[0]["max_mag"] == 12.411
    assert s[0]["url"].endswith("view=detail.top&oid=2356059") and s[0]["range"] == "12.411 Ic – (2.451) Ic"
    one = {"VSXObjects": {"VSXObject": SAMPLE["VSXObjects"]["VSXObject"][0]}}
    assert len(VSX.parse_vsx_json(one)) == 1
    assert VSX.parse_vsx_json([]) == [] and VSX.parse_vsx_json({"VSXObjects": []}) == []


def test_parse_vizier_vsx():
    tsv = ("#comment\n_RAJ\t_DEJ\tOID\tName\tType\tmax\tn_max\tf_min\tmin\tn_min\tPeriod\n"
           "deg\tdeg\t\t\t\tmag\t\t\tmag\t\td\n---\t---\t---\n"
           "264.93727\t-24.71258\t123\tV0544 Oph\tM\t11.8\tV\t<\t14.3\tV\t254\n"
           "264.9\t-24.7\t124\tX Oph\tEA\t13.0\tV\t(\t0.5\tV\t\n")
    s = VSX.parse_vizier_vsx(tsv)
    assert [x["name"] for x in s] == ["V0544 Oph", "X Oph"]
    assert s[0]["oid"] == 123 and s[0]["period"] == "254" and s[1]["range"] == "13.0 V – (0.5) V"


def test_query_falls_back_to_vizier():
    class R:
        def __init__(self, text, ok=True):
            self.text, self.ok = text, ok
        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError("down")
        def json(self):
            return json.loads(self.text)

    class S:
        def __init__(self):
            self.urls = []
        def get(self, url, params=None, timeout=None):
            self.urls.append(url)
            if "aavso" in url or "cds.unistra" in url:
                return R("", ok=False)
            return R("_RAJ\t_DEJ\tOID\tName\n264.9\t-24.7\t5\tY Oph\n")
    st, src = VSX.query(264.9, -24.7, 0.05, session=S())
    assert src == "VizieR CfA" and st[0]["name"] == "Y Oph"


def test_from_astap_and_offsets():
    var = {"ra": np.array([10.0, 10.01, 30.0]), "dec": np.array([5.0, 5.01, 5.0]),
           "label": np.array(["V0544_Oph 11.8V-<14.3V_Period_254", "NSV_1 -0.07V--0.02V#", "Far 1V-2V"])}
    s = VSX.offsets(VSX.from_astap(var, 10.0, 5.0, 0.05), 10.0, 5.0)
    assert [x["name"] for x in s] == ["V0544 Oph", "NSV 1"]
    assert s[0]["range"] == "11.8V – <14.3V" and s[0]["period"] == "254" and s[1]["range"] == "-0.07V – -0.02V"
    assert abs(s[1]["dn"] - 0.01) < 1e-9 and s[0]["url"].endswith("ident=V0544%20Oph")


def test_stsci_fits_resampled_north_up():
    import io
    import math
    from astropy.io import fits
    from PIL import Image
    from synthtrack import sky as SKY
    n, s, th = 300, 1.0 / 3600, math.radians(5.4)          # 1″/px, plate rotated by 5.4°
    ra0, de0 = 324.7, 73.05
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRPIX1"], hdr["CRPIX2"], hdr["CRVAL1"], hdr["CRVAL2"] = n / 2, n / 2, ra0, de0
    hdr["CD1_1"], hdr["CD1_2"] = -s * math.cos(th), -s * math.sin(th)
    hdr["CD2_1"], hdr["CD2_2"] = -s * math.sin(th), s * math.cos(th)
    from astropy.wcs import WCS
    w = WCS(hdr)
    img = np.zeros((n, n), np.float32)
    # a star 100″ north and 80″ east of the centre
    sra, sde = ra0 + 80 / 3600 / math.cos(math.radians(de0)), de0 + 100 / 3600
    x, y = w.wcs_world2pix([[sra, sde]], 0)[0]
    yy, xx = np.mgrid[0:n, 0:n]
    img += 1000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8.0)
    buf = io.BytesIO()
    fits.PrimaryHDU(img, hdr).writeto(buf)
    png = SKY.fits_north_up_png(buf.getvalue(), ra0, de0, 240 / 3600, 240)
    a = np.asarray(Image.open(io.BytesIO(png)), float)
    yy2, xx2 = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    wgt = np.where(a > 0.5 * a.max(), a, 0)                 # the star is saturated by the stretch
    px, py = float((wgt * xx2).sum() / wgt.sum()), float((wgt * yy2).sum() / wgt.sum())
    # north up, east left, 1″/px, centre at 120,120 → star at x 120 − 80, y 120 − 100
    assert abs(px - 40) <= 1.5 and abs(py - 20) <= 1.5, (px, py)
