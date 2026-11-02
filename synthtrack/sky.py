"""Reference sky cut-outs (DSS2, 2MASS) for visual comparison of a candidate.

Only used on request from the GUI and only if ``settings.online.sky_images``
is enabled. Three independent services are tried in turn, so one being
down (CDS hosts both VizieR and hips2fits) does not matter. The lightweight
STScI/SkyView endpoints go first; a blocked CDS endpoint must not add two
full connection timeouts to every preview:

1. STScI DSS server (DSS1/DSS2 plates, independent of CDS),
2. NASA SkyView,
3. CDS hips2fits (alasky, then the alaskybis mirror).

All return images with north up and east left. Results are cached in the
job's ``img`` folder, so a second look costs nothing.
"""
from __future__ import annotations

from urllib.parse import urlencode

SURVEYS = {
    # key: (label, hips id, STScI survey, SkyView survey)
    "DSS2c": ("DSS2 barevně", "CDS/P/DSS2/color", "poss2ukstu_red", "DSS2 Red"),
    "DSS2r": ("DSS2 červeně", "CDS/P/DSS2/red", "poss2ukstu_red", "DSS2 Red"),
    "DSS2b": ("DSS2 modře", "CDS/P/DSS2/blue", "poss2ukstu_blue", "DSS2 Blue"),
    "2MASSJ": ("2MASS J", "CDS/P/2MASS/J", None, "2MASS-J"),
}


def _urls(ra: float, dec: float, fov_deg: float, survey: str, pixels: int):
    label, hips, stsci, skyview = SURVEYS.get(survey, SURVEYS["DSS2c"])
    if stsci:
        am = min(max(fov_deg * 60.0, 1.0), 60.0)
        yield "STScI DSS", ("https://archive.stsci.edu/cgi-bin/dss_search?"
                            + urlencode({"v": stsci, "r": f"{ra:.6f}", "d": f"{dec:.6f}", "e": "J2000",
                                         "h": f"{am:.2f}", "w": f"{am:.2f}", "f": "gif", "c": "none"}))
    if skyview:
        yield "NASA SkyView", ("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?"
                               + urlencode({"Survey": skyview, "Position": f"{ra:.6f},{dec:.6f}",
                                            "Size": f"{fov_deg:.5f}", "Pixels": pixels, "Return": "JPEG"}))
    q = urlencode({"hips": hips, "width": pixels, "height": pixels, "fov": f"{fov_deg:.5f}", "projection": "TAN",
                   "coordsys": "icrs", "rotation_angle": 0, "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "format": "jpg"})
    yield "CDS hips2fits", f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?{q}"
    yield "CDS hips2fits (zrcadlo)", f"https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits?{q}"


def fetch(ra: float, dec: float, fov_deg: float, survey: str = "DSS2c", pixels: int = 400,
          timeout: float = 6.0, session=None) -> tuple[bytes, str, str]:
    """(image bytes, content type, source name); raises RuntimeError listing what was tried."""
    import requests
    s = session or requests
    tried = []
    for name, url in _urls(ra, dec, fov_deg, survey, pixels):
        try:
            r = s.get(url, timeout=(2.5, timeout))
            ct = r.headers.get("Content-Type", "").split(";")[0].strip()
            if r.status_code == 200 and ct.startswith("image/") and len(r.content) > 1500:
                return r.content, ct, name
            tried.append(f"{name}: HTTP {r.status_code} {ct or ''}".strip())
        except Exception as e:  # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("Žádná služba neodpověděla – " + "; ".join(tried))
