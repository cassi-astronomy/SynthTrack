"""Reference-survey URL priority and local sky matrix metadata."""
import math
from urllib.parse import parse_qs, urlparse

import pytest
from astropy.wcs import WCS

from synthtrack.pipeline import _local_sky_matrix
from synthtrack.sky import _urls, fetch


def test_fast_reference_services_are_tried_before_cds():
    urls = list(_urls(257.211583, -22.818194, 0.10333, "DSS2c", 400))
    assert [name for name, _ in urls] == ["STScI DSS", "NASA SkyView", "CDS hips2fits",
                                            "CDS hips2fits (zrcadlo)"]
    q = parse_qs(urlparse(urls[0][1]).query)
    assert q["r"] == ["257.211583"] and q["d"] == ["-22.818194"]
    assert q["h"] == ["6.20"] and q["w"] == ["6.20"]


def test_fetch_returns_first_valid_image_without_touching_slow_fallbacks():
    class Response:
        status_code = 200
        content = b"x" * 2000
        headers = {"Content-Type": "image/gif"}

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout):
            self.calls.append((url, timeout))
            return Response()

    session = Session()
    data, content_type, source = fetch(10, 20, 0.1, session=session)
    assert len(data) == 2000 and content_type == "image/gif" and source == "STScI DSS"
    assert len(session.calls) == 1 and session.calls[0][1] == (2.5, 6.0)


def test_local_sky_matrix_preserves_arbitrary_rotation_and_scale():
    w = WCS(naxis=2)
    w.wcs.crpix = [1000.0, 1000.0]
    w.wcs.crval = [257.2, -22.8]
    angle, scale = math.radians(37.0), 6.4 / 3600.0
    ca, sa = math.cos(angle), math.sin(angle)
    w.wcs.cd = [[-scale * ca, scale * sa], [scale * sa, scale * ca]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sky = w.pixel_to_world(1100, 900)
    matrix = _local_sky_matrix(w, 1100, 900, float(sky.ra.deg), float(sky.dec.deg))
    area_scale = abs(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) ** 0.5 * 3600
    assert area_scale == pytest.approx(6.4, rel=2e-4)
    assert abs(matrix[0][1]) > 0.5 * abs(matrix[0][0])  # not collapsed to a 90-degree transform
