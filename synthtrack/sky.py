"""Reference sky cut-outs (DSS2, 2MASS) for visual comparison of a candidate.

Only used on request from the GUI and only if ``settings.online.sky_images``
is enabled. Three independent services are tried in turn, so one being
down (CDS hosts both VizieR and hips2fits) does not matter. The lightweight
STScI/SkyView endpoints go first; a blocked CDS endpoint must not add two
full connection timeouts to every preview:

1. STScI DSS server (DSS1/DSS2 plates, independent of CDS),
2. NASA SkyView,
3. CDS hips2fits (alasky, then the alaskybis mirror).

All return images with north up and east left. STScI cuts its plates in the plate's own
orientation – far from the plate centre at high declination north is rotated (FRAM, Dec +73°,
5.3° in RA from the plate centre: 5.4°, VSX crosses visibly off): its FITS is resampled onto
a north-up TAN grid with the cut-out's WCS (CRVAL/CD the server writes, a local
approximation of the plate solution). Results are cached in the
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
    q = urlencode({"hips": hips, "width": pixels, "height": pixels, "fov": f"{fov_deg:.5f}", "projection": "TAN",
                   "coordsys": "icrs", "rotation_angle": 0, "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "format": "jpg"})
    if survey == "DSS2c":
        # colour exists only as the CDS HiPS: STScI / SkyView would silently give the red plate
        # (user 17. 11.: "barevný DSS není barevný, je stejný jako červený")
        yield "CDS hips2fits", f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?{q}"
        yield "CDS hips2fits (zrcadlo)", f"https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits?{q}"
        stsci, skyview = SURVEYS["DSS2r"][2], SURVEYS["DSS2r"][3]
        am = min(max(fov_deg * 60.0, 1.0), 60.0)
        yield "STScI DSS – jen červený (barevný z CDS nedostupný)", (
            "https://archive.stsci.edu/cgi-bin/dss_search?"
            + urlencode({"v": stsci, "r": f"{ra:.6f}", "d": f"{dec:.6f}", "e": "J2000",
                         "h": f"{am:.2f}", "w": f"{am:.2f}", "f": "fits", "c": "none"}))
        yield "NASA SkyView – jen červený (barevný z CDS nedostupný)", (
            "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?"
            + urlencode({"Survey": skyview, "Position": f"{ra:.6f},{dec:.6f}",
                         "Size": f"{fov_deg:.5f}", "Pixels": pixels, "Return": "JPEG"}))
        return
    if stsci:
        am = min(max(fov_deg * 60.0, 1.0), 60.0)
        yield "STScI DSS", ("https://archive.stsci.edu/cgi-bin/dss_search?"
                            + urlencode({"v": stsci, "r": f"{ra:.6f}", "d": f"{dec:.6f}", "e": "J2000",
                                         "h": f"{am:.2f}", "w": f"{am:.2f}", "f": "fits", "c": "none"}))
    if skyview:
        yield "NASA SkyView", ("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?"
                               + urlencode({"Survey": skyview, "Position": f"{ra:.6f},{dec:.6f}",
                                            "Size": f"{fov_deg:.5f}", "Pixels": pixels, "Return": "JPEG"}))
    q = urlencode({"hips": hips, "width": pixels, "height": pixels, "fov": f"{fov_deg:.5f}", "projection": "TAN",
                   "coordsys": "icrs", "rotation_angle": 0, "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "format": "jpg"})
    yield "CDS hips2fits", f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?{q}"
    yield "CDS hips2fits (zrcadlo)", f"https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits?{q}"


def fits_north_up_png(data: bytes, ra: float, dec: float, fov_deg: float, pixels: int = 400,
                      gamma: float = 0.6) -> bytes:
    """A FITS cut-out (with a celestial WCS) resampled onto a north-up / east-left TAN grid of
    ``fov_deg`` × ``fov_deg`` centred on (ra, dec), stretched to 8 bits, as PNG."""
    import io
    import warnings

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image
    from scipy import ndimage
    with fits.open(io.BytesIO(data)) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim == 2)
        img = np.asarray(hdu.data, np.float32)
        hdr = hdu.header
    keep = fits.Header()
    for k in ("NAXIS", "NAXIS1", "NAXIS2", "WCSAXES", "CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2", "CROTA2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
              "CUNIT1", "CUNIT2", "RADESYS", "EQUINOX"):
        if k in hdr:
            keep[k] = hdr[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_in = WCS(keep)
    s = fov_deg / pixels
    w_out = WCS(naxis=2)
    w_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_out.wcs.crval = [ra, dec]
    w_out.wcs.crpix = [pixels / 2 + 0.5, pixels / 2 + 0.5]
    w_out.wcs.cd = [[-s, 0.0], [0.0, s]]
    yy, xx = np.mgrid[0:pixels, 0:pixels]
    wr, wd = w_out.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
    px, py = w_in.wcs_world2pix(wr, wd, 0)
    out = ndimage.map_coordinates(img, [py.reshape(pixels, pixels), px.reshape(pixels, pixels)],
                                  order=1, mode="constant", cval=np.nan)
    fin = out[np.isfinite(out)]
    lo, hi = (np.percentile(fin, [0.5, 99.9]) if fin.size else (0.0, 1.0))
    v = np.clip((np.nan_to_num(out, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    # FITS rows go up (south → north); an image's rows go down
    im = Image.fromarray((255 * v[::-1]).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def fetch(ra: float, dec: float, fov_deg: float, survey: str = "DSS2c", pixels: int = 400,
          timeout: float = 6.0, session=None, gamma: float = 0.6) -> tuple[bytes, str, str]:
    """(image bytes, content type, source name); raises RuntimeError listing what was tried.
    ``gamma`` of the STScI FITS → PNG stretch (1 = linear, for measurements)."""
    import requests
    s = session or requests
    tried = []
    for name, url in _urls(ra, dec, fov_deg, survey, pixels):
        try:
            r = s.get(url, timeout=(2.5, timeout))
            ct = r.headers.get("Content-Type", "").split(";")[0].strip()
            if r.status_code == 200 and name.startswith("STScI DSS"):
                if len(r.content) > 2880 and r.content[:6] == b"SIMPLE":
                    return fits_north_up_png(r.content, ra, dec, fov_deg, pixels, gamma), "image/png", name
                tried.append(f"{name}: HTTP 200, není FITS")
                continue
            if r.status_code == 200 and ct.startswith("image/") and len(r.content) > 1500:
                return r.content, ct, name
            tried.append(f"{name}: HTTP {r.status_code} {ct or ''}".strip())
        except Exception as e:  # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("Žádná služba neodpověděla – " + "; ".join(tried))
