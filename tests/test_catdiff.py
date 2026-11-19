import math

import numpy as np

from synthtrack import catdiff as CD


def _field(seed=3, n_st=2500, size=512, fwhm=3.0, extra=(), k=2.5):
    """A dense star field: catalogue (positions, expected fluxes) and the image (flux k × catalogue,
    Gaussian PSF, white noise σ = 1), with ``extra`` = [(x, y, flux)] sources not in the catalogue."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(8, size - 8, (n_st, 2))
    mag = rng.uniform(9.5, 16.5, n_st)
    ef = 10 ** (-0.4 * (mag - 16.0)) * 30.0                       # catalogue flux (a 16 mag star = 30)
    sg = fwhm / 2.3548
    img = rng.normal(0, 1.0, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    pts = list(zip(xy[:, 0], xy[:, 1], k * ef)) + list(extra)
    for x, y, f in pts:
        x0, x1, y0, y1 = int(max(0, x - 12)), int(min(size, x + 13)), int(max(0, y - 12)), int(min(size, y + 13))
        img[y0:y1, x0:x1] += (f * np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * sg * sg))
                              / (2 * math.pi * sg * sg)).astype(np.float32)
    return img, xy, ef, mag


def test_finds_source_beside_star_and_nothing_else():
    img0, xy, ef, mag = _field()
    j = int(np.argmin(np.abs(mag - 12.5)))                        # a 12.5 mag catalogue star
    x, y = xy[j] + np.array([2.5, 0.0])
    img, *_ = _field(extra=[(x, y, 2.5 * ef[j])])                 # an equally bright new source 2.5 px away
    c = CD.CatalogueDiff(img, xy, ef, mag, 3.0, bright_mask_mag=0.0, cell=256, psf_cell=512)
    assert 0.0 <= c.eps < 0.2                                     # the model is exact here
    pk = c.peaks(5.0, 10)
    hit = [p for p in pk if math.hypot(p[0] - x, p[1] - y) < 1.5]
    assert hit
    # no other peak far from the inserted source
    c0 = CD.CatalogueDiff(img0, xy, ef, mag, 3.0, bright_mask_mag=0.0, cell=256, psf_cell=512)
    assert len(c0.peaks(5.0, 10)) <= 2


def test_flux_scale_and_brightened_star():
    img0, xy, ef, mag = _field(seed=5)
    j = int(np.argmin(np.abs(mag - 14.0)))
    img, *_ = _field(seed=5, extra=[(xy[j, 0], xy[j, 1], 2.5 * ef[j] * 5.3)])   # the star 2 mag brighter
    c = CD.CatalogueDiff(img, xy, ef, mag, 3.0, bright_mask_mag=0.0, cell=256, psf_cell=512)
    assert abs(np.median(c.flux / ef) / 2.5 - 1) < 0.15              # catalogue → image scale recovered
    pk = c.peaks(5.0, 10)
    assert any(math.hypot(p[0] - xy[j, 0], p[1] - xy[j, 1]) < 1.0 for p in pk)


def test_frame_scales_follow_throughput():
    img, xy, ef, mag = _field(seed=7)
    c = CD.CatalogueDiff(img, xy, ef, mag, 3.0, bright_mask_mag=0.0, cell=256, psf_cell=512)
    cube = np.stack([img * 0.5, img * 1.0, img * 1.5])
    s = c.frame_scales(cube)
    assert abs(s[0] / s[1] - 0.5) < 0.05 and abs(s[2] / s[1] - 1.5) < 0.08
