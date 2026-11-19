"""Hot-pixel mask with a small, clustered dithering (161P/Hartley-IRAS, Coihueco 6. 11. 2026)."""
import numpy as np

from synthtrack.defects import chip_fixed
from synthtrack import search as SR

# chip offsets (dy, dx) of the 18 frames as measured on the real sequence: three groups
D = np.array([[3, 2]] * 6 + [[1, 1]] * 2 + [[0, 0]] * 5 + [[-1, -2]] * 4 + [[0, -1]], float)


def _gauss(img, y, x, amp, fwhm=2.85):
    s = fwhm / 2.3548
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))


def _sequence(seed=3, h=160, w=160):
    rng = np.random.default_rng(seed)
    n = len(D)
    stars = rng.uniform(15, 145, (40, 2))
    amps = rng.uniform(12, 40, 40)
    hot = rng.integers(10, 150, (25, 2))
    cube = rng.normal(0, 1, (n, h, w)).astype(np.float32)
    for i in range(n):
        for (y, x), a in zip(stars, amps):
            _gauss(cube[i], y + D[i, 0], x + D[i, 1], a)
        cube[i, hot[:, 0], hot[:, 1]] += 20.0          # fixed on the chip
    return cube, stars, hot


def test_clustered_dither_keeps_stars_masks_hot_pixels():
    cube, stars, hot = _sequence()
    med = np.median(cube, axis=0)
    cy, cx = np.nonzero(med > 6.0)
    offs = np.repeat(D[:, None, :], len(cy), axis=1)
    fixed = chip_fixed(cube, med, np.stack([cy, cx], 1), offs)
    is_hot = np.zeros(med.shape, bool)
    is_hot[hot[:, 0], hot[:, 1]] = True
    cand_hot = is_hot[cy, cx]
    # the old rule (median > 6) took star cores too – that is what this guards against
    assert (~cand_hot).sum() > 20
    assert fixed[cand_hot].mean() > 0.9                 # hot pixels still masked
    near_star = np.zeros(med.shape, bool)
    for y, x in stars:
        near_star[max(0, int(y) - 6):int(y) + 7, max(0, int(x) - 6):int(x) + 7] = True
    star_px = ~cand_hot & near_star[cy, cx]
    assert star_px.sum() > 20
    assert fixed[star_px].mean() < 0.05                 # star cores are not "hot pixels"


def test_hole_without_bright_surroundings_is_only_missing_data():
    rng = np.random.default_rng(7)
    n, h, w = 18, 96, 96
    reg = rng.normal(0, 1, (n, h, w)).astype(np.float32)
    reg[:, 46:49, 46:49] = np.nan                       # masked hot pixel, missing in every frame
    pt = SR.prepare_tile(reg, 2.85, 40.0)
    ring = np.zeros((h, w), bool)
    ring[40:56, 40:56] = True
    ring[45:50, 45:50] = False
    assert np.abs(pt.f_raw[:, ring]).sum() > 0          # data around the hole are kept
    assert not pt.bright[ring].any()
    # a saturated core (hole ringed by a bright star) is still masked generously
    reg2 = rng.normal(0, 1, (n, h, w)).astype(np.float32)
    for i in range(n):
        _gauss(reg2[i], 47, 47, 5000.0)                  # saturated: wings ≫ 20 σ next to the core
    reg2[:, 45:50, 45:50] = np.nan
    pt2 = SR.prepare_tile(reg2, 2.85, 40.0)
    assert (pt2.f_raw[:, 42:53, 42:53] == 0).all()
    assert pt2.bright[47, 47]


def test_scattered_holes_next_to_bright_mover_are_not_saturation():
    """(7) Iris, 29P: a few pixels missing per frame next to a bright unsaturated object."""
    rng = np.random.default_rng(11)
    n, h, w = 13, 96, 96
    reg = rng.normal(0, 1, (n, h, w)).astype(np.float32)
    for i in range(n):
        _gauss(reg[i], 48 - 0.1 * (i - 6), 48 - 0.25 * (i - 6), 140.0)   # bright, unsaturated
        k = rng.integers(0, 4)
        reg[i, 49 + k:51 + k, 46:49] = np.nan                         # 6 px missing, moving
    pt = SR.prepare_tile(reg, 2.9, 40.0)
    assert np.abs(pt.f_raw[:, 46:51, 46:51]).sum() > 0                # the object is still measured
