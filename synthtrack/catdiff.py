"""Difference image against the catalogue for "zdroj navíc" (a reference image made from UCAC4).

In a dense Milky Way field (327-2/3: 3.4 UCAC4 stars within the 3.8 px match radius of any point) a
new source is almost always "explained" by catalogue stars next to it: it merges with a brighter star
into one detection, or sits in a clump of faint ones whose summed light is as bright. Instead of
matching detections to stars, the stars are *subtracted*:

1. an empirical star profile (PSF) from the median image;
2. a model image: every catalogue star at its (refined) position with its catalogue flux, the scale
   between catalogue flux and image calibrated on the stars themselves, cell by cell over the field;
3. residual = median − model; a matched filter on it, normalised by a noise that includes the
   model's own error (catalogue magnitudes, colour terms): σ² = σ_pix² + (ε · model)². ε is calibrated
   so that the filtered residual *at the catalogue stars* has unit spread;
4. peaks above the threshold: a new source beside / on stars, or a star much brighter than the
   catalogue (variables – an outburst of a dwarf / recurrent nova is wanted);
5. each peak measured in every frame on the same residual (frame model scaled by the frame's
   throughput): a cosmic / satellite in one frame, or a frame-to-frame mismatch, is rejected.

The synthetic-tracking search is not touched; this only adds candidates to "zdroj navíc".
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.fft import irfft2, next_fast_len, rfft2


def _fftconv(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """'same' convolution of a 2-D image with a small odd kernel (FFT)."""
    h, w = a.shape
    kh, kw = k.shape
    sh = (next_fast_len(h + kh - 1), next_fast_len(w + kw - 1))
    r = irfft2(rfft2(a, sh) * rfft2(k, sh), sh)
    oy, ox = kh // 2, kw // 2
    return np.ascontiguousarray(r[oy:oy + h, ox:ox + w]).astype(np.float32)


def gauss_kernel(sg: float, r: int | None = None) -> np.ndarray:
    r = int(math.ceil(3.5 * sg)) if r is None else r
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sg * sg))
    return (g / g.sum()).astype(np.float32)


def empirical_psf(img: np.ndarray, xy: np.ndarray, flux: np.ndarray, fwhm: float, max_n: int = 600) -> np.ndarray:
    """Median of flux-normalised, sub-pixel-centred cut-outs of well-exposed stars (sum = 1).
    Neighbours average out in the median; the wings beyond ~3 FWHM are set to zero."""
    r = int(math.ceil(3.0 * fwhm)) + 1
    h, w = img.shape
    order = np.argsort(-flux)
    sel = order[int(0.02 * len(order)):int(0.35 * len(order))]       # bright, not the saturated top
    cuts = []
    for i in sel:
        x, y = xy[i]
        xi, yi = int(round(x)), int(round(y))
        if not (r + 2 <= xi < w - r - 2 and r + 2 <= yi < h - r - 2):
            continue
        c = np.asarray(img[yi - r - 2:yi + r + 3, xi - r - 2:xi + r + 3], float)
        if not np.isfinite(c).all():
            continue
        c = ndimage.shift(c, (yi - y, xi - x), order=1, mode="nearest")[2:-2, 2:-2]
        b = np.median(np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]]))
        c = c - b
        s = c[r - 2:r + 3, r - 2:r + 3].sum()
        if s <= 0:
            continue
        cuts.append(c / s)
        if len(cuts) >= max_n:
            break
    if len(cuts) < 30:
        return gauss_kernel(fwhm / 2.3548, r)
    p = np.median(np.array(cuts), axis=0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    p[np.hypot(xx, yy) > r] = 0.0
    p = np.clip(p, 0, None)
    return (p / p.sum()).astype(np.float32)


def deposit(shape, xy: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Point fluxes split bilinearly into the four nearest pixels."""
    h, w = shape
    Z = np.zeros(shape, np.float64)
    x, y = xy[:, 0], xy[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1) & np.isfinite(flux)
    x, y, f = x[ok], y[ok], flux[ok]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    np.add.at(Z, (y0, x0), f * (1 - fx) * (1 - fy))
    np.add.at(Z, (y0, x0 + 1), f * fx * (1 - fy))
    np.add.at(Z, (y0 + 1, x0), f * (1 - fx) * fy)
    np.add.at(Z, (y0 + 1, x0 + 1), f * fx * fy)
    return Z.astype(np.float32)


def _ap_sum(a: np.ndarray, xy: np.ndarray, r: float) -> np.ndarray:
    """Sum of ``a`` in a disc of radius r around every point (via a disc-convolved image)."""
    R = int(math.ceil(r))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disc = (np.hypot(xx, yy) <= r).astype(np.float32)
    A = _fftconv(np.nan_to_num(a), disc)
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, a.shape[1] - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, a.shape[0] - 1)
    return A[yi, xi]


class ZonePSF:
    """Empirical PSFs on a grid of zones (the star image changes over a wide field: coma and
    astigmatism towards the corners); a star's PSF is the bilinear blend of the four nearest zone
    centres, done exactly by depositing its flux into each zone's image with that weight."""

    def __init__(self, img, xy, flux, fwhm, shape, cell: int = 1024, min_stars: int = 80):
        h, w = shape
        self.ny, self.nx = max(1, int(round(h / cell))), max(1, int(round(w / cell)))
        self.cy = (np.arange(self.ny) + 0.5) * h / self.ny
        self.cx = (np.arange(self.nx) + 0.5) * w / self.nx
        glob = empirical_psf(img, xy, flux, fwhm)
        self.glob = glob
        self.psfs = {}
        for j in range(self.ny):
            for i in range(self.nx):
                sel = ((xy[:, 0] >= i * w / self.nx) & (xy[:, 0] < (i + 1) * w / self.nx)
                       & (xy[:, 1] >= j * h / self.ny) & (xy[:, 1] < (j + 1) * h / self.ny))
                p = empirical_psf(img, xy[sel], flux[sel], fwhm) if sel.sum() >= min_stars * 4 else glob
                if p.shape != glob.shape:
                    p = glob
                self.psfs[(j, i)] = p

    def _weights(self, xy):
        """(ny, nx, N) bilinear weights of every point over the zone centres (clamped at the edges)."""
        def axis(v, c):
            n_ = len(c)
            if n_ == 1:
                return np.ones((1, len(v)))
            t = np.clip(np.interp(v, c, np.arange(n_)), 0, n_ - 1)
            i0 = np.clip(np.floor(t).astype(int), 0, n_ - 2)
            f = t - i0
            W = np.zeros((n_, len(v)))
            W[i0, np.arange(len(v))] = 1 - f
            W[i0 + 1, np.arange(len(v))] += f
            return W
        return axis(xy[:, 1], self.cy)[:, None, :] * axis(xy[:, 0], self.cx)[None, :, :]

    def model(self, shape, xy, flux):
        W = self._weights(xy)
        M = np.zeros(shape, np.float32)
        for (j, i), p in self.psfs.items():
            wz = W[j, i]
            if np.any(wz > 0):
                M += _fftconv(deposit(shape, xy, flux * wz), p)
        return M


class CatalogueDiff:
    """Model, residual and the normalised matched-filter map of one median image."""

    def __init__(self, img, cxy, exp_flux, cat_mag, fwhm: float, bright_mask_mag: float = 9.0,
                 cell: int = 512, eps_max: float = 1.0, psf_cell: int = 1024):
        img = np.asarray(img, np.float32)
        self.shape = h, w = img.shape
        self.fwhm = fwhm
        ok = np.isfinite(cxy).all(axis=1) & np.isfinite(exp_flux) & (exp_flux > 0)
        ok &= (cxy[:, 0] > -0.5) & (cxy[:, 0] < w - 0.5) & (cxy[:, 1] > -0.5) & (cxy[:, 1] < h - 0.5)
        self.cxy, self.ef, self.cmag = cxy[ok], exp_flux[ok], cat_mag[ok]
        self.idx = np.nonzero(ok)[0]                                 # into the caller's arrays
        fin = np.isfinite(img)
        im = np.where(fin, img, 0.0).astype(np.float32)
        # sky: coarse median (stars are a small part of any 64 px cell even in the Milky Way? not quite –
        # the lower quartile is used, then raised to the median of the noise)
        f = 16
        small = im[::f, ::f]
        bg = ndimage.percentile_filter(small, 35, size=9)
        bg = ndimage.zoom(bg, f, order=1)[:h, :w]
        if bg.shape != im.shape:
            bg = np.pad(bg, ((0, h - bg.shape[0]), (0, w - bg.shape[1])), mode="edge")
        self.zpsf = ZonePSF(im - bg, self.cxy, self.ef, fwhm, self.shape, cell=psf_cell)
        self.psf = self.zpsf.glob                                    # the matched filter
        p2 = (self.psf.astype(np.float64) ** 2).astype(np.float32)
        r_ap = 1.5 * fwhm
        # 1) catalogue flux → image flux as a function of the star's brightness (not linear: bright stars
        #    come out up to ~2× brighter than their catalogue flux, 327-2/3), from stars with a good S/N
        M1 = _fftconv(deposit(self.shape, self.cxy, self.ef), self.psf)   # calibration only: one PSF is enough
        a_img = _ap_sum(im - bg, self.cxy, r_ap)
        a_mod = _ap_sum(M1, self.cxy, r_ap)
        sky_n = 1.4826 * float(np.median(np.abs(im[::7, ::7] - bg[::7, ::7] - np.median(im[::7, ::7] - bg[::7, ::7]))))
        ap_noise = sky_n * math.sqrt(math.pi) * r_ap
        lef = np.log10(np.maximum(self.ef, 1e-12))
        edges = np.arange(np.floor(lef.min() * 4) / 4, lef.max() + 0.25, 0.25)
        cen, val = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (lef >= lo) & (lef < hi) & (a_mod > 0) & (self.cmag > bright_mask_mag)
            if sel.sum() < 25:
                continue
            q = a_img[sel] / a_mod[sel]
            # S/N of these stars in the aperture (from their model flux and the sky noise)
            if np.median(a_mod[sel]) * np.median(q) < 15 * ap_noise:
                continue
            cen.append(0.5 * (lo + hi)); val.append(float(np.median(q)))
        if len(cen) >= 2:
            cen, val = np.array(cen), np.array(val)
            corr = np.interp(lef, cen, val, left=val[0], right=val[-1])
        else:
            g = (a_mod > 0)
            corr = np.full(len(lef), float(np.median(a_img[g] / a_mod[g])) if g.any() else 1.0)
        self.mag_corr = (cen, val) if len(cen) >= 2 else None
        self.flux = self.ef * corr                                   # image units
        M2 = self.zpsf.model(self.shape, self.cxy, self.flux)
        # 2) what is left of the scale over the field (vignetting, zero-point map errors), cell by cell,
        #    from the well-exposed stars
        a_mod2 = _ap_sum(M2, self.cxy, r_ap)
        good = (a_mod2 > 15 * ap_noise) & (self.cmag > bright_mask_mag)
        ratio = a_img[good] / a_mod2[good]
        ny, nx = max(1, h // cell), max(1, w // cell)
        K = np.full((ny, nx), np.nan)
        cxg = np.clip((self.cxy[good, 0] / w * nx).astype(int), 0, nx - 1)
        cyg = np.clip((self.cxy[good, 1] / h * ny).astype(int), 0, ny - 1)
        for jj in range(ny):
            for ii in range(nx):
                v = ratio[(cxg == ii) & (cyg == jj)]
                if v.size >= 15:
                    K[jj, ii] = np.median(v)
        kglob = float(np.median(ratio)) if ratio.size else 1.0
        K = np.where(np.isfinite(K), K, kglob)
        self.k_global = kglob
        if ny > 1 or nx > 1:
            Kf = ndimage.zoom(K, (h / ny, w / nx), order=1, mode="nearest")[:h, :w]
            if Kf.shape != im.shape:
                Kf = np.pad(Kf, ((0, h - Kf.shape[0]), (0, w - Kf.shape[1])), mode="edge")
        else:
            Kf = np.full((h, w), kglob, np.float32)
        self.kmap = Kf.astype(np.float32)
        self.model = (M2 * self.kmap).astype(np.float32)
        R = im - bg - self.model
        # residual sky (what the quartile missed) from the residual itself, robust
        rb = ndimage.median_filter(R[::f, ::f], size=9)
        rb = ndimage.zoom(rb, f, order=1)[:h, :w]
        if rb.shape != R.shape:
            rb = np.pad(rb, ((0, h - rb.shape[0]), (0, w - rb.shape[1])), mode="edge")
        self.resid = (R - rb).astype(np.float32)
        self.bg = (bg + rb).astype(np.float32)
        self.fin = fin
        # bright, saturated stars: no model is good enough – masked with a radius that grows with them
        self.mask = ~fin
        br = np.nonzero(self.cmag <= bright_mask_mag)[0]
        m = np.zeros((h, w), bool)
        for i in br:
            rad = fwhm * (2.0 + 1.5 * (bright_mask_mag - self.cmag[i]))
            x0, y0 = self.cxy[i]
            X0, X1 = int(max(0, x0 - rad - 1)), int(min(w, x0 + rad + 2))
            Y0, Y1 = int(max(0, y0 - rad - 1)), int(min(h, y0 + rad + 2))
            if X1 > X0 and Y1 > Y0:
                yy, xx = np.ogrid[Y0:Y1, X0:X1]
                m[Y0:Y1, X0:X1] |= (xx - x0) ** 2 + (yy - y0) ** 2 <= rad * rad
        self.mask |= m
        # matched filter = the PSF itself
        self.S = _fftconv(np.where(self.mask, 0.0, self.resid), self.psf)
        quiet = (~self.mask) & (self.model < 0.05 * np.nanpercentile(self.model, 99.5))
        sp = self.resid[quiet][::7]
        sig_pix = 1.4826 * float(np.median(np.abs(sp - np.median(sp)))) if sp.size > 1000 else float(np.std(self.resid))
        self.sig_pix = sig_pix
        A0 = sig_pix ** 2 * float(p2.sum())
        # confusion noise (stars below the catalogue, unresolved): the spread of the whole map
        zz = (self.S[~self.mask][::11]) / math.sqrt(A0)
        self.conf = max(1.0, 1.4826 * float(np.median(np.abs(zz - np.median(zz)))))
        self.A = np.float32(A0 * self.conf ** 2)
        # model error: every star's flux is off by a relative ε, independently of its neighbours – the
        # variance of S is ε² Σ_j (F_j · ac(p − x_j))², ac = PSF ⊗ PSF (what one star gives in S)
        ac = _fftconv(np.pad(self.psf, self.psf.shape[0] // 2), self.psf)
        ix_ = np.clip(np.round(self.cxy[:, 0]).astype(int), 0, w - 1)
        iy_ = np.clip(np.round(self.cxy[:, 1]).astype(int), 0, h - 1)
        Fimg = self.flux * self.kmap[iy_, ix_]
        self.B = np.maximum(_fftconv(deposit(self.shape, self.cxy, Fimg.astype(np.float64) ** 2), ac * ac), 0.0)
        # ε (relative error of a catalogue star's flux) from well-exposed stars: unit spread there
        ix = np.clip(np.round(self.cxy[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(self.cxy[:, 1]).astype(int), 0, h - 1)
        sel = (~self.mask[iy, ix]) & (self.cmag > bright_mask_mag)
        sel &= self.B[iy, ix] > 0
        S_st, B_st = self.S[iy, ix][sel], self.B[iy, ix][sel]
        strong = S_st.size and (np.sqrt(B_st) * 0.2 > np.sqrt(self.A))         # model error would dominate
        S_st, B_st = (S_st[strong], B_st[strong]) if np.count_nonzero(strong) >= 50 else (S_st, B_st)
        eps = eps_max
        for e in np.linspace(0.0, eps_max, 201):
            z = S_st / np.sqrt(self.A + e * e * B_st)
            if z.size and 1.4826 * np.median(np.abs(z - np.median(z))) <= 1.0:
                eps = float(e)
                break
        self.eps = eps
        Z_ = self.S / np.sqrt(self.A + eps * eps * self.B)
        zz = Z_[~self.mask][::11]
        self.z_scale = max(1.0, 1.4826 * float(np.median(np.abs(zz - np.median(zz)))))
        self.z = np.where(self.mask, 0.0, Z_ / self.z_scale).astype(np.float32)
        # a point source of exp_flux 1 gives S = k · Σ psf²: the flux of a residual peak
        self.s_unit = float(p2.sum())
        self.psf_ac = _fftconv(np.pad(self.psf, self.psf.shape[0] // 2), self.psf)
        self._corr = (lambda f: np.interp(np.log10(max(f, 1e-12)), *self.mag_corr, left=self.mag_corr[1][0],
                                          right=self.mag_corr[1][-1])) if self.mag_corr else (lambda f: 1.0)

    def flux_at(self, x: float, y: float) -> float:
        """Residual flux in exp_flux units (as the catalogue stars)."""
        xi, yi = int(round(x)), int(round(y))
        f_img = float(self.S[yi, xi]) / max(float(self.kmap[yi, xi]) * self.s_unit, 1e-12)
        return f_img / max(float(self._corr(f_img)), 1e-6)       # (approx.: the correction at this flux)

    def peaks(self, nsigma: float, border: int, min_sep: float | None = None) -> np.ndarray:
        """(N, 3) x, y (sub-pixel), z of the residual peaks above ``nsigma``."""
        z = self.z
        size = max(3, int(round(min_sep or 1.5 * self.fwhm)) | 1)
        mx = (z == ndimage.maximum_filter(z, size=size)) & (z > nsigma)
        mx[:border] = mx[-border:] = False
        mx[:, :border] = mx[:, -border:] = False
        ys, xs = np.nonzero(mx)
        out = []
        for y, x in zip(ys, xs):
            def par(a, b, c):
                d = a - 2 * b + c
                return float(np.clip(0.5 * (a - c) / d, -1, 1)) if d < 0 else 0.0
            fx = par(z[y, x - 1], z[y, x], z[y, x + 1])
            fy = par(z[y - 1, x], z[y, x], z[y + 1, x])
            out.append((x + fx, y + fy, float(z[y, x])))
        return np.array(out) if out else np.zeros((0, 3))

    def frame_snr(self, frame: np.ndarray, scale: float, x: float, y: float) -> float | None:
        """The same residual statistic in one frame: frame − scale · (model + sky); its noise from an
        annulus of the frame's own residual."""
        r = int(math.ceil(4 * self.fwhm)) + self.psf.shape[0] // 2 + 2
        xi, yi = int(round(x)), int(round(y))
        h, w = self.shape
        if not (r <= xi < w - r and r <= yi < h - r):
            return None
        sl = (slice(yi - r, yi + r + 1), slice(xi - r, xi + r + 1))
        c = np.asarray(frame[sl], np.float32)
        fin = np.isfinite(c)
        if fin.mean() < 0.8:
            return None
        res = np.where(fin, c, 0.0) - scale * self.model[sl]
        res = res - float(np.median(res[fin]))
        res[~fin] = 0.0
        Sm = ndimage.convolve(res, self.psf, mode="constant")
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        rr = np.hypot(xx, yy)
        ann = (rr > 2.5 * self.fwhm) & (rr <= r - self.psf.shape[0] // 2) & ndimage.binary_erosion(fin, iterations=2)
        if ann.sum() < 40:
            return None
        v = Sm[ann]
        # noise: the annulus spread, but never below the white noise expected for this frame
        n_ = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        n_ = max(n_, scale * math.sqrt(float(self.A)))
        if not fin[r, r]:
            return None
        return float(Sm[r, r] / max(n_, 1e-9))

    def frame_scales(self, cube, n_stars: int = 800) -> list[float]:
        """Frame units per median-image unit for every frame (includes its throughput): the robust ratio
        of the frame's filtered peak to the median image's at well-exposed catalogue stars, both on
        their own coarse sky."""
        h, w = self.shape
        sel = np.argsort(-self.flux)
        sel = sel[(self.cmag[sel] > 9.5)][:n_stars * 4:4]
        ix = np.clip(np.round(self.cxy[sel, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(self.cxy[sel, 1]).astype(int), 0, h - 1)
        base = _fftconv(self.model, self.psf)[iy, ix]
        out = []
        f = 16
        for i in range(cube.shape[0]):
            fr = np.nan_to_num(np.asarray(cube[i], np.float32))
            bgf = ndimage.percentile_filter(fr[::f, ::f], 35, size=9)
            bgf = ndimage.zoom(bgf, f, order=1)[:h, :w]
            if bgf.shape != fr.shape:
                bgf = np.pad(bgf, ((0, h - bgf.shape[0]), (0, w - bgf.shape[1])), mode="edge")
            v = _fftconv(fr - bgf, self.psf)[iy, ix]
            g = base > 0
            out.append(float(np.median(v[g] / base[g])) if g.any() else 1.0)
        return out
