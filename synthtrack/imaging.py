"""Cutouts along a track: mean / median stacks and animated sub-stacks."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CROSS_RGB = (255, 196, 40)          # amber, visible on white stars and black sky

# 8-bit encoding of σ-unit frames for the interactive viewer (GUI):
#   q = C + B·asinh(x / A)  →  ~0.07 σ per level near the sky, ±50 σ range
ENC_A, ENC_B, ENC_C = 2.0, 32.0, 127.5


def _nanmean(a: np.ndarray, axis=0) -> np.ndarray:
    """NaN-aware mean without flooding logs for wholly masked pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a: np.ndarray, axis=0) -> np.ndarray:
    """NaN-aware median without flooding logs for wholly masked pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def track_crops(cube: np.ndarray, dts: np.ndarray, x: float, y: float, vx: float, vy: float,
                size: int) -> np.ndarray:
    """(N, size, size) crops following the motion (zero padded at edges)."""
    n, h, w = cube.shape
    half = size // 2
    out = np.zeros((n, size, size), np.float32)
    for i in range(n):
        cx = int(round(x + vx * dts[i]))
        cy = int(round(y + vy * dts[i]))
        y0, y1, x0, x1 = cy - half, cy - half + size, cx - half, cx - half + size
        sy0, sy1, sx0, sx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        out[i, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = cube[i, sy0:sy1, sx0:sx1]
    return out


def static_crops(cube: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """(N, size, size) crops at a fixed position (stars stay put), zero padded."""
    n, h, w = cube.shape
    half = size // 2
    out = np.zeros((n, size, size), np.float32)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    sy0, sy1, sx0, sx1 = max(0, y0), min(h, y0 + size), max(0, x0), min(w, x0 + size)
    if sy1 > sy0 and sx1 > sx0:
        out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = cube[:, sy0:sy1, sx0:sx1]
    return out


def _draw_cross(im: Image.Image, x: float, y: float, gap: float, arm: float, width: int = 2) -> Image.Image:
    """Crosshair with an open centre (the object itself stays visible)."""
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line([(x + dx * gap, y + dy * gap), (x + dx * (gap + arm), y + dy * (gap + arm))],
               fill=CROSS_RGB, width=width)
    return im


def write_motion_gif(out_dir: Path, tag: str, cube: np.ndarray, dts: np.ndarray, x: float, y: float,
                     vx: float, vy: float, size: int, fwhm: float = 2.5, scale: int = 3,
                     frames_max: int = 6, segments_min: int = 3, duration_ms: int = 350) -> str:
    """Animation in the star frame: stars stand still, the object moves and a
    crosshair follows it. Frames are sub-stacks of consecutive registered
    frames (more frames = smoother motion, fewer = less noise)."""
    n = cube.shape[0]
    t_mid = 0.5 * (float(np.min(dts)) + float(np.max(dts)))
    cx, cy = x + vx * t_mid, y + vy * t_mid
    span = abs(vx) + abs(vy)
    span *= float(np.max(dts) - np.min(dts))
    size = int(size + min(span, 2 * size))           # keep the whole path inside
    crops = static_crops(cube, cx, cy, size)
    half = size // 2
    k = int(np.clip(n // 3, segments_min, frames_max)) if n >= segments_min else max(1, n)
    subs, pos = [], []
    for idx in np.array_split(np.arange(n), k):
        subs.append(_nanmean(crops[idx], axis=0))
        t = float(np.mean(dts[idx]))
        ox = x + vx * t - (round(cx) - half)
        oy = y + vy * t - (round(cy) - half)
        pos.append(((ox + 0.5) * scale, (oy + 0.5) * scale))
    # limits from the noise of ONE sub-stack (not of the full mean, which is √k
    # times less noisy – that over-stretched the background of the animation)
    with np.errstate(all="ignore"):
        lo, hi = _stretch(subs[0], -1.5, 8.0)
    gap = max(2.0 * fwhm, 5.0) * scale
    arm = max(2.5 * fwhm, 6.0) * scale
    frames = [_draw_cross(to_u8(sb, lo, hi, scale=scale, gamma=1.0), px, py, gap, arm, width=max(2, scale - 1))
              for sb, (px, py) in zip(subs, pos)]
    p = out_dir / f"c{tag}_motion.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return p.name


def _stretch(img: np.ndarray, k_lo: float = 1.5, k_hi: float = 6.0):
    """Robust limits: background median −k_lo·σ … +k_hi·σ of the crop (σ from MAD)."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return -1.0, 1.0
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med))) or 1.0
    return med - abs(k_lo) * mad, med + k_hi * mad


def to_u8(img: np.ndarray, lo: float | None = None, hi: float | None = None, scale: int = 3,
          gamma: float = 2.0) -> Image.Image:
    """8-bit image; ``gamma`` 2 = square-root stretch (stacks), 1 = linear (animation)."""
    if lo is None or hi is None:
        lo, hi = _stretch(img)
    finite = img[np.isfinite(img)]
    fill = float(np.median(finite)) if finite.size else 0.0
    # Missing coverage is not signal.  Showing NaNs as saturated white made
    # masks look like real square sources in the animation.
    a = np.clip((np.nan_to_num(img, nan=fill) - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        a = a ** (1.0 / gamma)
    im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def local_star_shifts(crops: np.ndarray, max_shift: float = 3.0, min_peak: float = 0.3) -> list:
    """Residual shift (dx, dy) of the stars in every crop relative to the median crop.

    Registration is a global polynomial; locally (field corners, a partly
    blocked pupil, a poorly constrained frame) the stars of one frame can be
    off by 1–2 px, which makes the animation jump. Cross-correlation of the
    star field (high-passed, clipped) with sub-pixel parabolic peak; frames
    where the correlation is weak or the shift implausible get (0, 0)."""
    n, h, w = crops.shape
    c = np.nan_to_num(crops.astype(np.float32))
    c = np.clip(c, -3.0, 50.0)
    ref = np.median(c, axis=0)
    def prep(a):
        a = a - np.median(a)
        a[a < 2.0] = 0.0                               # stars only, no noise
        return a
    R = np.fft.rfft2(prep(ref))
    out = []
    for i in range(n):
        F = np.fft.rfft2(prep(c[i]))
        cc = np.fft.irfft2(F * np.conj(R), s=(h, w))
        denom = math.sqrt(float((prep(c[i]) ** 2).sum()) * float((prep(ref) ** 2).sum())) or 1.0
        k = int(np.argmax(cc))
        py, px = divmod(k, w)
        peak = float(cc[py, px]) / denom
        def sub(m, z, p):                              # parabolic sub-pixel offset of the peak
            d = m - 2 * z + p
            return float(np.clip(0.5 * (m - p) / d, -0.5, 0.5)) if d < 0 else 0.0
        z = cc[py, px]
        dy = py + sub(cc[(py - 1) % h, px], z, cc[(py + 1) % h, px])
        dx = px + sub(cc[py, (px - 1) % w], z, cc[py, (px + 1) % w])
        dy = dy - h if dy > h / 2 else dy
        dx = dx - w if dx > w / 2 else dx
        if peak < min_peak or math.hypot(dx, dy) > max_shift:
            dx = dy = 0.0
        out.append([round(float(dx), 2), round(float(dy), 2)])
    return out


def write_motion_frames(out_dir: Path, tag: str, cube: np.ndarray, dts: np.ndarray, x: float, y: float,
                        vx: float, vy: float, size: int, max_size: int = 128, amp=None) -> dict:
    """All registered frames around the track as one 8-bit strip for the GUI viewer.

    Stars stand still, the object moves along ``track`` (crop pixel coordinates
    per frame). The GUI builds the sub-stacks (2…n frames per animation) and
    the brightness/contrast itself, so nothing has to be recomputed here.
    Encoding: q = C + B·asinh(x/A) with x in σ units (``enc`` = [A, B, C])."""
    n = cube.shape[0]
    t_mid = 0.5 * (float(np.min(dts)) + float(np.max(dts)))
    cx, cy = x + vx * t_mid, y + vy * t_mid
    span = (abs(vx) + abs(vy)) * float(np.max(dts) - np.min(dts))
    size = int(min(max(size, size + span), max(size, max_size)))
    crops = static_crops(cube, cx, cy, size)
    half = size // 2
    ox, oy = round(cx) - half, round(cy) - half
    track = [[round(x + vx * float(t) - ox, 2), round(y + vy * float(t) - oy, 2)] for t in dts]
    # Registered data are in sigma units, therefore zero is the neutral
    # background value for pixels without coverage.
    q = np.clip(np.rint(ENC_C + ENC_B * np.arcsinh(np.nan_to_num(crops, nan=0.0) / ENC_A)), 0, 255).astype(np.uint8)
    p = out_dir / f"c{tag}_frames.png"
    Image.fromarray(q.reshape(n * size, size), mode="L").save(p, optimize=True)
    meta = {"file": p.name, "size": size, "n": n, "enc": [ENC_A, ENC_B, ENC_C], "track": track,
            "dt_min": [round(float(t), 3) for t in dts]}
    try:
        meta["shift"] = local_star_shifts(crops)
    except Exception:          # noqa: BLE001 – a viewer aid, never fail the job
        pass
    if amp is not None:        # throughput of every frame at the crop centre (obstruction, cirrus)
        yi = int(min(max(round(cy), 0), amp.shape[1] - 1))
        xi = int(min(max(round(cx), 0), amp.shape[2] - 1))
        meta["tp"] = [round(float(a), 3) for a in amp[:, yi, xi]]
    return meta


def write_candidate_images(out_dir: Path, cid: int | str, crops: np.ndarray, segments: int = 3,
                           scale: int = 3, diff_crops: np.ndarray | None = None,
                           fwhm: float = 2.5) -> dict[str, str]:
    """mean / median stacks along the track, animated sub-stacks and (optionally,
    ``diff_crops`` = 2-D SNR map along the track) the significance map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cid:04d}" if isinstance(cid, int) else str(cid)
    n = crops.shape[0]
    mean = _nanmean(crops, axis=0)
    med = _nanmedian(crops, axis=0)
    files = {}
    lo, hi = _stretch(mean)
    p = out_dir / f"c{tag}_mean.png"
    to_u8(mean, lo, hi, scale=scale).save(p)
    files["mean"] = p.name
    p = out_dir / f"c{tag}_median.png"
    to_u8(med, scale=scale).save(p)
    files["median"] = p.name
    frames = []
    subs = []
    for idx in np.array_split(np.arange(n), max(1, segments)):
        subs.append(_nanmean(crops[idx], axis=0))
    slo, shi = _stretch(subs[0])
    for sb in subs:
        frames.append(to_u8(sb, slo, shi, scale=scale))
    p = out_dir / f"c{tag}_sub.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=500, loop=0)
    files["gif"] = p.name
    if diff_crops is not None:
        # significance map: sum of the filtered unit-noise frames along the track / σ_sum
        d = np.nan_to_num(diff_crops)
        p = out_dir / f"c{tag}_snr.png"
        to_u8(d, -2.0, 8.0, scale=scale).save(p)
        files["snr"] = p.name
    return files
