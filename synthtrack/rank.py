"""Per-run ranking of passed candidates calibrated by injected synthetic movers.

The fixed SNR threshold and the time-shuffled noise estimate (``search.noise_threshold``)
do not reach the false-alarm level of dense Milky Way fields: 161P/Hartley-IRAS,
Coihueco 7. 11. 2026, 1 567 passing unknown candidates; the shuffled null said
0–2 per tile where 135–270 passed (star residuals follow time trends, a shuffle
breaks them). Injected movers are real-looking objects whose positions are known.
In every run with injections a small logistic model separates *passed injected*
objects (positives) from *passed non-injected* detections (almost all false,
"positive-unlabelled"); out-of-fold scores keep ``keep`` of the injections and
everything scoring lower is set aside. Identified known objects are never set
aside here; how many of them would fall under the cut is reported – it is the
check on real objects (injections are idealised point sources).

On the 161P run (5-fold CV, AUC 0.93): keeping 95 / 90 % of injections left
453 / 276 of 1 567 other candidates; the comet scored 0.997."""
from __future__ import annotations

import math

import numpy as np

# Only features with a physical reason. "independent check / search SNR" and "raw / clipped SNR"
# separated the (idealised) injections well but with the wrong sign for a real comet: with them
# 161P/Hartley-IRAS scored 0.32 (set aside at keep 0.9), without them 0.997.
# Full set; "object width / star FWHM" and "detector path / linear track" separate well only
# when the injections look like the field's real objects (star profile from the frame and
# the field's typical motion, inject.py). With Gaussian injections at random directions they
# put all 3 identified asteroids of the 29P field under the cut (0.07–0.34); the ranking then
# falls back to SAFE automatically (see calibrate).
NAMES = ["chi2", "reverse_rel", "frames_signal", "mover_T", "median_ratio", "log_star_ratio",
         "seg_min_over_max", "fwhm_rel", "chip_ratio"]
SAFE = [0, 1, 2, 3, 4, 5, 6]


def _g(c, k, default):
    v = getattr(c, k, None) if not isinstance(c, dict) else c.get(k)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def features(c, fwhm: float) -> list[float]:
    snr = max(_g(c, "snr", 6.0), 1.0)
    seg = (c.get("seg_snr") if isinstance(c, dict) else getattr(c, "seg_snr", None)) or [snr, snr, snr]
    seg = np.asarray(seg, float)
    fw = _g(c, "fwhm", fwhm) or fwhm
    return [min(_g(c, "chi2", 1.0), 6.0),
            float(np.clip(_g(c, "snr_reverse", 0.0) / snr, -2, 2)),
            _g(c, "frames_with_signal", 0.8),
            float(np.clip(_g(c, "mover_vs_static", 0.3), -3, 1)),
            float(np.clip(_g(c, "median_ratio", 1.0), -1, 3)),
            math.log1p(max(_g(c, "star_ratio", 0.0), 0.0)),
            float(np.clip(seg.min() / max(seg.max(), 0.1), -2, 1)),
            float(np.clip(_g(c, "obj_fwhm", fw) / max(fw, 0.5), 0.2, 3.0)),
            float(np.clip(_g(c, "chip_ratio", 0.3), -1, 2))]


def _fit(z, y, l2=1.0, iters=3000, lr=0.1):
    w = np.zeros(z.shape[1])
    b = 0.0
    pos = max(y.mean(), 1e-6)
    sw = np.where(y == 1, (1 - pos) / pos, 1.0)          # balanced classes
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z @ w + b)))
        g = (p - y) * sw
        w -= lr * (z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def calibrate(cands, fwhm: float, keep: float = 0.9, folds: int = 5, seed: int = 0,
              min_pos: int = 20, min_neg: int = 30, soft: float | None = 0.97) -> dict | None:
    """Score passed candidates (``c.inj_score``, 0–1) and set aside those under the cut.

    The full feature set is tried first; if an identified known object would fall
    under its cut, the SAFE set is used instead (and if even then, the cut is lowered).
    Returns a summary dict, or None when there are too few injections/candidates."""
    res = _calibrate(cands, fwhm, keep, folds, seed, min_pos, min_neg, list(range(len(NAMES))), dry=True, soft=soft)
    if res is None:
        return None
    cols, used = list(range(len(NAMES))), "full"
    if res["known_below_cut"]:
        cols, used = SAFE, "safe"
    out = _calibrate(cands, fwhm, keep, folds, seed, min_pos, min_neg, cols, dry=False, soft=soft)
    out["feature_set"] = used
    if used == "safe":
        out["full_set_known_below"] = res["known_below_cut"]
    return out


def _calibrate(cands, fwhm, keep, folds, seed, min_pos, min_neg, cols, dry, soft=None):
    passed = [c for c in cands if c.auto_status == "candidate"]
    pos = [c for c in passed if c.injected is not None]
    # the model only knows the SNR range of the injections: brighter detections are neither
    # judged nor used as the safety check ((7) Iris, SNR 73 vs injections ≤ 15, scored 0.56 and
    # dragged the cut down for 1 000 candidates of the 29P field)
    smax = 1.5 * max((_g(c, "snr", 0.0) for c in pos), default=0.0)
    inrange = lambda c: _g(c, "snr", 0.0) <= smax
    neg = [c for c in passed if c.injected is None and not c.known and inrange(c)]
    known = [c for c in passed if c.injected is None and c.known and inrange(c)]
    if len(pos) < min_pos or len(neg) < min_neg:
        return None
    train = pos + neg
    x = np.array([features(c, fwhm) for c in train], float)[:, cols]
    y = np.array([1] * len(pos) + [0] * len(neg), float)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mu) / sd
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)                        # stratified folds
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        fold[idx] = rng.permutation(np.arange(len(idx)) % folds)
    oof = np.zeros(len(y))
    for k in range(folds):
        w, b = _fit(z[fold != k], y[fold != k])
        oof[fold == k] = z[fold == k] @ w + b
    w, b = _fit(z, y)
    cut = float(np.quantile(oof[y == 1], 1.0 - keep))
    zk = (np.array([features(c, fwhm) for c in known], float)[:, cols] - mu) / sd if known else np.zeros((0, len(mu)))
    sk = zk @ w + b if known else np.zeros(0)
    sig = lambda s: 1.0 / (1.0 + np.exp(-(s - cut)))   # 0.5 = the cut
    if dry:
        return {"known_below_cut": [c.known.get("name", "?") for c, s in zip(known, sk) if s < cut]}
    for c, s in zip(train, oof):
        c.inj_score = round(float(sig(s)), 3)
    for c, s in zip(known, sk):
        c.inj_score = round(float(sig(s)), 3)
    # safety: an identified real object under the cut means the injections do not represent
    # this field – lower the cut below the weakest known object instead of losing real ones
    known_below = [c.known.get("name", "?") for c, s in zip(known, sk) if s < cut]
    lowered = None
    cut_orig = cut
    if len(sk) and float(sk.min()) < cut:
        lowered = round(float(sk.min()), 3)
        cut = float(sk.min()) - 0.25
        for c, s in zip(train, oof):
            c.inj_score = round(float(sig(s)), 3)
        for c, s in zip(known, sk):
            c.inj_score = round(float(sig(s)), 3)
    # second, looser cut: what lies between the two is set aside as "borderline" – its own
    # group in the GUI, to look through after the main list (29P: keep 0.9 → 122 left,
    # 0.97 → 275; the 10 % of injections under the main cut are ambiguous: moving with the
    # dither drift like warm pixels, or a third of the series at low throughput)
    cut2 = min(cut, float(np.quantile(oof[y == 1], 1.0 - soft))) if soft and soft > keep else cut
    if lowered is not None and soft and soft > keep:
        # the cut was lowered under a known object: keep a borderline band of the same width
        # below the new cut (03958/11P 8. 11.: the soft cut lay above the lowered one → 0 borderline)
        cut2 = cut - max(0.0, cut_orig - float(np.quantile(oof[y == 1], 1.0 - soft)))
    removed = borderline = 0
    for c, s in zip(neg, oof[len(pos):]):
        if s < cut:
            c.auto_status = "auto_rejected"
            if s >= cut2:
                c.flags.append("hraniční skóre podle injekcí")
                borderline += 1
            else:
                c.flags.append("nízké skóre podle injekcí")
            removed += 1
    auc = _auc(oof[y == 1], oof[y == 0])
    return {"keep": keep, "injected_passed": len(pos), "others": len(neg), "set_aside": removed,
            "remaining": len(neg) - removed, "borderline": borderline, "keep_soft": soft,
            "cv_auc": round(auc, 3), "known_scored": len(known),
            "known_below_cut": known_below, "cut_lowered_for_known": lowered is not None,
            "snr_max_judged": round(smax, 1),
            "weights": {NAMES[i]: round(float(v), 2) for i, v in zip(cols, w)}}


def _auc(p, n) -> float:
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    r = allv.argsort().argsort() + 1.0
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
