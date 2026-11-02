"""Candidate classifier learned from the user's decisions.

Training examples come from every finished job in the data directory:
  * positives – candidates the user *accepted*, detections identified with a
    known asteroid/comet, recovered injections (``injected``),
  * negatives – candidates the user *rejected* (full weight if reviewed by
    hand, reduced weight if they were only auto-rejected and never touched),
  * "unsure" and unreviewed candidates are not used.

Model: L2-regularised logistic regression on vetting features (NumPy, IRLS),
class-balanced. It is deliberately simple: with tens of positives a neural
network would only memorise them. The feature extraction and the stored
format (``kind``) are ready for an MLP / CNN on the frame strips once there are
hundreds of examples (injections make that cheap).

Labelled examples survive deleting a job: before a job is deleted (and at
every training) its labelled candidates are saved as feature vectors in
``<data>/models/examples/<job id>.json``; ``collect`` uses live jobs plus the
archives of jobs that no longer exist. Several observing series therefore
add up even when their images and results are cleaned away.

The model never deletes anything: it adds ``ml_score`` (probability of a real
object) that the GUI shows, sorts and filters by.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "classifier.json"
FEATURES = [
    "log_snr", "raw_over_snr", "reverse_over_snr", "chi2", "median_ratio",
    "seg_min_z", "seg_spread", "log_rate_over_min", "log_motion_fwhm", "star_contaminated",
    "frames_frac", "throughput_min", "fwhm_ratio", "substack_flux_cv", "n_soft_flags",
]
SOFT_FLAGS = {"u hvězdy", "snížená propustnost"}


def model_path(data_dir) -> Path:
    return Path(data_dir) / "models" / MODEL_NAME


def features(c: dict, ctx: dict) -> np.ndarray:
    """Feature vector of one candidate dict; ``ctx``: min_rate, fwhm, n_frames, span_min."""
    snr = max(float(c.get("snr") or 0.0), 1e-3)
    seg = np.asarray(c.get("seg_snr") or [snr], float)
    k = max(len(seg), 1)
    exp_seg = snr / math.sqrt(k)
    rate = float(c.get("rate_arcsec_min") or 0.0)
    rmin = max(float(ctx.get("min_rate") or 0.1), 1e-3)
    fw = float(c.get("fwhm") or ctx.get("fwhm") or 2.5)
    span = float(ctx.get("span_min") or 60.0)
    scale = float(ctx.get("pixel_scale") or 1.0)
    motion_px = rate / max(scale, 1e-6) * span
    fl = [p.get("flux") for p in (c.get("positions") or []) if p.get("flux") is not None]
    cv = float(np.std(fl) / max(abs(np.mean(fl)), 1e-6)) if len(fl) >= 2 else 0.5
    n = max(int(ctx.get("n_frames") or 1), 1)
    tp = c.get("throughput_min")
    v = [
        math.log(snr),
        float(c.get("snr_raw") or snr) / snr,
        float(c.get("snr_reverse") or 0.0) / snr,
        min(float(c.get("chi2") or 1.0), 6.0),
        float(np.clip(c.get("median_ratio") or 0.0, -1.0, 3.0)),
        float(np.clip((seg.min() - exp_seg), -8, 8)),
        float(np.clip(seg.std(), 0, 8)),
        math.log(max(rate, 1e-3) / rmin),
        math.log(max(motion_px, 1e-2) / max(fw, 0.5)),
        1.0 if c.get("star_contaminated") else 0.0,
        float(c.get("frames_used") or n) / n,
        1.0 if tp is None else float(np.clip(tp, 0, 1.5)),
        fw / max(float(ctx.get("fwhm") or fw), 0.5),
        float(np.clip(cv, 0, 3)),
        float(sum(1 for f in (c.get("flags") or []) if f in SOFT_FLAGS)),
    ]
    return np.asarray(v, float)


def job_context(summary: dict) -> dict:
    return {"min_rate": summary.get("min_rate_arcsec_min"), "fwhm": summary.get("fwhm_px"),
            "n_frames": summary.get("n_frames"), "span_min": summary.get("span_min"),
            "pixel_scale": summary.get("pixel_scale")}


def label(c: dict) -> tuple[float | None, float]:
    """(target, weight) of a candidate dict, target None = not used."""
    st = c.get("status")
    if c.get("injected") is not None:
        return 1.0, 1.0
    if st == "accepted":
        return 1.0, 1.0
    if c.get("known") and st != "rejected":
        return 1.0, 1.0
    if st == "rejected":
        return 0.0, (1.0 if c.get("reviewed") or c.get("auto_status") == "candidate" else 0.3)
    return None, 0.0


def _job_examples(jd: Path) -> dict | None:
    """Labelled examples of one finished job directory (None = nothing usable)."""
    try:
        job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
        cands = json.loads((jd / "candidates.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if job.get("status") != "done":
        return None
    ctx = job_context(job.get("summary") or {})
    ex = {"job_id": jd.name, "name": job.get("name"), "instrument": job.get("instrument_id"),
          "features_names": FEATURES, "X": [], "y": [], "w": [],
          "injected": 0, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    for c in cands:
        t, wt = label(c)
        if t is None or wt <= 0:
            continue
        ex["X"].append([round(float(v), 5) for v in features(c, ctx)])
        ex["y"].append(t)
        ex["w"].append(wt)
        ex["injected"] += int(c.get("injected") is not None)
    return ex if ex["y"] else None


def examples_dir(data_dir) -> Path:
    return Path(data_dir) / "models" / "examples"


def archive_job(data_dir, job_dir) -> bool:
    """Save the labelled examples of a job (call before deleting it)."""
    ex = _job_examples(Path(job_dir))
    if ex is None:
        return False
    d = examples_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ex['job_id']}.json").write_text(json.dumps(ex), encoding="utf-8")
    return True


def example_sets(data_dir) -> list[dict]:
    """Labelled example sets: live jobs plus archives of deleted jobs."""
    sets, live = [], set()
    for jd in sorted((Path(data_dir) / "jobs").glob("*")):
        ex = _job_examples(jd)
        if ex is not None:
            sets.append(ex)
            live.add(ex["job_id"])
    for p in sorted(examples_dir(data_dir).glob("*.json")):
        if p.stem in live:
            continue
        try:
            ex = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if ex.get("features_names") != FEATURES:
            continue                       # archived with another feature set
        ex["_archived"] = True
        sets.append(ex)
    return sets


def collect(data_dir, instrument: str | None = None, sets=None, with_groups: bool = False):
    """Examples (optionally of one instrument only), grouped by observing job."""
    if sets is None:
        sets = example_sets(data_dir)
    if instrument is not None:
        sets = [ex for ex in sets if ex.get("instrument") == instrument]
    X, y, w, groups = [], [], [], []
    info = {"jobs": 0, "archived_jobs": 0, "pos": 0, "neg": 0, "injected": 0}
    for ex in sets:
        X.extend(ex["X"])
        y.extend(ex["y"])
        w.extend(ex["w"])
        groups.extend([str(ex.get("job_id") or f"set-{len(groups)}")] * len(ex["y"]))
        info["pos"] += sum(1 for t in ex["y"] if t > 0.5)
        info["neg"] += sum(1 for t in ex["y"] if t <= 0.5)
        info["injected"] += int(ex.get("injected", 0))
        info["archived_jobs" if ex.get("_archived") else "jobs"] += 1
    if not X:
        base = (np.zeros((0, len(FEATURES))), np.zeros(0), np.zeros(0), info)
        return (*base, np.zeros(0, dtype=str)) if with_groups else base
    base = (np.array(X), np.array(y), np.array(w), info)
    return (*base, np.asarray(groups)) if with_groups else base


def _fit_logreg(X, y, w, lam=1.0, iters=50):
    """IRLS for weighted, L2-regularised logistic regression (bias not penalised)."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    R = lam * np.eye(d + 1)
    R[-1, -1] = 0.0
    for _ in range(iters):
        z = np.clip(A @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        s = np.maximum(w * p * (1 - p), 1e-9)
        g = A.T @ (w * (p - y)) + R @ beta
        H = (A * s[:, None]).T @ A + R
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def _auc(score, y):
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


MIN_POS_INSTRUMENT, MIN_NEG_INSTRUMENT = 8, 30


def instrument_path(data_dir, instrument: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(instrument))
    return Path(data_dir) / "models" / f"classifier-{safe}.json"


def train(data_dir, lam: float = 1.0) -> dict:
    """Global model on all examples + one model per instrument with enough examples.

    Nothing is lost by retraining: every training uses all stored examples
    (live jobs + archives), only the model files are replaced. A job is
    scored by the model of its instrument if there is one, else by the
    global model (features are normalised – rate vs. slowest searched rate,
    motion in FWHM, SNR – so it transfers reasonably, but artefacts of a
    lens and of a telescope differ)."""
    for jd in sorted((Path(data_dir) / "jobs").glob("*")):
        archive_job(data_dir, jd)          # keep the labels even if the job is deleted later
    sets = example_sets(data_dir)
    model = _train_one(data_dir, sets, None, lam)
    insts = sorted({ex.get("instrument") for ex in sets if ex.get("instrument")})
    per, lines = {}, []
    for inst in insts:
        X, y, _, _ = collect(data_dir, inst, sets)
        npos, nneg = int((y > 0.5).sum()), int((y <= 0.5).sum())
        if npos >= MIN_POS_INSTRUMENT and nneg >= MIN_NEG_INSTRUMENT:
            m = _train_one(data_dir, sets, inst, lam)
            per[inst] = {"n_pos": m["n_pos"], "n_neg": m["n_neg"], "cv_auc": m["cv_auc"]}
            auc = "–" if m["cv_auc"] is None else f"{m['cv_auc']:.2f}"
            lines.append(f"{inst}: vlastní model ({m['n_pos']} skutečných / {m['n_neg']} zamítnutých, AUC {auc})")
        else:
            instrument_path(data_dir, inst).unlink(missing_ok=True)
            lines.append(f"{inst}: zatím společný model (má {npos} skutečných / {nneg} zamítnutých, "
                         f"vlastní od {MIN_POS_INSTRUMENT} / {MIN_NEG_INSTRUMENT})")
    model["per_instrument"] = per
    if lines:
        model["report"] += " Přístroje – " + "; ".join(lines) + "."
    return model


def _train_one(data_dir, sets, instrument, lam):
    X, y, w, info, groups = collect(data_dir, instrument, sets, with_groups=True)
    X, y, w = np.asarray(X, float).reshape(-1, len(FEATURES)), np.asarray(y, float), np.asarray(w, float)
    npos, nneg = int((y > 0.5).sum()), int((y <= 0.5).sum())
    if npos < 3 or nneg < 10:
        raise ValueError(f"Málo označených příkladů (skutečných {npos}, zamítnutých {nneg}). "
                         f"Potřeba aspoň 3 skutečné a 10 zamítnutých. Nejrychleji: u některé série dejte "
                         f"„Spustit znovu“ s polem „Syntetické objekty (injekce)“ = 50 – nalezené umělé "
                         f"planetky se počítají jako skutečné. Pomohou i další série (označení se sčítají "
                         f"přes všechny úlohy, i smazané) a pole se známými planetkami.")
    # Grouped cross-validation: a complete observing series is held out.
    # Random candidate folds leaked field-specific star residuals into both
    # train and test and therefore overstated AUC.
    rng = np.random.default_rng(0)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    group_fold = {group: i % min(5, len(unique_groups)) for i, group in enumerate(unique_groups)}
    folds = np.asarray([group_fold[group] for group in groups])
    cv_score = np.full(len(y), np.nan)
    for f in range(min(5, len(unique_groups))):
        tr, te = folds != f, folds == f
        if (te.sum() == 0 or (y[tr] > 0.5).sum() < 2 or (y[tr] <= 0.5).sum() < 2):
            continue
        mu_f, sd_f = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd_f[sd_f < 1e-9] = 1.0
        ztr, zte = (X[tr] - mu_f) / sd_f, (X[te] - mu_f) / sd_f
        wb_f = _balanced_weights(y[tr], w[tr])
        b = _fit_logreg(ztr, y[tr], wb_f, lam)
        cv_score[te] = zte @ b[:-1] + b[-1]
    valid = np.isfinite(cv_score)
    auc = _auc(cv_score[valid], y[valid]) if valid.any() else None

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    wb = _balanced_weights(y, w)
    beta = _fit_logreg(Z, y, wb, lam)
    p = np.full(len(y), np.nan)
    p[valid] = 1 / (1 + np.exp(-np.clip(cv_score[valid], -30, 30)))
    # how many negatives would be hidden while keeping 95 % of the positives
    pos_cv, neg_cv = p[(y > 0.5) & valid], p[(y <= 0.5) & valid]
    thr95 = float(np.percentile(pos_cv, 5)) if len(pos_cv) else None
    hidden = float(np.mean(neg_cv < thr95)) if len(neg_cv) and thr95 is not None else None
    model = {"kind": "logreg", "version": 1, "instrument": instrument, "features": FEATURES,
             "mean": mu.tolist(), "std": sd.tolist(),
             "coef": beta[:-1].tolist(), "bias": float(beta[-1]), "lambda": lam,
             "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "n_pos": npos, "n_neg": nneg,
             "n_injected": info["injected"], "jobs": info["jobs"], "cv_auc": auc,
             "threshold_95": thr95, "hidden_at_95": hidden}
    path = model_path(data_dir) if instrument is None else instrument_path(data_dir, instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1), encoding="utf-8")
    top = sorted(zip(FEATURES, beta[:-1]), key=lambda t: -abs(t[1]))[:5]
    model["report"] = (f"Naučeno na {npos} skutečných ({info['injected']} z injekce) a {nneg} zamítnutých "
                       f"kandidátech z {info['jobs']} úloh"
                       + (f" a {info['archived_jobs']} smazaných (uložená označení)" if info['archived_jobs'] else "")
                       + ". Křížově ověřená AUC "
                       f"{'–' if auc is None else f'{auc:.3f}'} po celých pozorovacích sériích; "
                       + (f"při zachování 95 % skutečných by se skrylo {100 * hidden:.0f} % falešných "
                          f"(skóre < {thr95:.2f}). " if hidden is not None else
                          "pro odhad prahu je potřeba více nezávislých sérií. ")
                       + "Nejdůležitější: "
                       + ", ".join(f"{k} ({v:+.2f})" for k, v in top) + ".")
    if npos < 30:
        model["report"] += (" Pozor: skutečných příkladů je zatím málo – skóre berte jako orientační, "
                            "přidejte injekci nebo další označené úlohy.")
    return model


def _balanced_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Give both classes the same total weight without changing inputs."""
    out = w.copy()
    out[y > 0.5] *= w[y <= 0.5].sum() / max(w[y > 0.5].sum(), 1e-9)
    return out


def load(data_dir, instrument: str | None = None) -> dict | None:
    """Model of the instrument if it has one, else the global model."""
    paths = ([instrument_path(data_dir, instrument)] if instrument else []) + [model_path(data_dir)]
    for p in paths:
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("features") == FEATURES:      # other feature set → retrain
            return m
    return None


def score(model: dict, cands: list[dict], ctx: dict) -> None:
    """Add ``ml_score`` to candidate dicts in place."""
    if not model or not cands:
        return
    mu, sd = np.asarray(model["mean"]), np.asarray(model["std"])
    coef, b = np.asarray(model["coef"]), float(model["bias"])
    for c in cands:
        z = (features(c, ctx) - mu) / sd
        c["ml_score"] = round(float(1 / (1 + math.exp(-float(np.clip(z @ coef + b, -30, 30))))), 4)
