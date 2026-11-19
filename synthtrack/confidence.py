"""Confidence of a candidate: high / medium / low / none, with reasons.

A readable summary of the vetting numbers (similar in spirit to the
"Confidence" column of Tycho Tracker, whose exact criteria are not public):

* none   – auto-rejected by vetting (a hard flag),
* low    – passed vetting, but weak or with a warning,
* medium – passed, SNR ≥ 7, consistent sub-stacks and frames,
* high   – passed, SNR ≥ 9, every third of the sequence clearly carries the
           object, frames consistent, astrometry fits in all sub-stacks, no
           warning flag – and the classifier (if trained) does not object.

``reasons`` lists the criteria that kept the candidate from the next level
(Czech, shown in the GUI tooltip). Computed from the stored candidate dict,
so it also works for jobs processed by older versions.
"""
from __future__ import annotations

import math

LEVELS = ["none", "low", "medium", "high"]
LABELS = {"none": "žádná", "low": "nízká", "medium": "střední", "high": "vysoká"}


def _assess_twilight(c: dict) -> tuple[str, list[str]]:
    """"Zdroj navíc" (catalogue comparison, few frames): no motion test, so never "high". Its σ is
    the significance of a peak in one combined image (a glow: at its smoothing scale), not a
    synthetic-tracking SNR – an 8σ glow was empty sky at the frame edge (319-1/2 #612)."""
    snr = float(c.get("snr") or 0.0)
    flags = c.get("flags") or []
    kind = c.get("twilight_kind") or c.get("kind")
    reasons = ["zdroj navíc: bez testu pohybu, jen porovnání s katalogem"]
    moving = any(f.startswith("pohyb ") and "σ)" in f and _sigma_of(f) >= 5 for f in flags)
    dss_ok = any(f == "na DSS nic není" for f in flags)
    if kind == "mlhavý":
        if snr >= 15 and dss_ok:
            return "medium", reasons + ["silná mlhovina, na DSS nic není"]
        return "low", reasons + [f"mlhavý {snr:.0f}σ (náhodné struktury oblohy dosahují až ~13σ)"
                                 + ("" if dss_ok else ", nekontrolováno na DSS")]
    if moving and dss_ok:
        return "medium", reasons + ["pohyb ≥ 5σ, na DSS nic není"]
    if any(f.startswith("vzplanutí? VSX") for f in flags):
        return "medium", reasons + ["známá eruptivní proměnná (VSX) výrazně nad klidovou jasností – vzplanutí?"]
    return "low", reasons + (["bez měřitelného pohybu"] if not moving else ["nekontrolováno na DSS"])


def _sigma_of(flag: str) -> float:
    import re
    m = re.search(r"\((\d+(?:[.,]\d+)?)σ\)", flag)
    return float(m.group(1).replace(",", ".")) if m else 0.0


def assess(c: dict) -> tuple[str, list[str]]:
    if c.get("auto_status") != "candidate":
        flags = [f for f in (c.get("flags") or [])]
        return "none", ["zamítnuto prověřením" + (f": {', '.join(flags)}" if flags else "")]
    if c.get("source") == "twilight":
        return _assess_twilight(c)
    snr = float(c.get("snr") or 0.0)
    seg = [float(v) for v in (c.get("seg_snr") or [])]
    k = max(len(seg), 1)
    exp = snr / math.sqrt(k)
    seg_min_frac = (min(seg) / exp) if seg and exp > 0 else 0.0
    chi2 = float(c.get("chi2") or 1.0)
    med = float(c.get("median_ratio") or 0.0)
    pos = c.get("positions") or []
    pos_ok = bool(pos) and all(p.get("ok") for p in pos)
    soft = [f for f in (c.get("flags") or [])]
    ml = c.get("ml_score")

    # bright objects: χ² carries the crude model of a slow object's self-subtraction; when the
    # star-free sub-stacks agree (flux scatter ≤ 15 %) it says nothing about reliability
    fl = [float(p["flux"]) for p in pos if p.get("ok") and p.get("flux")]
    cv = (math.sqrt(sum((f - sum(fl) / len(fl)) ** 2 for f in fl) / len(fl)) / (sum(fl) / len(fl))
          if len(fl) >= 3 and sum(fl) > 0 else None)
    chi_med, chi_high = (2.0, 1.5)
    if snr >= 30:
        chi_med, chi_high = (4.0, 3.0)
        if cv is not None and cv <= 0.15 and pos_ok:
            chi_med = chi_high = float("inf")
    soft = [f for f in soft if f not in ("χ² vysoké, podstacky souhlasí", "segmenty nesouhlasí, podstacky souhlasí")]

    med_fail, high_fail = [], []
    if snr < 7:
        med_fail.append(f"SNR {snr:.1f} < 7")
    if seg and seg_min_frac < 0.25:
        med_fail.append("objekt chybí v některé třetině sekvence")
    if chi2 > chi_med:
        med_fail.append(f"nesouhlasné snímky (χ² {chi2:.1f})")
    if ml is not None and ml < 0.05:
        med_fail.append(f"klasifikátor {100 * ml:.0f} %")
    if "spíš proměnná hvězda" in soft:
        med_fail.append("vysvětlitelné i proměnnou statickou hvězdou")
    if "užší než hvězdy?" in soft:
        med_fail.append("užší než hvězdy (možný horký pixel)")
    if snr < 9:
        high_fail.append(f"SNR {snr:.1f} < 9")
    if seg and seg_min_frac < 0.5:
        high_fail.append("slabší v některé třetině sekvence")
    if chi2 > chi_high:
        high_fail.append(f"χ² {chi2:.1f} > {chi_high:.1f}".replace(".", ","))
    if med < 0.6:
        high_fail.append(f"medián/průměr {med:.2f} < 0,6")
    if not pos_ok:
        high_fail.append("astrometrie nesedí ve všech podstackách")
    if soft:
        high_fail.append("varování: " + ", ".join(soft))
    if ml is not None and ml < 0.5:
        high_fail.append(f"klasifikátor {100 * ml:.0f} %")
    if med_fail:
        return "low", med_fail
    if high_fail:
        return "medium", high_fail
    return "high", []


def annotate(cands: list[dict]) -> list[dict]:
    """Add ``confidence`` and ``confidence_reasons`` to candidate dicts (in place)."""
    for c in cands:
        lvl, why = assess(c)
        c["confidence"] = lvl
        c["confidence_reasons"] = why
    return cands
