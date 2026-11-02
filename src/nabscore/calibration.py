"""Corpus threshold vs per-series calibration, in-sample and leave-one-category-out.

Same raw scores and onset reporting as crossval.py; the only change is whether the
scores are ranked against each series' own history (detectors.calibrate) before the
single NAB threshold is applied.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import detectors as D
from .crossval import K_GRID, frames
from .run import load_corpus
from .scoring import PROFILES, load_windows, nab_score, raw_at


def scores(raw, det, k, calibrated):
    out = {}
    for r, v in raw.items():
        s = D.squash(v[det])
        out[r] = D.onset(D.calibrate(s) if calibrated else s, k)
    return out


def cross_validated(corpus, windows, per_k):
    cats = sorted({r.split("/")[0] for r in corpus})
    total_raw, total_windows, picks = 0.0, 0, {}
    for held in cats:
        train = [r for r in corpus if not r.startswith(held + "/")]
        test = [r for r in corpus if r.startswith(held + "/")]
        fits = {k: nab_score(frames(corpus, per_k[k], train), {r: windows[r] for r in train}) for k in per_k}
        k_best = max(fits, key=lambda k: fits[k]["score"])
        raw_score, n_win = raw_at(frames(corpus, per_k[k_best], test), {r: windows[r] for r in test},
                                  fits[k_best]["threshold"])
        total_raw += raw_score
        total_windows += n_win
        picks[held] = k_best
    w = PROFILES["standard"]
    null, perfect = -w["fn"] * total_windows, w["tp"] * total_windows
    return 100 * (total_raw - null) / (perfect - null), picks


def by_category(corpus, windows, s):
    out = {}
    for c in sorted({r.split("/")[0] for r in corpus}):
        files = [r for r in corpus if r.startswith(c + "/")]
        if sum(len(windows[r]) for r in files):
            out[c] = nab_score(frames(corpus, s, files), {r: windows[r] for r in files})["score"]
    return out


def main() -> None:
    windows = load_windows()
    corpus = load_corpus(windows)
    raw = pickle.loads(Path("data/cache/raw.pkl").read_bytes())
    out = {}
    for det in ("gaussian", "seasonal"):
        for calibrated in (False, True):
            name = f"{det} + {'per-series calibration + ' if calibrated else ''}onset"
            per_k = {k: scores(raw, det, k, calibrated) for k in K_GRID}
            fits = {k: nab_score(frames(corpus, per_k[k], corpus), windows) for k in K_GRID}
            k_in = max(fits, key=lambda k: fits[k]["score"])
            cv, picks = cross_validated(corpus, windows, per_k)
            out[name] = {"in_sample_by_k": {k: f["score"] for k, f in fits.items()}, "best_in_sample": fits[k_in]["score"],
                         "k_in_sample": k_in, "threshold": fits[k_in]["threshold"], "cross_validated": cv,
                         "k_picked": picks, "by_category": by_category(corpus, windows, per_k[k_in])}
            print(name, round(fits[k_in]["score"], 1), f"k={k_in}", "CV", round(cv, 1), picks,
                  {c: round(v, 1) for c, v in out[name]["by_category"].items()}, flush=True)
    Path("reports/calibration.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
