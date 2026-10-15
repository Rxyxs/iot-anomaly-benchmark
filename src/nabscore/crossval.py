"""Leave-one-category-out NAB score: tuning choices made without the scored data.

NAB picks the threshold on the same corpus it scores, and a detector's own
settings (here the onset window k) can be tuned the same way. For each of the
six categories, k and the threshold are chosen on the other five; the held-out
category is then scored at those fixed values. Raw scores are summed over the
six held-out categories and normalised as NAB does.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import detectors as D
from .run import load_corpus
from .scoring import PROFILES, load_windows, nab_score, raw_at

K_GRID = (25, 50, 100, 200, 400, 800)


def detector_scores(raw, name, k):
    if name == "gaussian":
        return {r: D.onset(D.squash(v["gaussian"]), k) for r, v in raw.items()}
    if name == "seasonal":
        return {r: D.onset(D.squash(v["seasonal"]), k) for r, v in raw.items()}
    return {r: D.onset(np.maximum(D.squash(v["gaussian"]), D.squash(v["seasonal"])), k) for r, v in raw.items()}


def frames(corpus, scores, files):
    return {r: pd.DataFrame({"timestamp": corpus[r]["timestamp"], "anomaly_score": scores[r]}) for r in files}


def main() -> None:
    windows = load_windows()
    corpus = load_corpus(windows)
    raw = pickle.loads(Path("data/cache/raw.pkl").read_bytes())
    cats = sorted({r.split("/")[0] for r in corpus})
    out = {}
    for name in ("gaussian", "seasonal", "gaussian_or_seasonal"):
        per_k = {k: detector_scores(raw, name, k) for k in K_GRID}
        in_sample = {k: nab_score(frames(corpus, per_k[k], corpus), windows)["score"] for k in K_GRID}
        total_raw, total_windows, picks = 0.0, 0, {}
        for held in cats:
            train = [r for r in corpus if not r.startswith(held + "/")]
            test = [r for r in corpus if r.startswith(held + "/")]
            fits = {k: nab_score(frames(corpus, per_k[k], train), {r: windows[r] for r in train}) for k in K_GRID}
            k_best = max(fits, key=lambda k: fits[k]["score"])
            raw_score, n_win = raw_at(frames(corpus, per_k[k_best], test), {r: windows[r] for r in test},
                                      fits[k_best]["threshold"])
            total_raw += raw_score
            total_windows += n_win
            picks[held] = k_best
        w = PROFILES["standard"]
        null, perfect = -w["fn"] * total_windows, w["tp"] * total_windows
        cv = 100 * (total_raw - null) / (perfect - null)
        out[name] = {"in_sample_by_k": in_sample, "best_in_sample": max(in_sample.values()),
                     "k_100_in_sample": in_sample[100], "cross_validated": cv, "k_picked": picks}
        print(name, {"best in-sample": round(max(in_sample.values()), 1), "k=100": round(in_sample[100], 1),
                     "cross-validated": round(cv, 1), "k picked": picks}, flush=True)
    Path("reports/crossval.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
