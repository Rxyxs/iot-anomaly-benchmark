"""Run every detector on the 58 NAB series and score it with the NAB scorer."""
from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import detectors as D
from .scoring import PROFILES, load_windows, nab_score

DATA = Path("data/raw")


def load_corpus(windows: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for rel in sorted(windows):
        cat, name = rel.split("/")
        out[rel] = pd.read_csv(DATA / cat / cat / name)
    return out


def raw_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    x, ts = df["value"].astype(float), df["timestamp"]
    probation = int(min(math.floor(0.15 * len(df)), 750))
    return {"gaussian": D.gaussian(x), "robust_diff": D.robust_diff(x),
            "seasonal": D.seasonal(x, ts, probation), "ks_shift": D.ks_shift(x, stride=1)}


def main() -> None:
    windows = load_windows()
    corpus = load_corpus(windows)
    t0 = time.time()
    cache = Path("data/cache/raw.pkl")
    if cache.exists():
        raw = pickle.loads(cache.read_bytes())
    else:
        raw = {rel: raw_scores(df) for rel, df in corpus.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(raw))
    print(f"raw scores in {time.time() - t0:.0f}s", flush=True)

    dets = ["gaussian", "robust_diff", "seasonal", "ks_shift"]
    variants = {}
    for det in dets:
        sq = {r: (D.squash(raw[r][det]) if det != "ks_shift" else raw[r][det]) for r in corpus}
        lk = {r: D.likelihood(raw[r][det]) for r in corpus}
        variants[f"{det}"] = sq
        variants[f"{det} + onset"] = {r: D.onset(sq[r]) for r in corpus}
        variants[f"{det} + likelihood"] = lk
        variants[f"{det} + likelihood + onset"] = {r: D.onset(lk[r]) for r in corpus}
    variants["ensemble (mean likelihood) + onset"] = {
        r: D.onset(np.mean([D.likelihood(raw[r][d]) for d in dets], axis=0)) for r in corpus}

    results = {}
    for name, per_file in variants.items():
        res = {r: pd.DataFrame({"timestamp": corpus[r]["timestamp"], "anomaly_score": per_file[r]}) for r in corpus}
        results[name] = {p: nab_score(res, windows, p) for p in PROFILES}
        print(name, {p: round(v["score"], 2) for p, v in results[name].items()}, flush=True)

    # Per-category breakdown for the best ensemble
    best = max(results, key=lambda k: results[k]["standard"]["score"])
    cats = sorted({r.split("/")[0] for r in corpus})
    by_cat = {}
    for c in cats:
        files = [r for r in corpus if r.startswith(c + "/")]
        res = {r: pd.DataFrame({"timestamp": corpus[r]["timestamp"], "anomaly_score": variants[best][r]})
               for r in files}
        w = {r: windows[r] for r in files}
        if sum(len(v) for v in w.values()):
            by_cat[c] = nab_score(res, w, "standard")["score"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps({"detectors": results, "best": best, "by_category": by_cat},
                                                       indent=2, default=float))
    print("best:", best, {c: round(v, 1) for c, v in by_cat.items()})


if __name__ == "__main__":
    main()
