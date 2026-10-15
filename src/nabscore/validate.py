"""Check the scorer against NAB's published results.

Downloads the published per-file anomaly scores of reference detectors from the
NAB repository and scores them with ``scoring.nab_score``. At NAB's stored thresholds
(``config/thresholds.json``) the numbers must match ``results/final_results.json``
exactly; an exhaustive threshold sweep can only match or beat them.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

from .scoring import PROFILES, load_windows, nab_score, nab_score_at

RAW = "https://raw.githubusercontent.com/numenta/NAB/master/results"
CACHE = Path("data/reference")


def fetch_detector(name: str, files: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for rel in files:
        cat, fname = rel.split("/")
        local = CACHE / name / cat / f"{name}_{fname}"
        if not local.exists():
            r = requests.get(f"{RAW}/{name}/{cat}/{name}_{fname}", timeout=60)
            r.raise_for_status()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(r.content)
        out[rel] = pd.read_csv(io.BytesIO(local.read_bytes()), usecols=["timestamp", "anomaly_score"])
    return out


def main() -> None:
    names = sys.argv[1:] or ["windowedGaussian", "random", "numenta"]
    windows = load_windows()
    published = json.loads(Path("data/labels/final_results.json").read_text())
    stored = json.loads(Path("data/labels/thresholds.json").read_text())
    report = {}
    for name in names:
        res = fetch_detector(name, sorted(windows))
        report[name] = {}
        for prof in PROFILES:
            at_stored = nab_score_at(res, windows, stored[name][prof]["threshold"], prof)
            swept = nab_score(res, windows, prof)["score"]
            report[name][prof] = {"published": round(published[name][prof], 3),
                                  "mine_at_stored_threshold": round(at_stored, 3),
                                  "mine_best_threshold": round(swept, 3)}
        print(name, report[name], flush=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/scorer_validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
