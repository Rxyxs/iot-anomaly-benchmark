"""The Numenta Anomaly Benchmark score, reimplemented from the NAB source.

Rules (nab/sweeper.py, nab/runner.py, numenta/NAB on GitHub):
- each labelled anomaly has a window around it; the first 15% of every file
  (at most 750 rows) is a probation period and is not scored;
- a point flagged inside a window is worth a scaled sigmoid of its position:
  near 1 at the left edge, 0 at the right edge, so early detection pays;
  only the best (earliest) detection in each window counts;
- a window with no detection costs ``fnWeight``;
- every flagged point outside a window costs ``fpWeight`` times a sigmoid of
  its distance past the previous window (a full ``fpWeight`` before any window);
- one threshold is chosen for the whole corpus, the one with the best total;
- the total is normalised: 100 * (score - null) / (perfect - null), where null
  flags nothing and perfect catches every window at its left edge.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROFILES = {
    "standard": {"tp": 1.0, "fp": 0.11, "fn": 1.0},
    "reward_low_FP_rate": {"tp": 1.0, "fp": 0.22, "fn": 1.0},
    "reward_low_FN_rate": {"tp": 1.0, "fp": 0.11, "fn": 2.0},
}
PROBATION = 0.15


def scaled_sigmoid(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    out = 2.0 / (1.0 + np.exp(5.0 * pos)) - 1.0
    return np.where(pos > 3.0, -1.0, out)


@dataclass
class FileRows:
    score: np.ndarray        # detector anomaly score per row
    sweep: np.ndarray        # weighted NAB value of flagging this row (per profile)
    window: np.ndarray       # global window id, -1 outside windows
    scored: np.ndarray       # False during probation


def file_rows(timestamps: pd.Series, scores: np.ndarray, windows: list[tuple[str, str]],
              weights: dict, window_offset: int) -> tuple[FileRows, int]:
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    n = len(ts)
    index = {t: i for i, t in enumerate(ts)}
    bounds = [(index[pd.Timestamp(a)], index[pd.Timestamp(b)]) for a, b in windows]
    max_tp = float(scaled_sigmoid(-1.0))
    sweep = np.empty(n)
    window = np.full(n, -1)
    prev_right, prev_width = None, None
    k = 0
    for i in range(n):
        if k < len(bounds) and bounds[k][0] <= i <= bounds[k][1]:
            left, right = bounds[k]
            width = right - left + 1
            sweep[i] = float(scaled_sigmoid(-(right - i + 1) / width)) * weights["tp"] / max_tp
            window[i] = window_offset + k
            if i == right:
                prev_right, prev_width = right, width
                k += 1
        else:
            if prev_right is None:
                sweep[i] = -1.0 * weights["fp"]
            else:
                sweep[i] = float(scaled_sigmoid(abs(prev_right - i) / (prev_width - 1))) * weights["fp"]
    probation = min(math.floor(PROBATION * n), PROBATION * 5000)
    scored = np.arange(n) >= probation
    return FileRows(np.asarray(scores, dtype=float), sweep, window, scored), window_offset + len(bounds)


def sweep_thresholds(rows: list[FileRows], fn_weight: float) -> tuple[float, float, int]:
    """Best (threshold, raw score, n_windows) over every distinct score in the corpus."""
    score = np.concatenate([r.score[r.scored] for r in rows])
    sweep = np.concatenate([r.sweep[r.scored] for r in rows])
    window = np.concatenate([r.window[r.scored] for r in rows])
    n_windows = int(window.max()) + 1 if (window >= 0).any() else 0
    order = np.argsort(-score, kind="stable")
    score, sweep, window = score[order], sweep[order], window[order]
    best_win = np.full(max(n_windows, 1), -fn_weight)
    fp_total = 0.0
    total = -fn_weight * n_windows
    best = (1.1, total)
    i, n = 0, len(score)
    while i < n:
        j = i
        while j < n and score[j] == score[i]:
            w = window[j]
            if w < 0:
                fp_total += sweep[j]
                total += sweep[j]
            elif sweep[j] > best_win[w]:
                total += sweep[j] - best_win[w]
                best_win[w] = sweep[j]
            j += 1
        if total > best[1]:
            best = (float(score[i]), total)
        i = j
    return best[0], best[1], n_windows


def nab_score(results: dict[str, pd.DataFrame], windows: dict[str, list], profile: str = "standard") -> dict:
    """results: relative data path -> DataFrame with timestamp and anomaly_score."""
    w = PROFILES[profile]
    rows, offset = [], 0
    for path in sorted(results):
        df = results[path]
        fr, offset = file_rows(df["timestamp"], df["anomaly_score"].to_numpy(), windows[path], w, offset)
        rows.append(fr)
    threshold, raw, n_windows = sweep_thresholds(rows, w["fn"])
    null = -w["fn"] * n_windows
    perfect = w["tp"] * n_windows
    return {"score": 100 * (raw - null) / (perfect - null), "threshold": threshold, "raw": raw,
            "windows": n_windows}


def load_windows(path: Path = Path("data/labels/combined_windows.json")) -> dict[str, list]:
    return json.loads(path.read_text())


def nab_score_at(results: dict[str, pd.DataFrame], windows: dict[str, list], threshold: float,
                 profile: str = "standard") -> float:
    """Normalised NAB score at a fixed threshold (how NAB reports its stored results)."""
    w = PROFILES[profile]
    rows, offset = [], 0
    for path in sorted(results):
        df = results[path]
        fr, offset = file_rows(df["timestamp"], df["anomaly_score"].to_numpy(), windows[path], w, offset)
        rows.append(fr)
    best = np.full(offset, -w["fn"])
    fp = 0.0
    for r in rows:
        hit = r.scored & (r.score >= threshold)
        fp += r.sweep[hit & (r.window < 0)].sum()
        inside = hit & (r.window >= 0)
        np.maximum.at(best, r.window[inside], r.sweep[inside])
    raw = fp + best.sum()
    null, perfect = -w["fn"] * offset, w["tp"] * offset
    return float(100 * (raw - null) / (perfect - null))


def raw_at(results: dict[str, pd.DataFrame], windows: dict[str, list], threshold: float,
           profile: str = "standard") -> tuple[float, int]:
    """Unnormalised NAB score and window count at a fixed threshold (for held-out subsets)."""
    w = PROFILES[profile]
    rows, offset = [], 0
    for path in sorted(results):
        df = results[path]
        fr, offset = file_rows(df["timestamp"], df["anomaly_score"].to_numpy(), windows[path], w, offset)
        rows.append(fr)
    best = np.full(offset, -w["fn"])
    fp = 0.0
    for r in rows:
        hit = r.scored & (r.score >= threshold)
        fp += r.sweep[hit & (r.window < 0)].sum()
        inside = hit & (r.window >= 0)
        np.maximum.at(best, r.window[inside], r.sweep[inside])
    return float(fp + best.sum()), offset
