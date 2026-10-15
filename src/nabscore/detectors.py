"""Online anomaly detectors. Every score at time t uses only data up to t.

Each detector returns a raw score per row. NAB uses one threshold for all 58
series, so raw scores must be comparable across series with very different
scales and noise; ``likelihood`` turns any raw score into how unusual it is
relative to the same detector's recent history (the idea behind Numenta's
"anomaly likelihood"), which is the step that makes a single threshold work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-9


def _past(s: pd.Series, window: int, fn: str) -> pd.Series:
    r = s.shift(1).rolling(window, min_periods=max(10, window // 10))
    return getattr(r, fn)()


def gaussian(x: pd.Series, window: int = 6400) -> np.ndarray:
    """|z| against the mean and std of the past window (NAB's windowedGaussian idea)."""
    mu, sd = _past(x, window, "mean"), _past(x, window, "std")
    return ((x - mu).abs() / (sd + EPS)).fillna(0).to_numpy()


def robust_diff(x: pd.Series, window: int = 1000) -> np.ndarray:
    """Size of the latest step relative to the typical step size (median / MAD)."""
    d = x.diff().fillna(0)
    med = _past(d, window, "median")
    mad = (d - med).abs().shift(0)
    mad = _past(mad, window, "median")
    return ((d - med).abs() / (1.4826 * mad + EPS)).fillna(0).to_numpy()


def period_of(x: pd.Series, ts: pd.Series, probation: int) -> int | None:
    """Daily or weekly period in rows, if the probation data shows it."""
    step = pd.to_datetime(ts).diff().median()
    best, best_r = None, 0.3
    head = x.iloc[:max(probation, 1)].to_numpy()
    for days in (1, 7):
        p = int(pd.Timedelta(days=days) / step) if step and step > pd.Timedelta(0) else 0
        if 2 <= p < len(head) // 2:
            a, b = head[:-p], head[p:]
            if a.std() > 0 and b.std() > 0:
                r = np.corrcoef(a, b)[0, 1]
                if r > best_r:
                    best, best_r = p, r
    return best


def seasonal(x: pd.Series, ts: pd.Series, probation: int, k: int = 4, window: int = 2000) -> np.ndarray:
    """Residual from the median of the same phase in the previous k periods, scaled by MAD."""
    p = period_of(x, ts, probation)
    if p is None:
        return gaussian(x, window)
    lags = pd.concat([x.shift(p * i) for i in range(1, k + 1)], axis=1)
    resid = x - lags.median(axis=1)
    med = _past(resid, window, "median")
    mad = _past((resid - med).abs(), window, "median")
    return ((resid - med).abs() / (1.4826 * mad + EPS)).fillna(0).to_numpy()


def ks_shift(x: pd.Series, recent: int = 30, reference: int = 300, stride: int = 1) -> np.ndarray:
    """Kolmogorov-Smirnov distance between the last `recent` points and the `reference` before them."""
    v = x.to_numpy(dtype=float)
    out = np.zeros(len(v))
    for t in range(recent + reference, len(v), stride):
        out[t] = ks_2samp(v[t - recent + 1:t + 1], v[t - recent - reference + 1:t - recent + 1]).statistic
    return out


def likelihood(raw: np.ndarray, long: int = 2000, short: int = 10) -> np.ndarray:
    """How far the recent average raw score sits above its own longer history, mapped to [0, 1).

    The z-score is squashed with a slow logistic rather than the normal CDF: with
    heavy-tailed scores the CDF returns exactly 1.0 for many rows, and those ties
    leave a single corpus-wide threshold nothing to separate.
    """
    s = pd.Series(raw, dtype=float)
    short_mean = s.rolling(short, min_periods=1).mean()
    mu = s.shift(short).rolling(long, min_periods=50).mean()
    sd = s.shift(short).rolling(long, min_periods=50).std()
    z = ((short_mean - mu) / (sd + EPS)).to_numpy()
    return np.nan_to_num(1 / (1 + np.exp(-z / 4)), nan=0.5)


def onset(score: np.ndarray, k: int = 100) -> np.ndarray:
    """Keep a score only when it beats every score of the previous k rows.

    NAB charges every flagged row outside a labelled window as a false positive,
    so a detector that stays high for fifty rows after one level shift is
    charged fifty times. Reporting only the start of an excursion fixes that
    without looking ahead.
    """
    s = pd.Series(score, dtype=float)
    prev_max = s.shift(1).rolling(k, min_periods=1).max().fillna(-np.inf)
    return np.where(s > prev_max, s, 0.0)


def squash(raw: np.ndarray) -> np.ndarray:
    """Map a non-negative raw score to [0, 1) without per-series calibration."""
    return 1 - np.exp(-np.asarray(raw, dtype=float) / 5.0)
