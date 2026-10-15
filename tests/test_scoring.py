import numpy as np
import pandas as pd
import pytest

from nabscore.detectors import onset
from nabscore.scoring import nab_score, raw_at, scaled_sigmoid

TS = pd.date_range("2020-01-01", periods=1000, freq="5min")
WIN = {"cat/a.csv": [(str(TS[400]), str(TS[449])), (str(TS[800]), str(TS[849]))]}


def run(scores):
    return nab_score({"cat/a.csv": pd.DataFrame({"timestamp": TS, "anomaly_score": scores})}, WIN)


def test_sigmoid_matches_nab_reference_points():
    assert scaled_sigmoid(-1.0) == pytest.approx(0.98661, abs=1e-5)
    assert scaled_sigmoid(-0.5) == pytest.approx(0.84828, abs=1e-5)
    assert scaled_sigmoid(0.0) == pytest.approx(0.0)
    assert scaled_sigmoid(4.0) == -1.0


def test_null_and_perfect_detectors():
    assert run(np.zeros(1000))["score"] == pytest.approx(0.0)
    perfect = np.zeros(1000)
    perfect[[400, 800]] = 1.0
    assert run(perfect)["score"] == pytest.approx(100.0)


def test_early_detection_beats_late_and_false_alarms_cost():
    early, late, noisy = np.zeros(1000), np.zeros(1000), np.zeros(1000)
    early[[405, 805]] = 1.0
    late[[445, 845]] = 1.0
    noisy[[405, 805]] = 1.0
    noisy[600:620] = 1.0                       # twenty false alarms between the windows
    assert run(early)["score"] > run(late)["score"] > 0
    assert run(noisy)["score"] < run(early)["score"]


def test_probation_rows_are_ignored():
    s = np.zeros(1000)
    s[10:100] = 1.0                            # all inside the first 15%
    assert run(s)["score"] == pytest.approx(0.0)


def test_raw_at_agrees_with_sweep_at_its_threshold():
    rng = np.random.default_rng(0)
    s = rng.random(1000)
    res = {"cat/a.csv": pd.DataFrame({"timestamp": TS, "anomaly_score": s})}
    best = nab_score(res, WIN)
    raw, n = raw_at(res, WIN, best["threshold"])
    assert raw == pytest.approx(best["raw"]) and n == 2


def test_onset_only_looks_back():
    s = np.array([0.1, 0.5, 0.4, 0.6, 0.6, 0.2, 0.9])
    out = onset(s, k=3)
    assert out.tolist() == [0.1, 0.5, 0.0, 0.6, 0.0, 0.0, 0.9]
    # changing the future never changes the past
    s2 = s.copy()
    s2[-1] = 0.0
    assert onset(s2, k=3)[:-1].tolist() == out[:-1].tolist()
