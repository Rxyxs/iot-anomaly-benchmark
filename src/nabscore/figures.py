"""README figures: leaderboard, onset effect, and the machine-failure example."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import detectors as D  # noqa: E402
from .run import load_corpus  # noqa: E402
from .scoring import load_windows, nab_score  # noqa: E402

OUT = Path("docs/img")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    corpus = load_corpus(windows)
    raw = pickle.loads(Path("data/cache/raw.pkl").read_bytes())
    cv = json.loads(Path("reports/crossval.json").read_text())
    pub = json.loads(Path("data/labels/final_results.json").read_text())

    # Final detector: seasonal residual + onset, k = 400 (the value cross-validation picked every time)
    final = {r: D.onset(D.squash(raw[r]["seasonal"]), 400) for r in corpus}
    res = {r: pd.DataFrame({"timestamp": corpus[r]["timestamp"], "anomaly_score": final[r]}) for r in corpus}
    summary = {p: nab_score(res, windows, p) for p in ("standard", "reward_low_FP_rate", "reward_low_FN_rate")}
    cats = sorted({r.split("/")[0] for r in corpus if windows[r]})
    by_cat = {}
    for c in cats:
        files = [r for r in corpus if r.startswith(c + "/")]
        by_cat[c] = nab_score({r: res[r] for r in files}, {r: windows[r] for r in files})["score"]
    Path("reports/final.json").write_text(json.dumps({"profiles": summary, "by_category": by_cat}, indent=2,
                                                     default=float))
    print({p: round(v["score"], 1) for p, v in summary.items()}, {c: round(v, 1) for c, v in by_cat.items()})

    # 1. Leaderboard
    board = {k: v["standard"] for k, v in pub.items() if k != "null"}
    board["this repo: seasonal + onset\n(in-sample, like NAB)"] = summary["standard"]["score"]
    board["this repo: seasonal + onset\n(cross-validated)"] = cv["seasonal"]["cross_validated"]
    items = sorted(board.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E15759" if "this repo" in k else "#BAB0AC" for k, _ in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("NAB score, standard profile")
    ax.set_title("Published NAB detectors and this repo (red)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "leaderboard.png", dpi=130)
    plt.close(fig)

    # 2. Onset effect, in-sample at k = 100 vs without
    res_json = json.loads(Path("reports/results.json").read_text())["detectors"]
    dets = ["gaussian", "seasonal", "robust_diff", "ks_shift"]
    before = [res_json[d]["standard"]["score"] for d in dets]
    after = [res_json[f"{d} + onset"]["standard"]["score"] for d in dets]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(dets))
    ax.bar(x - 0.2, before, 0.4, label="score every row", color="#BAB0AC")
    ax.bar(x + 0.2, after, 0.4, label="report only the onset", color="#4E79A7")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - 0.2, b + 0.8, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, a + 0.8, f"{a:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Gaussian", "seasonal", "robust diff", "KS shift"])
    ax.set_ylabel("NAB score, standard")
    ax.set_title("Every flagged row is a false positive: reporting only onsets")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "onset.png", dpi=130)
    plt.close(fig)

    # 3. Machine temperature example
    rel = "realKnownCause/machine_temperature_system_failure.csv"
    df = corpus[rel].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp)
    thr = summary["standard"]["threshold"]
    flagged = df[final[rel] >= thr]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[2, 1])
    ax.plot(df.timestamp, df.value, lw=0.6, color="#4E79A7")
    for a, b in windows[rel]:
        for axis in (ax, ax2):
            axis.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#F28E2B", alpha=0.25, lw=0)
    ax.scatter(flagged.timestamp, flagged.value, color="#E15759", s=22, zorder=3)
    ax.set_ylabel("temperature")
    ax.set_title("Industrial machine: labelled anomaly windows (orange), detector score and corpus threshold")
    ax2.plot(df.timestamp, final[rel], lw=0.8, color="#59A14F")
    ax2.axhline(thr, color="#E15759", lw=1, ls="--", label=f"single NAB threshold {thr:.3f}")
    ax2.set_ylabel("score")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "machine.png", dpi=130)
    plt.close(fig)
    peak = {f"{a[:10]}": float(np.max(final[rel][(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))]))
            for a, b in windows[rel]}
    print("machine peak score per window:", {k: round(v, 3) for k, v in peak.items()}, "threshold", round(thr, 3))
    print("figures written; detections on machine series:", len(flagged))


if __name__ == "__main__":
    main()
