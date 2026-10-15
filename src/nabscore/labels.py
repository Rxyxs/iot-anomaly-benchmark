"""Fetch NAB's labelled windows, scoring profiles and published results from GitHub."""
from __future__ import annotations

from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/numenta/NAB/master/"
FILES = ["labels/combined_windows.json", "labels/combined_labels.json", "config/profiles.json",
         "config/thresholds.json", "results/final_results.json"]


def main(out: Path = Path("data/labels")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        r = requests.get(BASE + f, timeout=60)
        r.raise_for_status()
        (out / Path(f).name).write_bytes(r.content)
        print("fetched", f)


if __name__ == "__main__":
    main()
