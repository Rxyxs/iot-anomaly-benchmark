**English** | [Español](README.es.md)

# Streaming anomaly detection on the Numenta Anomaly Benchmark

![CI](https://github.com/Rxyxs/iot-anomaly-benchmark/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

The Numenta Anomaly Benchmark (NAB) is 58 real and artificial time series (server CPU, machine
temperatures, traffic, ad clicks, tweet volumes) with labelled anomaly windows, and a scoring rule
built for monitoring: detections must come online, early detection is worth more, and every false
alarm costs. I reimplemented the scorer, validated it against NAB's published results, and built
simple online detectors to find out what actually moves the score.

## Results

| | NAB standard score |
|---|---|
| **Seasonal residual + onset reporting, cross-validated** | **42.4** |
| Same detector, settings and threshold chosen on the whole corpus (the NAB convention) | 49.5 |
| Published `windowedGaussian` baseline | 39.6 |
| Published Numenta HTM / best published (ARTime) | 70.1 / 74.9 |

![Leaderboard](docs/img/leaderboard.png)

### 1. A scorer you can trust before any detector

The NAB score is easy to get subtly wrong (scaled sigmoid inside windows, probation period, a
single corpus-wide threshold, normalisation against a null detector). Scoring NAB's own published
detector outputs with this implementation reproduces `final_results.json` **exactly, 9 of 9**
(windowedGaussian, numenta and random, three profiles each) at NAB's stored thresholds. An
exhaustive threshold sweep can do slightly better than NAB's stored thresholds (+0.1 to +0.8
points), so in-sample numbers here are comparable to the published ones to within a point.

### 2. What matters most is not the detector but what it reports

![Onset](docs/img/onset.png)

NAB charges every flagged row outside a window as a false positive. A detector that stays high for
fifty rows after one level shift is charged fifty times. Reporting a score only when it beats every
score of the previous k rows (the *onset* of an excursion), which needs no look-ahead, changes
everything: the seasonal detector goes from 2.4 to 41.0, the Gaussian from 36.4 to 42.6, the
robust-difference detector from 0 to 23.6.

### 3. Tuning on the benchmark inflates the score

The onset window k matters: the seasonal detector scores 20 at k = 25 and 49.5 at k = 400. Picking
k on the whole corpus is tuning on the test set, so I also score leave-one-category-out: k and the
threshold are chosen on the other categories, and the held-out one is scored at those fixed values.

| Detector | Best in-sample | Cross-validated | k chosen on held-out folds |
|---|---|---|---|
| Gaussian + onset | 44.6 | 30.4 | 100 or 800, depending on the fold |
| **Seasonal + onset** | 49.5 | **42.4** | 400 in every fold |
| Gaussian or seasonal + onset | 49.0 | 41.6 | 400 in all but one |

The seasonal detector keeps most of its score and picks the same setting every time; the Gaussian
detector loses 14 points and its best setting depends on which data it sees. Most published NAB
numbers are in-sample in this sense.

### 4. Where it fails: one threshold for everything

Scores by category (in-sample, standard profile): artificial 91.1, tweets 68.5, traffic 58.9,
AWS 55.8, ad exchange 46.7, **known-cause 27.1**.

![Machine example](docs/img/machine.png)

On the industrial machine-temperature series the detector reacts inside three of the four labelled
windows (peak scores 0.43-0.57), but similar scores occur in normal periods, and the single NAB
threshold of 0.753 sits above all of them, so nothing is flagged. For monitoring one asset, a
per-asset threshold or a baseline learned on that machine would matter more than a better detector.

A "likelihood" post-processing step (scoring how unusual a raw score is against its own history,
in the spirit of Numenta's anomaly likelihood) was tested and lowered every detector here; it is
kept in the code for completeness.

## Method

```
58 NAB series (Kaggle mirror) + labelled windows, profiles, published results (NAB on GitHub)
   │
   ├── scoring.py    NAB score: windows, scaled sigmoid, probation, corpus threshold sweep,
   │                 null/perfect normalisation; validated 9/9 against published results
   ├── detectors.py  online only: rolling Gaussian z, robust step size (median/MAD),
   │                 seasonal residual vs the same phase of past days/weeks, KS distribution shift,
   │                 likelihood post-processing, onset reporting
   ├── run.py        every detector x post-processing on the corpus
   └── crossval.py   leave-one-category-out choice of k and threshold
```

## Data

Numenta Anomaly Benchmark (Lavin & Ahmad, 2015; Ahmad et al., 2017). Series from the Kaggle mirror
[boltzmannbrain/nab](https://www.kaggle.com/datasets/boltzmannbrain/nab); labelled windows,
scoring profiles and published results from [numenta/NAB](https://github.com/numenta/NAB) (MIT).
Downloaded October 2026; not redistributed here.

## Reproduce

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
kaggle datasets download -d boltzmannbrain/nab -p data/raw --unzip
python -m nabscore.labels       # windows, profiles and published results from GitHub
python -m nabscore.validate     # scorer vs published results
python -m nabscore.run          # ~4 min the first time (raw scores are cached)
python -m nabscore.crossval
python -m nabscore.figures
pytest -q tests
```

## Limitations

- The detectors are deliberately simple and univariate; the point is what drives the score, not a
  new state of the art.
- Cross-validation over seven categories is coarse: categories differ a lot, and one of them has
  no anomalies at all.
- NAB windows are fixed-width and partly hand-labelled; a few labels are debated in the literature.

## License

Code: MIT. Data: NAB (Numenta), via Kaggle and GitHub, not redistributed.
