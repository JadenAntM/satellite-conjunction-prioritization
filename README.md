# Satellite Conjunction Prioritization

An applied machine learning project studying how to rank satellite conjunctions for review several days before closest approach.

**Research question:** At three days before closest approach, do fields in the latest conjunction data message (CDM) improve ranking of high-risk final-message endpoints over latest risk alone, and does visible message history add predictive value beyond that snapshot?

The target is a later **message estimate**, not an observed collision. Model probabilities refer to that binary endpoint. This repository contains the reproducible project code, fixed experiment settings, tests and essential data/method documentation. Generated data and experiment outputs stay local.

## Setup

Use Python **3.12 or newer**. Run commands from the repository root.

```bash
git clone https://github.com/JadenAntM/satellite-conjunction-prioritization.git
cd satellite-conjunction-prioritization
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. `requirements.txt` records the tested package versions. The package also exposes the `conjunctions` command; the examples use `python -m conjunctions` so the selected interpreter is explicit.

## Run the main experiment

```bash
python -m conjunctions download
python -m conjunctions prepare
python -m conjunctions run --name main
```

`download` retrieves the official ESA archive (about 221 MB) and verifies its SHA-256. If it is already present, it is verified and reused. An existing verified archive can also be supplied with `prepare --archive PATH`; no source directory needs to be part of this checkout.

`prepare` builds event-level representations at two, three and four days, retains the same four-day-eligible events at every horizon, and assigns identical grouped folds. For this release the common population is **8,694 events, including 160 positive endpoints (1.84%)**, grouped into 8,637 conservative related-event groups across 19 missions.

`run` trains the fixed Logistic Regression, Random Forest and Histogram Gradient Boosting comparisons, saves out-of-fold predictions, and calculates ranking, probability and review-budget metrics. It performs 180 fits for the default main experiment. Runtime varies by machine. Existing processed data or run directories are protected from overwriting; use a new directory/name when intentionally changing a configuration.

```bash
# Recompute summaries from saved predictions without refitting models
python -m conjunctions report --run results/runs/main

# Stronger domain partition: leave one mission out
python -m conjunctions run --name mission --scheme mission

# Optional: nested probability calibration of D at the primary horizon
python -m conjunctions calibrate --run results/runs/main
```

Mission evaluation uses risk-only Logistic Regression and snapshot/history-enhanced Random Forest comparisons. Calibration fits Platt/isotonic mappings using inner training folds; outer labels are only used for evaluation.

## What we compare

| Set | Information | Fields |
| --- | --- | --- |
| A | Latest visible log-risk only | 1 |
| B | Safe fields from the latest full CDM | 97 |
| C | Visible-message summaries, sequence length and time span | 50 |
| D | B + C | 147 |

**D minus B is the history-value comparison.** C includes historical levels as well as changes, so C minus A is not a pure trend-value test. Raw latest-risk ranking and training-fold prevalence are additional baselines. The design allows history to have no measurable extra benefit.

Primary evaluation uses Average Precision, paired uncertainty for D−B, and capture/precision under fixed review budgets. Accuracy is not the main metric for this rare label. Preprocessing is fitted within training folds; event and related-group boundaries are preserved. Three days is primary, while matched two/four-day comparisons show the horizon tradeoff. Details are in [the experiment protocol](docs/experiment.md).

## Repository layout

```text
configs/default.toml       Fixed horizons, information sets, seeds and budgets
src/conjunctions/          Data preparation, features, models, evaluation and CLI
tests/                    Timing, leakage, grouping, metrics and pipeline checks
docs/data.md              Source, attribution, label and field semantics
docs/experiment.md        Comparison design and interpretation limits
data/raw/                 Ignored downloaded archive
data/processed/           Ignored matched event tables and split/audit records
results/                  Ignored predictions, metrics, curves and figures
```

Each run stores input/source hashes, package versions, settings and completion metadata. Results include `metrics.csv`, `fold_metrics.csv`, `budgets.csv`, `workload.csv`, `paired_intervals.csv`, saved predictions and a horizon figure. Review fractions needed for 80/90/95% capture are retrospective curves, not deployment thresholds.

## Data and interpretation

Data: [ESA Collision Avoidance Challenge release](https://zenodo.org/records/4463683), distributed under **CC BY 4.0**. See [data documentation](docs/data.md) and [NOTICE.md](NOTICE.md) for attribution. The data are anonymized; activity status and usable absolute dates are unavailable. All object types are retained in the main experiment.

This is a previously explored public benchmark. These runs are exploratory internal validation, not a new independent final test. A different split seed cannot restore independence. Mission holdout provides limited domain evidence, not chronological validation. No automatic dismissal of alerts, physical collision prediction or manoeuvre/fuel-saving claim follows from the scores.

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md). Commit code, tests, configurations and concise documentation; keep downloaded data, fitted models and generated experiment artifacts out of Git.
