"""Nested endpoint-probability calibration; outer labels never fit calibrators."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .archive import sha256
from .experiment import META, fit_predict, load_cohorts
from .metrics import score_metrics


def logit(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrators(inner_scores, labels, outer_scores):
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration training requires both classes")
    platt = LogisticRegression(C=1, max_iter=1000, random_state=20261005)
    platt.fit(logit(inner_scores)[:, None], labels)
    predictions = {"raw": outer_scores, "platt": platt.predict_proba(logit(outer_scores)[:, None])[:, 1]}
    if min(int(np.sum(labels)), int(len(labels) - np.sum(labels))) >= 50:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(inner_scores, labels)
        predictions["isotonic"] = np.clip(isotonic.predict(outer_scores), 1e-6, 1 - 1e-6)
    return predictions


def calibrate_run(directory, output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["scheme"] != "event" or "completed_utc" not in manifest:
        raise ValueError("Calibration requires a completed grouped event run")
    if sha256(output / "predictions.csv") != manifest["prediction_sha256"]:
        raise ValueError("Saved predictions changed")
    config = manifest["config"]
    if "D" not in config["information_sets"]:
        raise ValueError("Calibration is defined for information set D")
    cutoff = config["primary_cutoff"]
    for name, expected in manifest["input_sha256"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError("Prepared training inputs changed since the experiment")
    events = load_cohorts(directory, config)[cutoff]
    original = pd.read_csv(output / "predictions.csv").query("cutoff == @cutoff").set_index("event_id")
    destination = output / "calibration"
    if destination.exists():
        raise FileExistsError("Calibration output already exists")
    destination.mkdir()
    predictions, audit, metrics, bins = [], [], [], []
    for outer in range(config["folds"]):
        train = events[events.fold != outer].reset_index(drop=True)
        test = events[events.fold == outer].reset_index(drop=True)
        part = test[META].copy()
        part["fold"] = outer
        part["cutoff"] = cutoff
        for family in config["families"]:
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=20261035 + outer)
            scores = np.full(len(train), np.nan)
            assignment = np.full(len(train), -1)
            for fold, (tr, te) in enumerate(inner.split(train, train.label, train.evaluation_group)):
                if set(train.evaluation_group.iloc[tr]) & set(train.evaluation_group.iloc[te]):
                    raise RuntimeError("Inner calibration groups overlap")
                scores[te] = fit_predict(train.iloc[tr], train.iloc[te], family, "D")
                assignment[te] = fold
            if not np.isfinite(scores).all() or set(train.evaluation_group) & set(test.evaluation_group):
                raise RuntimeError("Invalid calibration provenance")
            raw = fit_predict(train, test, family, "D")
            np.testing.assert_allclose(raw, original.loc[test.event_id, f"D_{family}"], rtol=1e-10, atol=1e-10)
            for method, p in fit_calibrators(scores, train.label.to_numpy(), raw).items():
                part[f"{family}__{method}"] = p
            train[["event_id", "evaluation_group", "label"]].assign(
                inner_fold=assignment, raw_inner_oof=scores
            ).to_csv(destination / f"development_{family}_fold{outer}.csv", index=False)
            audit.append({"family": family, "outer_fold": outer, "training_n": len(train),
                          "evaluation_n": len(test), "outer_group_overlap": 0, "raw_scores_reproduced": True})
        predictions.append(part)
        print(f"Nested calibration completed: outer fold={outer}", flush=True)
    combined = pd.concat(predictions)
    combined.to_csv(destination / "predictions.csv", index=False)
    for name in combined.columns:
        if "__" not in name or name.startswith("latest__"):
            continue
        family, method = name.split("__")
        valid = combined[name].notna()
        p, y = combined.loc[valid, name], combined.loc[valid, "label"]
        metrics.append({"family": family, "method": method, **score_metrics(y, p, budgets=config["budgets"])})
        groups = pd.DataFrame({"score": p, "label": y, "bin": pd.cut(p, [0, .01, .02, .05, .1, .2, .5, 1], include_lowest=True)})
        for label, group in groups.groupby("bin", observed=True):
            n, rate, z = len(group), group.label.mean(), 1.96
            denominator = 1 + z * z / n
            center = (rate + z * z / (2 * n)) / denominator
            half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
            bins.append({"family": family, "method": method, "bin": str(label), "n": n,
                         "positive": int(group.label.sum()), "mean_probability": group.score.mean(),
                         "observed_rate": rate, "wilson_lo": center - half, "wilson_hi": center + half})
    pd.DataFrame(metrics).to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(bins).to_csv(destination / "reliability.csv", index=False)
    (destination / "provenance.json").write_text(json.dumps(audit, indent=2))
