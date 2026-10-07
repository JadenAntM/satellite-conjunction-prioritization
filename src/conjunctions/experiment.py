"""Matched-horizon grouped experiments with reproducible, non-overwriting outputs."""

import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from .archive import sha256
from .features import feature_columns
from .metrics import budget_metrics, paired_bootstrap, score_metrics, workload_at_recall
from .models import make_model

META = ["event_id", "evaluation_group", "mission_id", "label", "final_risk", "latest__risk", "latest__c_object_type"]


def load_cohorts(directory, config):
    cohorts = {k: pd.read_csv(Path(directory) / f"matched_k{k}.csv") for k in config["cutoffs"]}
    anchor = cohorts[max(cohorts)]
    for k, data in cohorts.items():
        if data.event_id.duplicated().any() or not (data.latest__time_to_tca >= k).all():
            raise ValueError("Invalid event IDs or cutoff timing")
        if not data.final_time_to_tca.between(0, 1, inclusive="left").all():
            raise ValueError("Invalid final-message endpoint timing")
        if not (data.label == (data.final_risk >= -6).astype(int)).all():
            raise ValueError("Labels differ from the defined final-message threshold")
        pd.testing.assert_frame_equal(data[["event_id", "evaluation_group", "label", "fold"]],
                                      anchor[["event_id", "evaluation_group", "label", "fold"]])
        if set(data.fold.unique()) != set(range(config["folds"])):
            raise ValueError("Prepared fold IDs do not match the experiment")
        if data.groupby("evaluation_group").fold.nunique().max() != 1:
            raise ValueError("Related groups cross outer folds")
        if data.groupby("evaluation_group").mission_id.nunique().max() != 1:
            raise ValueError("Related group crosses missions; mission exclusion needs a stricter contract")
        if not set(feature_columns("D")).issubset(data.columns):
            raise ValueError("Prepared representation is incomplete")
    audit = json.loads((Path(directory) / "audit.json").read_text())
    for key in ("cutoffs", "folds", "split_seed"):
        if audit["preparation_config"][key] != config[key]:
            raise ValueError(f"Prepared {key} differs from the requested config; prepare a new directory")
    return cohorts


def fit_predict(train, test, family, information):
    if train.label.nunique() != 2:
        raise ValueError("Training fold lacks both classes")
    estimator = make_model(family, information)
    with threadpool_limits(limits=2):
        estimator.fit(train, train.label)
        scores = estimator.predict_proba(test)[:, 1]
    return scores


def run_experiment(directory, results, name, config, scheme="event"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run name must use only letters, digits, underscores and hyphens")
    output = Path(results) / name
    if output.exists():
        raise FileExistsError("Run already exists; choose a new name")
    if scheme not in ("event", "mission"):
        raise ValueError("Unknown evaluation scheme")
    cohorts = load_cohorts(directory, config)
    files = [Path(directory) / f"matched_k{k}.csv" for k in config["cutoffs"]]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "scheme": scheme, "config": config,
        "input_sha256": {p.name: sha256(p) for p in files},
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "evidence_status": "Exploratory internal validation on a previously explored public benchmark; not an independent fresh test.",
    }
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    predictions, fold_metrics = [], []
    started = time.monotonic()
    fits = 0
    for cutoff, events in cohorts.items():
        if scheme == "event":
            splits = [(i, np.flatnonzero(events.fold != i), np.flatnonzero(events.fold == i)) for i in range(config["folds"])]
            specs = [(s, f) for s in config["information_sets"] for f in config["families"]]
        else:
            splits = [(i, tr, te) for i, (tr, te) in enumerate(LeaveOneGroupOut().split(events, groups=events.mission_id))]
            specs = [("A", "logistic"), ("B", "random_forest"), ("D", "random_forest")]
        for fold, train_index, test_index in splits:
            train, test = events.iloc[train_index], events.iloc[test_index]
            if set(train.evaluation_group) & set(test.evaluation_group):
                raise RuntimeError("Related groups cross the evaluation boundary")
            part = test[META].copy()
            part["cutoff"] = cutoff
            part["fold"] = fold
            part["latest_risk"] = test.latest__risk
            part["training_prevalence"] = train.label.mean()
            for information, family in specs:
                score_name = f"{information}_{family}"
                part[score_name] = fit_predict(train, test, family, information)
                fits += 1
            for score_name in ["latest_risk", "training_prevalence", *[f"{s}_{f}" for s, f in specs]]:
                fold_metrics.append({"cutoff": cutoff, "fold": fold, "model": score_name,
                                     **score_metrics(test.label, part[score_name], score_name != "latest_risk", config["budgets"])})
            predictions.append(part)
            print(f"Completed {scheme} CV: cutoff={cutoff}, fold={fold}", flush=True)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), model_fits=fits,
                    elapsed_seconds=time.monotonic() - started,
                    prediction_sha256=sha256(output / "predictions.csv"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def summarize_run(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if "completed_utc" not in manifest:
        raise ValueError("Only completed runs can be summarized")
    if sha256(output / "predictions.csv") != manifest["prediction_sha256"]:
        raise ValueError("Saved predictions differ from the completed run")
    data = pd.read_csv(output / "predictions.csv")
    metrics, budgets, curves, intervals = [], [], [], []
    config = manifest["config"]
    score_names = [c for c in data if c == "latest_risk" or c == "training_prevalence" or re.fullmatch(r"[ABCD]_(logistic|random_forest|hgb)", c)]
    for cutoff, events in data.groupby("cutoff"):
        if events.event_id.duplicated().any():
            raise ValueError("An event has multiple evaluation scores at one cutoff")
        for score in score_names:
            metrics.append({"cutoff": cutoff, "model": score,
                            **score_metrics(events.label, events[score], score != "latest_risk", config["budgets"])})
            for fraction in config["budgets"]:
                b = budget_metrics(events.label, events[score], fraction)
                baseline = budget_metrics(events.label, events.latest_risk, fraction)
                budgets.append({"cutoff": cutoff, "model": score, **b,
                                "incremental_hits": b["hits"] - baseline["hits"],
                                "recall_lift": b["recall"] / baseline["recall"] if baseline["recall"] else np.nan})
            for recall in (.8, .9, .95):
                curves.append({"cutoff": cutoff, "model": score, **workload_at_recall(events.label, events[score], recall)})
        pairs = [(f"D_{family}", f"B_{family}") for family in config["families"]]
        pairs += [(f"{s}_random_forest", "latest_risk") for s in ("B", "D")]
        for left, right in pairs:
            if left not in events or right not in events:
                continue
            for cluster in ("evaluation_group", "mission_id"):
                intervals.extend({"cutoff": cutoff, **row} for row in paired_bootstrap(
                    events, left, right, cluster, config["bootstrap_repeats"], config["bootstrap_seed"] + int(cutoff)
                ))
        print(f"Scored cutoff={cutoff}", flush=True)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(budgets).to_csv(output / "budgets.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "workload.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "paired_intervals.csv", index=False)
    from .plots import plot_run
    plot_run(output)
