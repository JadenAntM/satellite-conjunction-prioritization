"""Clean raw messages and construct aligned pre-TCA event representations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .archive import read_messages, verify_archive
from .features import feature_columns, summarize_history
from .schema import LATEST_FIELDS


def clean_messages(messages):
    exact = int(messages.duplicated().sum())
    data = messages.drop_duplicates().copy()
    for field in data.columns:
        if field != "c_object_type":
            data[field] = pd.to_numeric(data[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
    identity = data[["event_id", "mission_id"]]
    if identity.isna().any().any() or ((identity % 1) != 0).any().any():
        raise ValueError("Event and mission IDs must be finite integers")
    data[["event_id", "mission_id"]] = identity.astype("int64")
    invalid = data[["time_to_tca", "risk"]].isna().any(axis=1)
    tied = data.duplicated(["event_id", "time_to_tca"], keep=False)
    missions = data.groupby("event_id").mission_id.nunique()
    bad = set(data.loc[invalid | tied, "event_id"]) | set(missions[missions != 1].index)
    audit = {
        "raw_rows": len(messages),
        "exact_duplicate_rows": exact,
        "conflicting_timestamp_rows": int(tied.sum()),
        "invalid_time_or_risk_rows": int(invalid.sum()),
        "quarantined_events": sorted(int(x) for x in bad),
    }
    clean = data.loc[~data.event_id.isin(bad)].sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False], kind="stable"
    )
    return clean, audit


def related_groups(messages):
    """Conservative shared-content grouping; no satellite identity is inferred."""
    parent = {int(x): int(x) for x in messages.event_id.unique()}

    def find(event):
        while parent[event] != event:
            parent[event] = parent[parent[event]]
            event = parent[event]
        return event

    signature = pd.DataFrame({
        "mission": messages.mission_id.astype("int64"),
        "t": messages.time_to_tca.round(10),
        "risk": messages.risk.round(10),
        "miss": messages.miss_distance.round(3),
    })
    keys = pd.util.hash_pandas_object(signature, index=False)
    lookup = pd.DataFrame({"event": messages.event_id, "key": keys})
    counts = lookup.groupby("key").event.nunique()
    for _, group in lookup[lookup.key.isin(counts[counts > 1].index)].groupby("key"):
        events = sorted(int(x) for x in group.event.unique())
        for event in events[1:]:
            parent[find(event)] = find(events[0])
    return {event: find(event) for event in parent}


def build_events(clean, cutoff):
    if not np.isfinite(cutoff) or cutoff < 1:
        raise ValueError("Cutoff must be at least one day before TCA")
    rows = []
    for event, group in clean.groupby("event_id", sort=True):
        group = group.sort_values("time_to_tca", ascending=False, kind="stable")
        final = group.iloc[-1]
        visible = group[group.time_to_tca >= cutoff]
        # Keep the original strict final-message contract, including excluding post-TCA endings.
        if len(group) < 2 or visible.empty or not 0 <= final.time_to_tca < 1:
            continue
        latest = visible.iloc[-1]
        row = {f"latest__{field}": latest[field] for field in LATEST_FIELDS}
        row.update(
            event_id=int(event), mission_id=int(latest.mission_id), cutoff_days=cutoff,
            final_risk=float(final.risk), final_time_to_tca=float(final.time_to_tca),
            label=int(final.risk >= -6),
        )
        row.update(summarize_history(visible))
        rows.append(row)
    metadata = ["event_id", "mission_id", "cutoff_days", "final_risk", "final_time_to_tca", "label"]
    return pd.DataFrame(rows, columns=feature_columns("B") + metadata + feature_columns("C"))


def assign_folds(events, folds=5, seed=20261005):
    if events.empty or min(events.label.sum(), (1 - events.label).sum()) < folds:
        raise ValueError("Insufficient class support for the requested folds")
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    assignment = np.full(len(events), -1)
    for fold, (train, test) in enumerate(splitter.split(events, events.label, events.evaluation_group)):
        if set(events.evaluation_group.iloc[train]) & set(events.evaluation_group.iloc[test]):
            raise RuntimeError("Related groups cross a training/evaluation boundary")
        if events.label.iloc[train].nunique() != 2 or events.label.iloc[test].nunique() != 2:
            raise ValueError("A grouped fold lacks both classes")
        assignment[test] = fold
    return assignment


def prepare(archive, directory, config):
    directory = Path(directory)
    if directory.exists() and any(p.name != ".gitkeep" for p in directory.iterdir()):
        raise FileExistsError("Processed directory is not empty; choose a new directory")
    raw = read_messages(archive)
    clean, audit = clean_messages(raw)
    # Raw source fields remain numeric after the official schema read.
    groups = related_groups(raw)
    native = {}
    for cutoff in config["cutoffs"]:
        events = build_events(clean, cutoff)
        events["evaluation_group"] = events.event_id.map(groups)
        native[cutoff] = events
    anchor = native[max(config["cutoffs"])].sort_values("event_id").reset_index(drop=True)
    folds = assign_folds(anchor, config["folds"], config["split_seed"])
    matched = {}
    for cutoff, events in native.items():
        events = events[events.event_id.isin(anchor.event_id)].sort_values("event_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(events[["event_id", "label", "evaluation_group"]],
                                      anchor[["event_id", "label", "evaluation_group"]])
        matched[cutoff] = events.assign(fold=folds)
    directory.mkdir(parents=True, exist_ok=True)
    for cutoff, events in matched.items():
        events.to_csv(directory / f"matched_k{cutoff}.csv", index=False)
    anchor[["event_id", "label", "evaluation_group"]].assign(fold=folds).to_csv(directory / "splits.csv", index=False)
    audit.update(
        source_sha256=verify_archive(archive),
        matched_events=len(anchor), matched_positives=int(anchor.label.sum()),
        matched_groups=int(anchor.evaluation_group.nunique()),
        missions=int(anchor.mission_id.nunique()),
        native_counts=[{"cutoff": k, "events": len(d), "positives": int(d.label.sum())} for k, d in native.items()],
        preparation_config=config,
    )
    (directory / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit
