"""Fixed latest-message and causal-history information partitions."""

import numpy as np
import pandas as pd

from .schema import HISTORY_FIELDS, LATEST_FIELDS

STATISTICS = ("mean", "std", "max", "change", "rate", "last_change")
HISTORY_COLUMNS = ("visible_count", "visible_span_days") + tuple(
    f"history__{field}__{stat}" for field in HISTORY_FIELDS for stat in STATISTICS
)
SNAPSHOT_COLUMNS = tuple(f"latest__{field}" for field in LATEST_FIELDS)


def feature_columns(information):
    """Allowlist features; never select unknown input columns by a prefix alone."""
    if information == "A":
        return ["latest__risk"]
    if information == "B":
        return list(SNAPSHOT_COLUMNS)
    if information == "C":
        return list(HISTORY_COLUMNS)
    if information == "D":
        return list(SNAPSHOT_COLUMNS + HISTORY_COLUMNS)
    raise ValueError(f"Unknown information set: {information}")


def summarize_history(visible):
    latest = visible.iloc[-1]
    span = float(visible.time_to_tca.iloc[0] - latest.time_to_tca)
    row = {"visible_count": len(visible), "visible_span_days": span}
    for field in HISTORY_FIELDS:
        values = pd.to_numeric(visible[field], errors="coerce").to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        valid = values[np.isfinite(values)]
        prefix = f"history__{field}__"
        row[prefix + "mean"] = float(valid.mean()) if len(valid) else np.nan
        row[prefix + "std"] = float(valid.std()) if len(valid) else np.nan
        row[prefix + "max"] = float(valid.max()) if len(valid) else np.nan
        row[prefix + "change"] = values[-1] - values[0]
        row[prefix + "rate"] = (values[-1] - values[0]) / span if span > 0 else np.nan
        row[prefix + "last_change"] = values[-1] - values[-2] if len(values) >= 2 else np.nan
    return row
