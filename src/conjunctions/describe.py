"""Read-only description of prepared event tables for the dataset and feature sections.

Everything here is descriptive: it reads `prepare` outputs, never fits a model and never
changes processed data. Label-based counts (prevalence by cutoff, object class, fold or
sequence length) describe the cohort; they are not used to select features or settings.
"""

import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .archive import sha256
from .features import HISTORY_COLUMNS, SNAPSHOT_COLUMNS, STATISTICS, feature_columns
from .models import transform_groups
from .schema import EXPECTED_COLUMNS, HISTORY_FIELDS, LATEST_FIELDS

EXCLUDED_SOURCE_FIELDS = {
    "event_id": "identifier",
    "mission_id": "identifier (grouping only)",
    "F10": "space-weather index excluded from predictors",
    "F3M": "space-weather index excluded from predictors",
    "SSN": "space-weather index excluded from predictors",
    "AP": "space-weather index excluded from predictors",
}
# sigma_*, covariance correlations such as c_crdot_t, and the covariance determinant. Anchored so
# that c_cr_area_over_mass (radiation-pressure coefficient) stays an orbit/OD attribute.
UNCERTAINTY = re.compile(r"^[tc]_(sigma_[rtn](dot)?|c[rtn](dot)?_[rtn](dot)?|position_covariance_det)$")
TRANSFORM_LABELS = {
    "categorical": "most-frequent imputation, one-hot",
    "plain": "median imputation + missing indicator, log10 risk kept",
    "logged": "signed log1p, median imputation + missing indicator",
}


def field_group(field):
    """Readable source category for a raw CDM field name (no prefix)."""
    if field in {"risk", "max_risk_estimate", "max_risk_scaling", "mahalanobis_distance"}:
        return "risk estimate"
    if field == "time_to_tca":
        return "timing"
    if field == "miss_distance" or field.startswith("relative_"):
        return "relative state"
    if field in {"geocentric_latitude", "azimuth", "elevation"}:
        return "encounter geometry"
    if field == "c_object_type":
        return "chaser object class"
    owner = {"t": "target", "c": "chaser"}.get(field[:1])
    if owner and field[1:2] == "_":
        kind = "uncertainty/covariance" if UNCERTAINTY.match(field) else "orbit and orbit determination"
        return f"{owner} {kind}"
    raise ValueError(f"Unclassified CDM field: {field}")


def feature_group(column):
    if column in ("visible_count", "visible_span_days"):
        return "sequence length/span"
    if column.startswith("history__"):
        return "history summary"
    if column.startswith("latest__"):
        return field_group(column.removeprefix("latest__"))
    raise ValueError(f"Unknown feature column: {column}")


def feature_inventory(events):
    """One row per D feature with its set, source group, preprocessing route and missingness."""
    columns = feature_columns("D")
    route = {c: name for name, cols in transform_groups(columns).items() for c in cols}
    sets = {c: "".join(s for s in "ABCD" if c in feature_columns(s)) for c in columns}
    rows = []
    for column in columns:
        values = events[column]
        rows.append({
            "feature": column,
            "information_sets": sets[column],
            "group": feature_group(column),
            "preprocessing": TRANSFORM_LABELS[route[column]],
            "missing_fraction": float(values.isna().mean()),
            "distinct_values": int(values.nunique(dropna=True)),
            "constant": bool(values.nunique(dropna=True) <= 1),
        })
    return pd.DataFrame(rows)


def read_processed(processed):
    processed = Path(processed)
    audit = json.loads((processed / "audit.json").read_text())
    config = audit["preparation_config"]
    if config["primary_cutoff"] not in config["cutoffs"]:
        raise ValueError("Primary cutoff is absent from the prepared cutoffs")
    # Hash before reading so provenance certifies the bytes that were analysed.
    hashes = {p.name: sha256(p) for p in sorted(processed.glob("*")) if p.suffix in (".csv", ".json")}
    tables = {}
    for cutoff in config["cutoffs"]:
        path = processed / f"matched_k{cutoff}.csv"
        tables[cutoff] = pd.read_csv(path, dtype={"latest__c_object_type": "string"})
    expected = set(feature_columns("D")) | {"event_id", "mission_id", "label", "final_risk", "evaluation_group", "fold"}
    reference = None
    for cutoff, events in tables.items():
        missing = expected - set(events.columns)
        if missing or events.empty:
            raise ValueError(f"matched_k{cutoff}.csv lacks prepared columns: {sorted(missing)[:5]}")
        if events.event_id.duplicated().any():
            raise ValueError(f"matched_k{cutoff}.csv repeats event IDs")
        if not (events.label == (events.final_risk >= -6).astype(int)).all():
            raise ValueError(f"matched_k{cutoff}.csv labels differ from the final-message endpoint")
        keys = events.sort_values("event_id")[["event_id", "label", "evaluation_group", "fold"]].reset_index(drop=True)
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise ValueError("Cutoff tables are not matched on events, labels, groups and folds")
    return processed, audit, config, tables, hashes


def population_table(audit, tables):
    native = {row["cutoff"]: row for row in audit.get("native_counts", [])}
    rows = []
    for cutoff, events in sorted(tables.items()):
        rows.append({
            "cutoff_days": cutoff,
            "native_events": native.get(cutoff, {}).get("events"),
            "native_positives": native.get(cutoff, {}).get("positives"),
            "matched_events": len(events),
            "matched_positives": int(events.label.sum()),
            "matched_prevalence": float(events.label.mean()),
            "related_groups": int(events.evaluation_group.nunique()),
            "missions": int(events.mission_id.nunique()),
            "median_visible_messages": float(events.visible_count.median()),
            "median_visible_span_days": float(events.visible_span_days.median()),
        })
    return pd.DataFrame(rows)


def by_category(events, column, name):
    values = events[column].astype("string").fillna("MISSING")
    table = (pd.DataFrame({name: values, "label": events.label}).groupby(name)
             .agg(events=("label", "size"), positives=("label", "sum")).reset_index())
    table["prevalence"] = table.positives / table.events
    table["share_of_events"] = table.events / len(events)
    return table.sort_values(["events", name], ascending=[False, True]).reset_index(drop=True)


def sequence_lengths(tables):
    rows = []
    for cutoff, events in sorted(tables.items()):
        for label, group in [("all", events), ("high endpoint", events[events.label == 1]),
                             ("low endpoint", events[events.label == 0])]:
            counts = group.visible_count
            rows.append({"cutoff_days": cutoff, "subset": label, "events": len(group),
                         "min": counts.min(), "q25": counts.quantile(.25), "median": counts.median(),
                         "q75": counts.quantile(.75), "max": counts.max(),
                         "single_message_share": float((counts == 1).mean()) if len(group) else np.nan})
    return pd.DataFrame(rows)


def group_sizes(events):
    sizes = events.groupby("evaluation_group").size().value_counts().sort_index()
    return pd.DataFrame({"events_in_group": sizes.index, "groups": sizes.values,
                         "events": sizes.index * sizes.values})


def attrition(clean, cutoffs):
    """Sequential event exclusions mirroring build_events, ending at the native count per cutoff."""
    per_event = clean.groupby("event_id").time_to_tca.agg(messages="size", first="max", final="min")
    rows = [{"step": "events after cleaning", "removed": 0, "remaining": len(per_event)}]
    remaining = per_event
    for step, keep in [("fewer than two messages", lambda d: d.messages >= 2),
                       ("final message not within the last day before TCA",
                        lambda d: (d.final >= 0) & (d.final < 1))]:
        kept = remaining[keep(remaining)]
        rows.append({"step": step, "removed": len(remaining) - len(kept), "remaining": len(kept)})
        remaining = kept
    for cutoff in sorted(cutoffs):
        visible = remaining[remaining["first"] >= cutoff]
        rows.append({"step": f"no message visible at {cutoff} days", "removed": len(remaining) - len(visible),
                     "remaining": len(visible), "cutoff_days": cutoff})
    return pd.DataFrame(rows)


def fold_table(events):
    table = (events.groupby("fold").agg(events=("label", "size"), positives=("label", "sum"),
                                        related_groups=("evaluation_group", "nunique"),
                                        missions=("mission_id", "nunique")).reset_index())
    table["prevalence"] = table.positives / table.events
    return table


def group_summary(inventory):
    """Counts by information set and source group, with median missingness."""
    rows = []
    for name in "ABCD":
        members = inventory[inventory.information_sets.str.contains(name)]
        for group, part in members.groupby("group", sort=True):
            rows.append({"information_set": name, "group": group, "features": len(part),
                         "median_missing_fraction": float(part.missing_fraction.median()),
                         "max_missing_fraction": float(part.missing_fraction.max())})
    return pd.DataFrame(rows)


LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def short_group(group):
    """Compact label so the feature-group table fits one template column."""
    return group.replace("orbit and orbit determination", "orbit/OD").replace("uncertainty/covariance", "uncertainty")


def latex_escape(text):
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in str(text))


def write_latex(population, primary_groups, objects, cutoff, output):
    """Plain single-column tables for the ACL-style course template."""
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\hline",
             r"Cutoff & Events & High endpoints & Prevalence \\", r"\hline"]
    for row in population.itertuples():
        native = "" if pd.isna(row.native_events) else f" ({int(row.native_events):,})"
        lines.append(f"{row.cutoff_days} days & {row.matched_events:,}{native} & {row.matched_positives} & "
                     f"{100 * row.matched_prevalence:.2f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Matched event population by cutoff (native eligible events in parentheses). "
              r"The same events, labels, related groups and folds are used at every cutoff.}",
              r"\label{tab:population}", r"\end{table}", ""]
    lines += [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\hline",
              r"Chaser class & Events & High & Prevalence \\", r"\hline"]
    for row in objects.itertuples():
        lines.append(f"{latex_escape(row.object_type)} & {row.events:,} & {row.positives} & "
                     f"{100 * row.prevalence:.2f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}",
              f"\\caption{{Chaser object class at the {cutoff}-day cutoff. Class is reported by the CDM and "
              r"does not establish whether a payload is active.}", r"\label{tab:object-types}",
              r"\end{table}", ""]
    lines += [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrr}", r"\hline",
              r"Set & Source group & Features & Missing \\", r"\hline"]
    previous = None
    for row in primary_groups[primary_groups.information_set != "D"].itertuples():
        if previous is not None and row.information_set != previous:
            lines.append(r"\hline")
        previous = row.information_set
        lines.append(f"{row.information_set} & {latex_escape(short_group(row.group))} & {row.features} & "
                     f"{100 * row.median_missing_fraction:.1f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}",
              f"\\caption{{Raw input columns per information set at the {cutoff}-day cutoff, before missing "
              r"indicators and one-hot encoding, with the median share of events missing each column. OD: orbit determination. A is contained in B; D is B plus C.}", r"\label{tab:feature-groups}",
              r"\end{table}"]
    (output / "dataset_tables.tex").write_text("\n".join(lines) + "\n")


def make_figure(events, cutoff, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    high, low = events[events.label == 1], events[events.label == 0]
    figure, axes = plt.subplots(2, 1, figsize=(4.8, 5.6))  # one column of the two-column template
    limit = 40
    clipped = events.visible_count.max() > limit
    bins = np.arange(.5, min(events.visible_count.max(), limit) + 1.5)
    for subset, name, color in [(low, "Low endpoint", "#4c72b0"), (high, "High endpoint", "#dd8452")]:
        axes[0].hist(subset.visible_count.clip(upper=bins[-1] - .5), bins=bins, density=True, alpha=.6,
                     color=color, label=f"{name} (n={len(subset):,})")
        risk = subset.latest__risk.clip(lower=-30)
        axes[1].hist(risk, bins=np.arange(-30, 0.5, 1), density=True, alpha=.6, color=color, label=name)
    axes[1].axvline(-6, color="black", linestyle="--", linewidth=1, label="Endpoint threshold (-6)")
    count_note = f" (counts above {limit} shown at {limit})" if clipped else ""
    risk_note = " (values below -30 shown at -30)" if (events.latest__risk < -30).any() else ""
    axes[0].set(xlabel=f"Visible messages at {cutoff} days{count_note}", ylabel="Share of events")
    axes[1].set(xlabel=f"Latest visible log10 risk at {cutoff} days{risk_note}", ylabel="Share of events")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        ax.xaxis.label.set_size(9)
    figure.tight_layout()
    figure.savefig(output / "dataset_overview.png", dpi=160)
    figure.savefig(output / "dataset_overview.svg")
    plt.close(figure)


def summary_markdown(audit, population, objects, inventory, lengths, cutoff, sizes, removals=None):
    primary = population[population.cutoff_days == cutoff].iloc[0]
    count = {k: int(primary[k]) for k in ("matched_events", "matched_positives", "related_groups", "missions")}
    single = lengths[(lengths.cutoff_days == cutoff) & (lengths.subset == "all")].iloc[0]
    always = inventory[inventory.missing_fraction == 0]
    mostly = inventory[inventory.missing_fraction > .5]
    constant = inventory[inventory.constant]
    lines = [
        "# Dataset and feature summary", "",
        f"Source SHA-256: `{audit.get('source_sha256', 'unknown')}`", "",
        "## Cleaning",
        f"- Raw messages: {audit['raw_rows']:,}",
        f"- Exact duplicate rows removed: {audit['exact_duplicate_rows']:,}",
        f"- Rows with conflicting timestamps: {audit['conflicting_timestamp_rows']:,}",
        f"- Rows with invalid time or risk: {audit['invalid_time_or_risk_rows']:,}",
        f"- Quarantined events: {len(audit['quarantined_events']):,}", "",
        "## Primary cohort",
        f"- {cutoff}-day cutoff: {count['matched_events']:,} events, {count['matched_positives']} high endpoints "
        f"({100 * primary.matched_prevalence:.2f}%), {count['related_groups']:,} related groups, "
        f"{count['missions']} missions.",
        f"- Median visible messages {primary.median_visible_messages:g}; median visible span "
        f"{primary.median_visible_span_days:.2f} days; {100 * single.single_message_share:.1f}% of events have "
        "a single visible message (history levels then equal the latest value, SD and change are 0, "
        "rate and last change are missing).",
        f"- Related groups: {int(sizes[sizes.events_in_group > 1].groups.sum())} contain more than one event "
        f"(largest has {int(sizes.events_in_group.max())} events); the rest are single events.",
        f"- Majority-class accuracy would be {100 * (1 - primary.matched_prevalence):.2f}%, which is why accuracy "
        "is not a headline metric.", "",
        "## Chaser classes", "",
        "| Class | Events | High | Prevalence |", "| --- | ---: | ---: | ---: |",
        *[f"| {r.object_type} | {r.events:,} | {r.positives} | {100 * r.prevalence:.2f}% |" for r in objects.itertuples()],
        "", "## Features",
        f"- {len(inventory)} D input columns ({len(feature_columns('B'))} snapshot + {len(feature_columns('C'))} history); "
        f"{len(always)} never missing, {len(mostly)} missing in over half of events, {len(constant)} constant "
        "in this cohort.",
        "- Constant columns are kept: the allowlist is fixed before evaluation, and imputers/encoders tolerate them.",
        "- Missing indicators and one-hot encoding add model columns beyond these counts; they are fitted per training fold.",
        *([] if removals is None else [
            "", "## Event eligibility (from the raw archive)", "",
            "Cutoff rows each start from the events left after the first two rules.", "",
            "| Step | Removed | Remaining |", "| --- | ---: | ---: |",
            *[f"| {r.step} | {r.removed:,} | {r.remaining:,} |" for r in removals.itertuples()]]),
        "", "Label-based counts here are cohort descriptions. No feature or setting was selected from them.", "",
    ]
    return "\n".join(lines)


def describe_dataset(processed, output, archive=None):
    processed, audit, config, tables, hashes = read_processed(processed)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output directory exists; choose a new directory")
    removals = None
    if archive is not None:
        from .archive import read_messages
        from .data import clean_messages
        clean, _ = clean_messages(read_messages(archive))
        removals = attrition(clean, config["cutoffs"])
        native = {row["cutoff"]: row["events"] for row in audit.get("native_counts", [])}
        final = removals.dropna(subset=["cutoff_days"])
        if any(native.get(int(r.cutoff_days)) != r.remaining for r in final.itertuples()):
            raise ValueError("Eligibility counts from the archive differ from the prepared audit")
    cutoff = config["primary_cutoff"]
    events = tables[cutoff]
    population = population_table(audit, tables)
    objects = by_category(events, "latest__c_object_type", "object_type")
    missions = by_category(events, "mission_id", "mission_id")
    lengths = sequence_lengths(tables)
    folds = fold_table(events)
    inventory = feature_inventory(events)
    groups = group_summary(inventory)
    sizes = group_sizes(events)
    excluded = pd.DataFrame([{"field": f, "reason": EXCLUDED_SOURCE_FIELDS.get(f, "latest-message predictor")}
                             for f in EXPECTED_COLUMNS if f not in LATEST_FIELDS])
    output.mkdir(parents=True)
    population.to_csv(output / "population.csv", index=False)
    objects.to_csv(output / "object_types.csv", index=False)
    missions.to_csv(output / "missions.csv", index=False)
    lengths.to_csv(output / "sequence_lengths.csv", index=False)
    folds.to_csv(output / "folds.csv", index=False)
    sizes.to_csv(output / "group_sizes.csv", index=False)
    if removals is not None:
        removals.to_csv(output / "attrition.csv", index=False)
    inventory.to_csv(output / "feature_inventory.csv", index=False)
    groups.to_csv(output / "feature_groups.csv", index=False)
    excluded.to_csv(output / "excluded_source_fields.csv", index=False)
    write_latex(population, groups, objects, cutoff, output)
    make_figure(events, cutoff, output)
    (output / "SUMMARY.md").write_text(summary_markdown(audit, population, objects, inventory, lengths, cutoff, sizes, removals))
    provenance = {
        "processed_directory": str(processed),
        "primary_cutoff": cutoff,
        "input_sha256": hashes,
        "history_fields": list(HISTORY_FIELDS),
        "history_statistics": list(STATISTICS),
        "snapshot_columns": len(SNAPSHOT_COLUMNS),
        "history_columns": len(HISTORY_COLUMNS),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return output
