import json

import pandas as pd
import pytest

from conjunctions.data import build_events, clean_messages
from conjunctions.describe import describe_dataset, feature_group, feature_inventory, field_group
from conjunctions.features import feature_columns
from conjunctions.models import make_model, transform_groups
from conjunctions.schema import LATEST_FIELDS


@pytest.fixture
def processed(messages, tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    clean, audit = clean_messages(messages)
    native = []
    for cutoff in (2, 3, 4):
        events = build_events(clean, cutoff)
        events["evaluation_group"] = events.event_id
        events["fold"] = [0, 1, 0, 1]
        events.to_csv(directory / f"matched_k{cutoff}.csv", index=False)
        native.append({"cutoff": cutoff, "events": len(events), "positives": int(events.label.sum())})
    audit.update(source_sha256="test", native_counts=native,
                 preparation_config={"cutoffs": [2, 3, 4], "primary_cutoff": 3, "folds": 2})
    (directory / "audit.json").write_text(json.dumps(audit))
    return directory


def test_every_snapshot_field_has_a_source_group():
    groups = {field: field_group(field) for field in LATEST_FIELDS}
    assert groups["risk"] == "risk estimate" and groups["c_object_type"] == "chaser object class"
    assert groups["t_sigma_r"] == "target uncertainty/covariance"
    assert groups["c_crdot_t"] == "chaser uncertainty/covariance"
    assert groups["c_position_covariance_det"] == "chaser uncertainty/covariance"
    assert groups["t_j2k_sma"] == "target orbit and orbit determination"
    assert groups["c_span"] == "chaser orbit and orbit determination"
    # Radiation-pressure and drag coefficients are orbit/OD attributes, not covariance terms.
    assert groups["t_cr_area_over_mass"] == groups["t_cd_area_over_mass"] == "target orbit and orbit determination"
    assert groups["c_ctdot_rdot"] == "chaser uncertainty/covariance"
    counts = pd.Series(groups).value_counts()
    assert counts["target uncertainty/covariance"] == counts["chaser uncertainty/covariance"] == 22
    assert counts["target orbit and orbit determination"] == 18
    assert feature_group("visible_count") == "sequence length/span"
    with pytest.raises(ValueError):
        field_group("label")


def test_inventory_matches_model_preprocessing(processed):
    events = pd.read_csv(processed / "matched_k3.csv")
    inventory = feature_inventory(events)
    assert inventory.feature.tolist() == feature_columns("D")
    assert inventory.information_sets.str.contains("D").all()
    assert inventory.loc[inventory.feature == "latest__risk", "information_sets"].item() == "ABD"
    routes = transform_groups(feature_columns("D"))
    # The model's own ColumnTransformer must route columns exactly as the inventory reports.
    blocks = {name: list(cols) for name, _, cols in make_model("hgb", "D").named_steps["features"].transformers}
    assert blocks == {k: v for k, v in routes.items() if v}
    assert inventory.missing_fraction.between(0, 1).all()


def test_describe_exports_tables_without_touching_inputs(processed, tmp_path):
    before = {p.name: p.read_bytes() for p in processed.iterdir()}
    output = describe_dataset(processed, tmp_path / "summary")
    assert {p.name: p.read_bytes() for p in processed.iterdir()} == before
    population = pd.read_csv(output / "population.csv")
    assert population.cutoff_days.tolist() == [2, 3, 4]
    assert population.matched_events.tolist() == [4] * 3 and population.matched_positives.tolist() == [2] * 3
    objects = pd.read_csv(output / "object_types.csv")
    assert objects.object_type.tolist() == ["UNKNOWN"] and objects.prevalence.item() == .5
    lengths = pd.read_csv(output / "sequence_lengths.csv")
    primary = lengths[(lengths.cutoff_days == 3) & (lengths.subset == "all")].iloc[0]
    assert primary["median"] == 1 and primary.single_message_share == 1
    assert len(pd.read_csv(output / "feature_inventory.csv")) == 147
    excluded = set(pd.read_csv(output / "excluded_source_fields.csv").field)
    assert excluded == {"event_id", "mission_id", "F10", "F3M", "SSN", "AP"}
    tex = (output / "dataset_tables.tex").read_text()
    assert r"\label{tab:population}" in tex and "50.00\\%" in tex and "uncertainty/covariance" in tex
    assert (output / "dataset_overview.png").stat().st_size > 0
    assert "4 events, 2 high endpoints" in (output / "SUMMARY.md").read_text()
    with pytest.raises(FileExistsError):
        describe_dataset(processed, output)


def test_describe_rejects_unmatched_or_relabelled_tables(processed, tmp_path):
    events = pd.read_csv(processed / "matched_k4.csv")
    events.loc[0, "fold"] = 1 - events.loc[0, "fold"]
    events.to_csv(processed / "matched_k4.csv", index=False)
    with pytest.raises(ValueError, match="not matched"):
        describe_dataset(processed, tmp_path / "a")
    events.loc[0, "fold"] = 1 - events.loc[0, "fold"]
    events.loc[0, "label"] = 1 - events.loc[0, "label"]
    events.to_csv(processed / "matched_k4.csv", index=False)
    with pytest.raises(ValueError, match="endpoint"):
        describe_dataset(processed, tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_missing_values_and_absent_classes_are_reported(processed, tmp_path):
    for cutoff in (2, 3, 4):
        path = processed / f"matched_k{cutoff}.csv"
        events = pd.read_csv(path)
        events.loc[0, "latest__c_object_type"] = None
        events.loc[[0, 1], "latest__t_sigma_r"] = None
        events.to_csv(path, index=False)
    output = describe_dataset(processed, tmp_path / "summary")
    objects = pd.read_csv(output / "object_types.csv", keep_default_na=False)
    assert set(objects.object_type) == {"UNKNOWN", "MISSING"} and objects.events.sum() == 4
    inventory = pd.read_csv(output / "feature_inventory.csv").set_index("feature")
    assert inventory.loc["latest__t_sigma_r", "missing_fraction"] == .5
    assert inventory.loc["latest__c_object_type", "missing_fraction"] == .25
    groups = pd.read_csv(output / "feature_groups.csv")
    target = groups[(groups.information_set == "B") & (groups.group == "target uncertainty/covariance")]
    assert target.max_missing_fraction.item() == .5


def test_sequence_lengths_tolerate_an_absent_endpoint_class():
    from conjunctions.describe import sequence_lengths
    events = pd.DataFrame({"visible_count": [1, 3], "label": [0, 0]})
    table = sequence_lengths({3: events}).set_index("subset")
    assert table.loc["high endpoint", "events"] == 0 and pd.isna(table.loc["high endpoint", "median"])
    assert table.loc["all", "median"] == 2
