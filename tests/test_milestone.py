import json

import numpy as np
import pandas as pd
import pytest

from conjunctions.archive import sha256
from conjunctions.metrics import budget_metrics
from conjunctions.milestone import export_milestone, review_weights


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({
        "event_id": [4, 2, 1, 3], "cutoff": [3] * 4, "fold": [0, 0, 1, 1],
        "mission_id": [1, 1, 2, 2], "evaluation_group": [4, 2, 1, 3],
        "label": [1, 0, 1, 0], "final_risk": [-5., -8., -5.5, -9.],
        "latest_risk": [-8., -8., -5., -9.],
        "B_random_forest": [.9, .9, .3, .1], "D_random_forest": [.7, .2, .8, .1],
    }).to_csv(run / "predictions.csv", index=False)
    manifest = {"completed_utc": "2026-10-07T22:08:27+00:00", "scheme": "event",
                "config": {"cutoffs": [3], "primary_cutoff": 3},
                "prediction_sha256": sha256(run / "predictions.csv")}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_tie_inclusion_matches_existing_budget_accounting():
    labels, scores = np.array([1, 0, 1, 0]), np.array([.9, .9, .3, .1])
    for fraction in (.01, .05, .5, 1.):
        weights = review_weights(scores, fraction)
        metric = budget_metrics(labels, scores, fraction)
        assert weights.sum() == metric["reviews"]
        assert np.dot(labels, weights) == metric["hits"]
    assert review_weights(scores, .05).tolist() == [.5, .5, 0, 0]


def test_export_accounts_for_tied_errors_and_preserves_source(saved_run, tmp_path):
    original = {p.name: p.read_bytes() for p in saved_run.iterdir()}
    output = export_milestone(saved_run, tmp_path / "milestone")
    table = pd.read_csv(output / "primary_results.csv").set_index("model")
    assert table.loc["B_random_forest", "top5_hits"] == .5
    cases = pd.read_csv(output / "error_summary.csv")
    missed = cases[(cases.model == "B_random_forest") & (cases.category == "high_endpoint_missed")].iloc[0]
    assert missed.events == 2 and missed.expected_not_reviewed == 1.5
    events = pd.read_csv(output / "event_analysis.csv").set_index("event_id")
    assert events.loc[4, "history_rank_gain"] < 0
    assert events.loc[1, "history_rank_gain"] > 0
    assert {p.name: p.read_bytes() for p in saved_run.iterdir()} == original
    demo = (output / "ranking_demo.html").read_text()
    assert "__DATA__" not in demo and 'id="reveal" type="checkbox">' in demo
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert all(sha256(output / name) == digest for name, digest in manifest["output_sha256"].items())
    with pytest.raises(FileExistsError):
        export_milestone(saved_run, output)


def test_tampered_predictions_rejected_without_output(saved_run, tmp_path):
    with (saved_run / "predictions.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differ"):
        export_milestone(saved_run, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_export_cannot_write_inside_source_run(saved_run):
    with pytest.raises(ValueError, match="outside"):
        export_milestone(saved_run, saved_run / "export")
