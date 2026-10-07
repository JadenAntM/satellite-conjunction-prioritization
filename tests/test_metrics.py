import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from conjunctions.metrics import WeightedRank, budget_metrics, paired_bootstrap, score_metrics, workload_at_recall


def test_boundary_ties_do_not_use_labels_to_order_events():
    y = np.array([1, 0, 0, 0])
    result = budget_metrics(y, np.ones(4), .5)
    assert result["reviews"] == 2 and result["hits"] == .5
    assert result["precision"] == .25 and result["recall"] == .5
    assert budget_metrics(y[::-1], np.ones(4), .5) == result


def test_budget_accounting():
    for fraction in (.01, .02, .05, .1):
        result = budget_metrics([1, 0, 1, 0], [.9, .8, .7, .1], fraction)
        assert result["hits"] + result["missed"] == 2
        assert result["hits"] + result["false_reviews"] == result["reviews"]


def test_retrospective_curve_keeps_whole_boundary_ties():
    assert workload_at_recall([1, 0, 1, 0], [.9, .9, .8, .1], .5)["reviews"] == 2
    assert workload_at_recall([1, 0, 1, 0], [.9, .9, .8, .1], .9)["reviews"] == 3


def test_classifier_probability_and_log_risk_have_different_metrics():
    raw = score_metrics([1, 0], [-5., -8.], probability=False)
    assert np.isnan(raw["brier"]) and np.isnan(raw["log_loss"])
    model = score_metrics([1, 0], [.3, .2])
    assert np.isfinite(model["brier"])


def test_no_positive_ranking_metrics_are_undefined():
    metrics = score_metrics([0, 0], [.1, .2])
    assert np.isnan(metrics["ap"]) and np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["top5_recall"])


@pytest.mark.parametrize("fraction", [0, -1, 1.1])
def test_invalid_budget_is_rejected(fraction):
    with pytest.raises(ValueError):
        budget_metrics([0, 1], [.2, .4], fraction)


def test_replication_weights_match_explicit_tied_bootstrap_rows():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([.8, .8, .4, .3, .1, .4])
    weights = np.array([2, 3, 0, 1, 2, 1])
    rows = np.repeat(np.arange(len(y)), weights)
    ap, recall = WeightedRank(y, scores).evaluate(weights)
    assert np.isclose(ap, average_precision_score(y[rows], scores[rows]))
    assert np.isclose(recall, budget_metrics(y[rows], scores[rows], .05)["recall"])


def test_paired_cluster_bootstrap_uses_identical_resamples():
    events = pd.DataFrame({"label": [1, 0, 1, 0], "evaluation_group": [1, 1, 2, 2],
                           "a": [.8, .5, .6, .1], "b": [.8, .5, .6, .1]})
    results = paired_bootstrap(events, "a", "b", "evaluation_group", repeats=100)
    assert all(x["difference"] == x["lo95"] == x["hi95"] == 0 for x in results)
