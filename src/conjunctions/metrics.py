"""Rare-event ranking, probability scores and retrospective review workloads."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def validate_scores(labels, scores, probability=False):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Labels and scores must be nonempty vectors of equal length")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if probability and ((scores < 0) | (scores > 1)).any():
        raise ValueError("Endpoint probabilities must be in [0, 1]")
    return labels, scores


def budget_metrics(labels, scores, fraction):
    labels, scores = validate_scores(labels, scores)
    if not 0 < fraction <= 1:
        raise ValueError("Review fraction must be in (0, 1]")
    reviews = math.ceil(len(labels) * fraction)
    boundary = np.partition(scores, len(scores) - reviews)[len(scores) - reviews]
    above, tied = scores > boundary, scores == boundary
    hits = float(labels[above].sum() + (reviews - above.sum()) * labels[tied].mean())
    positive = labels.sum()
    return {
        "budget": fraction, "reviews": reviews, "hits": hits,
        "missed": float(positive - hits), "false_reviews": reviews - hits,
        "precision": hits / reviews, "recall": hits / positive if positive else np.nan,
        "prevalence_lift": (hits / reviews) / labels.mean() if positive else np.nan,
        "false_per_true": (reviews - hits) / hits if hits else np.nan,
    }


def score_metrics(labels, scores, probability=True, budgets=(.01, .02, .05, .10)):
    labels, scores = validate_scores(labels, scores, probability)
    positive = int(labels.sum())
    result = {
        "n": len(labels), "positive": positive, "prevalence": labels.mean(),
        "ap": average_precision_score(labels, scores) if positive else np.nan,
        "roc_auc": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan,
        "brier": brier_score_loss(labels, scores) if probability else np.nan,
        "log_loss": log_loss(labels, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1]) if probability else np.nan,
    }
    for fraction in budgets:
        for key, value in budget_metrics(labels, scores, fraction).items():
            if key != "budget":
                result[f"top{int(fraction * 100)}_{key}"] = value
    return result


def workload_at_recall(labels, scores, recall):
    labels, scores = validate_scores(labels, scores)
    if not 0 < recall <= 1:
        raise ValueError("Capture target must be in (0, 1]")
    if not labels.sum():
        return {"target_recall": recall, "reviews": np.nan, "fraction": np.nan, "hits": 0}
    bins = pd.DataFrame({"score": scores, "label": labels}).groupby("score").label.agg(["size", "sum"]).sort_index(ascending=False)
    hits, reviews = bins["sum"].cumsum(), bins["size"].cumsum()
    boundary = hits[hits >= recall * labels.sum()].index[0]
    n, p = int(reviews.loc[boundary]), int(hits.loc[boundary])
    return {"target_recall": recall, "reviews": n, "fraction": n / len(labels), "hits": p,
            "recall": p / labels.sum(), "precision": p / n, "false_reviews": n - p}


class WeightedRank:
    """Fast, tie-aware metrics for replicating clusters in a paired bootstrap."""

    def __init__(self, labels, scores):
        self.labels, scores = validate_scores(labels, scores)
        unique, inverse = np.unique(scores, return_inverse=True)
        self.bin = len(unique) - 1 - inverse
        self.nbins = len(unique)

    def evaluate(self, weights):
        positive = np.dot(weights, self.labels)
        if positive == 0:
            return np.array([np.nan, np.nan])
        counts = np.bincount(self.bin, weights=weights, minlength=self.nbins)
        positives = np.bincount(self.bin, weights=weights * self.labels, minlength=self.nbins)
        reviewed, captured = counts.cumsum(), positives.cumsum()
        precision = np.divide(captured, reviewed, out=np.zeros_like(captured), where=reviewed > 0)
        ap = np.sum((positives / positive) * precision)
        capacity = math.ceil(.05 * weights.sum())
        boundary = np.searchsorted(reviewed, capacity)
        before_n = reviewed[boundary - 1] if boundary else 0
        before_p = captured[boundary - 1] if boundary else 0
        hits = before_p + (capacity - before_n) * positives[boundary] / counts[boundary]
        return np.array([ap, hits / positive])


def paired_bootstrap(events, left, right, cluster, repeats=1000, seed=20261055):
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    _, inverse = np.unique(events[cluster], return_inverse=True)
    clusters = int(inverse.max() + 1)
    a, b = WeightedRank(events.label, events[left]), WeightedRank(events.label, events[right])
    point = a.evaluate(np.ones(len(events))) - b.evaluate(np.ones(len(events)))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, clusters, size=clusters), minlength=clusters)
        weights = counts[inverse]
        values.append(a.evaluate(weights) - b.evaluate(weights))
    values = np.asarray(values)
    result = []
    for column, metric in enumerate(["ap", "top5_recall"]):
        valid = values[:, column][np.isfinite(values[:, column])]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
        result.append({"left": left, "right": right, "cluster": cluster, "clusters": clusters,
                       "metric": metric, "difference": point[column], "lo95": low, "hi95": high,
                       "resamples": repeats, "valid_resamples": len(valid)})
    return result
