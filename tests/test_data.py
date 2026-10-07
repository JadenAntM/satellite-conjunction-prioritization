import numpy as np
import pandas as pd
import pytest

from conjunctions.data import assign_folds, build_events, clean_messages, related_groups
from conjunctions.features import feature_columns


def test_future_values_cannot_change_predictors(messages):
    before = build_events(clean_messages(messages)[0], 2)
    changed = messages.copy()
    changed.loc[changed.time_to_tca < 2, ["risk", "miss_distance", "c_sigma_t"]] = [-30., 999999., 888888.]
    after = build_events(clean_messages(changed)[0], 2)
    pd.testing.assert_frame_equal(before[feature_columns("D")], after[feature_columns("D")])
    assert before.label.sum() != after.label.sum()


def test_future_rows_cannot_change_history(messages):
    before = build_events(clean_messages(messages)[0], 2)
    extra = messages[messages.time_to_tca == .5].copy()
    extra.time_to_tca = .7
    after = build_events(clean_messages(pd.concat([messages, extra]))[0], 2)
    pd.testing.assert_frame_equal(before[feature_columns("D")], after[feature_columns("D")])
    assert before.visible_count.tolist() == [2] * 4


def test_information_partition_and_allowlist():
    a, b, c, d = [set(feature_columns(s)) for s in "ABCD"]
    assert [len(x) for x in (a, b, c, d)] == [1, 97, 50, 147]
    assert a <= b and not b & c and b | c == d
    assert not {"label", "final_risk", "event_id", "mission_id", "fold"} & d
    assert not any(x in b for x in ["latest__F10", "latest__F3M", "latest__AP", "latest__SSN"])
    with pytest.raises(ValueError):
        feature_columns("future")


def test_cutoff_and_target_contract(messages):
    events = build_events(clean_messages(messages)[0], 2)
    assert events.latest__time_to_tca.tolist() == [2.] * 4
    assert events.label.tolist() == [1, 0, 1, 0]
    assert events.history__risk__max.tolist() == [-7.] * 4
    assert events.visible_span_days.tolist() == [2.] * 4


@pytest.mark.parametrize("endpoint", [-.1, 1., 2.])
def test_endpoint_eligibility_is_strict(messages, endpoint):
    event = messages.query("event_id == 1").copy()
    event.loc[event.time_to_tca == .5, "time_to_tca"] = endpoint
    assert build_events(clean_messages(event)[0], 2).empty


def test_duplicates_conflicts_and_missing_risk(messages):
    clean, audit = clean_messages(pd.concat([messages, messages.iloc[[0]]]))
    assert len(clean) == len(messages) and audit["exact_duplicate_rows"] == 1
    extra = messages.iloc[[0]].copy()
    extra.risk = -9.
    clean, audit = clean_messages(pd.concat([messages, extra]))
    assert 1 not in set(clean.event_id) and audit["quarantined_events"] == [1]
    invalid = messages.copy()
    invalid.loc[0, "risk"] = np.nan
    clean, _ = clean_messages(invalid)
    assert 1 not in set(clean.event_id)


def test_mission_and_identity_contract(messages):
    invalid = messages.copy()
    invalid.loc[0, "mission_id"] = 20
    clean, _ = clean_messages(invalid)
    assert 1 not in set(clean.event_id)
    invalid.loc[0, "event_id"] = np.nan
    with pytest.raises(ValueError, match="finite integers"):
        clean_messages(invalid)


def test_related_grouping_is_transitive_and_mission_specific(messages):
    extra = messages.query("event_id == 1 and time_to_tca == 4").copy()
    extra.event_id = 7
    extra.relative_position_r = 1234.
    groups = related_groups(pd.concat([messages, extra]))
    assert groups[1] == groups[7] and groups[1] != groups[2]


def test_grouped_folds_preserve_related_events():
    events = pd.DataFrame({"event_id": np.arange(40), "evaluation_group": np.repeat(np.arange(20), 2),
                           "label": np.repeat(np.arange(20) % 2, 2)})
    folds = assign_folds(events, folds=2, seed=42)
    assert events.assign(fold=folds).groupby("evaluation_group").fold.nunique().max() == 1
    for fold in range(2):
        assert not set(events.evaluation_group[folds == fold]) & set(events.evaluation_group[folds != fold])
