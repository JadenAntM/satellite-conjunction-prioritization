import numpy as np
import pandas as pd
import pytest

from conjunctions.features import HISTORY_COLUMNS, summarize_history
from conjunctions.schema import HISTORY_FIELDS


def visible(times, risks, **fields):
    data = {"time_to_tca": times, "risk": risks}
    for field in HISTORY_FIELDS:
        data.setdefault(field, fields.get(field, [1.] * len(times)))
    return pd.DataFrame(data)


def test_history_statistics_follow_documented_definitions():
    row = summarize_history(visible([6., 5., 3.], [-10., -8., -7.]))
    assert set(row) == set(HISTORY_COLUMNS)
    assert row["visible_count"] == 3 and row["visible_span_days"] == 3.
    assert row["history__risk__mean"] == pytest.approx(-25 / 3)
    assert row["history__risk__std"] == pytest.approx(np.std([-10., -8., -7.]))  # population SD
    assert row["history__risk__max"] == -7.
    assert row["history__risk__change"] == 3.  # first to latest
    assert row["history__risk__rate"] == 1.  # per visible day
    assert row["history__risk__last_change"] == 1.


def test_single_message_history_keeps_levels_but_not_changes():
    row = summarize_history(visible([4.], [-9.]))
    assert row["visible_count"] == 1 and row["visible_span_days"] == 0.
    assert row["history__risk__mean"] == row["history__risk__max"] == -9.
    assert row["history__risk__std"] == 0. and row["history__risk__change"] == 0.
    assert np.isnan(row["history__risk__rate"]) and np.isnan(row["history__risk__last_change"])


def test_missing_and_nonfinite_values_do_not_leak_into_summaries():
    sigma = [np.inf, 2., np.nan]
    row = summarize_history(visible([5., 4., 3.], [-9., -8., -7.], t_sigma_r=sigma))
    assert row["history__t_sigma_r__mean"] == row["history__t_sigma_r__max"] == 2.
    assert row["history__t_sigma_r__std"] == 0.
    # Change statistics require observed endpoints; a missing latest value stays missing.
    assert np.isnan(row["history__t_sigma_r__change"]) and np.isnan(row["history__t_sigma_r__last_change"])
    row = summarize_history(visible([5., 4.], [-9., -8.], c_sigma_n=[np.nan, np.nan]))
    assert all(np.isnan(row[f"history__c_sigma_n__{s}"]) for s in ("mean", "std", "max", "change", "rate"))


def test_history_uses_rows_in_given_order_latest_last():
    early_to_late = summarize_history(visible([6., 4.], [-10., -6.]))
    assert early_to_late["history__risk__change"] == 4. and early_to_late["visible_span_days"] == 2.
