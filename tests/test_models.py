import numpy as np
import pytest

from conjunctions.calibration import fit_calibrators
from conjunctions.data import build_events, clean_messages
from conjunctions.models import make_model


def test_preprocessing_never_refits_on_evaluation_data(messages):
    data = build_events(clean_messages(messages)[0], 2)
    estimator = make_model("logistic", "A")
    estimator.fit(data, data.label)
    scaler = estimator.named_steps["features"].named_transformers_["plain"].named_steps["scale"]
    mean = scaler.mean_.copy()
    estimator.predict_proba(data.assign(latest__risk=99999.))
    np.testing.assert_array_equal(mean, scaler.mean_)
    np.testing.assert_array_equal(mean, [-7.])


@pytest.mark.parametrize("family", ["logistic", "random_forest", "hgb"])
def test_snapshot_pipeline_accepts_unseen_object_type(messages, family):
    data = build_events(clean_messages(messages)[0], 2)
    estimator = make_model(family, "B")
    estimator.fit(data, data.label)
    score = estimator.predict_proba(data.assign(latest__c_object_type="PAYLOAD"))[:, 1]
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()


def test_calibrator_support_gate_and_raw_probability_preserved():
    labels = np.tile([0, 1], 60)
    inner = np.where(labels == 1, .4, .1)
    outer = np.array([.05, .2, .5])
    calibrated = fit_calibrators(inner, labels, outer)
    assert set(calibrated) == {"raw", "platt", "isotonic"}
    np.testing.assert_array_equal(calibrated["raw"], outer)
    assert np.isfinite(calibrated["platt"]).all()
    assert "isotonic" not in fit_calibrators(inner[:80], labels[:80], outer)
