import json

import pandas as pd
import pytest

from conjunctions.cli import read_config
from conjunctions.experiment import load_cohorts, run_experiment
from conjunctions.data import build_events, clean_messages


def test_existing_run_cannot_be_overwritten(tmp_path):
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path, tmp_path, "existing", {})


def test_run_name_cannot_escape_output_directory(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(tmp_path, tmp_path, "../outside", {})


def test_config_has_fixed_primary_and_information_sets():
    config = read_config("configs/default.toml")
    assert config["primary_cutoff"] == 3 and config["information_sets"] == list("ABCD")
    assert config["budgets"] == [.01, .02, .05, .1]


def test_prepared_data_must_retain_valid_labels_and_matching_folds(messages, tmp_path):
    config = read_config("configs/default.toml")
    config.update(cutoffs=[2, 3], primary_cutoff=3, folds=2)
    clean = clean_messages(messages)[0]
    for cutoff in (2, 3):
        data = build_events(clean, cutoff)
        data["evaluation_group"] = data.event_id
        data["fold"] = [0, 0, 1, 1]
        data.to_csv(tmp_path / f"matched_k{cutoff}.csv", index=False)
    (tmp_path / "audit.json").write_text(json.dumps({"preparation_config": config}))
    assert len(load_cohorts(tmp_path, config)[3]) == 4
    changed = pd.read_csv(tmp_path / "matched_k2.csv")
    changed.loc[0, "label"] = 0
    changed.to_csv(tmp_path / "matched_k2.csv", index=False)
    with pytest.raises(ValueError, match="Labels differ"):
        load_cohorts(tmp_path, config)
