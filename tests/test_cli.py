import pytest

from conjunctions.cli import main, read_config


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_is_valid():
    config = read_config("configs/default.toml")
    assert config["primary_cutoff"] == 3 and config["folds"] == 5


@pytest.mark.parametrize("change, message", [
    (lambda text: text.replace("folds = 5\n", ""), "missing: folds"),
    (lambda text: text.replace("folds = 5", "folds = 2.5"), "folds must be an integer"),
    (lambda text: text.replace("split_seed = 20261005", 'split_seed = "x"'), "split_seed must be an integer"),
])
def test_invalid_configs_give_clear_errors(tmp_path, change, message):
    text = change(open("configs/default.toml", encoding="utf-8").read())
    with pytest.raises(ValueError, match=message):
        read_config(write(tmp_path, text))


def test_cli_reports_config_errors_without_traceback(tmp_path, capsys):
    path = write(tmp_path, "cutoffs = [3]\n")
    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(path), "prepare", "--output", str(tmp_path / "out")])
    assert exit_info.value.code == 2 and "Config is missing" in capsys.readouterr().err
