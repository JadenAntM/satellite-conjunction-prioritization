"""Project command line: download, prepare, run, report and calibrate."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import json
from pathlib import Path
import tomllib

from .archive import download_archive
from .data import prepare
from .experiment import run_experiment, summarize_run


def read_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    cutoffs = config["cutoffs"]
    if not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(k not in (2, 3, 4) for k in cutoffs):
        raise ValueError("Use unique cutoffs drawn from 2, 3, 4 days")
    if config["primary_cutoff"] not in cutoffs:
        raise ValueError("Primary cutoff is absent from experiment cutoffs")
    if config["folds"] < 2:
        raise ValueError("Use at least two grouped folds")
    for key, choices in [("families", {"logistic", "random_forest", "hgb"}), ("information_sets", set("ABCD"))]:
        if not config[key] or len(set(config[key])) != len(config[key]) or not set(config[key]).issubset(choices):
            raise ValueError(f"Invalid {key}")
    if config["bootstrap_repeats"] < 100:
        raise ValueError("Use at least 100 bootstrap repeats")
    if not config["budgets"] or any(x not in (.01, .02, .05, .10) for x in config["budgets"]):
        raise ValueError("Supported budgets are 1%, 2%, 5% and 10%")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Satellite conjunction endpoint prioritization")
    parser.add_argument("--config", default="configs/default.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch and verify the official ESA archive")
    download.add_argument("--directory", default="data/raw")
    prep = commands.add_parser("prepare", help="Construct matched cutoff-safe events and group folds")
    prep.add_argument("--archive", default="data/raw/esa_collision_dataset.zip")
    prep.add_argument("--output", default="data/processed")
    run = commands.add_parser("run", help="Fit the fixed experiment and summarize saved OOF predictions")
    run.add_argument("--processed", default="data/processed")
    run.add_argument("--results", default="results/runs")
    run.add_argument("--name", required=True)
    run.add_argument("--scheme", choices=["event", "mission"], default="event")
    report = commands.add_parser("report", help="Recompute metrics/curves from completed saved predictions")
    report.add_argument("--run", required=True)
    calibration = commands.add_parser("calibrate", help="Nested calibration for D at the primary cutoff")
    calibration.add_argument("--run", required=True)
    calibration.add_argument("--processed", default="data/processed")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(download_archive(args.directory))
        elif args.command == "prepare":
            print(json.dumps(prepare(args.archive, args.output, read_config(args.config)), indent=2))
        elif args.command == "run":
            output = run_experiment(args.processed, args.results, args.name, read_config(args.config), args.scheme)
            summarize_run(output)
            print(f"Results: {output}")
        elif args.command == "report":
            summarize_run(args.run)
        elif args.command == "calibrate":
            from .calibration import calibrate_run
            calibrate_run(args.processed, args.run)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")
