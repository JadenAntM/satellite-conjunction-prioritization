# Working on the project

1. Create a branch for a focused change, then open a pull request for teammates to review.
2. Keep work within the data/feature/model/evaluation structure. Add a notebook only if it has a specific analysis purpose; production data preparation belongs in the package.
3. Run `python -m pytest -q` before publishing code changes. Tests use artificial correctness fixtures, not synthetic research outcomes.
4. Keep downloaded data, generated CSVs/figures, trained models, caches and environments local. `.gitignore` covers the project artifact directories.
5. Do not commit credentials, personal machine paths or environment-specific imports. Run commands from the repository root.
6. Record experiment changes in a config and choose a new run name. Preserve old run manifests/predictions locally. Do not change labels or claim a new random split is independent validation.
7. Separate model/information comparisons from deployment claims. Report weak candidates and null history/calibration results as well as improvements.

The default experiment is in `configs/default.toml`. A different config is supplied before the command:

```bash
python -m conjunctions --config configs/your_experiment.toml prepare --output data/processed/your_experiment
python -m conjunctions --config configs/your_experiment.toml run --processed data/processed/your_experiment --name your_experiment
```

Timing, event/group separation, class imbalance and metric contracts are documented in `docs/experiment.md`. Scope changes should update that documentation along with the code.
