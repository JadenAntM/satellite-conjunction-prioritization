"""Small figures generated from a run's saved predictions and scores."""

import os
from pathlib import Path

import pandas as pd


def plot_run(output):
    output = Path(output)
    os.environ["MPLCONFIGDIR"] = str(output / ".cache" / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(output / ".cache" / "xdg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(output / "metrics.csv")
    budgets = pd.read_csv(output / "budgets.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in ["latest_risk", "A_logistic", "B_random_forest", "C_random_forest", "D_random_forest"]:
        group = metrics[metrics.model == name].sort_values("cutoff")
        if len(group):
            axes[0].plot(group.cutoff, group.ap, "-o", label=name)
        group = budgets[(budgets.model == name) & (budgets.budget == .05)].sort_values("cutoff")
        if len(group):
            axes[1].plot(group.cutoff, group.recall, "-o", label=name)
    axes[0].set(xlabel="Cutoff before TCA (days)", ylabel="Average precision", ylim=(0, 1))
    axes[1].set(xlabel="Cutoff before TCA (days)", ylabel="Recall at 5% review capacity", ylim=(0, 1))
    axes[0].legend(fontsize=7)
    figure.suptitle("Matched events; exploratory internal validation")
    figure.tight_layout()
    figure.savefig(output / "horizons.png", dpi=160)
    plt.close(figure)
