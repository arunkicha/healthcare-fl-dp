import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["overall", "stigma_sensitive", "non_sensitive"]
GROUP_COLORS = {"overall": "#4c72b0", "stigma_sensitive": "#c44e52", "non_sensitive": "#55a868"}


def load_all():
    centralized = pd.read_csv(RESULTS_DIR / "centralized_baseline.csv")
    federated = pd.read_csv(RESULTS_DIR / "federated_no_dp.csv")
    dp_multiseed = pd.read_csv(RESULTS_DIR / "dp_federated_multiseed.csv")
    epsilons = pd.read_csv(RESULTS_DIR / "dp_federated_epsilons.csv")
    mia = pd.read_csv(RESULTS_DIR / "mia_multiseed.csv")
    return centralized, federated, dp_multiseed, epsilons, mia


def stage_summary(centralized, federated, dp_multiseed):
    rows = []
    cent_test = centralized[centralized["split"] == "test"]
    for group in GROUP_ORDER:
        val = cent_test.loc[cent_test["group"] == group, "roc_auc"]
        if len(val):
            rows.append({"stage": "centralized", "group": group, "mean": val.iloc[0], "std": 0.0})

    for group in GROUP_ORDER:
        val = federated.loc[federated["group"] == group, "roc_auc"]
        if len(val):
            rows.append({"stage": "federated_no_dp", "group": group, "mean": val.iloc[0], "std": 0.0})

    for stage in dp_multiseed["stage"].unique():
        sub = dp_multiseed[dp_multiseed["stage"] == stage]
        for group in GROUP_ORDER:
            g = sub[sub["group"] == group]["roc_auc"]
            if len(g):
                rows.append({"stage": stage, "group": group, "mean": g.mean(), "std": g.std()})

    return pd.DataFrame(rows)


def plot_stage_comparison(summary):
    stages = summary["stage"].unique()
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = range(len(stages))

    for i, group in enumerate(GROUP_ORDER):
        means, stds = [], []
        for stage in stages:
            row = summary[(summary["stage"] == stage) & (summary["group"] == group)]
            means.append(row["mean"].iloc[0] if len(row) else 0)
            stds.append(row["std"].iloc[0] if len(row) else 0)
        offset = (i - 1) * width
        ax.bar([p + offset for p in x], means, width, yerr=stds, capsize=4,
               label=group, color=GROUP_COLORS[group])

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC across pipeline stages, by tier")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage_comparison_roc_auc.png", dpi=150)
    plt.close(fig)


def plot_multiseed_stability(dp_multiseed):
    stages = dp_multiseed["stage"].unique()
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 5), sharey=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        sub = dp_multiseed[dp_multiseed["stage"] == stage]
        data = [sub[sub["group"] == g]["roc_auc"].values for g in GROUP_ORDER]
        ax.boxplot(data, tick_labels=GROUP_ORDER)
        ax.set_title(stage)
        ax.set_ylabel("ROC-AUC")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("ROC-AUC spread across seeds, by stage and tier")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multiseed_stability.png", dpi=150)
    plt.close(fig)


def plot_epsilon_by_tier(epsilons):
    summary = epsilons.groupby(["stage", "tier"])["epsilon"].agg(["mean", "std"]).reset_index()
    stages = summary["stage"].unique()
    tiers = ["stigma_sensitive", "non_sensitive"]

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = range(len(stages))

    for i, tier in enumerate(tiers):
        means, stds = [], []
        for stage in stages:
            row = summary[(summary["stage"] == stage) & (summary["tier"] == tier)]
            means.append(row["mean"].iloc[0] if len(row) else 0)
            stds.append(row["std"].iloc[0] if len(row) else 0)
        offset = (i - 0.5) * width
        color = GROUP_COLORS["stigma_sensitive"] if tier == "stigma_sensitive" else GROUP_COLORS["non_sensitive"]
        ax.bar([p + offset for p in x], means, width, yerr=stds, capsize=4, label=tier, color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Epsilon (privacy budget spent)")
    ax.set_title("Final-round epsilon by tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "epsilon_by_tier.png", dpi=150)
    plt.close(fig)


def plot_mia_attack_accuracy(mia):
    summary = mia.groupby(["stage", "subgroup"])["attack_accuracy"].agg(["mean", "std"]).reset_index()
    stages = summary["stage"].unique()

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = range(len(stages))

    for i, group in enumerate(GROUP_ORDER):
        means, stds = [], []
        for stage in stages:
            row = summary[(summary["stage"] == stage) & (summary["subgroup"] == group)]
            means.append(row["mean"].iloc[0] if len(row) else 0)
            stds.append(row["std"].iloc[0] if len(row) else 0)
        offset = (i - 1) * width
        ax.bar([p + offset for p in x], means, width, yerr=stds, capsize=4,
               label=group, color=GROUP_COLORS[group])

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Attack accuracy")
    ax.set_title("Membership inference attack accuracy, by stage and tier")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mia_attack_accuracy.png", dpi=150)
    plt.close(fig)


def save_summary_tables(summary, epsilons, mia):
    summary.to_csv(RESULTS_DIR / "summary_roc_auc_by_stage.csv", index=False)

    eps_summary = epsilons.groupby(["stage", "tier"])["epsilon"].agg(["mean", "std"]).reset_index()
    eps_summary.to_csv(RESULTS_DIR / "summary_epsilon_by_tier.csv", index=False)

    mia_summary = mia.groupby(["stage", "subgroup"])[
        ["attack_accuracy", "attack_auc", "n_members", "n_nonmembers"]
    ].agg(["mean", "std"])
    mia_summary.to_csv(RESULTS_DIR / "summary_mia_by_stage.csv")


def main():
    centralized, federated, dp_multiseed, epsilons, mia = load_all()

    summary = stage_summary(centralized, federated, dp_multiseed)
    plot_stage_comparison(summary)
    plot_multiseed_stability(dp_multiseed)
    plot_epsilon_by_tier(epsilons)
    plot_mia_attack_accuracy(mia)
    save_summary_tables(summary, epsilons, mia)

    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Summary tables saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()