import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_DIR / "noise_ablation.csv")
    summary = df.groupby("stigma_noise")[
        ["epsilon_stigma", "auc_overall", "auc_stigma_sensitive"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["stigma_noise", "eps_mean", "eps_std", "auc_o_mean", "auc_o_std",
                        "auc_s_mean", "auc_s_std"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.errorbar(summary["stigma_noise"], summary["eps_mean"], yerr=summary["eps_std"],
                 marker="o", color="#c44e52", label="epsilon (stigma-sensitive tier)")
    ax1.set_xlabel("Stigma-sensitive noise multiplier")
    ax1.set_ylabel("Epsilon (privacy budget)", color="#c44e52")
    ax1.tick_params(axis="y", labelcolor="#c44e52")

    ax2.errorbar(summary["stigma_noise"], summary["auc_s_mean"], yerr=summary["auc_s_std"],
                 marker="s", color="#4c72b0", label="ROC-AUC (stigma-sensitive tier)")
    ax2.set_ylabel("ROC-AUC", color="#4c72b0")
    ax2.tick_params(axis="y", labelcolor="#4c72b0")

    fig.suptitle("Privacy-utility tradeoff: stigma-sensitive tier noise ablation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "noise_ablation_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved to: {FIGURES_DIR / 'noise_ablation_curve.png'}")


if __name__ == "__main__":
    main()