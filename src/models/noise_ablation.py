import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.models.dp_federated import (
    load_data,
    build_features,
    standardize,
    run_config,
)
from src.models.baseline import VITAL_FEATURES

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
NON_SENSITIVE_NOISE_FIXED = 0.5
STIGMA_NOISE_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def main():
    df = load_data()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_medians = train_df[["age"] + VITAL_FEATURES].median(numeric_only=True)
    X_train_all, feature_cols = build_features(train_df, train_medians)
    X_test, _ = build_features(test_df, train_medians)
    y_train_all = train_df["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_test = test_df["hospital_expire_flag"].to_numpy(dtype=np.float32)

    mean = X_train_all.mean(axis=0, keepdims=True)
    std = X_train_all.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train_all = standardize(X_train_all, mean, std)
    X_test = standardize(X_test, mean, std)

    tiers_test = test_df["final_tier"].to_numpy()

    rows = []
    for stigma_noise in STIGMA_NOISE_VALUES:
        print(f"\n########## stigma_noise={stigma_noise}, non_sensitive_noise={NON_SENSITIVE_NOISE_FIXED} ##########")
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            test_results, mean_epsilons, _ = run_config(
                f"ablation_stigma{stigma_noise}", stigma_noise, NON_SENSITIVE_NOISE_FIXED,
                train_df, X_train_all, y_train_all, X_test, y_test, tiers_test, feature_cols,
                verbose=False,
            )
            row = {
                "stigma_noise": stigma_noise,
                "non_sensitive_noise": NON_SENSITIVE_NOISE_FIXED,
                "seed": seed,
                "epsilon_stigma": mean_epsilons.get("stigma_sensitive"),
                "epsilon_non_sensitive": mean_epsilons.get("non_sensitive"),
                "auc_overall": test_results["overall"]["roc_auc"],
                "auc_stigma_sensitive": test_results["stigma_sensitive"]["roc_auc"],
                "auc_non_sensitive": test_results["non_sensitive"]["roc_auc"],
            }
            rows.append(row)
            print(f"  seed {seed}: eps_stigma={row['epsilon_stigma']:.3f}  "
                  f"AUC_stigma={row['auc_stigma_sensitive']:.4f}")

    result_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "noise_ablation.csv"
    result_df.to_csv(out_path, index=False)

    summary = (
        result_df.groupby("stigma_noise")[["epsilon_stigma", "auc_overall", "auc_stigma_sensitive"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(f"\n{'='*70}\nABLATION SUMMARY (mean +/- std across {len(SEEDS)} seeds)\n{'='*70}")
    print(summary)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()