"""
Centralized (non-federated, non-private) PyTorch MLP severity/mortality
classifier. This is the accuracy ceiling -- federated.py and dp_federated.py
reuse this exact architecture, just trained differently.

Model inputs: demographics (gender, age) + 12 vital features.
final_tier is NEVER a model input -- it's used only to report metrics broken
down by tier, and later to set the DP noise budget per tier. Feeding tier
into the model would leak the exact attribute the project is trying to
protect.

Run from repo root:
    python -m src.models.baseline
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VITAL_FEATURES = [
    "heart_rate", "resp_rate", "spo2", "temp_fahrenheit",
    "nbp_systolic", "nbp_diastolic", "nbp_mean", "glucose",
    "gcs_eye", "gcs_verbal", "gcs_motor", "weight_kg",
]

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MortalityMLP(nn.Module):
    """Simple feedforward classifier. Same architecture reused by
    federated.py (plain FedAvg) and dp_federated.py (FedAvg + Opacus DP-SGD),
    so results across all three stages are comparable."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # raw logit -- use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_train_test():
    modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
    split = pd.read_csv(PROCESSED_DIR / "train_test_split.csv")
    df = modeling_table.merge(split, on="subject_id", how="inner")
    return df


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def evaluate(model, X, y, tiers, label=""):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)

    results = {}
    for group_name, mask in [
        ("overall", np.ones(len(y), dtype=bool)),
        ("stigma_sensitive", tiers == "stigma_sensitive"),
        ("non_sensitive", tiers == "non_sensitive"),
    ]:
        y_g, p_g, pred_g = y[mask], probs[mask], preds[mask]
        if len(np.unique(y_g)) < 2:
            roc_auc, pr_auc = float("nan"), float("nan")
        else:
            roc_auc = roc_auc_score(y_g, p_g)
            pr_auc = average_precision_score(y_g, p_g)
        f1 = f1_score(y_g, pred_g, zero_division=0)
        results[group_name] = {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "n": mask.sum()}

    print(f"--- {label} evaluation ---")
    for group_name, m in results.items():
        print(f"  {group_name:<18} n={m['n']:<4} ROC-AUC={m['roc_auc']:.4f}  "
              f"PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")
    return results


def main():
    df = load_train_test()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Impute vitals using TRAIN medians only (avoid test-set leakage), encode gender
    X_train_df = train_df[["age"] + VITAL_FEATURES].copy()
    train_medians = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(train_medians)
    X_train_df["gender"] = (train_df["gender"] == "M").astype(float)

    X_test_df = test_df[["age"] + VITAL_FEATURES].copy().fillna(train_medians)
    X_test_df["gender"] = (test_df["gender"] == "M").astype(float)

    feature_cols = ["age", "gender"] + VITAL_FEATURES
    X_train = X_train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = X_test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_test = test_df["hospital_expire_flag"].to_numpy(dtype=np.float32)

    X_train, X_test = standardize(X_train, X_test)

    tiers_train = train_df["final_tier"].to_numpy()
    tiers_test = test_df["final_tier"].to_numpy()

    print(f"Train: {len(X_train)} admissions | Test: {len(X_test)} admissions")
    print(f"Features ({len(feature_cols)}): {feature_cols}\n")

    model = MortalityMLP(n_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)

    N_EPOCHS = 100
    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} -- loss: {loss.item():.4f}")

    print()
    train_results = evaluate(model, X_train, y_train, tiers_train, label="TRAIN")
    print()
    test_results = evaluate(model, X_test, y_test, tiers_test, label="TEST")

    rows = []
    for split_name, results in [("train", train_results), ("test", test_results)]:
        for group, m in results.items():
            rows.append({"stage": "centralized", "split": split_name, "group": group, **m})
    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "centralized_baseline.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()