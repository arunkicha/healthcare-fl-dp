"""
Federated training (plain FedAvg, no privacy) across the 10 simulated clients
from partition_clients.py. Same MortalityMLP architecture as baseline.py, so
results are directly comparable -- this stage tells us what federation alone
costs in accuracy before any DP noise is added.

Simulation approach: since this is a small dataset (99 patients, 10 clients),
FL is simulated in-process (no real Flower server/client network) -- each
"client" is just a data partition, and FedAvg is implemented manually as
weighted parameter averaging after local training rounds. This keeps the
demo fast and dependency-light while still faithfully implementing the
FedAvg algorithm.

Run from repo root:
    python -m src.models.federated
"""

import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.models.baseline import MortalityMLP, VITAL_FEATURES

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_ROUNDS = 20          # federated communication rounds
LOCAL_EPOCHS = 5       # local training epochs per client per round
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data():
    modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
    split = pd.read_csv(PROCESSED_DIR / "train_test_split.csv")
    assignments = pd.read_csv(PROCESSED_DIR / "client_assignments.csv")

    df = modeling_table.merge(split, on="subject_id", how="inner")
    df = df.merge(assignments[["subject_id", "client_id"]], on="subject_id", how="left")
    return df


def build_features(df, train_medians):
    X = df[["age"] + VITAL_FEATURES].copy().fillna(train_medians)
    X["gender"] = (df["gender"] == "M").astype(float)
    feature_cols = ["age", "gender"] + VITAL_FEATURES
    return X[feature_cols].to_numpy(dtype=np.float32), feature_cols


def standardize(X, mean, std):
    return (X - mean) / std


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
        results[group_name] = {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1, "n": int(mask.sum())}

    print(f"--- {label} evaluation ---")
    for group_name, m in results.items():
        print(f"  {group_name:<18} n={m['n']:<4} ROC-AUC={m['roc_auc']:.4f}  "
              f"PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")
    return results


def get_flat_params(model):
    return [p.detach().clone() for p in model.parameters()]


def set_flat_params(model, flat_params):
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), flat_params):
            p.copy_(new_p)


def fedavg(client_params_list, client_weights):
    """Weighted average of parameters across clients (weight = local sample count)."""
    total_weight = sum(client_weights)
    n_params = len(client_params_list[0])
    avg_params = []
    for i in range(n_params):
        stacked = torch.stack([cp[i] * (w / total_weight) for cp, w in zip(client_params_list, client_weights)])
        avg_params.append(stacked.sum(dim=0))
    return avg_params


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

    # Split train data per client
    client_ids = sorted(train_df["client_id"].dropna().unique())
    client_data = {}
    for cid in client_ids:
        mask = train_df["client_id"] == cid
        client_data[cid] = {
            "X": torch.tensor(X_train_all[mask.to_numpy()]),
            "y": torch.tensor(y_train_all[mask.to_numpy()]),
            "n": int(mask.sum()),
        }
        print(f"Client {int(cid)}: {client_data[cid]['n']} training admissions")

    global_model = MortalityMLP(n_features=len(feature_cols))
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nStarting FedAvg: {N_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"{len(client_ids)} clients\n")

    for rnd in range(N_ROUNDS):
        client_params_list = []
        client_weights = []

        for cid in client_ids:
            local_model = MortalityMLP(n_features=len(feature_cols))
            set_flat_params(local_model, get_flat_params(global_model))
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3, weight_decay=1e-4)

            local_model.train()
            for _ in range(LOCAL_EPOCHS):
                optimizer.zero_grad()
                logits = local_model(client_data[cid]["X"])
                loss = criterion(logits, client_data[cid]["y"])
                loss.backward()
                optimizer.step()

            client_params_list.append(get_flat_params(local_model))
            client_weights.append(client_data[cid]["n"])

        new_global_params = fedavg(client_params_list, client_weights)
        set_flat_params(global_model, new_global_params)

        if (rnd + 1) % 5 == 0:
            print(f"Round {rnd+1}/{N_ROUNDS} complete")

    print()
    test_results = evaluate(global_model, X_test, y_test, tiers_test, label="TEST (federated, no privacy)")

    rows = []
    for group, m in test_results.items():
        rows.append({"stage": "federated_no_dp", "split": "test", "group": group, **m})
    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "federated_no_dp.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

    # Quick comparison vs centralized baseline, if it exists
    baseline_path = RESULTS_DIR / "centralized_baseline.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_test = baseline_df[(baseline_df["split"] == "test") & (baseline_df["group"] == "overall")]
        if len(baseline_test):
            centralized_auc = baseline_test["roc_auc"].iloc[0]
            federated_auc = test_results["overall"]["roc_auc"]
            retained_pct = 100 * federated_auc / centralized_auc if centralized_auc else float("nan")
            print(f"\nCentralized overall test ROC-AUC: {centralized_auc:.4f}")
            print(f"Federated overall test ROC-AUC:    {federated_auc:.4f}")
            print(f"Federated retains {retained_pct:.1f}% of centralized ROC-AUC")


if __name__ == "__main__":
    main()