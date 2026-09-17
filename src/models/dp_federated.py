import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy import stats
from opacus import PrivacyEngine

from src.models.baseline import MortalityMLP, VITAL_FEATURES
from src.privacy.tiered_dp import (
    NOISE_MULTIPLIER_STIGMA_SENSITIVE,
    NOISE_MULTIPLIER_NON_SENSITIVE,
    NOISE_MULTIPLIER_FLAT,
    MAX_GRAD_NORM,
    DELTA,
)
from src.privacy.mia import run_mia_for_run

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_ROUNDS = 20
LOCAL_EPOCHS = 5
SEEDS = [42, 43, 44, 45, 46]

CONFIGS = {
    "noise0_sanity_check": {"stigma": 0.0, "non_sensitive": 0.0},
    "flat_dp": {"stigma": NOISE_MULTIPLIER_FLAT, "non_sensitive": NOISE_MULTIPLIER_FLAT},
    "tiered": {"stigma": NOISE_MULTIPLIER_STIGMA_SENSITIVE, "non_sensitive": NOISE_MULTIPLIER_NON_SENSITIVE},
}


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


def get_flat_params(model):
    return [p.detach().clone() for p in model.parameters()]


def set_flat_params(model, flat_params):
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), flat_params):
            p.copy_(new_p)


def weighted_average_params(params_list, weights):
    total = sum(weights)
    if total == 0:
        return params_list[0]
    n_params = len(params_list[0])
    avg = []
    for i in range(n_params):
        stacked = torch.stack([p[i] * (w / total) for p, w in zip(params_list, weights)])
        avg.append(stacked.sum(dim=0))
    return avg


def train_dp_subset(base_params, X, y, noise_multiplier, n_features):
    n = len(X)
    if n == 0:
        return None

    model = MortalityMLP(n_features=n_features)
    set_flat_params(model, base_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=max(1, n // 4), shuffle=True)

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    epsilon = None
    if noise_multiplier > 0:
        try:
            epsilon = privacy_engine.get_epsilon(delta=DELTA)
        except Exception:
            epsilon = None

    final_params = [p.detach().clone() for p in model.parameters()]
    return final_params, n, epsilon


def evaluate(model, X, y, tiers, label="", verbose=True):
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

    if verbose:
        print(f"--- {label} evaluation ---")
        for group_name, m in results.items():
            print(f"  {group_name:<18} n={m['n']:<4} ROC-AUC={m['roc_auc']:.4f}  "
                  f"PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")
    return results


def run_config(config_name, noise_stigma, noise_non_sensitive, train_df, X_train_all,
                y_train_all, X_test, y_test, tiers_test, feature_cols, verbose=True):
    if verbose:
        print(f"\n{'='*70}\nCONFIG: {config_name} "
              f"(stigma_sensitive noise={noise_stigma}, non_sensitive noise={noise_non_sensitive})\n{'='*70}")

    client_ids = sorted(train_df["client_id"].dropna().unique())
    client_subsets = {}
    for cid in client_ids:
        cmask = (train_df["client_id"] == cid).to_numpy()
        for tier, noise in [("stigma_sensitive", noise_stigma), ("non_sensitive", noise_non_sensitive)]:
            tmask = cmask & (train_df["final_tier"] == tier).to_numpy()
            X_sub = torch.tensor(X_train_all[tmask])
            y_sub = torch.tensor(y_train_all[tmask])
            client_subsets[(cid, tier)] = {"X": X_sub, "y": y_sub, "noise": noise}

    global_model = MortalityMLP(n_features=len(feature_cols))
    last_epsilons = {"stigma_sensitive": [], "non_sensitive": []}

    for rnd in range(N_ROUNDS):
        client_params_list = []
        client_weights = []

        for cid in client_ids:
            base_params = get_flat_params(global_model)
            subset_results = []

            for tier in ["stigma_sensitive", "non_sensitive"]:
                sub = client_subsets[(cid, tier)]
                result = train_dp_subset(
                    base_params, sub["X"], sub["y"], sub["noise"], len(feature_cols)
                )
                if result is not None:
                    params, n, epsilon = result
                    subset_results.append((params, n))
                    if epsilon is not None and rnd == N_ROUNDS - 1:
                        last_epsilons[tier].append(epsilon)

            if not subset_results:
                continue

            params_list = [r[0] for r in subset_results]
            weights = [r[1] for r in subset_results]
            client_combined = weighted_average_params(params_list, weights)

            client_params_list.append(client_combined)
            client_weights.append(sum(weights))

        new_global_params = weighted_average_params(client_params_list, client_weights)
        set_flat_params(global_model, new_global_params)

        if verbose and (rnd + 1) % 5 == 0:
            print(f"Round {rnd+1}/{N_ROUNDS} complete")

    if verbose:
        print()
    test_results = evaluate(global_model, X_test, y_test, tiers_test,
                             label=f"TEST ({config_name})" if verbose else "", verbose=verbose)

    if verbose:
        print("\n--- Final-round epsilon (privacy spent), averaged across clients with data in that tier ---")
        for tier, eps_list in last_epsilons.items():
            if eps_list:
                print(f"  {tier}: mean epsilon = {np.mean(eps_list):.3f} (n_clients={len(eps_list)})")
            else:
                print(f"  {tier}: no epsilon reported (noise_multiplier was 0)")

    mean_epsilons = {tier: (np.mean(v) if v else None) for tier, v in last_epsilons.items()}
    return test_results, mean_epsilons, global_model


def print_significance_tests(results_df):
    print(f"\n{'='*70}\nSIGNIFICANCE TESTS (paired across {len(SEEDS)} seeds)\n{'='*70}")
    baseline_stage = "dp_federated_noise0_sanity_check"
    comparisons = [
        (baseline_stage, "dp_federated_flat_dp"),
        (baseline_stage, "dp_federated_tiered"),
        ("dp_federated_flat_dp", "dp_federated_tiered"),
    ]

    for group in ["overall", "stigma_sensitive", "non_sensitive"]:
        sub = results_df[results_df["group"] == group]
        pivot = sub.pivot(index="seed", columns="stage", values="roc_auc")
        print(f"\n--- group: {group} ---")
        for a_stage, b_stage in comparisons:
            if a_stage not in pivot.columns or b_stage not in pivot.columns:
                continue
            a_vals = pivot[a_stage].values
            b_vals = pivot[b_stage].values
            if len(a_vals) < 3:
                continue
            t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
            try:
                w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
            except ValueError:
                w_p = float("nan")
            print(f"{a_stage} vs {b_stage}: paired t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}")


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
    tiers_train = train_df["final_tier"].to_numpy()

    all_rows = []
    epsilon_rows = []
    all_mia_rows = []

    for config_name, noise_cfg in CONFIGS.items():
        print(f"\n\n########## CONFIG: {config_name} -- running {len(SEEDS)} seeds ##########")
        for seed_idx, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            verbose = (seed_idx == 0)

            print(f"\n--- seed {seed} ---")
            test_results, mean_epsilons, global_model = run_config(
                config_name, noise_cfg["stigma"], noise_cfg["non_sensitive"],
                train_df, X_train_all, y_train_all, X_test, y_test, tiers_test, feature_cols,
                verbose=verbose,
            )
            for group, m in test_results.items():
                all_rows.append({"stage": f"dp_federated_{config_name}", "seed": seed, "group": group, **m})
            for tier, eps in mean_epsilons.items():
                epsilon_rows.append({"stage": config_name, "seed": seed, "tier": tier, "epsilon": eps})

            mia_rows = run_mia_for_run(
                global_model, X_train_all, y_train_all, tiers_train, X_test, y_test, tiers_test
            )
            for row in mia_rows:
                row["stage"] = config_name
                row["seed"] = seed
                all_mia_rows.append(row)

            summary = {g: f"ROC-AUC={m['roc_auc']:.3f}" for g, m in test_results.items()}
            print(f"seed {seed} summary: {summary}")

    results_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "dp_federated_multiseed.csv"
    results_df.to_csv(out_path, index=False)

    epsilon_df = pd.DataFrame(epsilon_rows)
    epsilon_df.to_csv(RESULTS_DIR / "dp_federated_epsilons.csv", index=False)

    mia_df = pd.DataFrame(all_mia_rows)
    mia_out_path = RESULTS_DIR / "mia_multiseed.csv"
    mia_df.to_csv(mia_out_path, index=False)

    print(f"\n\n{'='*70}\nSTABILITY SUMMARY (mean +/- std across {len(SEEDS)} seeds)\n{'='*70}")
    summary = (
        results_df.groupby(["stage", "group"])[["roc_auc", "pr_auc", "f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary)

    print(f"\n{'='*70}\nMIA STABILITY SUMMARY (mean +/- std across {len(SEEDS)} seeds)\n{'='*70}")
    mia_summary = (
        mia_df.groupby(["stage", "subgroup"])[["attack_balanced_accuracy", "attack_auc", "n_members", "n_nonmembers"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(mia_summary)

    print_significance_tests(results_df)

    print(f"\nSaved per-seed results to: {out_path}")
    print(f"Saved epsilon tracking to: {RESULTS_DIR / 'dp_federated_epsilons.csv'}")
    print(f"Saved MIA results to: {mia_out_path}")


if __name__ == "__main__":
    main()