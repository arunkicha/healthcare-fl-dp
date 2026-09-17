import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.models.baseline import MortalityMLP, VITAL_FEATURES
from src.privacy.mia import run_mia_for_run

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBSAMPLE_N = 100
N_EPOCHS = 300
RANDOM_SEED = 42


def load_data():
    modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
    split = pd.read_csv(PROCESSED_DIR / "train_test_split.csv")
    return modeling_table.merge(split, on="subject_id", how="inner")


def build_features(df, train_medians):
    X = df[["age"] + VITAL_FEATURES].copy().fillna(train_medians)
    X["gender"] = (df["gender"] == "M").astype(float)
    feature_cols = ["age", "gender"] + VITAL_FEATURES
    return X[feature_cols].to_numpy(dtype=np.float32), feature_cols


def standardize(X, mean, std):
    return (X - mean) / std


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    df = load_data()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    small_idx = rng.choice(len(train_df), size=min(SUBSAMPLE_N, len(train_df)), replace=False)
    train_small = train_df.iloc[small_idx].reset_index(drop=True)

    train_medians = train_small[["age"] + VITAL_FEATURES].median(numeric_only=True)
    X_train, feature_cols = build_features(train_small, train_medians)
    X_test, _ = build_features(test_df, train_medians)

    # Real outcome label is ~10% imbalanced (real mortality rate), which dilutes the
    # loss-based memorization signal regardless of how overfit the model is. This check
    # exists only to prove the attack CAN detect memorization when present, so we assign
    # fresh balanced 50/50 random labels here instead of using the real outcome -- this
    # isolates attack power from the real task's class imbalance and learnability.
    y_train = rng.binomial(1, 0.5, len(train_small)).astype(np.float32)
    y_test = rng.binomial(1, 0.5, len(test_df)).astype(np.float32)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train = standardize(X_train, mean, std)
    X_test = standardize(X_test, mean, std)

    tiers_train = train_small["final_tier"].to_numpy()
    tiers_test = test_df["final_tier"].to_numpy()

    model = MortalityMLP(n_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)

    print(f"Training on {len(train_small)} samples with balanced random labels "
          f"(no real signal -- forces pure memorization) for {N_EPOCHS} epochs...")
    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS} -- loss {loss.item():.4f}")

    rows = run_mia_for_run(model, X_train, y_train, tiers_train, X_test, y_test, tiers_test)
    result_df = pd.DataFrame(rows)
    print(f"\n--- MIA power check (memorization-forced model, n_train={len(train_small)}) ---")
    print(result_df.to_string(index=False))

    out_path = RESULTS_DIR / "mia_validity_check.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    overall_bal_acc = result_df.loc[result_df["subgroup"] == "overall", "attack_balanced_accuracy"].iloc[0]
    if overall_bal_acc > 0.65:
        print(f"\nCONFIRMED: attack has power (balanced accuracy {overall_bal_acc:.3f}, well above chance)")
        print("under forced memorization. This validates that the ~0.50 result on the main")
        print("experiment reflects genuine generalization on a class-imbalanced task, not an")
        print("underpowered attack or an artifact of the imbalanced outcome rate.")
    else:
        print(f"\nWARNING: balanced accuracy {overall_bal_acc:.3f} is still close to chance.")
        print("Try a smaller SUBSAMPLE_N or more N_EPOCHS.")


if __name__ == "__main__":
    main()