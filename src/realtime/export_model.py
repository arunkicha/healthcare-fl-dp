"""Train one federated DP model on the full training split and export it.

Writes models/<config>/{model.pt, preprocessing.json, model_card.json}.
Run:  python -m src.realtime.export_model --config tiered
      python -m src.realtime.export_model --config noise0_sanity_check   (comparison model)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from src.models.baseline import VITAL_FEATURES
from src.models.dp_federated import CONFIGS, build_features, load_data, run_config, standardize
from src.privacy.tiered_dp import DELTA, MAX_GRAD_NORM

MODEL_DIR = Path("models")


def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiered", choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_data()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    medians = train_df[["age"] + VITAL_FEATURES].median(numeric_only=True)
    X_train, feature_cols = build_features(train_df, medians)
    X_test, _ = build_features(test_df, medians)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train, X_test = standardize(X_train, mean, std), standardize(X_test, mean, std)

    y_train = train_df["hospital_expire_flag"].to_numpy(dtype=np.float32)
    y_test = test_df["hospital_expire_flag"].to_numpy(dtype=np.float32)
    tiers_test = test_df["final_tier"].to_numpy()

    noise = CONFIGS[args.config]
    test_results, epsilons, model = run_config(
        args.config, noise["stigma"], noise["non_sensitive"], train_df, X_train, y_train,
        X_test, y_test, tiers_test, feature_cols,
    )

    out = MODEL_DIR / args.config
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

    preprocessing = {
        "feature_cols": feature_cols,
        "medians": {c: float(medians[c]) for c in ["age"] + VITAL_FEATURES},
        "mean": mean.flatten().tolist(),
        "std": std.flatten().tolist(),
    }
    card = {
        "config": args.config,
        "noise_multiplier_stigma_sensitive": noise["stigma"],
        "noise_multiplier_non_sensitive": noise["non_sensitive"],
        "max_grad_norm": MAX_GRAD_NORM,
        "delta": DELTA,
        "epsilon_per_round": epsilons,
        "note": "epsilon_per_round covers one round of local training; cumulative epsilon over all rounds is larger",
        "test_metrics": test_results,
        "n_train": int(len(train_df)),
        "exported": date.today().isoformat(),
        "seed": args.seed,
    }
    (out / "preprocessing.json").write_text(json.dumps(to_py(preprocessing), indent=2))
    (out / "model_card.json").write_text(json.dumps(to_py(card), indent=2))
    print(f"Exported model to {out}")


if __name__ == "__main__":
    main()
