"""
Loss-threshold membership inference attack (Yeom et al., 2018 style),
stratified by the project's existing `final_tier` column
("stigma_sensitive" / "non_sensitive").

Slots directly into dp_federated.py's existing structure: reuses the
`global_model` your run_config() already produces per (config, seed), plus
the X_train_all/y_train_all/X_test/y_test arrays and tiers already computed
in main(). No retraining needed -- this reads per-sample loss off models
you've already trained.

Answers: "is it harder to tell whether a stigma_sensitive patient's record
was used in training, than for a non_sensitive patient?" -- the empirical
privacy check to report alongside the epsilon numbers in dp_federated_epsilons.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("results/metrics")


@torch.no_grad()
def per_sample_loss(model, X, y):
    """
    X: numpy float32 array (n, n_features) -- same shape your evaluate() uses.
    y: numpy float32 array (n,)
    Returns: (n,) numpy array of per-sample BCE loss.

    Lower loss = model is more confident/accurate on that sample, which is
    the standard MIA signal that a sample was likely seen during training.
    """
    model.eval()
    logits = model(torch.tensor(X))
    y_t = torch.tensor(y, dtype=torch.float32)
    losses = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
    return losses.numpy()


def loss_threshold_attack(member_losses, nonmember_losses):
    """
    Attacker's optimal fixed threshold on -loss (Yeom-style). A well-protected
    model gives attack_accuracy near 0.5 and attack_auc near 0.5. Values near
    1.0 mean the model leaks membership badly.

    With very few samples (e.g. your stigma_sensitive test n=4), treat the
    result as illustrative, not statistically reliable -- report n_members/
    n_nonmembers alongside it so this is transparent in the paper.
    """
    n_mem, n_non = len(member_losses), len(nonmember_losses)
    if n_mem == 0 or n_non == 0:
        return {"attack_accuracy": np.nan, "attack_auc": np.nan,
                "n_members": n_mem, "n_nonmembers": n_non}

    all_losses = np.concatenate([member_losses, nonmember_losses])
    labels = np.concatenate([np.ones(n_mem), np.zeros(n_non)])
    scores = -all_losses  # lower loss -> higher "is member" score

    best_acc = 0.0
    for t in np.unique(scores):
        preds = (scores >= t).astype(int)
        best_acc = max(best_acc, (preds == labels).mean())

    try:
        attack_auc = roc_auc_score(labels, scores)
    except ValueError:
        attack_auc = np.nan  # only one class present -- degenerate, same failure
                              # mode you already saw in the ROC-AUC subgroup results

    return {"attack_accuracy": best_acc, "attack_auc": attack_auc,
            "n_members": n_mem, "n_nonmembers": n_non}


def run_mia_for_run(global_model, X_train_all, y_train_all, tiers_train,
                     X_test, y_test, tiers_test):
    """
    Runs the MIA overall and split by final_tier, for one already-trained
    global_model from one (config_name, seed) run.

    Returns: list of row dicts, ready to append to a results list the same
    way dp_federated.py already builds all_rows / epsilon_rows.
    """
    train_losses = per_sample_loss(global_model, X_train_all, y_train_all)
    test_losses = per_sample_loss(global_model, X_test, y_test)

    rows = []
    for subgroup, train_mask, test_mask in [
        ("overall", np.ones(len(y_train_all), dtype=bool), np.ones(len(y_test), dtype=bool)),
        ("stigma_sensitive", tiers_train == "stigma_sensitive", tiers_test == "stigma_sensitive"),
        ("non_sensitive", tiers_train == "non_sensitive", tiers_test == "non_sensitive"),
    ]:
        result = loss_threshold_attack(train_losses[train_mask], test_losses[test_mask])
        result["subgroup"] = subgroup
        rows.append(result)
    return rows


# --------------------------------------------------------------------------
# Integration into dp_federated.py -- two small changes needed there:
# --------------------------------------------------------------------------
#
# 1. In run_config(), change the final return line from:
#
#        return test_results, mean_epsilons
#
#    to also hand back the trained model:
#
#        return test_results, mean_epsilons, global_model
#
# 2. In main(), compute tiers_train once (alongside your existing tiers_test):
#
#        tiers_train = train_df["final_tier"].to_numpy()
#
#    Then inside the existing `for seed_idx, seed in enumerate(SEEDS):` loop,
#    update the run_config() call to capture the model, and add the MIA call
#    right after it:
#
#        test_results, mean_epsilons, global_model = run_config(
#            config_name, noise_cfg["stigma"], noise_cfg["non_sensitive"],
#            train_df, X_train_all, y_train_all, X_test, y_test, tiers_test, feature_cols,
#            verbose=verbose,
#        )
#
#        mia_rows = run_mia_for_run(
#            global_model, X_train_all, y_train_all, tiers_train, X_test, y_test, tiers_test
#        )
#        for row in mia_rows:
#            row["stage"] = config_name
#            row["seed"] = seed
#            all_mia_rows.append(row)   # initialize all_mia_rows = [] near all_rows/epsilon_rows
#
# 3. After the existing results_df / epsilon_df save block at the end of main(), add:
#
#        mia_df = pd.DataFrame(all_mia_rows)
#        mia_df.to_csv(RESULTS_DIR / "mia_multiseed.csv", index=False)
#        print(mia_df.groupby(["stage", "subgroup"])[["attack_accuracy", "attack_auc",
#                                                       "n_members", "n_nonmembers"]].agg(["mean", "std"]))
#
# That's it -- no retraining, reuses the global_model your existing loop already produces.
# --------------------------------------------------------------------------


# ----------------------- self-test on synthetic data -----------------------
if __name__ == "__main__":
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)

    n_features = 14  # matches age + gender + 12 vitals, roughly

    class TinyMortalityMLP(nn.Module):
        """Stand-in for MortalityMLP: forward returns (n,) raw logits."""
        def __init__(self, n_features):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

    def make_data(n, seed, n_stigma_frac=0.2):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, n_features)).astype(np.float32)
        y = rng.binomial(1, 0.5, n).astype(np.float32)
        tiers = np.where(rng.random(n) < n_stigma_frac, "stigma_sensitive", "non_sensitive")
        return X, y, tiers

    # Mimics the real scale: n=76 train / n=24 test, small stigma_sensitive subgroup
    X_train, y_train, tiers_train = make_data(76, seed=1)
    X_test, y_test, tiers_test = make_data(24, seed=2)

    model = TinyMortalityMLP(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    for _ in range(50):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt), yt)
        loss.backward()
        opt.step()

    rows = run_mia_for_run(model, X_train, y_train, tiers_train, X_test, y_test, tiers_test)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nSelf-test complete -- module runs end to end against real-shaped data.")