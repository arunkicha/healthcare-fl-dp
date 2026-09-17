from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("results/metrics")


@torch.no_grad()
def per_sample_loss(model, X, y):
    model.eval()
    logits = model(torch.tensor(X))
    y_t = torch.tensor(y, dtype=torch.float32)
    losses = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
    return losses.numpy()


def best_balanced_accuracy(scores, labels, n_mem, n_non):
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    i = np.arange(1, len(scores) + 1)
    fp = i - cum_tp
    tpr = cum_tp / n_mem
    tnr = (n_non - fp) / n_non
    balanced_acc = (tpr + tnr) / 2
    return max(balanced_acc.max(), 0.5)


def loss_threshold_attack(member_losses, nonmember_losses):
    n_mem, n_non = len(member_losses), len(nonmember_losses)
    if n_mem == 0 or n_non == 0:
        return {
            "attack_balanced_accuracy": np.nan,
            "attack_auc": np.nan,
            "n_members": n_mem,
            "n_nonmembers": n_non,
        }

    all_losses = np.concatenate([member_losses, nonmember_losses])
    labels = np.concatenate([np.ones(n_mem), np.zeros(n_non)])
    scores = -all_losses

    best_bal_acc = best_balanced_accuracy(scores, labels, n_mem, n_non)

    try:
        attack_auc = roc_auc_score(labels, scores)
    except ValueError:
        attack_auc = np.nan

    return {
        "attack_balanced_accuracy": best_bal_acc,
        "attack_auc": attack_auc,
        "n_members": n_mem,
        "n_nonmembers": n_non,
    }


def run_mia_for_run(global_model, X_train_all, y_train_all, tiers_train,
                     X_test, y_test, tiers_test):
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


if __name__ == "__main__":
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)

    n_features = 14

    class TinyMortalityMLP(nn.Module):
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