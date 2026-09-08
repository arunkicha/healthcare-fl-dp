"""
Cross-checks the client partition (from partition_clients.py) against outcome
balance, per client AND per tier. With only 2-3 stigma_sensitive patients per
client, it's realistic for a client's stigma_sensitive subgroup to collapse
to a single outcome class (all survived or all died) -- this would make local
training/eval for that tier meaningless on that client, so we need to know
about it before running FL training, not after.

Run from repo root (after dataset.py and partition_clients.py have both run):
    python audit_client_distribution.py
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def main():
    modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
    assignments = pd.read_csv(PROCESSED_DIR / "client_assignments.csv")

    # modeling_table is admission-level; assignments is patient-level -- merge on subject_id
    df = modeling_table.merge(
        assignments[["subject_id", "client_id"]], on="subject_id", how="left"
    )

    n_missing = df["client_id"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} admissions have no client assignment "
              f"(subject_id mismatch) -- check before training.\n")

    print("--- Admissions per client ---")
    print(df.groupby("client_id").size(), "\n")

    print("--- Outcome (died) distribution per client, per tier ---")
    grouped = (
        df.groupby(["client_id", "final_tier"])["hospital_expire_flag"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n_admissions", "sum": "n_died"})
    )
    grouped["mortality_pct"] = (100 * grouped["n_died"] / grouped["n_admissions"]).round(1)
    print(grouped, "\n")

    print("--- Sanity check: any client x tier combo with a single outcome class only? ---")
    problems = []
    for (client_id, tier), sub in df.groupby(["client_id", "final_tier"]):
        n_classes = sub["hospital_expire_flag"].nunique()
        if n_classes < 2:
            problems.append((client_id, tier, len(sub), sub["hospital_expire_flag"].iloc[0]))

    if problems:
        print("FOUND single-class client x tier combos (local eval will be degenerate for these):")
        for client_id, tier, n, only_class in problems:
            label = "all survived" if only_class == 0 else "all died"
            print(f"  client {client_id}, {tier}: {n} admissions, {label}")
    else:
        print("OK: every client x tier combination has both outcome classes present.")

    print("\n--- Vital feature missingness per client (flag if any client is much worse) ---")
    feature_cols = [c for c in df.columns if c in [
        "heart_rate", "resp_rate", "spo2", "temp_fahrenheit", "nbp_systolic",
        "nbp_diastolic", "nbp_mean", "glucose", "gcs_eye", "gcs_verbal",
        "gcs_motor", "weight_kg",
    ]]
    missing_by_client = df.groupby("client_id")[feature_cols].apply(
        lambda g: g.isna().mean().mean() * 100
    )
    print(missing_by_client.round(1).rename("avg_pct_missing"))


if __name__ == "__main__":
    main()