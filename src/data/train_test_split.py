"""
Splits patients into train/test sets, stratified by (tier x outcome) so the
test set has representation of both outcome classes in BOTH tiers -- critical
for the stigma_sensitive tier given how few patients (22) it has.

Splits by subject_id (patient), NOT hadm_id, so a patient's admissions all
land in the same split -- no leakage.

Patient-level outcome for stratification purposes = died in ANY of their
admissions (same aggregation logic as audit_class_balance.py).

Run from repo root:
    python -m src.data.train_test_split
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# Keep in sync with EXCLUDED_PATIENTS in dataset.py / partition_clients.py
EXCLUDED_PATIENTS = [41976]


def main():
    modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
    modeling_table = modeling_table[~modeling_table["subject_id"].isin(EXCLUDED_PATIENTS)]

    # Patient-level outcome: died in any admission
    patient_outcome = (
        modeling_table.groupby("subject_id")["hospital_expire_flag"].max().reset_index()
    )
    patient_tier = (
        modeling_table.groupby("subject_id")["final_tier"].first().reset_index()
    )
    patients = patient_outcome.merge(patient_tier, on="subject_id")
    patients["stratum"] = patients["final_tier"] + "_" + patients["hospital_expire_flag"].astype(str)

    rng = np.random.default_rng(RANDOM_SEED)
    split_rows = []

    print("--- Stratum sizes and split ---")
    for stratum, group in patients.groupby("stratum"):
        subject_ids = group["subject_id"].to_numpy().copy()
        rng.shuffle(subject_ids)

        n = len(subject_ids)
        n_test = max(1, round(n * TEST_FRACTION)) if n >= 2 else 0
        # never let a stratum go fully into test (train needs at least 1 too)
        n_test = min(n_test, n - 1) if n >= 2 else 0

        test_ids = subject_ids[:n_test]
        train_ids = subject_ids[n_test:]

        print(f"{stratum}: n={n}, train={len(train_ids)}, test={len(test_ids)}")

        for sid in train_ids:
            split_rows.append({"subject_id": sid, "split": "train"})
        for sid in test_ids:
            split_rows.append({"subject_id": sid, "split": "test"})

    split_df = pd.DataFrame(split_rows)
    merged = patients.merge(split_df, on="subject_id")

    print("\n--- Final split: tier x outcome counts (patient-level) ---")
    print(merged.groupby(["split", "final_tier", "hospital_expire_flag"]).size(), "\n")

    print("--- Sanity check: does test set have both outcome classes for both tiers? ---")
    problems = []
    for tier in merged["final_tier"].unique():
        test_sub = merged[(merged["split"] == "test") & (merged["final_tier"] == tier)]
        n_classes = test_sub["hospital_expire_flag"].nunique()
        status = "OK" if n_classes == 2 else "PROBLEM -- single class in test!"
        print(f"{tier} (test): {n_classes} classes present -- {status}")
        if n_classes < 2:
            problems.append(tier)

    out_path = PROCESSED_DIR / "train_test_split.csv"
    split_df.to_csv(out_path, index=False)
    print(f"\nSaved patient-level split to: {out_path}")
    print("Join this onto modeling_table.csv by subject_id to get admission-level train/test sets.")

    if problems:
        print(f"\nWARNING: {problems} has a single-class test set -- "
              f"consider lowering TEST_FRACTION or reconsidering evaluation approach for this tier.")


if __name__ == "__main__":
    main()