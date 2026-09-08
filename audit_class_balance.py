"""
Cross-checks the stigma_sensitive / non_sensitive tier split against the outcome
label (hospital_expire_flag = in-hospital mortality) that the model will actually
predict. Critical check: confirms neither tier collapses to a single class
(all-survived or all-died), which would make ROC-AUC/PR-AUC meaningless for that
subgroup.

ADMISSIONS.csv is per-admission; patient_tiers.csv (from audit_tier_mapping.py)
is per-patient. Outcome is aggregated to patient level: died = 1 if ANY of their
admissions has hospital_expire_flag == 1.

Run from repo root (after audit_tier_mapping.py has produced data/processed/patient_tiers.csv):
    python audit_class_balance.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")
PROCESSED_DIR = Path("data/processed")


def main():
    tiers_path = PROCESSED_DIR / "patient_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(
            f"{tiers_path} not found -- run audit_tier_mapping.py first."
        )

    tiers = pd.read_csv(tiers_path)

    adm = pd.read_csv(RAW_DIR / "ADMISSIONS.csv")
    # Aggregate to patient level: died in any admission -> outcome = 1
    outcome = (
        adm.groupby("subject_id")["hospital_expire_flag"]
        .max()
        .reset_index()
        .rename(columns={"hospital_expire_flag": "died"})
    )

    merged = tiers.merge(outcome, on="subject_id", how="left")

    n_missing = merged["died"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} patients in patient_tiers.csv have no matching "
              f"ADMISSIONS.csv record -- check for subject_id mismatches.\n")

    print("--- Patient count per tier ---")
    print(merged["final_tier"].value_counts(), "\n")

    print("--- Outcome (died) count per tier ---")
    print(merged.groupby("final_tier")["died"].value_counts().sort_index(), "\n")

    print("--- Mortality rate per tier ---")
    rates = merged.groupby("final_tier")["died"].mean()
    for tier, rate in rates.items():
        n = (merged["final_tier"] == tier).sum()
        n_died = int(merged.loc[merged["final_tier"] == tier, "died"].sum())
        print(f"{tier}: {n_died}/{n} died ({100*rate:.1f}%)")
    print()

    print("--- Sanity check: does either tier collapse to a single class? ---")
    for tier in merged["final_tier"].unique():
        sub = merged.loc[merged["final_tier"] == tier, "died"]
        n_classes = sub.nunique()
        status = "OK -- both classes present" if n_classes == 2 else "PROBLEM -- single class only!"
        print(f"{tier}: {n_classes} distinct outcome value(s) -- {status}")

    out_path = PROCESSED_DIR / "patient_tiers_with_outcome.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged tier+outcome table to: {out_path}")


if __name__ == "__main__":
    main()