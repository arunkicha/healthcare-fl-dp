import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-1.4")
PROCESSED_DIR = Path("data/processed")


def main():
    tiers_path = PROCESSED_DIR / "patient_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(
            f"{tiers_path} not found -- run audit_tier_mapping.py first."
        )

    tiers = pd.read_csv(tiers_path)

    adm = pd.read_csv(RAW_DIR / "ADMISSIONS.csv")
    adm.columns = adm.columns.str.lower()
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