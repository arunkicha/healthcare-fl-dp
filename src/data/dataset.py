"""
Builds the final modeling table: one row per hospital admission (hadm_id), with:
  - demographics: gender, age at admission
  - outcome label: hospital_expire_flag (in-hospital mortality)
  - privacy tier: stigma_sensitive / non_sensitive (from audit_tier_mapping.py)
  - 12 vital-sign features, aggregated per admission using the CareVue/MetaVision
    crosswalk validated in audit_vital_coverage.py

Assumes patient_tiers.csv already exists (run audit_tier_mapping.py first).

Run from repo root:
    python -m src.data.dataset
"""

import pandas as pd
from pathlib import Path
from src.data.vital_itemid_crosswalk import VITAL_ITEMID_CROSSWALK

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")
PROCESSED_DIR = Path("data/processed")

MAX_PLAUSIBLE_AGE = 90  # MIMIC convention: ages >89 are obscured via shifted DOB

# Excluded as a non-representative outlier: subject_id 41976 has 15 admissions
# (next highest stigma_sensitive patient has 3) and would dominate FedAvg
# aggregation for the stigma_sensitive tier if included. Confirmed via
# check_admission_outliers.py -- same exclusion logic as the mule-detection
# project's single distorting-outlier bank client.
EXCLUDED_PATIENTS = [41976]


def load_demographics_and_outcome():
    patients = pd.read_csv(RAW_DIR / "PATIENTS.csv", parse_dates=["dob"])
    admissions = pd.read_csv(
        RAW_DIR / "ADMISSIONS.csv", parse_dates=["admittime", "dischtime"]
    )

    df = admissions.merge(
        patients[["subject_id", "gender", "dob"]], on="subject_id", how="left"
    )

    age_years = (df["admittime"] - df["dob"]).dt.days / 365.25
    df["age"] = age_years.clip(upper=MAX_PLAUSIBLE_AGE).round(1)

    return df[
        [
            "subject_id",
            "hadm_id",
            "gender",
            "age",
            "hospital_expire_flag",
        ]
    ]


def load_tiers():
    tiers_path = PROCESSED_DIR / "patient_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(
            f"{tiers_path} not found -- run audit_tier_mapping.py first."
        )
    return pd.read_csv(tiers_path)


def build_itemid_to_feature_map():
    mapping = {}
    for feature, itemids in VITAL_ITEMID_CROSSWALK.items():
        for item in itemids:
            mapping[item] = feature
    return mapping


def load_vital_features():
    usecols = ["subject_id", "hadm_id", "itemid", "valuenum"]
    chart = pd.read_csv(RAW_DIR / "CHARTEVENTS.csv", usecols=usecols, low_memory=False)

    itemid_to_feature = build_itemid_to_feature_map()
    all_itemids = list(itemid_to_feature.keys())

    chart = chart[chart["itemid"].isin(all_itemids)].copy()
    chart["feature"] = chart["itemid"].map(itemid_to_feature)
    chart = chart.dropna(subset=["valuenum"])

    # Mean value per admission per feature -- simple, consistent aggregation
    # across all 12 features (a few, like GCS components, are ordinal scores
    # where mean is a coarse summary; revisit per-feature aggregation choice
    # later if model performance suggests it's worth the extra complexity).
    agg = (
        chart.groupby(["hadm_id", "feature"])["valuenum"]
        .mean()
        .reset_index()
    )

    wide = agg.pivot(index="hadm_id", columns="feature", values="valuenum").reset_index()
    wide.columns.name = None

    # Ensure all 12 expected feature columns exist even if one had zero rows
    for feature in VITAL_ITEMID_CROSSWALK.keys():
        if feature not in wide.columns:
            wide[feature] = pd.NA

    return wide


def main():
    demo_outcome = load_demographics_and_outcome()
    demo_outcome = demo_outcome[~demo_outcome["subject_id"].isin(EXCLUDED_PATIENTS)]

    tiers = load_tiers()
    tiers = tiers[~tiers["subject_id"].isin(EXCLUDED_PATIENTS)]

    vitals = load_vital_features()

    df = demo_outcome.merge(tiers, on="subject_id", how="left")
    df = df.merge(vitals, on="hadm_id", how="left")

    n_missing_tier = df["final_tier"].isna().sum()
    if n_missing_tier:
        print(f"WARNING: {n_missing_tier} admissions have no tier match "
              f"(subject_id mismatch with patient_tiers.csv) -- check before training.\n")

    print(f"Final modeling table shape: {df.shape}")
    print(f"Admissions: {len(df)} | Unique patients: {df['subject_id'].nunique()}\n")

    print("--- Tier distribution (admission-level) ---")
    print(df["final_tier"].value_counts(), "\n")

    print("--- Outcome distribution ---")
    print(df["hospital_expire_flag"].value_counts(), "\n")

    print("--- Missingness per vital feature ---")
    feature_cols = list(VITAL_ITEMID_CROSSWALK.keys())
    missing_pct = df[feature_cols].isna().mean().sort_values(ascending=False) * 100
    print(missing_pct.round(1))

    out_path = PROCESSED_DIR / "modeling_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved modeling table to: {out_path}")


if __name__ == "__main__":
    main()