import pandas as pd
from pathlib import Path
from src.data.vital_itemid_crosswalk import VITAL_ITEMID_CROSSWALK

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-1.4")
PROCESSED_DIR = Path("data/processed")
CHUNKSIZE = 2_000_000

MAX_PLAUSIBLE_AGE = 90

EXCLUDED_PATIENTS = [41976]


def load_demographics_and_outcome():
    patients = pd.read_csv(RAW_DIR / "PATIENTS.csv")
    patients.columns = patients.columns.str.lower()
    patients["dob"] = pd.to_datetime(patients["dob"])

    admissions = pd.read_csv(RAW_DIR / "ADMISSIONS.csv")
    admissions.columns = admissions.columns.str.lower()
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    df = admissions.merge(
        patients[["subject_id", "gender", "dob"]], on="subject_id", how="left"
    )

    age_years = (df["admittime"] - df["dob"]).dt.days / 365.25
    df["age"] = age_years.clip(upper=MAX_PLAUSIBLE_AGE).round(1)

    return df[["subject_id", "hadm_id", "gender", "age", "hospital_expire_flag"]]


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
    itemid_to_feature = build_itemid_to_feature_map()
    all_itemids = set(itemid_to_feature.keys())
    wanted = {"subject_id", "hadm_id", "itemid", "valuenum"}

    accum = None
    reader = pd.read_csv(
        RAW_DIR / "CHARTEVENTS.csv",
        usecols=lambda c: c.lower() in wanted,
        low_memory=False,
        chunksize=CHUNKSIZE,
    )
    for i, chunk in enumerate(reader):
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk[chunk["itemid"].isin(all_itemids)].copy()
        if chunk.empty:
            print(f"  scanned chunk {i+1}, no relevant rows")
            continue
        chunk["feature"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["valuenum"])
        partial = (
            chunk.groupby(["hadm_id", "feature"])["valuenum"]
            .agg(sum="sum", count="count")
        )
        accum = partial if accum is None else accum.add(partial, fill_value=0)
        print(f"  scanned chunk {i+1} ({(i+1)*CHUNKSIZE:,} rows)")

    accum["mean"] = accum["sum"] / accum["count"]
    wide = accum["mean"].unstack("feature").reset_index()
    wide.columns.name = None

    for feature in VITAL_ITEMID_CROSSWALK.keys():
        if feature not in wide.columns:
            wide[feature] = pd.NA

    return wide


def main():
    demo_outcome = load_demographics_and_outcome()
    demo_outcome = demo_outcome[~demo_outcome["subject_id"].isin(EXCLUDED_PATIENTS)]

    tiers = load_tiers()
    tiers = tiers[~tiers["subject_id"].isin(EXCLUDED_PATIENTS)]

    print("Extracting vital-sign features from CHARTEVENTS (chunked)...")
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