"""
Checks how many DIAGNOSES_ICD rows match ICD-9 codes in our stigma-sensitivity mapping.

Two-tier design (matches project goal: privacy-first for stigma-sensitive conditions,
utility-first for everything else):
  - stigma_sensitive: HIV/AIDS, substance use, mental health conditions
  - non_sensitive: everything else (including clinically severe but non-stigmatizing
    conditions like heart disease/diabetes, which are reported separately below for
    context but do NOT get their own privacy tier)

Note: this Kaggle mirror of the MIMIC-III demo uses lowercase snake_case column names.

Run from repo root:
    python audit_icd_coverage.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")

# --- Stigma-sensitive tier: the ONLY tier that gets tightened privacy ---
STIGMA_SENSITIVE_PREFIXES = [
    "042",              # HIV/AIDS
    "291", "292",       # substance-induced mental disorders
    "303", "304", "305",  # alcohol/drug dependence & abuse
    "295", "296", "298", "300", "301", "311",  # psychotic/mood/anxiety/personality disorders
]

# --- Informational only: NOT a separate privacy tier, just reported for context ---
CLINICALLY_SEVERE_NONSTIGMA_PREFIXES = [
    "410", "411", "412", "413", "414",  # ischemic heart disease
    "250",                               # diabetes mellitus
]


def load_diagnoses():
    diag = pd.read_csv(RAW_DIR / "DIAGNOSES_ICD.csv")
    d_diag = pd.read_csv(RAW_DIR / "D_ICD_DIAGNOSES.csv")

    diag["icd9_code"] = diag["icd9_code"].astype(str).str.strip()
    d_diag["icd9_code"] = d_diag["icd9_code"].astype(str).str.strip()

    merged = diag.merge(d_diag, on="icd9_code", how="left")
    return merged


def matches_any(code: str, prefixes: list) -> bool:
    padded = code.zfill(3)
    return any(code.startswith(p) or padded.startswith(p) for p in prefixes)


def assign_tier(code: str) -> str:
    if matches_any(code, STIGMA_SENSITIVE_PREFIXES):
        return "stigma_sensitive"
    return "non_sensitive"


def main():
    df = load_diagnoses()
    df["tier"] = df["icd9_code"].apply(assign_tier)
    df["is_clinically_severe_nonstigma"] = df["icd9_code"].apply(
        lambda c: matches_any(c, CLINICALLY_SEVERE_NONSTIGMA_PREFIXES)
    )

    total_rows = len(df)
    total_patients = df["subject_id"].nunique()

    print(f"Total diagnosis rows: {total_rows}")
    print(f"Total unique patients (with any diagnosis): {total_patients}\n")

    print("--- Rows per tier (privacy-relevant split) ---")
    print(df["tier"].value_counts(), "\n")

    print("--- Unique patients per tier ---")
    for tier in ["stigma_sensitive", "non_sensitive"]:
        n = df.loc[df["tier"] == tier, "subject_id"].nunique()
        print(f"{tier}: {n} patients")
    print()

    print("--- For context only: clinically severe but non-stigma conditions ---")
    n_severe = df.loc[df["is_clinically_severe_nonstigma"], "subject_id"].nunique()
    print(f"Patients with heart disease / diabetes codes: {n_severe} "
          f"(these stay in non_sensitive tier, informational only)\n")

    print("--- Coverage ---")
    mapped_pct = 100 * (df["tier"] == "stigma_sensitive").mean()
    print(f"{mapped_pct:.1f}% of diagnosis rows are stigma-sensitive\n")

    print("--- Codes matched under stigma_sensitive (sanity check they're real) ---")
    hs = df[df["tier"] == "stigma_sensitive"]
    print(hs[["icd9_code", "short_title"]].drop_duplicates().sort_values("icd9_code"))


if __name__ == "__main__":
    main()