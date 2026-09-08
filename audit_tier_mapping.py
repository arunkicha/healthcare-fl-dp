"""
Resolves per-patient privacy tier from DIAGNOSES_ICD.csv using the two-tier design:
  - stigma_sensitive: patient has ANY diagnosis in the stigma-sensitive code list
                       (HIV/AIDS, substance use, mental health) -> tighter DP budget
  - non_sensitive:    everyone else (including heart disease/diabetes and all
                       generic ICU comorbidities) -> looser DP budget, utility-first

This is a strict two-category split: "stigma-sensitive wins" if present at all;
there is no medium/third tier by design (see README for rationale -- kept as a
noted future-work extension, not implemented now given dataset size).

Run from repo root:
    python audit_tier_mapping.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STIGMA_SENSITIVE_PREFIXES = [
    "042",
    "291", "292",
    "303", "304", "305",
    "295", "296", "298", "300", "301", "311",
]


def is_stigma_sensitive(code: str) -> bool:
    padded = code.zfill(3)
    return any(code.startswith(p) or padded.startswith(p) for p in STIGMA_SENSITIVE_PREFIXES)


def main():
    diag = pd.read_csv(RAW_DIR / "DIAGNOSES_ICD.csv")
    diag["icd9_code"] = diag["icd9_code"].astype(str).str.strip()
    diag["is_stigma_sensitive"] = diag["icd9_code"].apply(is_stigma_sensitive)

    # Per patient: do they have ANY stigma-sensitive diagnosis at all?
    patient_flags = (
        diag.groupby("subject_id")["is_stigma_sensitive"]
        .any()
        .reset_index()
    )
    patient_flags["final_tier"] = patient_flags["is_stigma_sensitive"].map(
        {True: "stigma_sensitive", False: "non_sensitive"}
    )

    print("--- Final per-patient tier counts (two-tier design) ---")
    print(patient_flags["final_tier"].value_counts(), "\n")

    n_stigma = (patient_flags["final_tier"] == "stigma_sensitive").sum()
    n_total = len(patient_flags)
    print(f"{n_stigma} / {n_total} patients ({100*n_stigma/n_total:.1f}%) "
          f"are stigma_sensitive -> get the tighter DP budget")
    print(f"{n_total - n_stigma} / {n_total} patients "
          f"({100*(n_total-n_stigma)/n_total:.1f}%) are non_sensitive -> looser DP budget\n")

    out_path = PROCESSED_DIR / "patient_tiers.csv"
    patient_flags[["subject_id", "final_tier"]].to_csv(out_path, index=False)
    print(f"Saved resolved per-patient tiers to: {out_path}")


if __name__ == "__main__":
    main()