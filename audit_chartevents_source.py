"""
Confirms whether the ~50/99 patient coverage ceiling seen in
audit_chartevents_coverage.py is caused by the CareVue vs MetaVision
ICU data-source split (a known MIMIC-III quirk: these two systems use
different itemids for the SAME vital sign, e.g. Heart Rate = 211 under
CareVue, 220045 under MetaVision).

Checks:
  1. Does ICUSTAYS.csv (or another table) expose a dbsource column?
  2. Does the classic CareVue Heart Rate itemid (211) exist in this data,
     and does it cover the OTHER ~50 patients that itemid 220045 doesn't?

Run from repo root:
    python audit_chartevents_source_split.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")


def main():
    # --- Check 1: dbsource column, if present ---
    icustays_path = RAW_DIR / "ICUSTAYS.csv"
    if icustays_path.exists():
        icu = pd.read_csv(icustays_path)
        print("ICUSTAYS.csv columns:", list(icu.columns))
        if "dbsource" in icu.columns:
            print("\n--- dbsource counts (patients) ---")
            print(icu.groupby("dbsource")["subject_id"].nunique())
        print()

    # --- Check 2: classic CareVue Heart Rate itemid vs MetaVision Heart Rate itemid ---
    usecols = ["subject_id", "itemid"]
    chart = pd.read_csv(RAW_DIR / "CHARTEVENTS.csv", usecols=usecols, low_memory=False)

    carevue_hr_patients = set(chart.loc[chart["itemid"] == 211, "subject_id"].unique())
    metavision_hr_patients = set(chart.loc[chart["itemid"] == 220045, "subject_id"].unique())

    print(f"Patients with CareVue Heart Rate (itemid 211): {len(carevue_hr_patients)}")
    print(f"Patients with MetaVision Heart Rate (itemid 220045): {len(metavision_hr_patients)}")
    overlap = carevue_hr_patients & metavision_hr_patients
    print(f"Overlap between the two: {len(overlap)}")
    print(f"Combined (union) coverage: {len(carevue_hr_patients | metavision_hr_patients)} patients")


if __name__ == "__main__":
    main()