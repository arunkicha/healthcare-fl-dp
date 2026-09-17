import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-1.4")


def main():
    icustays_path = RAW_DIR / "ICUSTAYS.csv"
    if icustays_path.exists():
        icu = pd.read_csv(icustays_path)
        icu.columns = icu.columns.str.lower()
        print("ICUSTAYS.csv columns:", list(icu.columns))
        if "dbsource" in icu.columns:
            print("\n--- dbsource counts (patients) ---")
            print(icu.groupby("dbsource")["subject_id"].nunique())
        print()

    wanted = {"subject_id", "itemid"}
    chart = pd.read_csv(
        RAW_DIR / "CHARTEVENTS.csv",
        usecols=lambda c: c.lower() in wanted,
        low_memory=False,
    )
    chart.columns = chart.columns.str.lower()

    carevue_hr_patients = set(chart.loc[chart["itemid"] == 211, "subject_id"].unique())
    metavision_hr_patients = set(chart.loc[chart["itemid"] == 220045, "subject_id"].unique())

    print(f"Patients with CareVue Heart Rate (itemid 211): {len(carevue_hr_patients)}")
    print(f"Patients with MetaVision Heart Rate (itemid 220045): {len(metavision_hr_patients)}")
    overlap = carevue_hr_patients & metavision_hr_patients
    print(f"Overlap between the two: {len(overlap)}")
    print(f"Combined (union) coverage: {len(carevue_hr_patients | metavision_hr_patients)} patients")


if __name__ == "__main__":
    main()