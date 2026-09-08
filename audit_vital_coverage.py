"""
Validates VITAL_ITEMID_CROSSWALK against real CHARTEVENTS.csv coverage:
for each vital, reports patient coverage using ONLY the CareVue itemid,
ONLY the MetaVision itemid, and the COMBINED (coalesced) coverage.

This confirms the crosswalk pairings are correct and complete BEFORE
they're used in dataset.py's feature extraction -- a wrong or missing
itemid here would silently produce a feature that's mostly null for
half the patients.

Run from repo root:
    python audit_vital_coverage.py
"""

import pandas as pd
from pathlib import Path
from src.data.vital_itemid_crosswalk import VITAL_ITEMID_CROSSWALK

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")


def main():
    usecols = ["subject_id", "itemid"]
    chart = pd.read_csv(RAW_DIR / "CHARTEVENTS.csv", usecols=usecols, low_memory=False)
    total_patients = chart["subject_id"].nunique()

    print(f"Total patients in CHARTEVENTS: {total_patients}\n")
    print(f"{'feature':<18} {'itemids':<30} {'combined_n':<12} {'combined_pct':<12}")
    print("-" * 75)

    results = []
    for feature, itemids in VITAL_ITEMID_CROSSWALK.items():
        subset = chart[chart["itemid"].isin(itemids)]
        n_combined = subset["subject_id"].nunique()
        pct_combined = 100 * n_combined / total_patients
        results.append((feature, itemids, n_combined, pct_combined))
        print(f"{feature:<18} {str(itemids):<30} {n_combined:<12} {pct_combined:<12.1f}")

    print("\n--- Per-itemid breakdown (to catch a dead/wrong itemid in the crosswalk) ---")
    for feature, itemids, n_combined, pct_combined in results:
        print(f"\n{feature}:")
        for item in itemids:
            n = chart.loc[chart["itemid"] == item, "subject_id"].nunique()
            flag = "  <- contributes 0, check this itemid" if n == 0 else ""
            print(f"    itemid {item}: {n} patients{flag}")

    print("\n--- Features below 90% combined coverage (may need an additional itemid) ---")
    weak = [r for r in results if r[3] < 90]
    if weak:
        for feature, itemids, n_combined, pct_combined in weak:
            print(f"{feature}: {pct_combined:.1f}%")
    else:
        print("None -- all features reach >= 90% combined coverage.")


if __name__ == "__main__":
    main()