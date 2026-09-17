import pandas as pd
from pathlib import Path
from src.data.vital_itemid_crosswalk import VITAL_ITEMID_CROSSWALK

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-1.4")
CHUNKSIZE = 2_000_000


def main():
    all_itemids = set()
    for itemids in VITAL_ITEMID_CROSSWALK.values():
        all_itemids.update(itemids)

    wanted = {"subject_id", "itemid"}
    total_patients_seen = set()
    itemid_to_patients = {item: set() for item in all_itemids}

    reader = pd.read_csv(
        RAW_DIR / "CHARTEVENTS.csv",
        usecols=lambda c: c.lower() in wanted,
        low_memory=False,
        chunksize=CHUNKSIZE,
    )
    for i, chunk in enumerate(reader):
        chunk.columns = chunk.columns.str.lower()
        total_patients_seen.update(chunk["subject_id"].unique())
        relevant = chunk[chunk["itemid"].isin(all_itemids)]
        for item, group in relevant.groupby("itemid"):
            itemid_to_patients[item].update(group["subject_id"].unique())
        print(f"  scanned chunk {i+1} ({(i+1)*CHUNKSIZE:,} rows)")

    total_patients = len(total_patients_seen)
    print(f"\nTotal patients in CHARTEVENTS: {total_patients}\n")
    print(f"{'feature':<18} {'itemids':<30} {'combined_n':<12} {'combined_pct':<12}")
    print("-" * 75)

    results = []
    for feature, itemids in VITAL_ITEMID_CROSSWALK.items():
        combined_patients = set()
        for item in itemids:
            combined_patients |= itemid_to_patients[item]
        n_combined = len(combined_patients)
        pct_combined = 100 * n_combined / total_patients
        results.append((feature, itemids, n_combined, pct_combined))
        print(f"{feature:<18} {str(itemids):<30} {n_combined:<12} {pct_combined:<12.1f}")

    print("\n--- Per-itemid breakdown (to catch a dead/wrong itemid in the crosswalk) ---")
    for feature, itemids, n_combined, pct_combined in results:
        print(f"\n{feature}:")
        for item in itemids:
            n = len(itemid_to_patients[item])
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