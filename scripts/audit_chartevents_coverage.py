import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-1.4")


def main():
    print("Loading CHARTEVENTS.csv (this is the largest file, may take a moment)...")
    wanted = {"subject_id", "hadm_id", "itemid", "valuenum", "valueuom"}
    chart = pd.read_csv(
        RAW_DIR / "CHARTEVENTS.csv",
        usecols=lambda c: c.lower() in wanted,
        low_memory=False,
    )
    chart.columns = chart.columns.str.lower()

    d_items = pd.read_csv(RAW_DIR / "D_ITEMS.csv")
    d_items.columns = d_items.columns.str.lower()

    total_patients = chart["subject_id"].nunique()
    print(f"Total patients in CHARTEVENTS: {total_patients}")
    print(f"Total chartevent rows: {len(chart)}\n")

    coverage = (
        chart.groupby("itemid")["subject_id"]
        .nunique()
        .reset_index(name="n_patients")
    )
    coverage["pct_patients"] = 100 * coverage["n_patients"] / total_patients

    coverage = coverage.merge(
        d_items[["itemid", "label", "category", "unitname"]],
        on="itemid",
        how="left",
    )

    coverage = coverage.sort_values("n_patients", ascending=False)

    print("--- Top 40 itemids by patient coverage ---")
    cols = ["itemid", "label", "category", "unitname", "n_patients", "pct_patients"]
    print(coverage.head(40)[cols].to_string(index=False))

    print("\n--- Coverage thresholds ---")
    for pct in [90, 75, 50, 25]:
        n_items = (coverage["pct_patients"] >= pct).sum()
        print(f"itemids present in >= {pct}% of patients: {n_items}")

    out_path = Path("data/processed/chartevents_itemid_coverage.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_path, index=False)
    print(f"\nFull coverage table saved to: {out_path}")
    print("Review this CSV to pick the final feature itemid list for dataset.py.")


if __name__ == "__main__":
    main()