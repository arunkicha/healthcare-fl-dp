import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

modeling_table = pd.read_csv(PROCESSED_DIR / "modeling_table.csv")
assignments = pd.read_csv(PROCESSED_DIR / "client_assignments.csv")

df = modeling_table.merge(assignments[["subject_id", "client_id"]], on="subject_id", how="left")

print("--- Admission count per stigma_sensitive patient, sorted descending ---")
stigma = df[df["final_tier"] == "stigma_sensitive"]
counts = stigma.groupby(["subject_id", "client_id"]).size().reset_index(name="n_admissions")
counts = counts.sort_values("n_admissions", ascending=False)
print(counts.to_string(index=False))