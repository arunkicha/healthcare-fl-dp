import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/mimic-iii-clinical-database-demo-1.4")

diag = pd.read_csv(RAW_DIR / "DIAGNOSES_ICD.csv", nrows=5)
print("DIAGNOSES_ICD.csv columns:")
print(list(diag.columns))
print(diag.head(), "\n")

d_diag = pd.read_csv(RAW_DIR / "D_ICD_DIAGNOSES.csv", nrows=5)
print("D_ICD_DIAGNOSES.csv columns:")
print(list(d_diag.columns))
print(d_diag.head())