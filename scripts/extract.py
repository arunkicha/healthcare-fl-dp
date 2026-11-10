import gzip, shutil
from pathlib import Path

folder = Path("data/raw/mimic-iii-clinical-database-1.4")
files = ["ADMISSIONS", "PATIENTS", "CHARTEVENTS", "D_ITEMS",
         "DIAGNOSES_ICD", "D_ICD_DIAGNOSES", "ICUSTAYS"]

for f in files:
    src = folder / f"{f}.csv.gz"
    dst = folder / f"{f}.csv"
    print(f"Extracting {f}...")
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
print("Done.")