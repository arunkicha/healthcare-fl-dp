# Data

This folder is intentionally empty in the repository. MIMIC-III is restricted-access
data and must not be committed or shared.

## Get the data

1. Complete CITI "Data or Specimens Only Research" training.
2. Sign the PhysioNet credentialed data use agreement.
3. Download MIMIC-III Clinical Database v1.4: https://physionet.org/content/mimiciii/1.4/

## Where to put it

```
data/
  raw/
    mimic-iii-clinical-database-1.4/     <- downloaded .csv.gz files go here
  processed/                             <- created by python -m src.data.dataset
```

Files used by this project: ADMISSIONS, PATIENTS, ICUSTAYS, CHARTEVENTS, D_ITEMS,
DIAGNOSES_ICD, D_ICD_DIAGNOSES. Run `python -m scripts.extract` to unzip them, and
optionally `python -m scripts.verify` to check them against the checksums that ship
with the download.

Everything under `data/raw/` and `data/processed/` is git-ignored.
