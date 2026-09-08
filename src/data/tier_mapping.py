"""
Maps ICD-9 codes (from DIAGNOSES_ICD.csv / D_ICD_DIAGNOSES.csv) to sensitivity tiers.

Tier definitions (draft, confirm against audit_icd_coverage.py output before finalizing):
  - high-stigma: HIV/AIDS (042), substance use (291-292, 303-305), mental health (290-319 subset)
  - low-stigma: cardiac (410-414), diabetes (250.x)
"""
