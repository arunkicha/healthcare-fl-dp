"""
Maps each vital sign / feature to its CareVue + MetaVision itemid pair.
MIMIC-III's two ICU source systems use different itemids for the same
measurement (confirmed via audit_chartevents_source_split.py: zero itemid
overlap between systems, ~98-99% combined coverage vs ~50% for either alone).

These pairings are the standard, widely-used MIMIC-III concept mappings
(matching the itemid crosswalks used in the mimic-code reference repo's
pivoted vital-sign views) -- validate them against this demo's actual
coverage using audit_vital_coverage.py before trusting them blindly.

Structure: feature_name -> list of itemids to coalesce (take first non-null
per admission/charttime across whichever itemids are present for that patient).
"""

VITAL_ITEMID_CROSSWALK = {
    "heart_rate": [211, 220045],
    "resp_rate": [615, 618, 220210, 224690],
    "spo2": [646, 220277],
    "temp_fahrenheit": [678, 679, 223761],
    "nbp_systolic": [455, 220179],
    "nbp_diastolic": [8441, 220180],
    "nbp_mean": [456, 220181],
    "glucose": [807, 811, 220621, 225664],
    "gcs_eye": [184, 220739],
    "gcs_verbal": [723, 223900],
    "gcs_motor": [454, 223901],
    "weight_kg": [762, 763, 224639, 226512],  # 3723/3580-3582 dropped: 0 patients in this demo
}