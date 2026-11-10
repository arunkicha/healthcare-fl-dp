import pandas as pd
import time

start = time.time()
chart = pd.read_csv(
    "data/raw/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv",
    usecols=["subject_id", "hadm_id", "itemid", "valuenum"],
    low_memory=False,
)
print(f"Loaded {len(chart):,} rows in {time.time()-start:.1f}s")
print(f"Memory usage: {chart.memory_usage(deep=True).sum() / 1e9:.2f} GB")