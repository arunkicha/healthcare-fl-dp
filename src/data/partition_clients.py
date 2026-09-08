"""
Partitions patients into N simulated FL clients, stratified by privacy tier
(stigma_sensitive / non_sensitive), so every client gets a near-even share of
both tiers. A naive random split risks some clients getting zero
stigma_sensitive patients given how few there are (23/100) -- this avoids that.

Partitions by subject_id (patient), NOT hadm_id (admission), so a patient's
multiple admissions always stay together in the same client -- prevents
patient leakage across clients.

Note: this is a stratified near-IID split by design, not a fully realistic
non-IID hospital simulation -- a deliberate tradeoff given the small tier
sizes in this demo dataset. Document this as a known limitation.

Run from repo root:
    python -m src.data.partition_clients
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
N_CLIENTS = 10
RANDOM_SEED = 42

# Keep in sync with EXCLUDED_PATIENTS in src/data/dataset.py
EXCLUDED_PATIENTS = [41976]


def main():
    tiers_path = PROCESSED_DIR / "patient_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(f"{tiers_path} not found -- run audit_tier_mapping.py first.")

    tiers = pd.read_csv(tiers_path)
    tiers = tiers[~tiers["subject_id"].isin(EXCLUDED_PATIENTS)]
    rng = np.random.default_rng(RANDOM_SEED)

    assignments = []

    for tier_name, group in tiers.groupby("final_tier"):
        subject_ids = group["subject_id"].to_numpy().copy()
        rng.shuffle(subject_ids)
        # Round-robin assignment within this tier across all N_CLIENTS
        client_ids = np.arange(len(subject_ids)) % N_CLIENTS
        for sid, cid in zip(subject_ids, client_ids):
            assignments.append({"subject_id": sid, "final_tier": tier_name, "client_id": int(cid)})

    assign_df = pd.DataFrame(assignments)

    print(f"Partitioned {len(assign_df)} patients into {N_CLIENTS} clients (stratified by tier)\n")

    print("--- Patients per client ---")
    print(assign_df.groupby("client_id").size(), "\n")

    print("--- Tier breakdown per client ---")
    breakdown = assign_df.groupby(["client_id", "final_tier"]).size().unstack(fill_value=0)
    print(breakdown, "\n")

    empty_tier_clients = breakdown[(breakdown == 0).any(axis=1)]
    if len(empty_tier_clients) > 0:
        print("WARNING: some clients have ZERO patients in one tier:")
        print(empty_tier_clients)
    else:
        print("OK: every client has at least 1 patient in both tiers.")

    out_path = PROCESSED_DIR / "client_assignments.csv"
    assign_df.to_csv(out_path, index=False)
    print(f"\nSaved client assignments to: {out_path}")


if __name__ == "__main__":
    main()