"""Real-time mortality-risk scoring with a privately trained federated model.

Loads an exported model (see src/realtime/export_model.py) and scores a single
admission in milliseconds. The deployed weights come from tiered DP-SGD
federated training, so serving the model adds no further privacy cost.

Model directory layout (models/<config>/):
    model.pt              torch state_dict of MortalityMLP
    preprocessing.json    feature order, training medians, mean, std
    model_card.json       config, noise multipliers, epsilon, test metrics
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.baseline import MortalityMLP, VITAL_FEATURES

FEATURE_COLS = ["age", "gender"] + VITAL_FEATURES
FEATURE_RANGES = {
    "age": (0.0, 120.0),
    "heart_rate": (20.0, 250.0),
    "resp_rate": (4.0, 60.0),
    "spo2": (50.0, 100.0),
    "temp_fahrenheit": (80.0, 110.0),
    "nbp_systolic": (40.0, 260.0),
    "nbp_diastolic": (20.0, 160.0),
    "nbp_mean": (30.0, 200.0),
    "glucose": (20.0, 1000.0),
    "gcs_eye": (1.0, 4.0),
    "gcs_verbal": (1.0, 5.0),
    "gcs_motor": (1.0, 6.0),
    "weight_kg": (20.0, 300.0),
}
BAND_CUTOFFS = (0.10, 0.30)
BAND_NAMES = ("low", "moderate", "high")


def risk_band(p, cutoffs=BAND_CUTOFFS):
    if p < cutoffs[0]:
        return BAND_NAMES[0]
    if p < cutoffs[1]:
        return BAND_NAMES[1]
    return BAND_NAMES[2]


def available_models(model_dir="models"):
    root = Path(model_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "model.pt").exists())


class RiskPredictor:
    def __init__(self, model_dir="models", name="tiered"):
        path = Path(model_dir) / name
        self.name = name
        self.pre = json.loads((path / "preprocessing.json").read_text())
        self.card = json.loads((path / "model_card.json").read_text())
        self.feature_cols = self.pre["feature_cols"]
        self.medians = self.pre["medians"]
        self.mean = np.asarray(self.pre["mean"], dtype=np.float32)
        self.std = np.asarray(self.pre["std"], dtype=np.float32)
        self.model = MortalityMLP(n_features=len(self.feature_cols))
        self.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        self.model.eval()

    def encode(self, record):
        values, imputed, warnings = [], [], []
        for col in self.feature_cols:
            raw = record.get(col)
            if col == "gender":
                values.append(1.0 if str(raw).upper().startswith("M") else 0.0)
                continue
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                values.append(float(self.medians[col]))
                imputed.append(col)
                continue
            lo, hi = FEATURE_RANGES.get(col, (-np.inf, np.inf))
            if not lo <= float(raw) <= hi:
                warnings.append(f"{col}={raw} outside plausible range [{lo}, {hi}]")
            values.append(float(raw))
        return np.asarray(values, dtype=np.float32), imputed, warnings

    def predict(self, record):
        start = time.perf_counter()
        x, imputed, warnings = self.encode(record)
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            prob = float(torch.sigmoid(self.model(torch.from_numpy(x).unsqueeze(0)))[0])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "risk": prob,
            "band": risk_band(prob),
            "latency_ms": latency_ms,
            "imputed": imputed,
            "warnings": warnings,
        }

    def records_to_matrix(self, records):
        return np.stack([self.encode(r)[0] for r in records])

    def predict_matrix(self, X_raw):
        X = (np.asarray(X_raw, dtype=np.float32) - self.mean) / self.std
        with torch.inference_mode():
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy()

    def default_record(self):
        record = {c: float(self.medians[c]) for c in self.feature_cols if c != "gender"}
        record["gender"] = "M"
        return record
