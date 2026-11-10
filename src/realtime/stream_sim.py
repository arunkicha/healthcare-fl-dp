"""Replay test admissions as a stream and measure real-time scoring performance.

Scores one admission at a time (as in a bedside system) and reports latency
percentiles, throughput, and a check that streamed scores match batch scores.
Only aggregate statistics and latencies are saved; no patient rows are written.

Run:  python -m src.realtime.stream_sim --name tiered --n 2000 --budget-ms 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.dp_federated import load_data
from src.realtime.predictor import FEATURE_COLS, RiskPredictor

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="tiered")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--budget-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    predictor = RiskPredictor(name=args.name)
    df = load_data()
    test_df = df[df["split"] == "test"].sample(frac=1.0, random_state=args.seed).head(args.n)
    records = test_df[FEATURE_COLS].to_dict("records")
    y = test_df["hospital_expire_flag"].to_numpy()

    for record in records[:20]:
        predictor.predict(record)

    results = [predictor.predict(r) for r in records]
    latency = np.array([r["latency_ms"] for r in results])
    stream_risk = np.array([r["risk"] for r in results])

    X_raw = predictor.records_to_matrix(records)
    start = time.perf_counter()
    batch_risk = predictor.predict_matrix(X_raw)
    batch_seconds = time.perf_counter() - start

    summary = {
        "model": args.name,
        "n_scored": len(records),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_p99_ms": float(np.percentile(latency, 99)),
        "latency_max_ms": float(latency.max()),
        "stream_throughput_per_s": float(len(records) / (latency.sum() / 1000.0)),
        "batch_throughput_per_s": float(len(records) / batch_seconds),
        "budget_ms": args.budget_ms,
        "within_budget_pct": float((latency <= args.budget_ms).mean() * 100.0),
        "max_abs_stream_vs_batch_diff": float(np.abs(stream_risk - batch_risk).max()),
        "auc_on_stream": float(roc_auc_score(y, stream_risk)) if len(np.unique(y)) > 1 else float("nan"),
        "high_risk_alert_rate_pct": float(np.mean([r["band"] == "high" for r in results]) * 100.0),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "realtime_latency.csv", index=False)
    pd.DataFrame({"latency_ms": latency}).to_csv(RESULTS_DIR / "realtime_latency_samples.csv", index=False)

    for key, value in summary.items():
        print(f"{key:<32} {value}")


if __name__ == "__main__":
    main()
