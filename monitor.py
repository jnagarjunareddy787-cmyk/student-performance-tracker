"""
Model drift monitoring using Kolmogorov-Smirnov test.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def detect_drift(reference_df, live_df, threshold=0.05):
    report = {}
    common_cols = [c for c in reference_df.columns if c in live_df.columns]

    for col in common_cols:
        stat, p_value = ks_2samp(reference_df[col].dropna(),
                                 live_df[col].dropna())
        report[col] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": bool(p_value < threshold)
        }

    drift_count = sum(1 for v in report.values() if v["drift_detected"])
    drift_ratio = drift_count / max(len(report), 1)

    print("=" * 60)
    print("📊 DRIFT MONITORING REPORT")
    print("=" * 60)
    print(f"Features analyzed : {len(report)}")
    print(f"Features drifted  : {drift_count}")
    print(f"Drift ratio       : {drift_ratio:.2%}")
    print("=" * 60)

    if drift_ratio > 0.3:
        print("🚨 RETRAINING RECOMMENDED (drift ratio > 30%)")
    else:
        print("✅ Model stable — no retraining needed")

    return report


if __name__ == "__main__":
    df = pd.read_csv("data/student-mat.csv", sep=";")
    numeric = df.select_dtypes(include=["int64"])

    np.random.seed(42)
    live = numeric.sample(frac=0.5).copy()
    live["age"] += np.random.randint(-2, 3, size=len(live))

    report = detect_drift(numeric, live)
    for feat, info in report.items():
        status = "⚠️ DRIFT" if info["drift_detected"] else "✅ OK"
        print(f"  {feat:15s} | p={info['p_value']:.4f} | {status}")