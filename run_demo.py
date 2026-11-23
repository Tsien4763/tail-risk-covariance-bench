from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from covariance_bench import (
    linear_shrinkage_covariance,
    min_variance_weights,
    sample_covariance,
    tail_conditioned_covariance,
)
from prepare_panels import dateindexed


def main() -> None:
    rng = np.random.default_rng(101)
    n_days, n_assets = 250, 8
    factor = rng.normal(0.0, 0.01, size=(n_days, 1))
    # Inject a left-tail stress regime on 10% of days.
    stress = rng.random(n_days) < 0.10
    factor[stress] -= 0.025
    idio = rng.normal(0.0, 0.008, size=(n_days, n_assets))
    returns = factor * rng.uniform(0.6, 1.4, size=(1, n_assets)) + idio

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "synthetic_panel.csv"
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        df = pd.DataFrame(returns, columns=[f"A{i}" for i in range(n_assets)])
        df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
        df.to_csv(csv_path, index=False)
        panel = dateindexed(csv_path)

    arr = panel.to_numpy()
    cov_sample = sample_covariance(arr)
    cov_shrink = linear_shrinkage_covariance(arr, alpha=0.25)
    cov_tail = tail_conditioned_covariance(arr, tail_quantile=0.15)

    w_sample = min_variance_weights(cov_sample)
    w_shrink = min_variance_weights(cov_shrink)
    w_tail = min_variance_weights(cov_tail)

    print("=== Tail-Risk Covariance Benchmark MVP (synthetic demo) ===")
    print(f"Panel shape: {panel.shape} | Stress days: {int(stress.sum())}")
    print(f"Condition numbers: sample={np.linalg.cond(cov_sample):.1f} | shrink={np.linalg.cond(cov_shrink):.1f}")
    print(f"Tail-regime variance (w_sample vs w_tail): {float(w_sample @ cov_tail @ w_sample):.6f} vs {float(w_tail @ cov_tail @ w_tail):.6f}")
    print(f"Min-var weight sum check: sample={w_sample.sum():.4f}, shrink={w_shrink.sum():.4f}, tail={w_tail.sum():.4f}")


if __name__ == "__main__":
    main()
