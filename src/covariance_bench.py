from __future__ import annotations

import numpy as np


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    return np.cov(x, rowvar=False, ddof=1)


def linear_shrinkage_covariance(returns: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Shrink sample covariance toward a scaled identity target."""
    s = sample_covariance(returns)
    mu = float(np.trace(s)) / s.shape[0]
    return (1.0 - alpha) * s + alpha * mu * np.eye(s.shape[0])


def tail_conditioned_covariance(returns: np.ndarray, tail_quantile: float = 0.15) -> np.ndarray:
    """Estimate covariance on left-tail days defined by equal-weight market returns."""
    x = np.asarray(returns, dtype=float)
    ew = x.mean(axis=1)
    cutoff = float(np.quantile(ew, tail_quantile))
    tail_rows = x[ew <= cutoff]
    if len(tail_rows) < 2:
        return sample_covariance(x)
    return np.cov(tail_rows, rowvar=False, ddof=1)


def min_variance_weights(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Closed-form global minimum-variance weights normalized to sum to 1."""
    n = cov.shape[0]
    reg = cov + ridge * np.eye(n)
    ones = np.ones(n)
    w = np.linalg.solve(reg, ones)
    return w / w.sum()
