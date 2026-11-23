from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covariance_bench import (
    linear_shrinkage_covariance,
    min_variance_weights,
    sample_covariance,
    tail_conditioned_covariance,
)


class TestTailRiskCovariance(unittest.TestCase):
    def test_shrinkage_improves_conditioning(self) -> None:
        rng = np.random.default_rng(5)
        r = rng.normal(scale=0.01, size=(30, 10))
        c_s = sample_covariance(r)
        c_sh = linear_shrinkage_covariance(r, alpha=0.3)
        self.assertLess(np.linalg.cond(c_sh), np.linalg.cond(c_s))

    def test_tail_min_var_weights(self) -> None:
        rng = np.random.default_rng(6)
        r = rng.normal(scale=0.01, size=(120, 6))
        c_t = tail_conditioned_covariance(r, tail_quantile=0.2)
        w = min_variance_weights(c_t)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertEqual(w.shape, (6,))


if __name__ == "__main__":
    unittest.main()
