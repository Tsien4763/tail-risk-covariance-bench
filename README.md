# Tail-Risk Covariance Benchmark

[中文说明](README_ZH.md)

How much portfolio variance do full-sample covariance matrices miss during market drawdowns, and how do you benchmark tail-conditioned estimators without introducing forward-fill or survivorship artifacts?

This repository pairs strict daily return panel construction with tail-risk covariance benchmarking. Panel ingestion rejects non-monotonic dates, disables forward-filling on missing closes, masks Fama-French sentinel values, and logs SHA-256 provenance before comparing sample, linear-shrinkage, and left-tail conditioned minimum-variance covariance estimators.

- `src/prepare_panels.py`: strict date-indexed CSV ingestion, missing-sentinel masking, and SHA-256 quality reporting.
- `src/covariance_bench.py`: sample, linear-shrinkage, and left-tail conditioned covariance estimators plus closed-form minimum-variance weights.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` writes a 250-day synthetic multi-asset return panel with injected stress days, loads it through `prepare_panels.dateindexed`, and compares condition numbers and tail-regime variance across estimators.

## Scope

This is the public MVP on synthetic returns. Full benchmark scripts over Kenneth French Data Library and ETF panels are available upon request.
