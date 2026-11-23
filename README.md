# Tail-Risk Covariance Benchmark

Daily return panel preparation and tail-conditioned covariance estimators for public equity and ETF risk benchmarks.

Standard full-sample covariance matrices understate co-movement during market drawdowns; this codebase pairs strict no-forward-fill panel construction with shrinkage and left-tail covariance diagnostics.

- `src/prepare_panels.py`: strict date-indexed CSV ingestion, missing-sentinel masking, and SHA-256 provenance reporting for Fama-French and ETF panels.
- `src/covariance_bench.py`: sample, linear-shrinkage, and tail-conditioned covariance estimators plus minimum-variance weights.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` writes a 250-day synthetic multi-asset return panel with injected stress days, loads it via `prepare_panels.dateindexed`, and compares sample, shrinkage, and tail-conditioned covariance risk.

## Scope

The demo uses synthetic return arrays; reproducing historical benchmarks requires downloading the public Kenneth French Data Library files.
