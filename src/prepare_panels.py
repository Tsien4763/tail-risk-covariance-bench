#!/usr/bin/env python3
"""Rebuild daily research panels from the four frozen starter CSVs.

Requires: python>=3.10, pandas>=2.1, numpy>=1.24.
No network access required. This is data preparation, not backtest fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FRENCH49 = 'French_49Industries_EW_Daily_2004-2020.csv'
FF3 = 'French_FF3_Daily_2004-2020.csv'
ETF = 'ETF_Daily_Prices_2004-2026_Imported.csv'
NON_ETF = {'^IRX', '^VIX', '^VIX3M'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def dateindexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Date' not in df:
        raise ValueError(f'Date column missing: {path}')
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')
    if not df['Date'].is_monotonic_increasing or not df['Date'].is_unique:
        raise ValueError(f'Date must be strictly increasing and unique: {path}')
    return df.set_index('Date').sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--french-dir', type=Path, required=True)
    ap.add_argument('--etf-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    f49_source = args.french_dir / FRENCH49
    ff3_source = args.french_dir / FF3
    etf_source = args.etf_dir / ETF
    f49 = dateindexed(f49_source).apply(pd.to_numeric, errors='raise')
    ff3 = dateindexed(ff3_source).apply(pd.to_numeric, errors='raise')
    pxall = dateindexed(etf_source).apply(pd.to_numeric, errors='coerce')

    if len(f49.columns) != 49:
        raise ValueError(f'Expected 49 industry portfolios, found {len(f49.columns)}')
    if not f49.index.equals(ff3.index):
        raise ValueError('Fama-French industry/factor calendar mismatch')
    if not {'Mkt-RF', 'SMB', 'HML', 'RF'}.issubset(ff3.columns):
        raise ValueError('Unexpected FF3 factor column names')

    french_missing = f49.isin((-99.99, -999.0))
    factors_missing = ff3.isin((-99.99, -999.0))
    french_returns = f49.mask(french_missing) / 100.0
    factor_returns = ff3.mask(factors_missing) / 100.0
    french_returns.index.name = 'Date'
    factor_returns.index.name = 'Date'
    french_returns.to_csv(out / 'french49_ew_returns_decimal_2004_2020.csv', float_format='%.12g')
    factor_returns.to_csv(out / 'ff3_daily_factors_decimal_2004_2020.csv', float_format='%.12g')

    ticker_cols = [c for c in pxall.columns if c not in NON_ETF]
    price = pxall[ticker_cols].where(pxall[ticker_cols] > 0)
    # Explicitly disable forward fill: missing current or previous close -> missing return.
    etf_returns = price.pct_change(fill_method=None)
    etf_returns.index.name = 'Date'
    # The final 2026-08-11 raw line has auxiliary fields but no ETF prices;
    # likewise the first session has no previous close for a return.
    etf_valid_day = etf_returns.notna().any(axis=1)
    etf_returns = etf_returns.loc[etf_valid_day]
    etf_returns.to_csv(out / 'etf_daily_simple_returns_decimal_2004_2026.csv', float_format='%.12g')
    train_universe = [c for c in ticker_cols if pd.notna(price.loc['2008-01-02':, c]).mean() > 0.95]
    # This is a descriptive eligibility scan ONLY. Fix investible universe using
    # training-period data alone before running any predictive backtest.
    report = {
        'sources_sha256': {str(p.name): sha256(p) for p in (f49_source, ff3_source, etf_source)},
        'industries_n': len(f49.columns),
        'industry_days': len(f49),
        'industry_date_min': str(f49.index.min().date()),
        'industry_date_max': str(f49.index.max().date()),
        'industry_missing_sentinel_count': int(french_missing.sum().sum()),
        'ff3_days': len(ff3),
        'etf_tickers': ticker_cols,
        'etf_n_tickers': len(ticker_cols),
        'etf_raw_dates': len(pxall),
        'etf_valid_return_dates': len(etf_returns),
        'etf_date_min': str(etf_returns.index.min().date()),
        'etf_date_max': str(etf_returns.index.max().date()),
        'etf_missing_returns': {k: int(v) for k, v in etf_returns.isna().sum().to_dict().items()},
        'candidate_eligible_after_2008_descriptive_only': train_universe,
        'warn': [
            'ETF original snapshot is not point-in-time: third-party adjusted close mirror.',
            'Do not choose investible universe using future periods or survivorship information.',
            'Oxford-Man mirror has no dates and is intentionally not aligned to this panel.',
            'Daily price returns do not reconstruct intraday realized covariance.'
        ]
    }
    (out / 'quality_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('industries_n','industry_days','etf_n_tickers','etf_valid_return_dates','etf_date_max')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
