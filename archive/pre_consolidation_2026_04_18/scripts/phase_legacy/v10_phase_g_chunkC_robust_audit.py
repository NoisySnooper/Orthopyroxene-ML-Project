#!/usr/bin/env python3
"""Phase G Chunk C robust audit: two-axis honesty bar for the per-regime
opx_liq claims.

Two axes of uncertainty:
    Axis 1 - Test-set sampling noise. Quantified by bootstrap 95% CI on
             RMSE over paired (y_true, y_pred) rows (Chunk B, 500 resamples,
             seed=SEED_BOOTSTRAP). Already in
             results/v10_opx_per_regime_benchmark.csv (columns v10_rmse_lo,
             v10_rmse_hi, putirka_rmse_lo, putirka_rmse_hi).
    Axis 2 - Model-fit stochasticity across seeds. Quantified by the
             empirical per-seed per-regime RMSE distribution from the
             Chunk C probe (results/v10_chunkC_perseed_regime_rmse.csv),
             20 seeds (42..61). Putirka equations are deterministic so
             their seed axis collapses to a point.

Robust verdict rule:
    "v10 outperforms Putirka (robust)" requires ALL of:
        (i)   n >= P_REGIME_MIN_N_FOR_CLAIMS
        (ii)  v10_rmse_hi < putirka_rmse_lo            (axis 1 non-overlap)
        (iii) seed_rmse_hi < putirka_rmse_lo           (axis 2 non-overlap)
              where seed_rmse_hi = 97.5th percentile of the 20-seed RMSE
              distribution FOR THE SAME CELL that Chunk B reported for that
              regime row.
    Otherwise downgrade to:
        "v10 outperforms Putirka under favorable seed (not robust)"
            when (ii) holds but (iii) fails.
        "competitive with Putirka"
            when (ii) fails and the two CIs overlap.
        "insufficient data (n < 20)"
            when (i) fails.

Output:
    results/v10_opx_per_regime_claims_audit_robust.csv

    Columns (in addition to Chunk B's set):
        v10_method                  - the cell used by Chunk B for this row
        seed_rmse_mean              - mean of 20-seed RMSE for that cell
        seed_rmse_std               - std of 20-seed RMSE for that cell
        seed_rmse_lo                - 2.5th percentile of 20-seed RMSE
        seed_rmse_hi                - 97.5th percentile of 20-seed RMSE
        seed_rmse_min, seed_rmse_max- min/max across seeds
        seed_is_deterministic       - True if seed_rmse_std == 0
        axis1_nonoverlap            - v10_rmse_hi < putirka_rmse_lo
        axis2_nonoverlap            - seed_rmse_hi < putirka_rmse_lo
        robust_verdict              - string per rule above
        robust_outperforms          - True only for "robust" verdict

Note: Chunk C probe covers ALL cells that Chunk B used as per-regime best.
If a regime row has no matching cell in the probe (shouldn't happen), the
robust columns are left NaN and the row is flagged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import RESULTS, P_REGIME_MIN_N_FOR_CLAIMS


BENCHMARK_CSV = RESULTS / 'v10_opx_per_regime_benchmark.csv'
PROBE_CSV     = RESULTS / 'v10_chunkC_perseed_regime_rmse.csv'
OUT_CSV       = RESULTS / 'v10_opx_per_regime_claims_audit_robust.csv'


def classify(n, axis1_nonoverlap, axis2_nonoverlap):
    if n < P_REGIME_MIN_N_FOR_CLAIMS:
        return 'insufficient data (n < 20)', False
    if axis1_nonoverlap and axis2_nonoverlap:
        return 'v10 outperforms Putirka (robust)', True
    if axis1_nonoverlap and not axis2_nonoverlap:
        return ('v10 outperforms Putirka under favorable seed '
                '(not robust)'), False
    return 'competitive with Putirka', False


def parse_v10_method(s):
    # format "Model/feature_set" e.g. "ElasticNet/raw"
    parts = str(s).split('/')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def main():
    bench = pd.read_csv(BENCHMARK_CSV)
    probe = pd.read_csv(PROBE_CSV)

    required_bench = {
        'regime', 'target', 'n', 'sample_size_limited', 'v10_method',
        'v10_rmse', 'v10_rmse_lo', 'v10_rmse_hi',
        'putirka_method', 'putirka_rmse',
        'putirka_rmse_lo', 'putirka_rmse_hi',
    }
    missing = required_bench - set(bench.columns)
    if missing:
        raise RuntimeError(f'benchmark csv missing columns: {missing}')

    rows = []
    unmatched = []
    for _, br in bench.iterrows():
        model, fs = parse_v10_method(br['v10_method'])
        match = probe[
            (probe['regime'] == br['regime']) &
            (probe['target'] == br['target']) &
            (probe['model']  == model) &
            (probe['feature_set'] == fs)
        ]
        if len(match) == 0:
            unmatched.append((br['regime'], br['target'], br['v10_method']))
            seed_mean = seed_std = np.nan
            seed_lo = seed_hi = seed_min = seed_max = np.nan
            seed_determ = None
        else:
            vals = match['rmse'].to_numpy(dtype=float)
            seed_mean = float(vals.mean())
            seed_std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            seed_lo   = float(np.percentile(vals, 2.5))
            seed_hi   = float(np.percentile(vals, 97.5))
            seed_min  = float(vals.min())
            seed_max  = float(vals.max())
            seed_determ = bool(seed_std == 0.0)

        n = int(br['n'])
        putirka_lo = float(br['putirka_rmse_lo'])
        axis1 = bool(float(br['v10_rmse_hi']) < putirka_lo)
        axis2 = (not np.isnan(seed_hi)) and bool(seed_hi < putirka_lo)
        verdict, robust = classify(n, axis1, axis2)

        rows.append({
            'regime':               br['regime'],
            'target':               br['target'],
            'n':                    n,
            'sample_size_limited':  bool(br['sample_size_limited']),
            'v10_method':           br['v10_method'],
            'v10_rmse':             float(br['v10_rmse']),
            'v10_rmse_lo':          float(br['v10_rmse_lo']),
            'v10_rmse_hi':          float(br['v10_rmse_hi']),
            'putirka_method':       br['putirka_method'],
            'putirka_rmse':         float(br['putirka_rmse']),
            'putirka_rmse_lo':      putirka_lo,
            'putirka_rmse_hi':      float(br['putirka_rmse_hi']),
            'seed_rmse_mean':       seed_mean,
            'seed_rmse_std':        seed_std,
            'seed_rmse_lo':         seed_lo,
            'seed_rmse_hi':         seed_hi,
            'seed_rmse_min':        seed_min,
            'seed_rmse_max':        seed_max,
            'seed_is_deterministic': seed_determ,
            'axis1_nonoverlap':     axis1,
            'axis2_nonoverlap':     axis2,
            'robust_verdict':       verdict,
            'robust_outperforms':   robust,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}: {len(out)} rows')
    if unmatched:
        print(f'WARNING: {len(unmatched)} rows had no probe match:')
        for u in unmatched:
            print('  ', u)
    print()
    cols = ['regime', 'target', 'n', 'v10_method',
            'v10_rmse', 'v10_rmse_lo', 'v10_rmse_hi',
            'seed_rmse_mean', 'seed_rmse_lo', 'seed_rmse_hi',
            'putirka_rmse', 'putirka_rmse_lo',
            'axis1_nonoverlap', 'axis2_nonoverlap', 'robust_verdict']
    print(out[cols].round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
