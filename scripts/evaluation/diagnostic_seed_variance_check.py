#!/usr/bin/env python3
"""Phase 1 diagnostic: scan opx and cpx multiseed summaries for suspiciously
small RMSE-over-seeds standard deviation. Write
`results/diagnostic_seed_variance_check.csv` with classification of each
row as deterministic_fit, seed_bug, or rounding_artifact.

Deterministic fit = ElasticNet at frozen hyperparameters (expected
zero-std). Anything else with std <= 1e-8 is a likely seed_bug.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

SUMMARIES = [
    ('opx', 'results/opx_multiseed_summary.csv'),
    ('cpx', 'results/cpx_multiseed_summary.csv'),
    ('tabpfn', 'results/tabpfn_multiseed_summary.csv'),
]


def classify(std_val: float, model: str) -> tuple[bool, bool, str]:
    exactly_zero = (std_val == 0.0)
    rounded_zero = (std_val != 0.0) and (std_val < 1e-4)
    if exactly_zero:
        src = 'deterministic_fit' if model == 'ElasticNet' else 'seed_bug'
    elif rounded_zero:
        src = 'rounding_artifact'
    else:
        src = 'stochastic_fit_ok'
    return exactly_zero, rounded_zero, src


def main():
    rows = []
    for pipeline, path in SUMMARIES:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            std_val = float(r['std'])
            exact, rounded, src = classify(std_val, r['model'])
            rows.append({
                'pipeline':      pipeline if pipeline != 'tabpfn' else r.get('pipeline', 'tabpfn'),
                'model':         r['model'],
                'feature_set':   r['feature_set'],
                'track':         r['track'],
                'target':        r['target'],
                'n_seeds':       int(r.get('count', 20)),
                'rmse_std_8dp':  f'{std_val:.8f}',
                'is_exactly_zero': exact,
                'is_rounded_zero': rounded,
                'inferred_source': src,
            })

    out = pd.DataFrame(rows)
    outp = PROJECT_ROOT / 'results' / 'diagnostic_seed_variance_check.csv'
    out.to_csv(outp, index=False)
    n_zero = int(out['is_exactly_zero'].sum())
    n_bug = int((out['inferred_source'] == 'seed_bug').sum())
    print(f'Wrote {outp}')
    print(f'  total rows: {len(out)}')
    print(f'  exactly-zero std: {n_zero}')
    print(f'  classified as seed_bug (non-ElasticNet zero-std): {n_bug}')
    if n_bug == 0:
        print('  => no seed_bug cases. All zero-std are ElasticNet deterministic fits.')


if __name__ == '__main__':
    main()
