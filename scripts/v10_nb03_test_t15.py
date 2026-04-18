#!/usr/bin/env python3
"""T15 (v2): pre-registered regime-stratified pass condition for the opx-liq
pressure head, two-axis honesty bar.

Reads  results/v10_opx_per_regime_claims_audit_robust.csv (Chunk C)
Checks Section 12 of docs/v10_nb03_test_protocol.md pass condition (v2):
    at least one row with target='P_kbar' AND n >= 20 AND
    axis1_nonoverlap=True AND axis2_nonoverlap=True
    (equivalently robust_outperforms=True).
Appends one row to results/v10_nb03_test_log.csv.

Exit 0 on success (test ran), regardless of pass/fail outcome. Exit 1 on
missing-input or logging errors.

Backwards compatibility: the original T15 (Chunk B) read the axis-1-only
audit at results/v10_opx_per_regime_claims_audit.csv. Chunk C revises the
pass condition to require axis 2 (20-seed RMSE spread non-overlap) as well.
If the robust audit is missing, the script falls back to the axis-1-only
audit with a WARNING tag in the log `details` JSON, so this script remains
runnable even when Chunk C artifacts are absent (e.g. on a cold checkout).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import RESULTS

ROBUST_CSV = RESULTS / 'v10_opx_per_regime_claims_audit_robust.csv'
LEGACY_CSV = RESULTS / 'v10_opx_per_regime_claims_audit.csv'
LOG_CSV    = RESULTS / 'v10_nb03_test_log.csv'


def main() -> int:
    if ROBUST_CSV.exists():
        source_tag = 'robust'
        audit = pd.read_csv(ROBUST_CSV)
        mask = (
            (audit['target'] == 'P_kbar')
            & (audit['n'] >= 20)
            & audit['axis1_nonoverlap'].astype(bool)
            & audit['axis2_nonoverlap'].astype(bool)
        )
    elif LEGACY_CSV.exists():
        source_tag = 'legacy_axis1_only_WARNING'
        audit = pd.read_csv(LEGACY_CSV)
        mask = (
            (audit['target'] == 'P_kbar')
            & (~audit['sample_size_limited'].astype(bool))
            & (~audit['ci_overlap'].astype(bool))
            & (audit['v10_rmse'] < audit['putirka_rmse'])
        )
    else:
        print(f'missing {ROBUST_CSV} and {LEGACY_CSV}; run Chunk C or Chunk B '
              f'audit scripts first', file=sys.stderr)
        return 1

    hits = audit[mask]
    passed = bool(len(hits))

    if passed:
        winner = hits.iloc[0]
        details = {
            'source':           source_tag,
            'regime':           winner.get('regime'),
            'n':                int(winner['n']),
            'v10_method':       winner.get('v10_method', ''),
            'v10_rmse':         float(winner['v10_rmse']),
            'v10_rmse_lo':      float(winner.get('v10_rmse_lo', float('nan'))),
            'v10_rmse_hi':      float(winner.get('v10_rmse_hi', float('nan'))),
            'putirka_rmse':     float(winner['putirka_rmse']),
            'putirka_rmse_lo':  float(winner.get('putirka_rmse_lo',
                                                 float('nan'))),
            'putirka_rmse_hi':  float(winner.get('putirka_rmse_hi',
                                                 float('nan'))),
            'seed_rmse_mean':   float(winner.get('seed_rmse_mean',
                                                 float('nan'))),
            'seed_rmse_lo':     float(winner.get('seed_rmse_lo',
                                                 float('nan'))),
            'seed_rmse_hi':     float(winner.get('seed_rmse_hi',
                                                 float('nan'))),
            'seed_is_deterministic': bool(winner.get(
                'seed_is_deterministic', False)),
            'verdict':          winner.get('robust_verdict',
                                           winner.get('verdict', '')),
            'n_qualifying_regimes': int(len(hits)),
        }
        value = float(winner['putirka_rmse'] - winner['v10_rmse'])
    else:
        details = {
            'source':               source_tag,
            'n_qualifying_regimes': 0,
            'audit_rows':           int(len(audit)),
        }
        value = 0.0

    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'pipeline':  'opx',
        'test_id':   'T15',
        'target':    'P_kbar',
        'track':     'opx_liq',
        'passed':    str(bool(passed)),
        'value':     value,
        'threshold': '',
        'details':   json.dumps(details),
    }

    header = not LOG_CSV.exists()
    with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)

    print(f'T15 source={source_tag} passed={passed} value={value:.3f}')
    print(f'appended to {LOG_CSV}')
    if passed:
        print(f'winning regime: {details["regime"]} (n={details["n"]}, '
              f'delta={value:.3f} kbar)')
        if source_tag == 'robust':
            print(f'  seed_is_deterministic={details["seed_is_deterministic"]}'
                  f'  seed_rmse=[{details["seed_rmse_lo"]:.3f}, '
                  f'{details["seed_rmse_hi"]:.3f}]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
