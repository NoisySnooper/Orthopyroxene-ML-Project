#!/usr/bin/env python3
"""T15: pre-registered regime-stratified pass condition for the opx-liq
pressure head.

Reads  results/v10_opx_per_regime_claims_audit.csv (produced by nb04)
Checks Section 12 of docs/v10_nb03_test_protocol.md pass condition:
    at least one row with target='P_kbar', sample_size_limited=False,
    ci_overlap=False, v10_rmse < putirka_rmse.
Appends one row to results/v10_nb03_test_log.csv.

Exit 0 on success (test ran), regardless of pass/fail outcome. Exit 1 on
missing-input or logging errors.
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

AUDIT_CSV = RESULTS / 'v10_opx_per_regime_claims_audit.csv'
LOG_CSV   = RESULTS / 'v10_nb03_test_log.csv'


def main() -> int:
    if not AUDIT_CSV.exists():
        print(f'missing {AUDIT_CSV}; run nb04 per-regime benchmark first',
              file=sys.stderr)
        return 1
    audit = pd.read_csv(AUDIT_CSV)

    mask = (
        (audit['target'] == 'P_kbar')
        & (~audit['sample_size_limited'].astype(bool))
        & (~audit['ci_overlap'].astype(bool))
        & (audit['v10_rmse'] < audit['putirka_rmse'])
    )
    hits = audit[mask]
    passed = bool(len(hits))

    if passed:
        winner = hits.iloc[0]
        details = {
            'regime':        winner['regime'],
            'n':             int(winner['n']),
            'v10_rmse':      float(winner['v10_rmse']),
            'putirka_rmse':  float(winner['putirka_rmse']),
            'ci_overlap':    bool(winner['ci_overlap']),
            'verdict':       winner['verdict'],
            'n_qualifying_regimes': int(len(hits)),
        }
        value = float(winner['putirka_rmse'] - winner['v10_rmse'])
    else:
        details = {
            'n_qualifying_regimes': 0,
            'audit_rows': int(len(audit)),
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

    print(f'T15 passed={passed} value={value:.3f}')
    print(f'appended to {LOG_CSV}')
    if passed:
        print(f'winning regime: {details["regime"]} '
              f'(n={details["n"]}, delta={value:.3f} kbar)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
