#!/usr/bin/env python3
"""v10 T01-T12 test protocol logger (primary pipelines).

Per docs/master_plan.md Section 7 and docs/nb03_test_protocol.md.

Each primary pipeline (opx, cpx, twopx) runs tests T01-T12. Results go
to a shared CSV keyed by (pipeline, test_id, target). The logger is
intentionally a thin recorder: test logic lives in NB03 and Phase C/D/E
analysis cells, which compute metrics and call `log_test_result` once
per (test, pipeline, target, track) cell.

Public API:
  log_test_result(test_id, pipeline, target, track, passed, value,
                  threshold=None, details=None) -> dict
  summarize_tests(pipeline=None) -> pd.DataFrame
  test_catalog() -> pd.DataFrame

Companion to: docs/master_plan.md Section 7.
Author: NQTa (with Claude)
Date: 2026-04-16
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS  # noqa: E402

TEST_LOG_CSV = RESULTS / 'v10_nb03_test_log.csv'
UNIVERSAL_TEST_LOG_CSV = RESULTS / 'universal' / 'v10_nb03_test_log.csv'

TEST_CATALOG = [
    ('T01', 'Boosted is primary',                                         'primary'),
    ('T02', 'Ensemble beats best base (4 methods compared)',              'primary'),
    ('T03', 'Resampling hurts (replicates v9 finding per pipeline)',      'primary'),
    ('T04', 'N_AUG=1 beats N_AUG=5',                                      'primary'),
    ('T05', 'Feature set winners reproduce v9 or are stable',             'primary'),
    ('T06', 'Mineral-only pipeline is worse than +liq',                   'primary'),
    ('T07', 'Composition-conditional bias correction improves ArcPL T',   'primary'),
    ('T08', 'P piecewise bias correction improves test P',                'primary'),
    ('T09', 'CV-predict stacking beats full-fit stacking',                'primary'),
    ('T10', 'IsolationForest OOD correlates with residual magnitude',     'primary'),
    ('T11', 'CatBoost or LightGBM beats XGB on any pipeline (NEW)',       'primary'),
    ('T12', 'MLP or ElasticNet beats tree models on any pipeline (NEW)',  'primary'),
    ('T13', 'Universal degrades gracefully (1<2<3 phase R^2 ordering)',   'universal'),
    ('T14', 'Universal beats specialized models on same sample',          'universal'),
]

PRIMARY_PIPELINES = ('opx', 'cpx', 'twopx')
COLUMNS = (
    'timestamp', 'pipeline', 'test_id', 'target', 'track',
    'passed', 'value', 'threshold', 'details',
)


def test_catalog() -> pd.DataFrame:
    return pd.DataFrame(TEST_CATALOG, columns=('test_id', 'hypothesis', 'scope'))


def _log_path(test_id: str) -> Path:
    for tid, _, scope in TEST_CATALOG:
        if tid == test_id:
            return UNIVERSAL_TEST_LOG_CSV if scope == 'universal' else TEST_LOG_CSV
    raise ValueError(f'unknown test_id: {test_id!r}')


def _append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row], columns=COLUMNS)
    if path.exists():
        df_row.to_csv(path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def log_test_result(test_id: str,
                    pipeline: str,
                    target: str,
                    track: str,
                    passed: bool,
                    value: float,
                    threshold: Optional[float] = None,
                    details: Optional[dict] = None) -> dict:
    """Append one result row to the relevant test log CSV.

    Parameters
    ----------
    test_id : str
        One of T01-T14 (see test_catalog()).
    pipeline : str
        'opx', 'cpx', 'twopx', or 'universal'.
    target : str
        'T_C' or 'P_kbar' (or 'both' for aggregate tests).
    track : str
        e.g. 'opx_only', 'opx_liq', 'cpx_liq', 'twopx', 'N/A'.
    passed : bool
        Hypothesis met (True) or not (False).
    value : float
        Primary statistic (RMSE delta, correlation, etc.).
    threshold : float, optional
        Pre-registered pass threshold, if any.
    details : dict, optional
        Any extra metadata (sample sizes, CI bounds, sub-metrics).

    Returns
    -------
    The row as a dict.
    """
    valid_tests = {t[0] for t in TEST_CATALOG}
    if test_id not in valid_tests:
        raise ValueError(f'unknown test_id: {test_id!r}; valid: {sorted(valid_tests)}')

    row = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'pipeline':  pipeline,
        'test_id':   test_id,
        'target':    target,
        'track':     track,
        'passed':    bool(passed),
        'value':     float(value) if value is not None else None,
        'threshold': float(threshold) if threshold is not None else None,
        'details':   json.dumps(details, default=str) if details else None,
    }
    _append_row(_log_path(test_id), row)
    return row


def summarize_tests(pipeline: Optional[str] = None) -> pd.DataFrame:
    """Return a summary of test results across one or all pipelines."""
    frames = []
    for path in (TEST_LOG_CSV, UNIVERSAL_TEST_LOG_CSV):
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    if pipeline is not None:
        df = df[df['pipeline'] == pipeline]
    return df.sort_values(['pipeline', 'test_id', 'target', 'track']).reset_index(drop=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='T01-T14 test protocol logger')
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('catalog', help='Show test catalog')

    s = sub.add_parser('summary', help='Show current test log')
    s.add_argument('--pipeline', default=None)

    args = ap.parse_args()
    if args.cmd == 'catalog':
        print(test_catalog().to_string(index=False))
    elif args.cmd == 'summary':
        df = summarize_tests(pipeline=args.pipeline)
        if df.empty:
            print('(no test results logged yet)')
        else:
            print(df.to_string(index=False))
