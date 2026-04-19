#!/usr/bin/env python3
"""v10 NB03 tests T16-T18: Phase G.7 bias-correction decision tests.

Appends three rows to results/v10_nb03_test_log.csv:

  T16 -- at least one cell ships a correction at canonical seed=42
  T17 -- shipped corrections are stable across seeds (>10 of 20 ship)
  T18 -- parameter stability (Form A edge swing + Form B CV-reseed std)

All three tests are pure consistency checks against the Phase G.7
output CSVs. No models are refit. The script is idempotent: each run
appends one new row per test with the current timestamp.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import LOGS, RESULTS

TEST_LOG = RESULTS / 'v10_nb03_test_log.csv'
LOG_PATH = LOGS / 'v10_nb03_test_t16_t18.log'

T18_EDGE_THRESH_P = 0.5   # kbar
T18_EDGE_THRESH_T = 5.0   # deg C
T18_ALPHA_STD_MAX = 0.1   # quantile units

TEST_LOG_COLUMNS = [
    'timestamp', 'pipeline', 'test_id', 'target', 'track',
    'passed', 'value', 'threshold', 'details',
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _append_row(row: dict):
    new = not TEST_LOG.exists()
    with open(TEST_LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=TEST_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def test_t16(ship: pd.DataFrame, fh) -> dict:
    n_ship = int((ship.winner.isin(['A', 'B'])).sum())
    passed = n_ship >= 1
    winners = ship[ship.winner.isin(['A', 'B'])][[
        'track', 'target', 'winner']].to_dict('records')
    details = {'n_cells_shipped': n_ship, 'winners': winners}
    _log(f'T16: {n_ship} cell(s) ship -> passed={passed}', fh)
    return {
        'timestamp': _now(), 'pipeline': 'all', 'test_id': 'T16',
        'target': 'ALL', 'track': 'ALL',
        'passed': passed, 'value': n_ship, 'threshold': 1,
        'details': json.dumps(details),
    }


def test_t17(ship: pd.DataFrame, pseed: pd.DataFrame, fh) -> list[dict]:
    """Count seeds whose winner matches the canonical shipping form.

    The per_seed CSV does not carry an explicit `ships` column. Winner
    per seed is already the output of ship_decision + choose_winner, so
    `winner == canonical_form` is the correct per-seed ship signal."""
    rows = []
    ps_all = pseed[pseed.regime == 'ALL'].drop_duplicates(
        ['pipeline', 'track', 'target', 'seed'])
    for _, r in ship.iterrows():
        w = r['winner']
        if w not in ('A', 'B'):
            continue
        ps = ps_all[(ps_all.track == r['track']) & (ps_all.target == r['target'])]
        n_ship = int((ps['winner'] == w).sum())
        n_tot = len(ps)
        passed = n_ship > (n_tot / 2.0) if n_tot > 0 else False
        _log(f'T17 {r["track"]}/{r["target"]} form={w}: '
             f'{n_ship}/{n_tot} seeds pick {w} -> passed={passed}', fh)
        rows.append({
            'timestamp': _now(),
            'pipeline': r['pipeline'] if 'pipeline' in r else 'all',
            'test_id': 'T17',
            'target': r['target'], 'track': r['track'],
            'passed': passed, 'value': n_ship, 'threshold': n_tot // 2 + 1,
            'details': json.dumps({
                'form': w, 'n_seeds_total': n_tot, 'n_seeds_ship': n_ship,
            }),
        })
    return rows


def test_t18(edge: pd.DataFrame | None, stab: pd.DataFrame | None,
             fh) -> list[dict]:
    rows = []
    # T18a: Form A edge sensitivity.
    if edge is not None:
        ecell = edge.drop_duplicates(
            ['pipeline', 'track', 'target', 'perturbation'])
        for k, g in ecell.groupby(['pipeline', 'track', 'target']):
            base = g[g.perturbation == 'base']
            if base.empty:
                continue
            b_delta = float(base['overall_delta'].iloc[0])
            swings = []
            for es in ('inner_m1', 'inner_p1'):
                ss = g[g.perturbation == es]
                if not ss.empty:
                    swings.append(abs(float(ss['overall_delta'].iloc[0]) - b_delta))
            max_swing = max(swings) if swings else np.nan
            threshold = T18_EDGE_THRESH_P if k[2] == 'P_kbar' else T18_EDGE_THRESH_T
            passed = bool(np.isfinite(max_swing) and max_swing < threshold)
            _log(f'T18a(edge) {k[1]}/{k[2]}: swing={max_swing:.3f} '
                 f'threshold={threshold:.2f} -> passed={passed}', fh)
            rows.append({
                'timestamp': _now(), 'pipeline': k[0], 'test_id': 'T18a',
                'target': k[2], 'track': k[1],
                'passed': passed, 'value': round(float(max_swing), 4) if np.isfinite(max_swing) else None,
                'threshold': threshold,
                'details': json.dumps({'base_delta': round(b_delta, 4)}),
            })

    # T18b: Form B CV-reseed stability.
    if stab is not None:
        if 'form_b_ok' in stab.columns:
            stab = stab[stab.form_b_ok]
        for k, g in stab.groupby(['pipeline', 'track', 'target']):
            s_L = float(g['alpha_L'].std(ddof=0)) if not g['alpha_L'].isna().all() else np.nan
            s_R = float(g['alpha_R'].std(ddof=0)) if not g['alpha_R'].isna().all() else np.nan
            worst = max([x for x in (s_L, s_R) if np.isfinite(x)], default=np.nan)
            passed = bool(np.isfinite(worst) and worst < T18_ALPHA_STD_MAX)
            _log(f'T18b(stab) {k[1]}/{k[2]}: max alpha std={worst:.3f} -> '
                 f'passed={passed}', fh)
            rows.append({
                'timestamp': _now(), 'pipeline': k[0], 'test_id': 'T18b',
                'target': k[2], 'track': k[1],
                'passed': passed, 'value': round(float(worst), 4) if np.isfinite(worst) else None,
                'threshold': T18_ALPHA_STD_MAX,
                'details': json.dumps({
                    'alpha_L_std': round(s_L, 4) if np.isfinite(s_L) else None,
                    'alpha_R_std': round(s_R, 4) if np.isfinite(s_R) else None,
                }),
            })
    return rows


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        ship_p = RESULTS / 'v10_bias_correction_shipped.csv'
        pseed_p = RESULTS / 'v10_bias_correction_per_seed.csv'
        edge_p = RESULTS / 'v10_bias_correction_edge_sensitivity.csv'
        stab_p = RESULTS / 'v10_bias_correction_form_b_stability.csv'

        if not (ship_p.exists() and pseed_p.exists()):
            _log('ERROR: required Phase G.7 D2 outputs missing', fh)
            return 1

        ship = pd.read_csv(ship_p)
        pseed = pd.read_csv(pseed_p)
        edge = pd.read_csv(edge_p) if edge_p.exists() else None
        stab = pd.read_csv(stab_p) if stab_p.exists() else None

        all_rows = []
        all_rows.append(test_t16(ship, fh))
        all_rows.extend(test_t17(ship, pseed, fh))
        all_rows.extend(test_t18(edge, stab, fh))

        for r in all_rows:
            _append_row(r)
        _log(f'appended {len(all_rows)} row(s) to {TEST_LOG}', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
