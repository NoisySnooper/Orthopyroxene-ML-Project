#!/usr/bin/env python3
"""Fit Form A / Form B bias correction on TabPFN OOF residuals.

Inputs:
    results/tabpfn_oof_predictions.csv   (from C1, 5 seeds x 10 folds)
    results/tabpfn_predictions.csv        (existing 20-seed test preds)

For each of 4 opx combinations at each of seeds 42-46:
    1. Assemble OOF (y_true, y_oof_pred, regime) on training split
    2. fit_form_a, fit_form_b
    3. Apply to that seed's test predictions -> post RMSE + per-regime
    4. ship_decision for A and B, choose_winner

Outputs:
    results/tabpfn_bias_correction_perseed.csv   (5 seeds x 4 combos)
    results/tabpfn_bias_correction_summary.csv   (4 rows, aggregated)

Also updates the four `TabPFN` rows (winner='excluded') in
    results/bias_correction_shipped.csv
to reflect the actual winner decision, using the canonical seed 42's
Form A/B parameters. n_seeds_done = 5 (the OOF fit scope).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import RESULTS
from src.bias_correction import (
    SHIP_TOL, FormBParams, apply_form_a, apply_form_b,
    choose_winner, evaluate_pre_post, fit_form_a, fit_form_b,
    ship_decision,
)
from src.evaluation import _rmse, assign_p_regime

COMBOS = [
    ('opx', 'opx_liq',  'T_C'),
    ('opx', 'opx_liq',  'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]
SEEDS = [42, 43, 44, 45, 46]
CANONICAL_SEED = 42

OOF_CSV = RESULTS / 'tabpfn_oof_predictions.csv'
TEST_CSV = RESULTS / 'tabpfn_predictions.csv'
OUT_PERSEED = RESULTS / 'tabpfn_bias_correction_perseed.csv'
OUT_SUMMARY = RESULTS / 'tabpfn_bias_correction_summary.csv'
SHIPPED_CSV = RESULTS / 'bias_correction_shipped.csv'


def _process_seed(oof_df: pd.DataFrame, test_df: pd.DataFrame,
                  pipeline: str, track: str, target: str, seed: int
                  ) -> dict:
    oof_mask = ((oof_df['pipeline'] == pipeline) & (oof_df['track'] == track)
                & (oof_df['target'] == target) & (oof_df['seed'] == seed))
    oof_sub = oof_df[oof_mask].sort_values('sample_idx').reset_index(drop=True)
    if oof_sub.empty:
        return {'skip': True, 'reason': 'no OOF rows'}

    y_tr = oof_sub['y_true'].to_numpy(float)
    y_oof = oof_sub['y_oof_pred'].to_numpy(float)
    regimes_tr = oof_sub['regime'].to_numpy()

    params_a = fit_form_a(y_tr, y_oof, regimes_tr)
    params_b = fit_form_b(y_tr, y_oof)

    test_mask = ((test_df['pipeline'] == pipeline)
                 & (test_df['track'] == track)
                 & (test_df['target'] == target)
                 & (test_df['seed'] == seed))
    test_sub = test_df[test_mask].sort_values('sample_idx').reset_index(drop=True)
    if test_sub.empty:
        return {'skip': True, 'reason': 'no test rows'}
    y_te = test_sub['y_true'].to_numpy(float)
    y_pred_te = test_sub['y_pred'].to_numpy(float)
    regimes_te = test_sub['regime'].to_numpy()

    y_corr_a = apply_form_a(y_pred_te, regimes_te, params_a)
    y_corr_b = (apply_form_b(y_pred_te, params_b)
                if params_b is not None else y_pred_te.copy())

    rows_a = evaluate_pre_post(y_te, y_pred_te, y_corr_a, regimes_te,
                                n_boot_ci=500)
    rows_b = evaluate_pre_post(y_te, y_pred_te, y_corr_b, regimes_te,
                                n_boot_ci=500)

    pre_regime = {r['regime']: r['pre_rmse'] for r in rows_a
                  if r['regime'] != 'ALL'}
    post_regime_a = {r['regime']: r['post_rmse'] for r in rows_a
                     if r['regime'] != 'ALL'}
    post_regime_b = {r['regime']: r['post_rmse'] for r in rows_b
                     if r['regime'] != 'ALL'}
    pre_all = next(r['pre_rmse'] for r in rows_a if r['regime'] == 'ALL')
    post_all_a = next(r['post_rmse'] for r in rows_a if r['regime'] == 'ALL')
    post_all_b = next(r['post_rmse'] for r in rows_b if r['regime'] == 'ALL')

    sa = ship_decision('A', pre_all, post_all_a,
                       pre_regime, post_regime_a, tol=SHIP_TOL)
    if params_b is None:
        from src.bias_correction import ShipDecision
        sb = ShipDecision(form='B', ships=False,
                          overall_delta=0.0,
                          max_regime_degradation=0.0,
                          reason='no valid Form B fit on TabPFN OOF')
    else:
        sb = ship_decision('B', pre_all, post_all_b,
                           pre_regime, post_regime_b, tol=SHIP_TOL)
    winner = choose_winner(sa, sb)

    return {
        'skip': False,
        'pipeline': pipeline, 'track': track, 'target': target,
        'seed': seed,
        'pre_rmse_all': pre_all,
        'post_rmse_a': post_all_a, 'post_rmse_b': post_all_b,
        'ship_a': sa.ships, 'ship_b': sb.ships,
        'winner': winner,
        'form_a_overall_delta': sa.overall_delta,
        'form_a_max_regime_degr': sa.max_regime_degradation,
        'form_b_overall_delta': sb.overall_delta,
        'form_b_max_regime_degr': sb.max_regime_degradation,
        'params_a': {r: {'a': a, 'b': b} for r, (a, b) in params_a.items()},
        'params_b': (asdict(params_b) if params_b is not None else None),
        'decision_a': sa.to_dict(),
        'decision_b': sb.to_dict(),
    }


def _update_shipped(shipped_df: pd.DataFrame, summary_rows: list[dict],
                    canonical: dict[tuple, dict]) -> pd.DataFrame:
    df = shipped_df.copy()
    for _, row in df.iterrows():
        pass
    for i, row in df.iterrows():
        if row['model'] != 'TabPFN' or row['pipeline'] != 'opx':
            continue
        key = (row['pipeline'], row['track'], row['target'])
        if key not in canonical:
            continue
        c = canonical[key]
        s = next((s for s in summary_rows
                  if (s['pipeline'], s['track'], s['target']) == key), None)
        if s is None:
            continue
        df.at[i, 'canonical_seed'] = CANONICAL_SEED
        df.at[i, 'winner'] = str(c['winner'])
        df.at[i, 'ship_a'] = str(bool(c['ship_a']))
        df.at[i, 'ship_b'] = str(bool(c['ship_b']))
        df.at[i, 'form_a_params'] = json.dumps(c['decision_a'])
        df.at[i, 'form_b_params'] = json.dumps(c['decision_b'])
        df.at[i, 'n_seeds_done'] = len(SEEDS)
    return df


def main() -> int:
    if not OOF_CSV.exists():
        print(f'ERROR: {OOF_CSV} not found. Run C1 first.')
        return 2
    if not TEST_CSV.exists():
        print(f'ERROR: {TEST_CSV} not found.')
        return 2

    oof_df = pd.read_csv(OOF_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f'OOF rows: {len(oof_df)}; test rows: {len(test_df)}')

    perseed_rows: list[dict] = []
    canonical: dict[tuple, dict] = {}

    for (pipeline, track, target) in COMBOS:
        for seed in SEEDS:
            out = _process_seed(oof_df, test_df, pipeline, track, target, seed)
            if out.get('skip'):
                print(f'  [skip] {pipeline}/{track}/{target} seed={seed}: '
                      f'{out.get("reason")}')
                continue
            perseed_rows.append({
                'pipeline': out['pipeline'], 'track': out['track'],
                'target': out['target'], 'seed': out['seed'],
                'pre_rmse_all': out['pre_rmse_all'],
                'post_rmse_a': out['post_rmse_a'],
                'post_rmse_b': out['post_rmse_b'],
                'ship_a': out['ship_a'], 'ship_b': out['ship_b'],
                'winner': out['winner'],
                'form_a_overall_delta': out['form_a_overall_delta'],
                'form_a_max_regime_degr': out['form_a_max_regime_degr'],
                'form_b_overall_delta': out['form_b_overall_delta'],
                'form_b_max_regime_degr': out['form_b_max_regime_degr'],
                'params_a_json': json.dumps(out['params_a']),
                'params_b_json': json.dumps(out['params_b']),
                'decision_a_json': json.dumps(out['decision_a']),
                'decision_b_json': json.dumps(out['decision_b']),
            })
            if seed == CANONICAL_SEED:
                canonical[(pipeline, track, target)] = out
            print(f'  {pipeline}/{track}/{target} seed={seed} winner={out["winner"]} '
                  f'pre={out["pre_rmse_all"]:.3f} '
                  f'A={out["post_rmse_a"]:.3f} B={out["post_rmse_b"]:.3f}')

    if not perseed_rows:
        print('ERROR: no per-seed rows produced. Bailing.')
        return 3

    perseed_df = pd.DataFrame(perseed_rows)
    perseed_df.to_csv(OUT_PERSEED, index=False)
    print(f'Wrote {OUT_PERSEED} ({len(perseed_df)} rows)')

    summary_rows = []
    for (pipeline, track, target), grp in perseed_df.groupby(
            ['pipeline', 'track', 'target'], sort=False):
        winner_counts = grp['winner'].value_counts().to_dict()
        summary_rows.append({
            'pipeline': pipeline, 'track': track, 'target': target,
            'n_seeds': len(grp),
            'ship_a_count': int((grp['ship_a'] == True).sum()),
            'ship_b_count': int((grp['ship_b'] == True).sum()),
            'winner_A': int(winner_counts.get('A', 0)),
            'winner_B': int(winner_counts.get('B', 0)),
            'winner_none': int(winner_counts.get('none', 0)),
            'pre_rmse_mean': float(grp['pre_rmse_all'].mean()),
            'post_rmse_a_mean': float(grp['post_rmse_a'].mean()),
            'post_rmse_b_mean': float(grp['post_rmse_b'].mean()),
            'form_a_overall_delta_mean': float(grp['form_a_overall_delta'].mean()),
            'form_b_overall_delta_mean': float(grp['form_b_overall_delta'].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f'Wrote {OUT_SUMMARY} ({len(summary_df)} rows)')

    shipped_df = pd.read_csv(SHIPPED_CSV)
    shipped_updated = _update_shipped(shipped_df, summary_rows, canonical)
    shipped_updated.to_csv(SHIPPED_CSV, index=False)
    print(f'Updated {SHIPPED_CSV}')

    print('\nSummary:')
    print(summary_df[['pipeline', 'track', 'target', 'ship_a_count',
                      'ship_b_count', 'winner_A', 'winner_B', 'winner_none',
                      'pre_rmse_mean', 'post_rmse_a_mean', 'post_rmse_b_mean']]
          .to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
