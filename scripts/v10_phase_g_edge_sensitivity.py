#!/usr/bin/env python3
"""Phase G.7 A4: regime-edge sensitivity for Form A bias correction.

The pre-registered P-regime edges are [0, 5, 15, 30, 100] kbar. This
script checks that the Form A fit is not knife-edge sensitive to those
choices by perturbing the inner edges (5, 15, 30) by +/- 1 kbar and
re-fitting Form A on the same cached OOF predictions.

Perturbations tested (inner edges only; 0 and 100 stay fixed):
    base      = [0, 5, 15, 30, 100]     (pre-registered)
    inner_m1  = [0, 4, 14, 29, 100]     (all inner edges -1 kbar)
    inner_p1  = [0, 6, 16, 31, 100]     (all inner edges +1 kbar)

For each (cell, perturbation) we:
    1. Re-assign train + test regimes under the perturbed edges.
    2. Re-fit Form A per-regime OLS on the cached OOF predictions.
    3. Re-apply Form A to the cached test predictions.
    4. Re-compute overall + per-regime pre/post RMSE.
    5. Re-run ship_decision.

Output:
    results/v10_bias_correction_edge_sensitivity.csv

One row per (cell, perturbation, regime), plus a summary row per
(cell, perturbation) with the ship decision. Consumes seed=42
checkpoints from results/v10_bias_correction/checkpoints/.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import LOGS, P_CEILING_KBAR, RESULTS
from src.bias_correction import (
    apply_form_a,
    fit_form_a,
    ship_decision,
)
from src.data import (
    load_cpx_liq, load_cpx_only, load_opx_liq, load_opx_only, load_splits,
)
from src.evaluation import _bootstrap_stat, _rmse

LOG_PATH = LOGS / 'v10_phase_g_edge_sensitivity.log'
CKPT_DIR = RESULTS / 'v10_bias_correction' / 'checkpoints'
OUT_CSV = RESULTS / 'v10_bias_correction_edge_sensitivity.csv'
CANONICAL_SEED = 42

LOADERS = {
    'opx_liq':  (load_opx_liq,  'opx_liq'),
    'opx_only': (load_opx_only, 'opx_only'),
    'cpx_liq':  (load_cpx_liq,  'cpx_liq'),
    'cpx_only': (load_cpx_only, 'cpx_only'),
}

PERTURBATIONS = {
    'base':     [0.0,  5.0, 15.0, 30.0, P_CEILING_KBAR],
    'inner_m1': [0.0,  4.0, 14.0, 29.0, P_CEILING_KBAR],
    'inner_p1': [0.0,  6.0, 16.0, 31.0, P_CEILING_KBAR],
}

LABELS = ['shallow_crustal', 'deep_crustal_MASH',
          'lithospheric_mantle', 'deeper_mantle']


def assign_regime_with_edges(p_kbar: np.ndarray, edges: list) -> np.ndarray:
    """Same structure as src.evaluation.assign_p_regime but parametrised
    on the edge list so we can test perturbations."""
    p = np.asarray(p_kbar, dtype=float)
    labels = np.array(['unassigned'] * len(p), dtype=object)
    finite = np.isfinite(p)
    if not finite.any():
        return labels
    p_valid = p[finite]
    p_clipped = np.clip(p_valid, 0.0, edges[-1] - 1e-9)
    bin_idx = np.digitize(p_clipped, edges[1:-1], right=False)
    labels[finite] = np.array([LABELS[i] for i in bin_idx], dtype=object)
    return labels


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def refit_and_evaluate(ckpt, perturbation_name, edges,
                       y_tr, p_tr, y_te, p_te):
    """Refit Form A on cached OOF under perturbed regimes, apply to
    cached test preds, and compute per-regime + ALL RMSE with 95% CI."""
    oof_tr = ckpt.oof_tr.astype(np.float64)
    y_pred_te = ckpt.y_pred_te.astype(np.float64)

    regimes_tr = assign_regime_with_edges(p_tr, edges)
    regimes_te = assign_regime_with_edges(p_te, edges)
    mask_fin = np.isfinite(oof_tr)

    params_a = fit_form_a(y_tr[mask_fin], oof_tr[mask_fin],
                          regimes_tr[mask_fin])
    y_corr_te = apply_form_a(y_pred_te, regimes_te, params_a)

    rows = []
    pre_regime = {}
    post_regime = {}

    for r in LABELS + ['ALL']:
        if r == 'ALL':
            mask = np.ones(len(y_te), dtype=bool)
        else:
            mask = (regimes_te == r)
        n_r = int(mask.sum())
        if n_r < 5:
            rows.append(dict(perturbation=perturbation_name, regime=r,
                             n=n_r, edges=str(edges),
                             a=np.nan, b=np.nan,
                             pre_rmse=np.nan, pre_lo=np.nan, pre_hi=np.nan,
                             post_rmse=np.nan, post_lo=np.nan, post_hi=np.nan,
                             delta_rmse=np.nan))
            continue
        pre, pre_lo, pre_hi = _bootstrap_stat(
            y_te[mask], y_pred_te[mask], _rmse, n_bootstrap=500, seed=42)
        post, post_lo, post_hi = _bootstrap_stat(
            y_te[mask], y_corr_te[mask], _rmse, n_bootstrap=500, seed=42)
        if r != 'ALL':
            pre_regime[r] = pre
            post_regime[r] = post
            a_r, b_r = params_a.get(r, (np.nan, np.nan))
        else:
            a_r, b_r = (np.nan, np.nan)
        rows.append(dict(perturbation=perturbation_name, regime=r,
                         n=n_r, edges=str(edges),
                         a=a_r, b=b_r,
                         pre_rmse=pre, pre_lo=pre_lo, pre_hi=pre_hi,
                         post_rmse=post, post_lo=post_lo, post_hi=post_hi,
                         delta_rmse=pre - post))

    pre_all = next(r['pre_rmse'] for r in rows if r['regime'] == 'ALL')
    post_all = next(r['post_rmse'] for r in rows if r['regime'] == 'ALL')
    sd = ship_decision('A', pre_all, post_all, pre_regime, post_regime)
    return rows, sd


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'A4 edge sensitivity: perturbations={list(PERTURBATIONS)}', fh)

        all_rows = []
        ckpt_files = sorted(CKPT_DIR.glob(f'*_s{CANONICAL_SEED}.pkl'))
        if not ckpt_files:
            _log(f'No seed={CANONICAL_SEED} checkpoints in {CKPT_DIR}. '
                 f'Run the bias-correction driver first.', fh)
            return 1

        data_cache = {}

        for ckpt_path in ckpt_files:
            with open(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
            track = ckpt.track
            target = ckpt.target
            cell_tag = (f'{ckpt.pipeline}/{track}/{target}/{ckpt.model}/'
                        f'{ckpt.feature_set}')
            _log(f'{cell_tag}', fh)

            if track not in data_cache:
                loader, split_key = LOADERS[track]
                df = loader()
                tr_idx, te_idx = load_splits(split_key)
                data_cache[track] = {
                    'p_tr': df['P_kbar'].to_numpy(dtype=float)[tr_idx],
                    'p_te': df['P_kbar'].to_numpy(dtype=float)[te_idx],
                    'y_tr_map': {
                        'T_C':    df['T_C'].to_numpy(dtype=float)[tr_idx],
                        'P_kbar': df['P_kbar'].to_numpy(dtype=float)[tr_idx],
                    },
                    'y_te_map': {
                        'T_C':    df['T_C'].to_numpy(dtype=float)[te_idx],
                        'P_kbar': df['P_kbar'].to_numpy(dtype=float)[te_idx],
                    },
                }
            dc = data_cache[track]
            y_tr = dc['y_tr_map'][target]
            y_te = dc['y_te_map'][target]

            for name, edges in PERTURBATIONS.items():
                rows, sd = refit_and_evaluate(
                    ckpt, name, edges,
                    y_tr, dc['p_tr'], y_te, dc['p_te'])
                for r in rows:
                    r.update({
                        'pipeline': ckpt.pipeline, 'track': track,
                        'target': target, 'model': ckpt.model,
                        'feature_set': ckpt.feature_set,
                        'seed': ckpt.seed,
                        'ships': sd.ships,
                        'overall_delta': sd.overall_delta,
                        'max_regime_degradation': sd.max_regime_degradation,
                    })
                    all_rows.append(r)
                _log(f'  {name:9s} ships={sd.ships} '
                     f'overall_delta={sd.overall_delta:+.3f} '
                     f'worst_regime_delta={sd.max_regime_degradation:+.3f}', fh)

        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV}  rows={len(df)}', fh)

        # Print a compact cross-perturbation ship-decision stability summary.
        ship_pivot = (df[df.regime == 'ALL']
                      .pivot_table(index=['pipeline', 'track', 'target'],
                                   columns='perturbation',
                                   values='ships', aggfunc='first'))
        _log('ship-decision stability across perturbations:', fh)
        for line in ship_pivot.to_string().split('\n'):
            _log('  ' + line, fh)

        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
