#!/usr/bin/env python3
"""Phase 1 C3: extend postcorrection CSVs with TabPFN rows/cols.

Two extensions:

1. results/regime_allmodels_postcorrection.csv
   - Append `method_family='tabpfn'` rows for the 4 opx combos x 5 P-regimes
     from results/tabpfn_regime_rmse.csv (already computed, 20-seed mean).
   - Append `method_family='tabpfn_corrected'` rows for opx_only/T_C +
     opx_only/P_kbar (the 2 combos whose TabPFN Form A ships at canonical
     seed 42). Use canonical seed 42's test predictions (from
     tabpfn_predictions.csv) + Form A params (from tabpfn_bias_correction
     _perseed.csv), computing bootstrap 95% CI per regime with n_boot=500.

2. results/preregistered_scorecard_postcorrection.csv
   - Add 4 columns: tabpfn_post_rmse, tabpfn_post_rmse_lo,
     tabpfn_post_rmse_hi, tabpfn_correction_form. For non-shipped combos,
     tabpfn_post_rmse = tabpfn_rmse (copied) and tabpfn_correction_form
     = 'none'. For shipped combos, tabpfn_post_rmse comes from the
     corrected-test RMSE with bootstrap CI.
   - Re-evaluate `winner` using 5 candidates:
     {v10_pre, v10_corrected, external, tabpfn, tabpfn_corrected}.

Idempotent-ish: if tabpfn_corrected rows already present in
regime_allmodels_postcorrection, drops them first before re-adding.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS
from src.bias_correction import apply_form_a
from src.evaluation import _rmse, assign_p_regime

CANONICAL_SEED = 42
OPX_COMBOS = [
    ('opx', 'opx_liq',  'T_C'),
    ('opx', 'opx_liq',  'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]
P_REGIMES = ['shallow_crustal', 'deep_crustal_MASH',
             'lithospheric_mantle', 'deeper_mantle', 'ALL']
N_BOOT = 500


def _rmse_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = 0
                       ) -> tuple[float, float, float]:
    if len(y_true) == 0:
        return float('nan'), float('nan'), float('nan')
    point = _rmse(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = _rmse(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def _mae_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                      n_boot: int = N_BOOT, seed: int = 0
                      ) -> tuple[float, float, float]:
    if len(y_true) == 0:
        return float('nan'), float('nan'), float('nan')
    point = float(np.mean(np.abs(y_true - y_pred)))
    rng = np.random.default_rng(seed + 1)
    n = len(y_true)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi)


def extend_regime_allmodels():
    """Append tabpfn + tabpfn_corrected rows to regime_allmodels_postcorrection.csv."""
    reg_path = RESULTS / 'regime_allmodels_postcorrection.csv'
    tf_path = RESULTS / 'tabpfn_regime_rmse.csv'
    pred_path = RESULTS / 'tabpfn_predictions.csv'
    perseed_path = RESULTS / 'tabpfn_bias_correction_perseed.csv'

    reg = pd.read_csv(reg_path)
    n_initial = len(reg)

    # Idempotency: drop any prior tabpfn rows on opx.
    mask_drop = ((reg['pipeline'] == 'opx')
                 & (reg['method_family'].isin(['tabpfn', 'tabpfn_corrected'])))
    if mask_drop.any():
        reg = reg[~mask_drop].copy()
        print(f'  dropped {int(mask_drop.sum())} prior tabpfn rows on opx')

    tf = pd.read_csv(tf_path)
    perseed = pd.read_csv(perseed_path)
    preds = pd.read_csv(pred_path)

    # --- Block 1: tabpfn (uncorrected) rows for 4 opx combos x 5 P-regimes.
    tf_opx = tf[tf['pipeline'] == 'opx'].copy()
    # Need to add `correction_form` = 'none' column to match postcorrection schema.
    tf_opx['correction_form'] = 'none'

    # Ensure column alignment with regime_allmodels_postcorrection.
    missing_cols = set(reg.columns) - set(tf_opx.columns)
    extra_cols = set(tf_opx.columns) - set(reg.columns)
    for c in missing_cols:
        tf_opx[c] = np.nan if 'rmse' in c or 'mae' in c else ''
    tf_opx = tf_opx[reg.columns]

    # --- Block 2: tabpfn_corrected rows for combos whose Form A ships at seed=42.
    corrected_rows = []
    shipping_combos = perseed[(perseed['seed'] == CANONICAL_SEED)
                              & (perseed['ship_a'] == True)].copy()
    shipping_keys = set((r['pipeline'], r['track'], r['target'])
                        for _, r in shipping_combos.iterrows())
    print(f'  shipping combos @ seed={CANONICAL_SEED}: {shipping_keys}')

    for _, pr in shipping_combos.iterrows():
        pipeline, track, target = pr['pipeline'], pr['track'], pr['target']
        params_a_raw = json.loads(pr['params_a_json'])
        params_a = {k: (v['a'], v['b']) for k, v in params_a_raw.items()}

        # Test predictions at canonical seed.
        sub = preds[(preds['pipeline'] == pipeline)
                    & (preds['track'] == track)
                    & (preds['target'] == target)
                    & (preds['seed'] == CANONICAL_SEED)].copy()
        if sub.empty:
            print(f'  WARN: no test preds for {pipeline}/{track}/{target}@{CANONICAL_SEED}')
            continue
        sub = sub.sort_values('sample_idx').reset_index(drop=True)
        y_true = sub['y_true'].to_numpy(float)
        y_pred = sub['y_pred'].to_numpy(float)
        regimes = sub['regime'].to_numpy()
        y_corr = apply_form_a(y_pred, regimes, params_a)

        # Template row from any existing v10_corrected in this combo for regime_lo/hi etc.
        v10_corr = reg[(reg['pipeline'] == pipeline)
                       & (reg['track'] == track)
                       & (reg['target'] == target)
                       & (reg['method_family'] == 'v10_corrected')]
        tmpl_by_regime = {r['regime']: r for _, r in v10_corr.iterrows()}

        for regime in P_REGIMES:
            if regime == 'ALL':
                m = np.ones(len(y_true), dtype=bool)
            else:
                m = (regimes == regime)
            n = int(m.sum())
            if n == 0:
                continue
            rmse, rmse_lo, rmse_hi = _rmse_bootstrap_ci(y_true[m], y_corr[m],
                                                        n_boot=N_BOOT, seed=42)
            mae, mae_lo, mae_hi = _mae_bootstrap_ci(y_true[m], y_corr[m],
                                                    n_boot=N_BOOT, seed=42)

            # Template for regime_lo/hi/regime_type/n_used
            if regime in tmpl_by_regime:
                tmpl = tmpl_by_regime[regime]
                regime_type = tmpl['regime_type']
                regime_lo = tmpl['regime_lo']
                regime_hi = tmpl['regime_hi']
            else:
                # Fallback: P regime
                regime_type = 'P'
                bounds = {
                    'shallow_crustal': (0.0, 5.0),
                    'deep_crustal_MASH': (5.0, 10.0),
                    'lithospheric_mantle': (10.0, 20.0),
                    'deeper_mantle': (20.0, 1000.0),
                    'ALL': (0.0, 1000.0),
                }
                regime_lo, regime_hi = bounds[regime]

            corrected_rows.append({
                'pipeline': pipeline, 'track': track, 'target': target,
                'regime_type': regime_type, 'regime': regime,
                'regime_lo': regime_lo, 'regime_hi': regime_hi,
                'n': n,
                'method_family': 'tabpfn_corrected',
                'method': 'TabPFN/raw',
                'method_label': 'TabPFN v2 (corr=A)',
                'rmse': rmse, 'rmse_lo': rmse_lo, 'rmse_hi': rmse_hi,
                'mae': mae, 'mae_lo': mae_lo, 'mae_hi': mae_hi,
                'n_used': n, 'correction_form': 'A',
            })

    corrected_df = pd.DataFrame(corrected_rows)
    if not corrected_df.empty:
        corrected_df = corrected_df[reg.columns]

    out = pd.concat([reg, tf_opx, corrected_df], ignore_index=True)
    out.to_csv(reg_path, index=False)
    print(f'regime_allmodels_postcorrection: {n_initial} -> {len(out)} rows '
          f'(+{len(tf_opx)} tabpfn +{len(corrected_df)} tabpfn_corrected)')

    return corrected_df


def extend_scorecard(corrected_df: pd.DataFrame):
    """Add tabpfn_post_rmse columns to scorecard; re-evaluate winner."""
    sc_path = RESULTS / 'preregistered_scorecard_postcorrection.csv'
    sc = pd.read_csv(sc_path)
    n_initial = len(sc)

    # Drop prior if present.
    for c in ('tabpfn_post_rmse', 'tabpfn_post_rmse_lo', 'tabpfn_post_rmse_hi',
              'tabpfn_correction_form'):
        if c in sc.columns:
            sc = sc.drop(columns=c)

    # Default: tabpfn_post = tabpfn (pre) with form='none'.
    sc['tabpfn_post_rmse'] = sc['tabpfn_rmse'].astype(float)
    sc['tabpfn_post_rmse_lo'] = sc['tabpfn_rmse_lo'].astype(float)
    sc['tabpfn_post_rmse_hi'] = sc['tabpfn_rmse_hi'].astype(float)
    sc['tabpfn_correction_form'] = 'none'

    # For shipped opx combos, overwrite with corrected values.
    for _, r in corrected_df.iterrows():
        mask = ((sc['track'] == r['track']) & (sc['target'] == r['target'])
                & (sc['regime'] == r['regime']))
        if not mask.any():
            continue
        sc.loc[mask, 'tabpfn_post_rmse'] = r['rmse']
        sc.loc[mask, 'tabpfn_post_rmse_lo'] = r['rmse_lo']
        sc.loc[mask, 'tabpfn_post_rmse_hi'] = r['rmse_hi']
        sc.loc[mask, 'tabpfn_correction_form'] = 'A'

    # Re-evaluate winner with 5 candidates.
    def pick_winner(row: pd.Series) -> str:
        ext = row.get('best_external_rmse')
        cand = {
            'v10_pre': row['v10_pre_rmse'],
            'v10_corrected': row['v10_post_rmse'],
            'external': ext if pd.notna(ext) else np.inf,
            'tabpfn': row['tabpfn_rmse'] if pd.notna(row['tabpfn_rmse']) else np.inf,
            'tabpfn_corrected': (row['tabpfn_post_rmse']
                                  if pd.notna(row['tabpfn_post_rmse']) else np.inf),
        }
        # If tabpfn_correction_form == 'none', tabpfn_corrected == tabpfn: collapse.
        if row['tabpfn_correction_form'] == 'none':
            cand.pop('tabpfn_corrected')
        return min(cand, key=cand.get)

    sc['winner'] = sc.apply(pick_winner, axis=1)

    sc.to_csv(sc_path, index=False)
    wc = sc['winner'].value_counts().to_dict()
    print(f'scorecard: {n_initial} rows; winners={wc}')
    print(f'tabpfn_correction_form counts: '
          f'{sc["tabpfn_correction_form"].value_counts().to_dict()}')


def main() -> int:
    print('=== C3: extend postcorrection CSVs with TabPFN ===')
    corrected_df = extend_regime_allmodels()
    extend_scorecard(corrected_df)
    print('done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
