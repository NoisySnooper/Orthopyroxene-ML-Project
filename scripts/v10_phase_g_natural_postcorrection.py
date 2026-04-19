#!/usr/bin/env python3
"""Phase G.7 D3b: GEOROC natural-sample post-correction inference.

Runs the opx-only aggregate-best v10 model (canonical seed=42) on the
GEOROC natural orthopyroxene compilation, then applies the shipped
bias correction (Form A or Form B, whichever won for the cell) and
reports pre/post predictions for T_C and P_kbar.

Natural samples have no ground truth, so we:
  * Predict raw (pre-correction) y_pred.
  * Assign regime using the *predicted* P_kbar (consistent with how
    Agreda-Lopez 2024 applies their correction at inference).
  * Apply the correction under that assigned regime (Form A) or under
    the fitted quantile thresholds (Form B).
  * Report both pre and post predictions in the output so downstream
    natural-sample analyses can compare them.

Output:
    results/v10_natural_opx_post_correction_inference.csv

Columns:
    citation, sample, tectonic_setting, location,
    T_C_pre, T_C_post, T_C_correction_form,
    P_kbar_pre, P_kbar_post, P_kbar_correction_form,
    regime_pre_from_P_pre, regime_post_from_P_post

Requires the canonical-seed checkpoints for:
    opx_only / T_C
    opx_only / P_kbar
from results/v10_bias_correction/checkpoints/.

We additionally train a fresh final model on the full training split
(the checkpoint only stores test-set predictions, not the fitted
estimator) before scoring GEOROC.
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

from config import DATA_NATURAL, LIN2023_NATURAL, LOGS, RESULTS
from src.bias_correction import (
    apply_form_a, apply_form_b, FormBParams,
)
from src.evaluation import assign_p_regime
from src.features import (
    add_engineered_features, build_feature_matrix, cation_recalc_6oxy,
)
from src.models import build_model
from src.prepare_train_test import prepare_train_test
from src.v10_phase_c_analysis import load_best_params

LOG_PATH = LOGS / 'v10_phase_g_natural_postcorrection.log'
CKPT_DIR = RESULTS / 'v10_bias_correction' / 'checkpoints'
OUT_CSV = RESULTS / 'v10_natural_opx_post_correction_inference.csv'
CANONICAL_SEED = 42

TARGETS = ['T_C', 'P_kbar']


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _load_natural():
    df = pd.read_csv(LIN2023_NATURAL)
    return df


def _build_natural_features(df_nat: pd.DataFrame, feature_set: str) -> np.ndarray:
    """Natural GEOROC CSV already uses training oxide names (SiO2,
    TiO2, Al2O3, Cr2O3, FeO_total, MnO, MgO, CaO, Na2O). Add cation
    recalc + engineered features, then build the requested feature
    matrix (use_liq=False, opx-only)."""
    dfx = cation_recalc_6oxy(df_nat)
    dfx = add_engineered_features(dfx)
    X, feat_names = build_feature_matrix(dfx, feature_set, use_liq=False)
    return np.asarray(X, dtype=float), feat_names


def _load_canonical_checkpoint(pipeline: str, track: str, target: str):
    """Find the canonical (seed=CANONICAL_SEED) checkpoint for this cell."""
    pattern = f'{pipeline}_{track}_{target}_*_s{CANONICAL_SEED}.pkl'
    hits = list(CKPT_DIR.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f'No seed={CANONICAL_SEED} checkpoint matching {pattern} '
            f'in {CKPT_DIR}. Run the D2 driver first.')
    with open(hits[0], 'rb') as f:
        return pickle.load(f)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('D3b: GEOROC natural opx post-correction inference', fh)

        nat = _load_natural()
        _log(f'natural samples: n={len(nat)}', fh)

        # Load canonical checkpoints for both targets.
        ckpts = {}
        for tg in TARGETS:
            ckpts[tg] = _load_canonical_checkpoint('opx', 'opx_only', tg)
            _log(f'  {tg}: {ckpts[tg].model}/{ckpts[tg].feature_set} '
                 f'winner={ckpts[tg].winner}', fh)

        # Train fresh final models at the canonical seed, each using its
        # own feature_set. (Checkpoints only store test-set predictions.)
        predictions = {}
        for tg in TARGETS:
            ck = ckpts[tg]
            best = load_best_params(RESULTS / 'v10_optuna_best_params_opx.json')
            bp = best[(ck.model, ck.target, ck.track, ck.feature_set)]['best_params']
            prep = prepare_train_test('opx', 'opx_only', tg, ck.feature_set)
            est = build_model(ck.model, bp, seed=CANONICAL_SEED)
            est.fit(prep['X_tr'], prep['y_tr'])
            X_nat, _ = _build_natural_features(nat, ck.feature_set)
            y_hat = est.predict(X_nat)
            predictions[tg] = (y_hat, ck)
            _log(f'  {tg}: predicted {len(y_hat)} natural samples '
                 f'(pre-correction)', fh)

        # Assign regime from predicted P_kbar (pre-correction).
        y_p_pre = predictions['P_kbar'][0]
        regime_pre = assign_p_regime(y_p_pre)

        # Apply correction per target using regime-from-predicted-P (Form A)
        # or the fitted quantile thresholds (Form B).
        out_rows = []
        for tg in TARGETS:
            y_pre, ck = predictions[tg]
            form = ck.winner
            if form == 'A':
                form_a_params = {
                    r: (v['a'], v['b']) for r, v in ck.form_a_params.items()
                }
                y_post = apply_form_a(y_pre, regime_pre, form_a_params)
            elif form == 'B':
                fb = FormBParams(**ck.form_b_params)
                y_post = apply_form_b(y_pre, fb)
            else:  # none
                y_post = y_pre.copy()

            predictions[tg] = (y_pre, ck, y_post, form)

        y_p_post = predictions['P_kbar'][2]
        regime_post = assign_p_regime(y_p_post)

        # Build output rows.
        for i in range(len(nat)):
            out_rows.append({
                'citation':         nat.iloc[i]['CITATION'],
                'sample':           nat.iloc[i]['SAMPLE NAME'],
                'tectonic_setting': nat.iloc[i]['TECTONIC SETTING'],
                'location':         nat.iloc[i]['LOCATION'],
                'T_C_pre':          float(predictions['T_C'][0][i]),
                'T_C_post':         float(predictions['T_C'][2][i]),
                'T_C_correction_form': predictions['T_C'][3],
                'P_kbar_pre':       float(y_p_pre[i]),
                'P_kbar_post':      float(y_p_post[i]),
                'P_kbar_correction_form': predictions['P_kbar'][3],
                'regime_pre_from_P_pre':   regime_pre[i],
                'regime_post_from_P_post': regime_post[i],
            })
        df_out = pd.DataFrame(out_rows)
        df_out.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV}  rows={len(df_out)}', fh)

        # Print per-regime distribution pre vs post.
        _log('regime distribution pre vs post (from predicted P_kbar):', fh)
        pre_counts = pd.Series(regime_pre).value_counts().to_dict()
        post_counts = pd.Series(regime_post).value_counts().to_dict()
        all_regs = set(pre_counts) | set(post_counts)
        for r in sorted(all_regs):
            _log(f'  {r:20s} pre={pre_counts.get(r, 0):6d}  '
                 f'post={post_counts.get(r, 0):6d}', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
