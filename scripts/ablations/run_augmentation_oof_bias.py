"""Sections 5-7 for nb04b_aug_test: OOF predictions + bias correction +
ship-if-better verdict, under the 15x augmented protocol.

Runs for the 4 opx combinations at their non-augmented winning cell
(model, feature_set) pulled from results/opx_multiseed_summary.csv, at
each seed in SPLIT_SEEDS.

Outputs:
    results/augmentation_ablation_opx_bias_correction.csv
    results/augmentation_ablation_opx_regime_rmse.csv
    results/augmentation_ablation_opx_oof.pkl  (OOF arrays for fig03/04)

The augmented OOF is computed via
src.ablations.augmentation.oof_predict_augmented, which splits CV on the
original (un-augmented) data so augmented rows never leak into a held-
out fold.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import SPLIT_SEEDS
from src.ablations.augmentation import (
    augment_gaussian, oof_predict_augmented,
)
from src.bias_correction import (
    SHIP_TOL, ShipDecision, apply_form_a, apply_form_b,
    choose_winner, evaluate_pre_post, fit_form_a, fit_form_b,
    paired_bootstrap_delta_rmse, ship_decision,
)
from src.evaluation import assign_p_regime, _rmse
from src.models import build_model
from src.prepare_train_test import prepare_train_test

AUG_N_COPIES = 15
AUG_REL_NOISE = 0.03
N_FOLDS = 10

TRACK_TARGET_COMBOS = [
    ('opx_liq', 'T_C'),
    ('opx_liq', 'P_kbar'),
    ('opx_only', 'T_C'),
    ('opx_only', 'P_kbar'),
]

OUT_BIAS = ROOT / 'results' / 'augmentation_ablation_opx_bias_correction.csv'
OUT_REGIME = ROOT / 'results' / 'augmentation_ablation_opx_regime_rmse.csv'
OUT_OOF = ROOT / 'results' / 'augmentation_ablation_opx_oof.pkl'
BEST_PARAMS_JSON = ROOT / 'results' / 'optuna_best_params_opx.json'
OPX_SUMMARY_CSV = ROOT / 'results' / 'opx_multiseed_summary.csv'


def load_best_params_idx():
    with open(BEST_PARAMS_JSON) as f:
        payload = json.load(f)
    idx = {}
    for r in payload['results']:
        idx[(r['model'], r['target'], r['track'], r['feature_set'])] = r['best_params']
    return idx


def pick_winners_from_baseline() -> list[dict]:
    """Lowest mean RMSE over 8 models x 3 feature_sets per (track, target)."""
    df = pd.read_csv(OPX_SUMMARY_CSV)
    # Exclude TabPFN (not in MODELS); keep 8 tuned families only.
    df = df[df['model'] != 'TabPFN'].copy()
    winners = []
    for (track, target), grp in df.groupby(['track', 'target']):
        best = grp.loc[grp['mean'].idxmin()]
        winners.append({
            'track': track, 'target': target,
            'model': best['model'], 'feature_set': best['feature_set'],
            'baseline_mean_rmse': float(best['mean']),
            'baseline_std': float(best['std']),
        })
    return winners


def p_true_arrays(track: str, target: str):
    """Get the true P_kbar on train and test for regime assignment.
    For T_C targets, load P_kbar separately from the same split."""
    from src.data import load_splits
    from src.opx_tb_analysis import load_track
    df = load_track(track)
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    return df_tr['P_kbar'].to_numpy(float), df_te['P_kbar'].to_numpy(float)


def main():
    best_params_idx = load_best_params_idx()
    winners = pick_winners_from_baseline()
    print(f'Selected {len(winners)} winners:')
    for w in winners:
        print(f"  {w['track']}/{w['target']}: {w['model']}/{w['feature_set']} "
              f"(baseline {w['baseline_mean_rmse']:.3f})")

    bias_rows = []
    regime_rows = []
    oof_store: dict = {}

    wall_start = time.time()
    n_done = 0
    n_total = len(winners) * len(SPLIT_SEEDS)

    for w in winners:
        track = w['track']; target = w['target']
        model = w['model']; feature_set = w['feature_set']
        params_key = (model, target, track, feature_set)
        best_params = best_params_idx[params_key]

        splits = prepare_train_test('opx', track, target, feature_set)
        X_tr = splits['X_tr']; y_tr = splits['y_tr']
        groups_tr = splits['groups_tr']
        X_te = splits['X_te']; y_te = splits['y_te']

        # Regime labels (from true P_kbar) for both splits.
        p_true_tr, p_true_te = p_true_arrays(track, target)
        regimes_tr = assign_p_regime(p_true_tr)
        regimes_te = assign_p_regime(p_true_te)

        oof_store[(track, target)] = {
            'model': model, 'feature_set': feature_set,
            'y_tr': y_tr, 'regimes_tr': regimes_tr,
            'y_te': y_te, 'regimes_te': regimes_te,
            'per_seed': {},
        }

        for seed in SPLIT_SEEDS:
            t0 = time.time()
            X_aug, y_aug, cit_aug = augment_gaussian(
                X_tr, y_tr, groups_tr,
                n_copies=AUG_N_COPIES, rel_noise=AUG_REL_NOISE, seed=seed,
                clip_nonneg=True,
            )

            oof = oof_predict_augmented(
                model, best_params,
                X_tr, y_tr, groups_tr,
                X_aug, y_aug, cit_aug,
                seed=seed, n_folds=N_FOLDS,
            )

            params_a = fit_form_a(y_tr, oof, regimes_tr)
            params_b = fit_form_b(y_tr, oof)

            est = build_model(model, best_params, seed=seed)
            est.fit(X_aug, y_aug)
            y_pred_te = est.predict(X_te)
            y_corr_a_te = apply_form_a(y_pred_te, regimes_te, params_a)
            y_corr_b_te = (apply_form_b(y_pred_te, params_b)
                           if params_b is not None else y_pred_te.copy())

            form_a_rows = evaluate_pre_post(
                y_te, y_pred_te, y_corr_a_te, regimes_te,
                n_boot_ci=500)
            form_b_rows = evaluate_pre_post(
                y_te, y_pred_te, y_corr_b_te, regimes_te,
                n_boot_ci=500)

            pre_regime = {r['regime']: r['pre_rmse'] for r in form_a_rows
                          if r['regime'] != 'ALL'}
            post_regime_a = {r['regime']: r['post_rmse'] for r in form_a_rows
                             if r['regime'] != 'ALL'}
            post_regime_b = {r['regime']: r['post_rmse'] for r in form_b_rows
                             if r['regime'] != 'ALL'}
            pre_all = next(r['pre_rmse'] for r in form_a_rows if r['regime'] == 'ALL')
            post_all_a = next(r['post_rmse'] for r in form_a_rows if r['regime'] == 'ALL')
            post_all_b = next(r['post_rmse'] for r in form_b_rows if r['regime'] == 'ALL')

            sa = ship_decision('A', pre_all, post_all_a,
                               pre_regime, post_regime_a, tol=SHIP_TOL)
            if params_b is None:
                sb = ShipDecision(form='B', ships=False,
                                  overall_delta=0.0,
                                  max_regime_degradation=0.0,
                                  reason='no valid Form B fit on OOF (aug)')
            else:
                sb = ship_decision('B', pre_all, post_all_b,
                                   pre_regime, post_regime_b, tol=SHIP_TOL)
            winner = choose_winner(sa, sb)

            bias_rows.append({
                'pipeline': 'opx',
                'track': track, 'target': target,
                'model': model, 'feature_set': feature_set,
                'seed': int(seed),
                'aug_n_copies': AUG_N_COPIES, 'aug_rel_noise': AUG_REL_NOISE,
                'pre_rmse_all': pre_all,
                'post_rmse_a': post_all_a,
                'post_rmse_b': post_all_b,
                'ship_a': sa.ships, 'ship_b': sb.ships,
                'winner': winner,
                'form_a_overall_delta': sa.overall_delta,
                'form_a_max_regime_degr': sa.max_regime_degradation,
                'form_b_overall_delta': sb.overall_delta,
                'form_b_max_regime_degr': sb.max_regime_degradation,
                'form_b_alpha_L': (params_b.alpha_L if params_b else np.nan),
                'form_b_alpha_R': (params_b.alpha_R if params_b else np.nan),
                'form_b_a_L': (params_b.a_L if params_b else np.nan),
                'form_b_a_R': (params_b.a_R if params_b else np.nan),
                'form_b_s_L': (params_b.s_L if params_b else np.nan),
                'form_b_s_R': (params_b.s_R if params_b else np.nan),
            })

            for fr, form_name in ((form_a_rows, 'A'), (form_b_rows, 'B')):
                for r in fr:
                    regime_rows.append({
                        'track': track, 'target': target, 'model': model,
                        'feature_set': feature_set, 'form': form_name,
                        'seed': int(seed),
                        'regime': r['regime'], 'n': r['n'],
                        'pre_rmse': r['pre_rmse'], 'pre_lo': r['pre_lo'], 'pre_hi': r['pre_hi'],
                        'post_rmse': r['post_rmse'], 'post_lo': r['post_lo'], 'post_hi': r['post_hi'],
                        'delta_rmse': r['delta_rmse'], 'delta_lo': r['delta_lo'], 'delta_hi': r['delta_hi'],
                    })

            oof_store[(track, target)]['per_seed'][int(seed)] = {
                'oof': np.asarray(oof, dtype=np.float32),
                'y_pred_te': np.asarray(y_pred_te, dtype=np.float32),
                'y_corr_a_te': np.asarray(y_corr_a_te, dtype=np.float32),
                'y_corr_b_te': (np.asarray(y_corr_b_te, dtype=np.float32)
                                 if params_b is not None else None),
                'form_a_params': {r: (a, b) for r, (a, b) in params_a.items()},
                'form_b_params': (params_b.to_dict() if params_b is not None else None),
            }

            elapsed = time.time() - t0
            n_done += 1
            remaining = (n_total - n_done) * (time.time() - wall_start) / max(n_done, 1)
            print(
                f'[{n_done:3d}/{n_total}] {track}/{target}/{model}/{feature_set} '
                f'seed={seed} winner={winner} '
                f'pre={pre_all:.3f} A={post_all_a:.3f} B={post_all_b:.3f} '
                f'({elapsed:.1f}s) ETA {remaining/60:.1f}m'
            )

    pd.DataFrame(bias_rows).to_csv(OUT_BIAS, index=False)
    pd.DataFrame(regime_rows).to_csv(OUT_REGIME, index=False)
    with open(OUT_OOF, 'wb') as f:
        pickle.dump(oof_store, f)

    print('\nWrote:')
    print(f'  {OUT_BIAS}')
    print(f'  {OUT_REGIME}')
    print(f'  {OUT_OOF}')


if __name__ == '__main__':
    main()
