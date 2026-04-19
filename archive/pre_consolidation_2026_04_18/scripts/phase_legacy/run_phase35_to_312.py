"""Recovery script: NB03 Phase 3.5 -> 3.12.

Reconstructs state that the notebook's Phase 3.4 cell failed to persist
(the `seed42_models` dict was held in memory only, causing the Phase 3.5
cell to crash with a NameError). Then executes everything from Phase 3.5
onward against on-disk artifacts only.

Inputs consumed from disk:
  results/nb03_multi_seed_summary.csv
  results/nb03_multi_seed_results.csv
  results/nb03_optuna_best_params.json
  results/optuna_studies/*.joblib
  data/processed/opx_clean_opx_liq.parquet
  data/processed/opx_clean_opx_only.parquet
  data/splits/{train,test}_indices_opx{,_liq}.npy

Outputs (idempotent -- safe to rerun):
  models/model_{model}_{target}_{track}_{feature_set}.joblib (48 seed-42)
  models/model_{target}_{track}_{family}.joblib             (8 canonical)
  models/model_{target}_{track}_{family}_resampled.joblib   (8 resampled)
  models/meta_ridge_{target}_{track}_stacked.joblib         (4 stacked)
  results/nb03_per_family_winners.json
  results/nb03_canonical_test_predictions.npz
  results/nb03_stacking_diagnostics.json
  results/nb03_stacking_oof_matrix_{target}_{track}.npz (4 files)
  results/nb03_stacked_members_{target}_{track}.json    (4 files)
  results/nb03_resampling_diagnostics_{track}.csv       (2 files)
  results/nb03_resampling_summary_{track}.json          (2 files)
  results/nb03_resampling_impact.csv
  figures/fig_nb03_resampling_pt_distribution.{png,pdf}
  figures/fig_nb03_resampling_impact_on_metrics.{png,pdf}
  figures/fig_nb03_optuna_search_progress.{png,pdf}
  figures/fig_nb03_optuna_hyperparameter_importance.{png,pdf}
  figures/fig_nb03_stacking_weights.{png,pdf}
  figures/fig_nb03_stacking_oof_correlation.{png,pdf}
  figures/fig_nb03_stacking_vs_base_comparison.{png,pdf}
  figures/fig_nb03_three_family_comparison.{png,pdf}
  figures/fig_nb03c_multiseed_rmse.{png,pdf}
"""
from __future__ import annotations

import ast
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from config import (  # noqa: E402
    DATA_PROC, DATA_SPLITS, MODELS, RESULTS, FIGURES, LOGS,
    SPLIT_SEEDS, FEATURE_METHODS, MODEL_FAMILIES, TIEBREAKER_RULE,
    FAMILY_COLORS,
    OPTUNA_STUDIES_DIR, OPTUNA_MODELS, OPTUNA_TARGETS, OPTUNA_TRACKS,
    OPTUNA_FEATURE_SETS,
    RESAMPLING_N_P_BINS, RESAMPLING_N_T_BINS, RESAMPLING_SEED,
    STACKING_ALPHAS, STACKING_CV_FOLDS, STACKING_SEED, STACKING_BASE_ORDER,
    STACKING_OOF_MATRIX_TEMPLATE, STACKING_MANIFEST_TEMPLATE,
    STACKING_META_TEMPLATE, STACKING_DIAGNOSTICS_FILE,
)
from src.features import build_feature_matrix  # noqa: E402
from src.models import BASE_MODELS, build_model, predict_median  # noqa: E402
from src.resampling import tempered_resample  # noqa: E402
from src.stacking import (  # noqa: E402
    generate_oof_predictions, fit_ridge_meta_model,
    compute_oof_correlation_matrix, BASE_ORDER as STACK_BASE_ORDER,
)

# -----------------------------------------------------------------------------

LOGS.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('recovery')
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(LOGS / 'run_phase35_to_312.log', mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(sh)


def _parse_params(s):
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return json.loads(s)


def _frozen_lookup(frozen_store, track, feat, model, target):
    return frozen_store.get('||'.join([track, feat, model, target]))


def _load_tracks_and_splits():
    df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
    df_opx = pd.read_parquet(DATA_PROC / 'opx_clean_opx_only.parquet')
    splits = {}
    for track, df_track, trkfile, tefile in [
        ('opx_liq', df_liq, 'train_indices_opx_liq.npy', 'test_indices_opx_liq.npy'),
        ('opx_only', df_opx, 'train_indices_opx.npy', 'test_indices_opx.npy'),
    ]:
        tr = np.load(DATA_SPLITS / trkfile)
        te = np.load(DATA_SPLITS / tefile)
        df_tr = df_track.loc[tr].reset_index(drop=True)
        df_te = df_track.loc[te].reset_index(drop=True)
        splits[track] = {'df': df_track, 'df_train': df_tr, 'df_test': df_te,
                          'use_liq': (track == 'opx_liq')}
    return splits


def _fit_one(model_name, params, X_tr, y_tr, bootstrap_override=None):
    """Clone the base factory, set params, honor bootstrap override for RF/ERT."""
    est = clone(BASE_MODELS[model_name]())
    est.set_params(**params)
    if bootstrap_override is not None and model_name in ('RF', 'ERT'):
        est.set_params(bootstrap=bootstrap_override)
    est.fit(X_tr, y_tr)
    return est


# =============================================================================
# Phase 3.4 completion: fit and save all 48 seed-42 base models
# =============================================================================
def phase_3_4_save_seed42_models(splits, frozen_store):
    logger.info('=' * 72)
    logger.info('PHASE 3.4 COMPLETION: fit+save 48 seed-42 base models')
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    saved = 0
    skipped = 0
    for track in OPTUNA_TRACKS:
        s = splits[track]
        for feat in OPTUNA_FEATURE_SETS:
            X_tr, _ = build_feature_matrix(s['df_train'], feat, use_liq=s['use_liq'])
            for model_name in OPTUNA_MODELS:
                for target in OPTUNA_TARGETS:
                    params = _frozen_lookup(frozen_store, track, feat, model_name, target)
                    if params is None:
                        logger.warning('missing frozen params: %s', (track, feat, model_name, target))
                        skipped += 1
                        continue
                    y_tr = s['df_train'][target].values
                    fname = f'model_{model_name}_{target}_{track}_{feat}.joblib'
                    est = _fit_one(model_name, params, X_tr, y_tr)
                    joblib.dump(est, MODELS / fname)
                    saved += 1
    logger.info('Phase 3.4 save complete: %d models saved, %d skipped, %.1fs',
                saved, skipped, time.time() - t0)
    return saved


# =============================================================================
# Phase 3.5: per-family winner selection
# =============================================================================
def phase_3_5_select_winners(summary_df):
    logger.info('=' * 72)
    logger.info('PHASE 3.5: winner selection per family')

    def _choose(fam_subset, pref_model):
        if fam_subset.empty:
            return None
        min_row = fam_subset.loc[fam_subset['rmse_test_mean'].idxmin()]
        tol = float(min_row['rmse_test_std']) if pd.notna(min_row['rmse_test_std']) else 0.0
        band = fam_subset[fam_subset['rmse_test_mean'] <= min_row['rmse_test_mean'] + tol]
        if pref_model in band['model_name'].values:
            pref_rows = band[band['model_name'] == pref_model]
            return pref_rows.loc[pref_rows['rmse_test_mean'].idxmin()]
        return min_row

    winners = {
        'forest_family': {},
        'boosted_family': {},
        'tiebreaker_rule': TIEBREAKER_RULE,
        'selection_metadata': {
            'n_seeds': int(len(SPLIT_SEEDS)),
            'split_seeds': list(SPLIT_SEEDS),
            'feature_methods': list(FEATURE_METHODS),
        },
    }
    canonical_files = []
    for family_name, fam_cfg in MODEL_FAMILIES.items():
        if family_name == 'stacked':
            continue  # filled in Phase 3.10
        fam_key = f'{family_name}_family'
        for track in ['opx_only', 'opx_liq']:
            for target in ['T_C', 'P_kbar']:
                subset = summary_df[
                    (summary_df['track'] == track) &
                    (summary_df['target'] == target) &
                    (summary_df['model_name'].isin(fam_cfg['candidates']))
                ]
                chosen = _choose(subset, fam_cfg['tiebreaker_preferred'])
                if chosen is None:
                    logger.warning('no candidates: %s/%s/%s', family_name, track, target)
                    continue
                model_name = str(chosen['model_name'])
                feat_set = str(chosen['feature_set'])
                canonical_name = f'model_{target}_{track}_{family_name}.joblib'
                src_name = f'model_{model_name}_{target}_{track}_{feat_set}.joblib'
                src_path = MODELS / src_name
                dst_path = MODELS / canonical_name
                if not src_path.exists():
                    logger.error('missing seed-42 model: %s', src_path)
                    continue
                # Copy bytes (joblib load/dump roundtrip is safest)
                joblib.dump(joblib.load(src_path), dst_path)
                canonical_files.append(canonical_name)
                winners[fam_key][f'{track}_{target}'] = {
                    'model_name': model_name,
                    'feature_set': feat_set,
                    'filename': canonical_name,
                    'rmse_test_mean': float(chosen['rmse_test_mean']),
                    'rmse_test_std': float(chosen['rmse_test_std']) if pd.notna(chosen['rmse_test_std']) else None,
                    'r2_test_mean': float(chosen['r2_test_mean']),
                    'r2_test_std': float(chosen['r2_test_std']) if pd.notna(chosen['r2_test_std']) else None,
                }
                logger.info('  %s %s/%s: %s|%s rmse=%.3f +- %.3f',
                            family_name, track, target, model_name, feat_set,
                            float(chosen['rmse_test_mean']),
                            float(chosen['rmse_test_std']) if pd.notna(chosen['rmse_test_std']) else 0.0)

    with open(RESULTS / 'nb03_per_family_winners.json', 'w') as f:
        json.dump(winners, f, indent=2, default=str)
    logger.info('Phase 3.5 complete: %d canonical models, winners JSON written',
                len(canonical_files))
    return winners


# =============================================================================
# Phase 3.6: canonical test predictions (.npz)
# =============================================================================
def phase_3_6_canonical_predictions(splits, winners):
    logger.info('=' * 72)
    logger.info('PHASE 3.6: canonical test predictions')
    arrs = {}
    for family_name in ('forest', 'boosted'):
        fam_key = f'{family_name}_family'
        for track in ['opx_only', 'opx_liq']:
            s = splits[track]
            for target in ['T_C', 'P_kbar']:
                block = winners[fam_key].get(f'{track}_{target}')
                if block is None:
                    continue
                est = joblib.load(MODELS / block['filename'])
                X_te, _ = build_feature_matrix(s['df_test'], block['feature_set'],
                                                use_liq=s['use_liq'])
                y_true = s['df_test'][target].values
                y_pred = predict_median(est, X_te)
                key_pred = f'{family_name}_{target}_{track}_pred'
                key_true = f'{family_name}_{target}_{track}_true'
                arrs[key_pred] = np.asarray(y_pred, dtype=float)
                arrs[key_true] = np.asarray(y_true, dtype=float)
    np.savez(RESULTS / 'nb03_canonical_test_predictions.npz', **arrs)
    logger.info('Phase 3.6 complete: %d arrays in npz', len(arrs))


# =============================================================================
# Phase 3.7: verification
# =============================================================================
def phase_3_7_verify(winners):
    logger.info('=' * 72)
    logger.info('PHASE 3.7: verification')
    expected = []
    for fam in ('forest_family', 'boosted_family'):
        for track in ('opx_only', 'opx_liq'):
            for target in ('T_C', 'P_kbar'):
                key = f'{track}_{target}'
                if key in winners[fam]:
                    expected.append(winners[fam][key]['filename'])
    missing = [f for f in expected if not (MODELS / f).exists()]
    if missing:
        raise RuntimeError(f'Phase 3.7: missing canonical models: {missing}')
    logger.info('Phase 3.7 complete: all %d canonical models present', len(expected))


# =============================================================================
# Phase 3.9: tempered resampling + resampled models (both tracks, 8 models)
# =============================================================================
def phase_3_9_resampling(splits, frozen_store, winners):
    logger.info('=' * 72)
    logger.info('PHASE 3.9: tempered resampling + resampled canonical models')
    t0 = time.time()

    resampling_summaries = {}
    resampled_trains = {}
    for track in ('opx_liq', 'opx_only'):
        s = splits[track]
        df_tr_res, diag = tempered_resample(
            s['df_train'],
            target_col_p='P_kbar', target_col_t='T_C',
            n_p_bins=RESAMPLING_N_P_BINS, n_t_bins=RESAMPLING_N_T_BINS,
            seed=RESAMPLING_SEED,
        )
        resampled_trains[track] = df_tr_res
        resampling_summaries[track] = {
            'bin_edges': diag['bin_edges'],
            'actions': diag['actions'],
            'summary': diag['summary'],
        }
        diag['actions'].to_csv(RESULTS / f'nb03_resampling_diagnostics_{track}.csv', index=False)
        summ = dict(diag['summary'])
        summ['p_edges'] = diag['bin_edges'][0].tolist()
        summ['t_edges'] = diag['bin_edges'][1].tolist()
        with open(RESULTS / f'nb03_resampling_summary_{track}.json', 'w') as f:
            json.dump(summ, f, indent=2)
        logger.info('  %s: n_in=%d -> n_out=%d, cells=%d, grown=%d shrunk=%d held=%d',
                    track, summ['n_in'], summ['n_out'], summ['n_occupied_cells'],
                    summ['n_cells_grown'], summ['n_cells_shrunk'], summ['n_cells_held'])

    # Fit resampled models for each (family, track, target) using that winner's
    # model_name + feature_set + frozen params, with bootstrap=False for RF/ERT.
    impact_rows = []
    for family_name in ('forest', 'boosted'):
        fam_key = f'{family_name}_family'
        for track in ('opx_liq', 'opx_only'):
            s = splits[track]
            df_tr_res = resampled_trains[track]
            for target in ('T_C', 'P_kbar'):
                block = winners[fam_key].get(f'{track}_{target}')
                if block is None:
                    continue
                mname = block['model_name']
                feat = block['feature_set']
                params = _frozen_lookup(frozen_store, track, feat, mname, target)
                X_tr, _ = build_feature_matrix(df_tr_res, feat, use_liq=s['use_liq'])
                X_te, _ = build_feature_matrix(s['df_test'], feat, use_liq=s['use_liq'])
                y_tr = df_tr_res[target].values
                y_te = s['df_test'][target].values
                est = _fit_one(mname, params, X_tr, y_tr, bootstrap_override=False)
                fname = f'model_{target}_{track}_{family_name}_resampled.joblib'
                joblib.dump(est, MODELS / fname)
                pred = predict_median(est, X_te)
                rmse_res = float(np.sqrt(mean_squared_error(y_te, pred)))
                # Canonical (non-resampled) comparison using seed-42 trained model
                est_can = joblib.load(MODELS / block['filename'])
                pred_can = predict_median(est_can, X_te)
                rmse_can = float(np.sqrt(mean_squared_error(y_te, pred_can)))
                impact_rows.append({
                    'target': target, 'track': track, 'family': family_name,
                    'model_name': mname, 'feature_set': feat,
                    'rmse_canonical': rmse_can, 'rmse_resampled': rmse_res,
                    'delta': rmse_res - rmse_can,
                })
                logger.info('  %s %s %s %s: canonical=%.3f resampled=%.3f (%+.3f)',
                            family_name, track, target, mname,
                            rmse_can, rmse_res, rmse_res - rmse_can)

    pd.DataFrame(impact_rows).to_csv(RESULTS / 'nb03_resampling_impact.csv', index=False)
    logger.info('Phase 3.9 complete: 8 resampled models saved, impact CSV written. (%.1fs)',
                time.time() - t0)
    return resampling_summaries, impact_rows


# =============================================================================
# Phase 3.10: Ridge stacking
# =============================================================================
def phase_3_10_stacking(splits, summary_df, frozen_store, winners):
    logger.info('=' * 72)
    logger.info('PHASE 3.10: Ridge stacking meta-models')
    t0 = time.time()

    stacking_diagnostics = {}
    stacked_test_preds = {}
    stacked_winners = {}

    for track in ('opx_liq', 'opx_only'):
        s = splits[track]
        groups_tr = s['df_train']['Citation'].values
        cv = GroupKFold(n_splits=STACKING_CV_FOLDS)

        for target in ('T_C', 'P_kbar'):
            y_train = s['df_train'][target].values
            y_test = s['df_test'][target].values

            # Per-base-model best feature_set from summary CSV.
            base_feats = {}
            oof_dict = {}
            test_pred_dict = {}
            for base in STACK_BASE_ORDER:
                sub = summary_df[
                    (summary_df['track'] == track) &
                    (summary_df['target'] == target) &
                    (summary_df['model_name'] == base)
                ]
                if sub.empty:
                    raise RuntimeError(f'Phase 3.10: no summary rows for {base}/{track}/{target}')
                best_row = sub.loc[sub['rmse_test_mean'].idxmin()]
                feat = str(best_row['feature_set'])
                base_feats[base] = feat
                params = _frozen_lookup(frozen_store, track, feat, base, target)

                X_tr, _ = build_feature_matrix(s['df_train'], feat, use_liq=s['use_liq'])
                X_te, _ = build_feature_matrix(s['df_test'], feat, use_liq=s['use_liq'])

                def _make_ctor(mname, p):
                    def _ctor(seed):
                        est = clone(BASE_MODELS[mname]())
                        est.set_params(**p)
                        try:
                            est.set_params(random_state=seed)
                        except Exception:
                            pass
                        return est
                    return _ctor

                ctor = _make_ctor(base, params)

                oof = generate_oof_predictions(
                    ctor, X_tr, y_train, groups_tr, cv, seed=STACKING_SEED,
                )
                oof_dict[base] = oof

                est_full = ctor(STACKING_SEED)
                est_full.fit(X_tr, y_train)
                test_pred_dict[base] = np.asarray(est_full.predict(X_te), dtype=float)

            # Stack and fit meta
            oof_matrix = np.column_stack([oof_dict[k] for k in STACK_BASE_ORDER])
            meta = fit_ridge_meta_model(oof_matrix, y_train,
                                         alphas=STACKING_ALPHAS,
                                         cv=STACKING_CV_FOLDS)
            corr = compute_oof_correlation_matrix(oof_dict, base_order=STACK_BASE_ORDER)

            # Test RMSE via stacked predict
            test_matrix = np.column_stack([test_pred_dict[k] for k in STACK_BASE_ORDER])
            y_stacked = meta.predict(test_matrix)
            rmse_stacked = float(np.sqrt(mean_squared_error(y_test, y_stacked)))

            # Best base RMSE on test
            base_rmse = {}
            for base in STACK_BASE_ORDER:
                base_rmse[base] = float(np.sqrt(mean_squared_error(y_test, test_pred_dict[base])))
            best_base = min(base_rmse, key=base_rmse.get)

            # Sanity checks
            coef = meta.coef_.tolist()
            alpha = float(meta.alpha_)
            sanity = {
                'alpha_at_endpoint': bool(alpha <= min(STACKING_ALPHAS) + 1e-12
                                          or alpha >= max(STACKING_ALPHAS) - 1e-12),
                'single_model_collapse': any(abs(c) > 0.9 for c in coef)
                                         and sum(abs(c) > 0.1 for c in coef) == 1,
                'negative_weights': [STACK_BASE_ORDER[i]
                                     for i, c in enumerate(coef) if c < 0],
            }

            # Persist meta + manifest + oof matrix
            meta_fname = STACKING_META_TEMPLATE.format(target=target, track=track)
            joblib.dump(meta, MODELS / meta_fname)

            manifest = {
                'target': target, 'track': track,
                'base_order': list(STACK_BASE_ORDER),
                'members': {
                    base: {
                        'filename': f'model_{base}_{target}_{track}_{base_feats[base]}.joblib',
                        'feature_set': base_feats[base],
                    } for base in STACK_BASE_ORDER
                },
                'meta_filename': meta_fname,
                'alpha': alpha,
                'coef': dict(zip(STACK_BASE_ORDER, coef)),
                'intercept': float(meta.intercept_),
            }
            mpath = RESULTS / STACKING_MANIFEST_TEMPLATE.format(target=target, track=track)
            with open(mpath, 'w') as f:
                json.dump(manifest, f, indent=2)

            oof_path = RESULTS / STACKING_OOF_MATRIX_TEMPLATE.format(target=target, track=track)
            np.savez(oof_path,
                     oof_matrix=oof_matrix,
                     y_train=y_train,
                     base_order=np.array(STACK_BASE_ORDER),
                     corr=corr.values)

            stacking_diagnostics[f'{target}_{track}'] = {
                'alpha': alpha,
                'coef': dict(zip(STACK_BASE_ORDER, coef)),
                'intercept': float(meta.intercept_),
                'base_feats': base_feats,
                'corr': corr.values.tolist(),
                'base_test_rmse': base_rmse,
                'best_base': best_base,
                'stacked_test_rmse': rmse_stacked,
                'sanity': sanity,
            }
            stacked_test_preds[f'{target}_{track}'] = {
                'y_true': y_test.tolist(),
                'y_stacked': y_stacked.tolist(),
                'base_test_pred': {k: v.tolist() for k, v in test_pred_dict.items()},
            }

            stacked_winners[f'{track}_{target}'] = {
                'model_name': 'STACKED',
                'feature_set': 'per-base',
                'filename': meta_fname,
                'rmse_test_mean': rmse_stacked,
                'rmse_test_std': None,
                'r2_test_mean': float(r2_score(y_test, y_stacked)),
                'r2_test_std': None,
            }

            logger.info('  %s %s: alpha=%.4f stacked=%.3f best_base=%s(%.3f) (%+.3f)',
                        target, track, alpha, rmse_stacked,
                        best_base, base_rmse[best_base],
                        rmse_stacked - base_rmse[best_base])
            if sanity['alpha_at_endpoint']:
                logger.warning('    SANITY: alpha at endpoint for %s/%s', target, track)
            if sanity['single_model_collapse']:
                logger.warning('    SANITY: single-model collapse for %s/%s', target, track)
            if sanity['negative_weights']:
                logger.info('    (negative weights: %s)', sanity['negative_weights'])

    # Update winners JSON with stacked_family
    winners['stacked_family'] = stacked_winners
    with open(RESULTS / 'nb03_per_family_winners.json', 'w') as f:
        json.dump(winners, f, indent=2, default=str)

    with open(RESULTS / STACKING_DIAGNOSTICS_FILE, 'w') as f:
        json.dump(stacking_diagnostics, f, indent=2, default=str)

    logger.info('Phase 3.10 complete: 4 stacked meta-models saved (%.1fs)',
                time.time() - t0)
    return stacking_diagnostics, stacked_test_preds


def _fit_and_predict_oof(ctor, X, y, groups, cv, seed):
    """Helper used inside generate_oof_predictions lambda fallback."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    for tr, va in cv.split(X, y, groups):
        est = ctor(seed)
        est.fit(X[tr], y[tr])
        oof[va] = est.predict(X[va])
    return oof


# =============================================================================
# Phase 3.11-3.12: Figures
# =============================================================================
def _save_fig(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info('  saved figure: %s.{png,pdf}', stem)


def fig_resampling_pt_distribution(splits, resampling_summaries):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, track in enumerate(('opx_liq', 'opx_only')):
        s = splits[track]
        summ = resampling_summaries[track]
        df_tr = s['df_train']
        # Regenerate resampled for plotting (cheap: reuse same seed)
        df_tr_res, _ = tempered_resample(df_tr, seed=RESAMPLING_SEED)
        ax = axes[row, 0]
        ax.hexbin(df_tr['P_kbar'], df_tr['T_C'], gridsize=25, cmap='Blues')
        ax.set_title(f'{track}: original train (n={len(df_tr)})')
        ax.set_xlabel('P (kbar)'); ax.set_ylabel('T (C)')
        ax = axes[row, 1]
        ax.hexbin(df_tr_res['P_kbar'], df_tr_res['T_C'], gridsize=25, cmap='Oranges')
        ax.set_title(f'{track}: resampled (n={len(df_tr_res)})')
        ax.set_xlabel('P (kbar)'); ax.set_ylabel('T (C)')
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_resampling_pt_distribution')


def fig_resampling_impact(impact_rows):
    df = pd.DataFrame(impact_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, target in enumerate(('T_C', 'P_kbar')):
        ax = axes[i]
        sub = df[df['target'] == target]
        labels = [f"{r['family']}\n{r['track']}" for _, r in sub.iterrows()]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub['rmse_canonical'], w, label='canonical',
               color=FAMILY_COLORS['forest'])
        ax.bar(x + w / 2, sub['rmse_resampled'], w, label='resampled',
               color=FAMILY_COLORS['boosted'])
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f'test RMSE ({target})')
        ax.set_title(target); ax.legend()
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_resampling_impact_on_metrics')


def fig_optuna_search_progress():
    # One panel per (model, target) showing best-so-far across trials for each
    # (track, feature_set) study.
    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    for mi, model in enumerate(OPTUNA_MODELS):
        for ti, target in enumerate(OPTUNA_TARGETS):
            ax = axes[mi, ti]
            for track in OPTUNA_TRACKS:
                for feat in OPTUNA_FEATURE_SETS:
                    study_path = OPTUNA_STUDIES_DIR / f'study_{model}_{target}_{track}_{feat}.joblib'
                    if not study_path.exists():
                        continue
                    try:
                        study = joblib.load(study_path)
                    except Exception as e:
                        logger.warning('could not load %s: %s', study_path.name, e)
                        continue
                    values = [t.value for t in study.trials if t.value is not None]
                    if not values:
                        continue
                    best = np.minimum.accumulate(values)
                    ax.plot(np.arange(1, len(best) + 1), best, alpha=0.7,
                            label=f'{track[-3:]}/{feat}')
            ax.set_title(f'{model} {target}')
            ax.set_xlabel('trial'); ax.set_ylabel('best-so-far RMSE')
            ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_optuna_search_progress')


def fig_optuna_hparam_importance():
    import optuna
    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    for mi, model in enumerate(OPTUNA_MODELS):
        for ti, target in enumerate(OPTUNA_TARGETS):
            ax = axes[mi, ti]
            # Use the winning (track, feat) study per (model, target) from summary
            # (just use opx_liq/pwlr as representative if no winners loaded)
            study_path = OPTUNA_STUDIES_DIR / f'study_{model}_{target}_opx_liq_pwlr.joblib'
            if not study_path.exists():
                continue
            try:
                study = joblib.load(study_path)
                imp = optuna.importance.get_param_importances(study)
            except Exception as e:
                logger.warning('importance failed for %s: %s', study_path.name, e)
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center')
                ax.set_title(f'{model} {target}')
                continue
            names = list(imp.keys())
            vals = list(imp.values())
            ax.barh(names[::-1], vals[::-1], color=FAMILY_COLORS['forest'])
            ax.set_title(f'{model} {target}')
            ax.set_xlabel('importance')
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_optuna_hyperparameter_importance')


def fig_stacking_weights(stacking_diagnostics):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    keys = list(stacking_diagnostics.keys())
    for idx, key in enumerate(keys):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        coef = stacking_diagnostics[key]['coef']
        bases = list(coef.keys())
        ax.bar(bases, [coef[b] for b in bases], color=FAMILY_COLORS['forest'])
        ax.axhline(0.25, ls='--', color='gray', alpha=0.6, label='equal (0.25)')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(key); ax.set_ylabel('Ridge weight'); ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_stacking_weights')


def fig_stacking_oof_correlation(stacking_diagnostics):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    keys = list(stacking_diagnostics.keys())
    for idx, key in enumerate(keys):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        corr = np.array(stacking_diagnostics[key]['corr'])
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(list(STACK_BASE_ORDER)); ax.set_yticklabels(list(STACK_BASE_ORDER))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(corr[i, j]) > 0.5 else 'black')
        ax.set_title(key)
        fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_stacking_oof_correlation')


def fig_stacking_vs_base(stacking_diagnostics, summary_df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    keys = list(stacking_diagnostics.keys())
    for idx, key in enumerate(keys):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        d = stacking_diagnostics[key]
        target, track = key.rsplit('_', 1) if key.count('_') == 2 else key.split('_', 1)
        # reliable target/track split
        if key.startswith('T_C'):
            target = 'T_C'; track = key[len('T_C_'):]
        else:
            target = 'P_kbar'; track = key[len('P_kbar_'):]
        # base 20-seed std from summary
        stds = {}
        for b in STACK_BASE_ORDER:
            sub = summary_df[(summary_df['track'] == track) & (summary_df['target'] == target)
                             & (summary_df['model_name'] == b)]
            if sub.empty:
                stds[b] = 0.0
            else:
                stds[b] = float(sub['rmse_test_std'].min())
        names = list(STACK_BASE_ORDER) + ['stacked']
        vals = [d['base_test_rmse'][b] for b in STACK_BASE_ORDER] + [d['stacked_test_rmse']]
        errs = [stds[b] for b in STACK_BASE_ORDER] + [0.0]
        colors = [FAMILY_COLORS['forest']] * 2 + [FAMILY_COLORS['boosted']] * 2 + ['#009E73']
        ax.bar(names, vals, yerr=errs, color=colors, capsize=3)
        ax.set_title(key); ax.set_ylabel(f'test RMSE ({target})')
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_stacking_vs_base_comparison')


def fig_three_family(winners, stacking_diagnostics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, target in enumerate(('T_C', 'P_kbar')):
        ax = axes[i]
        tracks = ('opx_liq', 'opx_only')
        x = np.arange(len(tracks))
        w = 0.25
        forest_vals = [winners['forest_family'][f'{tr}_{target}']['rmse_test_mean'] for tr in tracks]
        boosted_vals = [winners['boosted_family'][f'{tr}_{target}']['rmse_test_mean'] for tr in tracks]
        stacked_vals = [stacking_diagnostics[f'{target}_{tr}']['stacked_test_rmse'] for tr in tracks]
        ax.bar(x - w, forest_vals, w, label='forest', color=FAMILY_COLORS['forest'])
        ax.bar(x, boosted_vals, w, label='boosted', color=FAMILY_COLORS['boosted'])
        ax.bar(x + w, stacked_vals, w, label='stacked', color='#009E73')
        ax.set_xticks(x); ax.set_xticklabels(tracks)
        ax.set_title(target); ax.set_ylabel('test RMSE'); ax.legend()
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03_three_family_comparison')


def fig_multiseed_rmse(summary_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, target in enumerate(('T_C', 'P_kbar')):
        for j, track in enumerate(('opx_liq', 'opx_only')):
            ax = axes[i, j]
            sub = summary_df[(summary_df['target'] == target) & (summary_df['track'] == track)]
            pivot = sub.pivot(index='model_name', columns='feature_set', values='rmse_test_mean')
            pivot = pivot.reindex(['RF', 'ERT', 'XGB', 'GB'])
            im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
            ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    v = pivot.values[r, c]
                    if not np.isnan(v):
                        ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                                color='white' if v > np.nanmedian(pivot.values) else 'black',
                                fontsize=8)
            ax.set_title(f'{target} {track}')
            fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    _save_fig(fig, 'fig_nb03c_multiseed_rmse')


def phase_3_11_3_12_figures(splits, summary_df, resampling_summaries,
                             impact_rows, stacking_diagnostics, winners):
    logger.info('=' * 72)
    logger.info('PHASE 3.11 + 3.12: figures')
    t0 = time.time()
    fig_multiseed_rmse(summary_df)
    fig_resampling_pt_distribution(splits, resampling_summaries)
    fig_resampling_impact(impact_rows)
    fig_optuna_search_progress()
    try:
        fig_optuna_hparam_importance()
    except Exception as e:
        logger.warning('hparam importance figure failed: %s', e)
    fig_stacking_weights(stacking_diagnostics)
    fig_stacking_oof_correlation(stacking_diagnostics)
    fig_stacking_vs_base(stacking_diagnostics, summary_df)
    fig_three_family(winners, stacking_diagnostics)
    logger.info('Phase 3.11+3.12 complete (%.1fs)', time.time() - t0)


# =============================================================================
# Orchestrator
# =============================================================================
def main():
    logger.info('Recovery run starting. Repo: %s', REPO)
    t0 = time.time()

    summary_df = pd.read_csv(RESULTS / 'nb03_multi_seed_summary.csv')
    with open(RESULTS / 'nb03_optuna_best_params.json') as f:
        frozen_raw = json.load(f)
    # Coerce param dicts (they may be JSON-safe already)
    frozen_store = {k: (_parse_params(v) if not isinstance(v, dict) else v)
                    for k, v in frozen_raw.items()}

    splits = _load_tracks_and_splits()

    phase_3_4_save_seed42_models(splits, frozen_store)
    winners = phase_3_5_select_winners(summary_df)
    phase_3_6_canonical_predictions(splits, winners)
    phase_3_7_verify(winners)
    resampling_summaries, impact_rows = phase_3_9_resampling(splits, frozen_store, winners)
    stacking_diagnostics, _ = phase_3_10_stacking(splits, summary_df, frozen_store, winners)
    phase_3_11_3_12_figures(splits, summary_df, resampling_summaries,
                             impact_rows, stacking_diagnostics, winners)

    logger.info('=' * 72)
    logger.info('RECOVERY COMPLETE in %.1f min', (time.time() - t0) / 60)


if __name__ == '__main__':
    main()
