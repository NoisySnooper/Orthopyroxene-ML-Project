#!/usr/bin/env python3
"""Phase G.2: NB05 generalization rebuild for opx_liq.

Replaces the v8 nb05_loso_validation.ipynb with a reproducible script that
evaluates four grouped cross-validation strategies on the opx_liq canonical
cells:

  1. LOSO         -- Leave-One-Study-Out on Citation (93 unique studies).
                     The strictest generalization test; directly addresses
                     whether tree-based models memorize study-specific
                     idiosyncrasies.
  2. Cluster-KFold -- GroupKFold where each group is a KMeans cluster on
                     standardized composition. Tests whether the model
                     generalizes across composition clusters.
  3. TargetBinKFold -- Leave-one-P-regime-out using the pre-registered
                     regime labels [0,5,15,30,100] kbar. Tests whether the
                     model generalizes across calibration-domain pressure
                     regimes.
  4. LeaveOneRegionOut -- Leave-one-study-type-out using the petrologic
                     study-type categorization inferred from Citation text
                     (MORB / mantle_melting / arc_silicic / basalt /
                     primitive_mafic / partitioning / metamorphic / other).
                     Weaker than LOSO because groups contain many studies,
                     but tests whether the model generalizes across
                     experimental-context categories.

Canonical cells evaluated:
    ElasticNet/raw  T_C      (aggregate T_C winner)
    MLP/raw         P_kbar   (aggregate P_kbar winner)
    ElasticNet/raw  P_kbar   (shallow_crustal regime winner, deterministic)

Bootstrap 95% CI on pooled OOF RMSE / MAE (500 resamples, seed=SEED_BOOTSTRAP).

Outputs:
    results/v10_opx_liq_generalization.csv          (one row per strategy x cell)
    results/v10_opx_liq_generalization_predictions.csv  (long, pooled OOF)
    figures/fig26_generalization_opx_liq.{pdf,png}
    tables/S8_6_generalization_opx_liq.{md,csv}
    logs/v10_phase_g_nb05_generalization.log
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from config import FIGURES, LOGS, RESULTS, SEED_BOOTSTRAP
from src.data import load_opx_liq, load_splits
from src.evaluation import (
    _bootstrap_stat,
    _rmse,
    assign_p_regime,
    infer_study_type,
    leave_one_region_out_splits,
    target_bin_kfold_splits,
)
from src.models import build_model
from src.opx_tb_analysis import load_best_params, prepare_train_test

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_nb05_generalization.log'
MANIFEST = RESULTS / 'v10_optuna_best_params_opx.json'

CANONICAL_CELLS = [
    ('ElasticNet', 'T_C',    'opx_liq', 'raw'),
    ('MLP',        'P_kbar', 'opx_liq', 'raw'),
    ('ElasticNet', 'P_kbar', 'opx_liq', 'raw'),
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def build_combined_track(track, target, feature_set):
    """Rebuild the combined (train+test) X, y plus aligned Citation / P_kbar
    vectors. Uses the same feature preparation as prepare_train_test."""
    pt = prepare_train_test(track, target, feature_set)
    X = np.vstack([pt['X_tr'], pt['X_te']])
    y = np.concatenate([pt['y_tr'], pt['y_te']])

    df = load_opx_liq()
    tr_idx, te_idx = load_splits(track)
    idx_combined = np.concatenate([tr_idx, te_idx])
    combined_df = df.iloc[idx_combined]
    citations = combined_df['Citation'].to_numpy()
    p_true = combined_df['P_kbar'].to_numpy(dtype=float)
    return X, y, citations, p_true, combined_df


def run_loso(X, y, groups, model_name, best_params, seed=42):
    """Return pooled OOF predictions and per-fold RMSE list."""
    logo = LeaveOneGroupOut()
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_rmses = []
    for tr, te in logo.split(X, y, groups=groups):
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X[tr], y[tr])
        y_hat = est.predict(X[te])
        oof[te] = y_hat
        fold_rmses.append((
            groups[te[0]] if len(te) else None,
            int(len(te)),
            float(np.sqrt(mean_squared_error(y[te], y_hat))),
        ))
    return oof, fold_rmses


def run_generic_splits(X, y, splits, model_name, best_params, seed=42):
    """`splits` is a list of (train_idx, test_idx, label) triples."""
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_rmses = []
    for tr, te, label in splits:
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X[tr], y[tr])
        y_hat = est.predict(X[te])
        oof[te] = y_hat
        fold_rmses.append((
            label,
            int(len(te)),
            float(np.sqrt(mean_squared_error(y[te], y_hat))),
        ))
    return oof, fold_rmses


def pooled_metrics(y, oof):
    mask = np.isfinite(oof)
    if mask.sum() < 20:
        return dict(n=int(mask.sum()), rmse=float('nan'),
                    rmse_lo=float('nan'), rmse_hi=float('nan'),
                    mae=float('nan'), mae_lo=float('nan'), mae_hi=float('nan'))
    y_m, p_m = y[mask], oof[mask]
    rmse, rmse_lo, rmse_hi = _bootstrap_stat(y_m, p_m, _rmse,
                                              n_bootstrap=500,
                                              seed=SEED_BOOTSTRAP)
    mae, mae_lo, mae_hi = _bootstrap_stat(y_m, p_m, _mae,
                                          n_bootstrap=500,
                                          seed=SEED_BOOTSTRAP)
    return dict(n=int(mask.sum()), rmse=rmse, rmse_lo=rmse_lo, rmse_hi=rmse_hi,
                mae=mae, mae_lo=mae_lo, mae_hi=mae_hi)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'START canonical_cells={CANONICAL_CELLS}', fh)

        best = load_best_params(MANIFEST)
        rows = []
        pred_rows = []

        for (model_name, target, track, feature_set) in CANONICAL_CELLS:
            key = (model_name, target, track, feature_set)
            best_params = best[key]['best_params']
            X, y, citations, p_true, combined = build_combined_track(
                track, target, feature_set)
            _log(f'\n=== {model_name}/{feature_set}/{target}/{track} '
                 f'combined n={len(y)} ===', fh)

            # --- LOSO ---------------------------------------------------------
            t0 = time.time()
            oof_loso, fold_loso = run_loso(
                X, y, citations, model_name, best_params, seed=42)
            m = pooled_metrics(y, oof_loso)
            m.update(strategy='LOSO', model=model_name, target=target,
                     track=track, feature_set=feature_set,
                     n_folds=len(fold_loso))
            rows.append(m)
            _log(f'LOSO n_folds={len(fold_loso)} rmse={m["rmse"]:.3f} '
                 f'[{m["rmse_lo"]:.3f},{m["rmse_hi"]:.3f}] '
                 f'elapsed={time.time()-t0:.1f}s', fh)

            # --- Cluster-KFold ------------------------------------------------
            t0 = time.time()
            n_clusters = 5
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(Xs)
            splits_ck = [
                (np.where(clusters != k)[0],
                 np.where(clusters == k)[0],
                 f'cluster_{k}')
                for k in range(n_clusters)
                if (clusters != k).sum() >= 50 and (clusters == k).sum() >= 1
            ]
            oof_ck, fold_ck = run_generic_splits(
                X, y, splits_ck, model_name, best_params, seed=42)
            m = pooled_metrics(y, oof_ck)
            m.update(strategy='ClusterKFold', model=model_name, target=target,
                     track=track, feature_set=feature_set,
                     n_folds=len(fold_ck))
            rows.append(m)
            _log(f'ClusterKFold k={n_clusters} n_folds={len(fold_ck)} '
                 f'rmse={m["rmse"]:.3f} [{m["rmse_lo"]:.3f},{m["rmse_hi"]:.3f}] '
                 f'elapsed={time.time()-t0:.1f}s', fh)

            # --- TargetBinKFold ----------------------------------------------
            t0 = time.time()
            regimes = assign_p_regime(p_true)
            splits_tb = target_bin_kfold_splits(X, y, regimes, min_train_fold=50)
            oof_tb, fold_tb = run_generic_splits(
                X, y, splits_tb, model_name, best_params, seed=42)
            m = pooled_metrics(y, oof_tb)
            m.update(strategy='TargetBinKFold', model=model_name, target=target,
                     track=track, feature_set=feature_set,
                     n_folds=len(fold_tb))
            rows.append(m)
            _log(f'TargetBinKFold n_folds={len(fold_tb)} '
                 f'rmse={m["rmse"]:.3f} [{m["rmse_lo"]:.3f},{m["rmse_hi"]:.3f}] '
                 f'folds={[(f[0], f[1]) for f in fold_tb]} '
                 f'elapsed={time.time()-t0:.1f}s', fh)

            # --- LeaveOneRegionOut -------------------------------------------
            t0 = time.time()
            study_types = infer_study_type(citations)
            splits_lr = leave_one_region_out_splits(
                X, y, study_types, min_train_fold=50)
            oof_lr, fold_lr = run_generic_splits(
                X, y, splits_lr, model_name, best_params, seed=42)
            m = pooled_metrics(y, oof_lr)
            m.update(strategy='LeaveOneRegionOut', model=model_name,
                     target=target, track=track, feature_set=feature_set,
                     n_folds=len(fold_lr))
            rows.append(m)
            _log(f'LORO n_folds={len(fold_lr)} rmse={m["rmse"]:.3f} '
                 f'[{m["rmse_lo"]:.3f},{m["rmse_hi"]:.3f}] '
                 f'folds={[(f[0], f[1]) for f in fold_lr]} '
                 f'elapsed={time.time()-t0:.1f}s', fh)

            # Record pooled predictions per strategy.
            for strategy, oof in [('LOSO', oof_loso), ('ClusterKFold', oof_ck),
                                  ('TargetBinKFold', oof_tb),
                                  ('LeaveOneRegionOut', oof_lr)]:
                for i, val in enumerate(oof):
                    if not np.isfinite(val):
                        continue
                    pred_rows.append({
                        'strategy':    strategy,
                        'model':       model_name,
                        'target':      target,
                        'track':       track,
                        'feature_set': feature_set,
                        'sample_idx':  int(i),
                        'P_kbar_true': float(p_true[i]),
                        'y_true':      float(y[i]),
                        'y_pred':      float(val),
                        'regime':      str(regimes[i]),
                        'study_type':  str(study_types[i]),
                    })

        # Write outputs.
        out_df = pd.DataFrame(rows)
        pred_df = pd.DataFrame(pred_rows)

        col_order = ['strategy', 'model', 'target', 'track', 'feature_set',
                     'n', 'n_folds', 'rmse', 'rmse_lo', 'rmse_hi',
                     'mae', 'mae_lo', 'mae_hi']
        out_df = out_df[col_order]

        out_csv = RESULTS / 'v10_opx_liq_generalization.csv'
        out_pred_csv = RESULTS / 'v10_opx_liq_generalization_predictions.csv'
        out_df.to_csv(out_csv, index=False)
        pred_df.to_csv(out_pred_csv, index=False)
        _log(f'wrote {out_csv} rows={len(out_df)}', fh)
        _log(f'wrote {out_pred_csv} rows={len(pred_df)}', fh)

        # --- Figure -----------------------------------------------------------
        import matplotlib.pyplot as plt
        cells = out_df[['model', 'target', 'feature_set']].drop_duplicates()
        fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4),
                                 squeeze=False)
        strategies = ['LOSO', 'ClusterKFold', 'TargetBinKFold',
                      'LeaveOneRegionOut']
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

        for ax, (_, cell) in zip(axes[0], cells.iterrows()):
            sub = out_df[(out_df.model == cell.model) &
                         (out_df.target == cell.target) &
                         (out_df.feature_set == cell.feature_set)]
            sub = sub.set_index('strategy').reindex(strategies)
            xs = np.arange(len(strategies))
            ax.bar(xs, sub['rmse'], color=colors, edgecolor='black',
                   linewidth=0.5)
            ax.errorbar(xs, sub['rmse'],
                        yerr=[sub['rmse'] - sub['rmse_lo'],
                              sub['rmse_hi'] - sub['rmse']],
                        fmt='none', ecolor='black', capsize=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(strategies, rotation=30, ha='right', fontsize=8)
            unit = '°C' if cell.target == 'T_C' else 'kbar'
            ax.set_ylabel(f'Pooled OOF RMSE ({unit})')
            ax.set_title(f'{cell.model}/{cell.feature_set}\n{cell.target}',
                         fontsize=10)
            ax.grid(axis='y', linestyle=':', alpha=0.4)

        fig.suptitle('opx-liq generalization: 4 grouped CV strategies '
                     '(95% bootstrap CI)', fontsize=11)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGURES / f'fig26_generalization_opx_liq.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote figures/fig26_generalization_opx_liq.{{pdf,png}}', fh)

        # --- SI table ---------------------------------------------------------
        tables_dir = PROJECT_ROOT / 'tables'
        tables_dir.mkdir(exist_ok=True)
        md_lines = ['# Table S8.6: opx-liq generalization (4 grouped CV strategies)',
                    '',
                    'Pooled OOF RMSE and MAE with bootstrap 95% CI over '
                    'out-of-fold predictions. Strategies are ordered from '
                    'strictest to weakest generalization test. LOSO leaves '
                    'one Citation out at a time (93 folds). ClusterKFold uses '
                    'k=5 KMeans clusters on standardized composition. '
                    'TargetBinKFold uses the pre-registered P-regime labels '
                    '(0/5/15/30/100 kbar). LeaveOneRegionOut uses a petrologic '
                    'study-type categorization inferred from Citation text.',
                    '']
        md_lines.append(
            '| Model/feat | Target | Strategy | n | folds | RMSE [95% CI] | MAE [95% CI] |')
        md_lines.append('|---|---|---|---|---|---|---|')
        for _, r in out_df.iterrows():
            unit = '°C' if r.target == 'T_C' else 'kbar'
            md_lines.append(
                f'| {r.model}/{r.feature_set} | {r.target} | {r.strategy} | '
                f'{r.n} | {r.n_folds} | '
                f'{r.rmse:.2f} [{r.rmse_lo:.2f}, {r.rmse_hi:.2f}] {unit} | '
                f'{r.mae:.2f} [{r.mae_lo:.2f}, {r.mae_hi:.2f}] {unit} |')
        md_path = tables_dir / 'S8_6_generalization_opx_liq.md'
        csv_path = tables_dir / 'S8_6_generalization_opx_liq.csv'
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')
        out_df.to_csv(csv_path, index=False)
        _log(f'wrote {md_path}', fh)
        _log(f'wrote {csv_path}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
