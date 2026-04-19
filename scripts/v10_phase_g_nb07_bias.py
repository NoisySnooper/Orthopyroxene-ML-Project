#!/usr/bin/env python3
"""Phase G.4: NB07 bias correction rebuild for opx_liq canonical cells.

Replaces the v8 nb07_bias_correction.ipynb. Fits an in-sample piecewise
linear bias correction on CV-predicted values from the training set, then
evaluates whether the correction improves or degrades held-out test-set
RMSE.

Strategy:
  1. For each canonical cell (ElasticNet/raw T_C, MLP/raw P_kbar,
     ElasticNet/raw P_kbar), compute OOF predictions on the training
     split using StratifiedGroupKFold on Citation (10 folds).
  2. Per pre-registered P regime (shallow_crustal / deep_crustal_MASH /
     lithospheric_mantle / deeper_mantle), fit an ordinary least-squares
     correction y_true = a * y_pred + b using the OOF predictions.
  3. Apply the fitted per-regime correction to held-out test predictions;
     the regime label for each test sample comes from its true P_kbar
     (the same assignment used everywhere in Phase G).
  4. Compare pre vs post RMSE (on the test split) with bootstrap 95% CI.
     Call the correction 'useful' if (post_rmse_hi < pre_rmse_lo) on at
     least one regime and (post_rmse_lo < pre_rmse_hi) elsewhere, i.e. it
     helps somewhere without hurting anywhere with 95% certainty.

Scope note: the v9 framing included an ArcPL external bias probe.
Regenerating that probe on v10 requires a clean opx-subset of ArcPL which
is not in the v10 data tree (the available ArcPL xlsx is cpx-liq only).
This script therefore scopes to the in-sample correction only; ArcPL
probe is deferred and logged in the execution log.

Outputs:
    results/v10_opx_liq_bias_correction.csv       per (cell, regime, pre/post)
    results/v10_opx_liq_bias_correction_params.csv  fitted a, b per cell x regime
    figures/fig28_bias_correction_opx_liq.{pdf,png}
    tables/S8_8_bias_correction_opx_liq.{md,csv}
    logs/v10_phase_g_nb07_bias.log
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

from config import FIGURES, LOGS, RESULTS, SEED_BOOTSTRAP
from src.data import load_opx_liq, load_splits
from src.evaluation import (
    _bootstrap_stat,
    _rmse,
    assign_p_regime,
    stratify_labels,
)
from src.models import build_model
from src.opx_tb_analysis import load_best_params, prepare_train_test

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_nb07_bias.log'
MANIFEST = RESULTS / 'v10_optuna_best_params_opx.json'

CELLS = [
    ('ElasticNet', 'T_C',    'opx_liq', 'raw'),
    ('MLP',        'P_kbar', 'opx_liq', 'raw'),
    ('ElasticNet', 'P_kbar', 'opx_liq', 'raw'),
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def oof_predict(model_name, best_params, X, y, groups, seed=42, n_folds=10):
    """Stratified-Group-KFold OOF predictions. Preserves the training
    distribution of y across folds (stratified on y quintiles)."""
    y_strat = stratify_labels(y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                random_state=seed)
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    for tr, va in sgkf.split(X, y_strat, groups=groups):
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X[tr], y[tr])
        oof[va] = est.predict(X[va])
    return oof


def fit_piecewise(y_true, y_pred, regimes):
    """Fit per-regime OLS correction. Returns {regime: (a, b)} such that
    y_corrected = a * y_pred + b."""
    params = {}
    for r in np.unique(regimes):
        mask = (regimes == r)
        if mask.sum() < 5:
            params[r] = (1.0, 0.0)
            continue
        lr = LinearRegression()
        lr.fit(y_pred[mask].reshape(-1, 1), y_true[mask])
        params[r] = (float(lr.coef_[0]), float(lr.intercept_))
    return params


def apply_piecewise(y_pred, regimes, params):
    y_corr = y_pred.copy()
    for r, (a, b) in params.items():
        mask = (regimes == r)
        y_corr[mask] = a * y_pred[mask] + b
    return y_corr


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'START cells={CELLS}', fh)
        best = load_best_params(MANIFEST)

        # Load combined data for regime + group info. We only need to run OOF
        # on the train split; the test split stays fully held out.
        df = load_opx_liq()
        tr_idx, te_idx = load_splits('opx_liq')
        p_kbar_full = df['P_kbar'].to_numpy(dtype=float)
        regimes_tr = assign_p_regime(p_kbar_full[tr_idx])
        regimes_te = assign_p_regime(p_kbar_full[te_idx])

        rows_metric = []
        rows_params = []

        for (model_name, target, track, feature_set) in CELLS:
            key = (model_name, target, track, feature_set)
            best_params = best[key]['best_params']
            pt = prepare_train_test(track, target, feature_set)
            X_tr, y_tr = pt['X_tr'], pt['y_tr']
            X_te, y_te = pt['X_te'], pt['y_te']
            groups_tr = pt['groups_tr']

            t0 = time.time()
            oof_tr = oof_predict(model_name, best_params, X_tr, y_tr,
                                 groups_tr, seed=42, n_folds=10)
            mask_fin = np.isfinite(oof_tr)
            params = fit_piecewise(y_tr[mask_fin], oof_tr[mask_fin],
                                   regimes_tr[mask_fin])
            _log(f'[{model_name}/{feature_set}/{target}] OOF elapsed='
                 f'{time.time()-t0:.1f}s params='
                 f'{ {r: (round(a,3), round(b,3)) for r,(a,b) in params.items()} }',
                 fh)

            # Fit final model on full train, predict test.
            est = build_model(model_name, best_params, seed=42)
            est.fit(X_tr, y_tr)
            y_pred_te = est.predict(X_te)
            y_corr_te = apply_piecewise(y_pred_te, regimes_te, params)

            # Per-regime pre vs post metrics with bootstrap CI.
            for r in np.unique(regimes_te):
                mask = (regimes_te == r)
                n_r = int(mask.sum())
                if n_r < 5:
                    rows_metric.append(dict(
                        model=model_name, target=target, track=track,
                        feature_set=feature_set, regime=r, n=n_r,
                        pre_rmse=float('nan'), pre_rmse_lo=float('nan'),
                        pre_rmse_hi=float('nan'),
                        post_rmse=float('nan'), post_rmse_lo=float('nan'),
                        post_rmse_hi=float('nan'),
                        improvement=float('nan')))
                    continue
                pre, pre_lo, pre_hi = _bootstrap_stat(
                    y_te[mask], y_pred_te[mask], _rmse,
                    n_bootstrap=500, seed=SEED_BOOTSTRAP)
                post, post_lo, post_hi = _bootstrap_stat(
                    y_te[mask], y_corr_te[mask], _rmse,
                    n_bootstrap=500, seed=SEED_BOOTSTRAP)
                rows_metric.append(dict(
                    model=model_name, target=target, track=track,
                    feature_set=feature_set, regime=r, n=n_r,
                    pre_rmse=pre, pre_rmse_lo=pre_lo, pre_rmse_hi=pre_hi,
                    post_rmse=post, post_rmse_lo=post_lo, post_rmse_hi=post_hi,
                    improvement=float(pre - post)))
                _log(f'  {r} n={n_r} pre={pre:.3f}[{pre_lo:.3f},{pre_hi:.3f}] '
                     f'post={post:.3f}[{post_lo:.3f},{post_hi:.3f}] '
                     f'improvement={pre-post:+.3f}', fh)

            # ALL aggregate
            pre, pre_lo, pre_hi = _bootstrap_stat(
                y_te, y_pred_te, _rmse, n_bootstrap=500, seed=SEED_BOOTSTRAP)
            post, post_lo, post_hi = _bootstrap_stat(
                y_te, y_corr_te, _rmse, n_bootstrap=500, seed=SEED_BOOTSTRAP)
            rows_metric.append(dict(
                model=model_name, target=target, track=track,
                feature_set=feature_set, regime='ALL', n=len(y_te),
                pre_rmse=pre, pre_rmse_lo=pre_lo, pre_rmse_hi=pre_hi,
                post_rmse=post, post_rmse_lo=post_lo, post_rmse_hi=post_hi,
                improvement=float(pre - post)))
            _log(f'  ALL n={len(y_te)} pre={pre:.3f} post={post:.3f} '
                 f'improvement={pre-post:+.3f}', fh)

            for r, (a, b) in params.items():
                rows_params.append(dict(
                    model=model_name, target=target, track=track,
                    feature_set=feature_set, regime=r, a=a, b=b))

        metric_df = pd.DataFrame(rows_metric)
        params_df = pd.DataFrame(rows_params)

        out_metric = RESULTS / 'v10_opx_liq_bias_correction.csv'
        out_params = RESULTS / 'v10_opx_liq_bias_correction_params.csv'
        metric_df.to_csv(out_metric, index=False)
        params_df.to_csv(out_params, index=False)
        _log(f'wrote {out_metric}', fh)
        _log(f'wrote {out_params}', fh)

        # --- Figure -----------------------------------------------------------
        import matplotlib.pyplot as plt
        cells_unique = metric_df[['model', 'target', 'feature_set']].drop_duplicates()
        fig, axes = plt.subplots(1, len(cells_unique),
                                 figsize=(4 * len(cells_unique), 4),
                                 squeeze=False)
        regime_order = ['shallow_crustal', 'deep_crustal_MASH',
                        'lithospheric_mantle', 'deeper_mantle', 'ALL']
        for ax, (_, cell) in zip(axes[0], cells_unique.iterrows()):
            sub = metric_df[(metric_df.model == cell.model) &
                            (metric_df.target == cell.target) &
                            (metric_df.feature_set == cell.feature_set)]
            sub = sub.set_index('regime').reindex(
                [r for r in regime_order if r in sub.regime.values])
            xs = np.arange(len(sub))
            w = 0.38
            ax.bar(xs - w/2, sub['pre_rmse'], w, label='pre',
                   color='#E69F00', edgecolor='black', linewidth=0.5)
            ax.bar(xs + w/2, sub['post_rmse'], w, label='post',
                   color='#0072B2', edgecolor='black', linewidth=0.5)
            ax.errorbar(xs - w/2, sub['pre_rmse'],
                        yerr=[sub['pre_rmse'] - sub['pre_rmse_lo'],
                              sub['pre_rmse_hi'] - sub['pre_rmse']],
                        fmt='none', ecolor='black', capsize=2)
            ax.errorbar(xs + w/2, sub['post_rmse'],
                        yerr=[sub['post_rmse'] - sub['post_rmse_lo'],
                              sub['post_rmse_hi'] - sub['post_rmse']],
                        fmt='none', ecolor='black', capsize=2)
            ax.set_xticks(xs)
            ax.set_xticklabels(sub.index, rotation=30, ha='right', fontsize=8)
            unit = '°C' if cell.target == 'T_C' else 'kbar'
            ax.set_ylabel(f'RMSE ({unit})')
            ax.set_title(f'{cell.model}/{cell.feature_set}\n{cell.target}',
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
        fig.suptitle('opx-liq piecewise bias correction (pre vs post, 95% CI)',
                     fontsize=11)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGURES / f'fig28_bias_correction_opx_liq.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote figures/fig28_bias_correction_opx_liq.{{pdf,png}}', fh)

        # --- SI table ---------------------------------------------------------
        tables_dir = PROJECT_ROOT / 'tables'
        tables_dir.mkdir(exist_ok=True)
        md_lines = ['# Table S8.8: opx-liq in-sample piecewise bias correction',
                    '',
                    'Per-regime OLS correction fit on 10-fold StratifiedGroupKFold '
                    'out-of-fold training-set predictions, evaluated on the '
                    'held-out test split. `pre` = raw model prediction RMSE; '
                    '`post` = corrected RMSE; both with bootstrap 95% CI '
                    '(500 resamples). A useful correction shows post CI '
                    'strictly below pre CI on at least one regime and no '
                    'worse than pre CI elsewhere.', '']
        md_lines.append('| Cell | Regime | n | pre RMSE [95% CI] | post RMSE [95% CI] | Δ |')
        md_lines.append('|---|---|---|---|---|---|')
        for _, r in metric_df.iterrows():
            if not np.isfinite(r.pre_rmse):
                continue
            unit = '°C' if r.target == 'T_C' else 'kbar'
            cell = f'{r.model}/{r.feature_set}/{r.target}'
            md_lines.append(
                f'| {cell} | {r.regime} | {r.n} | '
                f'{r.pre_rmse:.2f} [{r.pre_rmse_lo:.2f}, {r.pre_rmse_hi:.2f}] {unit} | '
                f'{r.post_rmse:.2f} [{r.post_rmse_lo:.2f}, {r.post_rmse_hi:.2f}] {unit} | '
                f'{r.improvement:+.2f} {unit} |')
        md_path = tables_dir / 'S8_8_bias_correction_opx_liq.md'
        csv_path = tables_dir / 'S8_8_bias_correction_opx_liq.csv'
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')
        metric_df.to_csv(csv_path, index=False)
        _log(f'wrote {md_path}', fh)
        _log(f'wrote {csv_path}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
