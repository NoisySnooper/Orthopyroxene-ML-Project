#!/usr/bin/env python3
"""supp_fig_6: ML winner vs Putirka 2008 equation, both against the
experimental ground truth.

For each (track, target) cell we plot two prediction clouds against the
held-out experimental P or T:
  · ML winner predictions (colored, larger markers)
  · Putirka 2008 equation predictions (gray squares)

Each panel reports a standard ML predicted-vs-actual metric panel:
RMSE, R^2, regression slope (ideal = 1), and signed bias (ideal = 0).
A least-squares best-fit line is drawn for each method to make the
slope visually obvious against the 1:1 reference.

Equations:
  · opx-liq T   →  Putirka eq 28a
  · opx-liq P   →  Putirka eq 29a
  · opx-only P  →  Putirka eq 29c
  · opx-only T  →  no opx-only T calibration in Thermobar 1.0.70 →
                   panel shows ML cloud only with a "no Putirka equivalent"
                   annotation.

Output:
  results/classical_equivalence_regression.csv
  figures/opx_only/supp_fig_6.{pdf,png,txt}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_train_test import prepare_train_test  # noqa: E402
from src.external_models import (predict_putirka_opx_liq,                 # noqa: E402
                                  predict_putirka_opx_only)
from src.external.arcpl_opx import to_thermobar_schema                    # noqa: E402
from scripts.figures._style import (apply_pub_style, resolve_out_dir,     # noqa: E402
                                     jgr_top, jgr_bottom)
from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

apply_pub_style()

ML_COLOR = OKABE_ITO['blue']
PUTIRKA_COLOR = '#666666'
ID_LINE = '#333333'

LIMS = {
    'T_C':    (700.0, 1900.0),
    'P_kbar': (-5.0, 50.0),
}
UNITS = {'T_C': '°C', 'P_kbar': 'kbar'}
TARGETS = {'T_C': 'T', 'P_kbar': 'P'}

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OUT_CSV = PROJECT_ROOT / 'results' / 'classical_equivalence_regression.csv'
OUT_DIR = resolve_out_dir(PROJECT_ROOT)
FIG_STEM = OUT_DIR / 'supp_fig_6'

PARQUET_PATHS = {
    'opx_liq':  'data/processed/opx_clean_opx_liq.parquet',
    'opx_only': 'data/processed/opx_clean_opx_only.parquet',
}
SPLIT_TAGS = {'opx_liq': 'opx_liq', 'opx_only': 'opx'}

CELLS = [
    ('opx', 'opx_liq',  'T_C',    'a'),
    ('opx', 'opx_liq',  'P_kbar', 'b'),
    ('opx', 'opx_only', 'T_C',    'c'),
    ('opx', 'opx_only', 'P_kbar', 'd'),
]

PUTIRKA_EQ = {
    ('opx_liq',  'T_C'):    'Putirka 28a',
    ('opx_liq',  'P_kbar'): 'Putirka 29a',
    ('opx_only', 'T_C'):    None,
    ('opx_only', 'P_kbar'): 'Putirka 29c',
}


def canonical_path(pipeline, model, target, track, fs):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{fs}.joblib')


def predict_putirka(track, target, test_tb, T_K_obs, P_kbar_obs):
    if track == 'opx_liq':
        if target == 'T_C':
            return predict_putirka_opx_liq(test_tb, target='T',
                                           P_kbar=P_kbar_obs)
        return predict_putirka_opx_liq(test_tb, target='P',
                                       T_K=T_K_obs)
    # opx_only
    if target == 'T_C':
        # No opx-only T equation in Thermobar 1.0.70 — returns NaN.
        return predict_putirka_opx_only(test_tb, target='T')
    return predict_putirka_opx_only(test_tb, target='P', T_K=T_K_obs)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     mask: np.ndarray) -> dict:
    """Return standard predicted-vs-actual metrics over `mask`.

    n: count of finite paired points; rmse: sqrt mean squared residual;
    mae: mean absolute residual; r2: coefficient of determination;
    bias: signed mean residual (y_pred - y_true);
    slope, intercept: OLS fit of y_pred on y_true.
    """
    n = int(mask.sum())
    nan = float('nan')
    if n < 2:
        return {'n': n, 'rmse': nan, 'mae': nan, 'r2': nan,
                'bias': nan, 'slope': nan, 'intercept': nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    resid = yp - yt
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    if np.var(yt) > 0:
        r2 = float(r2_score(yt, yp))
        slope, intercept = np.polyfit(yt, yp, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        r2 = slope = intercept = nan
    return {'n': n, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'bias': bias, 'slope': slope, 'intercept': intercept}


def _format_metrics_line(m: dict, unit: str, sig: int = 2) -> str:
    """One-line metrics summary: RMSE, R^2, slope, bias."""
    if not np.isfinite(m['rmse']):
        return 'no finite predictions'
    r2_str = f'{m["r2"]:+.2f}' if np.isfinite(m['r2']) else 'n/a'
    slope_str = f'{m["slope"]:.2f}' if np.isfinite(m['slope']) else 'n/a'
    bias_str = f'{m["bias"]:+.{sig}f}' if np.isfinite(m['bias']) else 'n/a'
    return (f'RMSE={m["rmse"]:.{sig}f}{unit}  R²={r2_str}  '
            f'slope={slope_str}  bias={bias_str}{unit}')


def main():
    boot = pd.read_csv(BOOT_CSV)
    rows: list[dict] = []
    cell_data: dict = {}

    for pipe, track, tgt, idx_letter in CELLS:
        winner = (boot[(boot.pipeline == pipe) & (boot.track == track)
                       & (boot.target == tgt)]
                  .sort_values('rmse_point').iloc[0])
        family = winner['model']
        fs = winner['feature_set']
        print(f'\n=== {track}/{tgt} winner {family}/{fs} ===')

        # ML predictions on the held-out test partition.
        if family == 'TabPFN':
            tab = pd.read_csv('results/tabpfn_predictions.csv')
            m = tab[(tab.track == track) & (tab.target == tgt)
                    & (tab.seed == 42)]
            y_ml = m['y_pred'].to_numpy(dtype=float)
            y_true = m['y_true'].to_numpy(dtype=float)
        else:
            mp = canonical_path(pipe, family, tgt, track, fs)
            est = joblib.load(mp)
            d = prepare_train_test(pipe, track, tgt, fs)
            y_ml = np.asarray(est.predict(d['X_te']), dtype=float)
            y_true = d['y_te']

        # Putirka predictions on the same test rows (same indexing).
        df_full = pd.read_parquet(PARQUET_PATHS[track])
        te_idx = np.load(f'data/splits/test_indices_{SPLIT_TAGS[track]}.npy')
        test_rows = df_full.iloc[te_idx].reset_index(drop=True)
        test_tb = to_thermobar_schema(test_rows)
        T_K_obs = test_rows['T_C'].to_numpy(dtype=float) + 273.15
        P_kbar_obs = test_rows['P_kbar'].to_numpy(dtype=float)
        try:
            y_put = predict_putirka(track, tgt, test_tb, T_K_obs, P_kbar_obs)
        except Exception as e:
            print(f'  Putirka raised: {e}; falling back to NaN')
            y_put = np.full(len(test_rows), np.nan, dtype=float)

        n_total = len(y_true)
        ml_mask = np.isfinite(y_ml) & np.isfinite(y_true)
        put_mask = np.isfinite(y_put) & np.isfinite(y_true)
        ml_metrics = _compute_metrics(y_true, y_ml, ml_mask)
        put_metrics = _compute_metrics(y_true, y_put, put_mask)

        unit = UNITS[tgt]
        print(f'  ML  ({family}/{fs}):  '
              f'{_format_metrics_line(ml_metrics, unit)}  n={ml_metrics["n"]}/{n_total}')
        eq_name = PUTIRKA_EQ[(track, tgt)] or '— no equation —'
        print(f'  Put ({eq_name}):  '
              f'{_format_metrics_line(put_metrics, unit)}  n={put_metrics["n"]}/{n_total}')

        rows.append({
            'pipeline': pipe, 'track': track, 'target': tgt,
            'winner_family': family, 'winner_feature_set': fs,
            'putirka_equation': eq_name,
            'n_total': n_total,
            'ml_n':         ml_metrics['n'],
            'ml_rmse':      ml_metrics['rmse'],
            'ml_mae':       ml_metrics['mae'],
            'ml_r2':        ml_metrics['r2'],
            'ml_bias':      ml_metrics['bias'],
            'ml_slope':     ml_metrics['slope'],
            'ml_intercept': ml_metrics['intercept'],
            'putirka_n':         put_metrics['n'],
            'putirka_rmse':      put_metrics['rmse'],
            'putirka_mae':       put_metrics['mae'],
            'putirka_r2':        put_metrics['r2'],
            'putirka_bias':      put_metrics['bias'],
            'putirka_slope':     put_metrics['slope'],
            'putirka_intercept': put_metrics['intercept'],
        })
        cell_data[(track, tgt)] = {
            'family': family, 'fs': fs, 'eq': eq_name, 'idx_letter': idx_letter,
            'y_true': y_true, 'y_ml': y_ml, 'y_put': y_put,
            'ml_mask': ml_mask, 'put_mask': put_mask,
            'ml_metrics': ml_metrics, 'put_metrics': put_metrics,
        }

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}')

    # ----------------------------- figure ---------------------------------
    # 2x2 grid: rows = tracks, cols = targets. Each panel overlays ML and
    # Putirka prediction clouds against the experimental ground truth.
    from scripts.figures._style import JGR_2COL_IN, is_jgr_mode  # noqa
    target_w = JGR_2COL_IN if is_jgr_mode() else 11.0
    fig, axes = plt.subplots(2, 2, figsize=(target_w, 8.5))

    rows_order = [('opx_liq', 'opx-liq'), ('opx_only', 'opx-only')]
    cols_order = [('T_C', 'T'), ('P_kbar', 'P')]

    for ri, (track, track_lbl) in enumerate(rows_order):
        for ci, (tgt, tgt_lbl) in enumerate(cols_order):
            ax = axes[ri, ci]
            data = cell_data[(track, tgt)]
            unit = UNITS[tgt]
            lim = LIMS[tgt]

            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.plot(lim, lim, ls='--', lw=1, color=ID_LINE,
                    alpha=0.8, zorder=1, label='1:1')

            # ML cloud first, then Putirka on top so the gray squares
            # don't bury the colored ML markers in dense regions.
            mlm = data['ml_mask']
            mlmet = data['ml_metrics']
            if mlm.sum() >= 2:
                ax.scatter(data['y_true'][mlm], data['y_ml'][mlm],
                           s=22, c=ML_COLOR, alpha=0.75,
                           edgecolor='black', linewidths=0.3,
                           label=f'ML  ({data["family"]}/{data["fs"]})',
                           zorder=3)
                # OLS best-fit line of predicted on actual.
                if np.isfinite(mlmet['slope']):
                    xs = np.linspace(lim[0], lim[1], 50)
                    ys = mlmet['slope'] * xs + mlmet['intercept']
                    ax.plot(xs, ys, color=ML_COLOR, lw=1.0, alpha=0.85,
                            zorder=4)

            putm = data['put_mask']
            putmet = data['put_metrics']
            if putm.sum() >= 2:
                ax.scatter(data['y_true'][putm], data['y_put'][putm],
                           s=20, c=PUTIRKA_COLOR, alpha=0.55,
                           edgecolor='black', linewidths=0.3,
                           marker='s',
                           label=f'Putirka  ({data["eq"]})',
                           zorder=2)
                if np.isfinite(putmet['slope']):
                    xs = np.linspace(lim[0], lim[1], 50)
                    ys = putmet['slope'] * xs + putmet['intercept']
                    ax.plot(xs, ys, color=PUTIRKA_COLOR, lw=1.0,
                            alpha=0.85, ls='-', zorder=2.5)

            # Title with panel index + cell.
            title = f'({data["idx_letter"]}) {track_lbl} · {tgt_lbl} ({unit})'
            ax.set_title(title, loc='left', pad=6, fontsize=10)
            ax.set_xlabel(f'experimental {tgt_lbl} ({unit})', fontsize=9)
            ax.set_ylabel(f'predicted {tgt_lbl} ({unit})', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True); ax.set_axisbelow(True)

            # Stats box (top-left): two-line entries per method, plus n.
            ml_header = f'ML ({data["family"]}/{data["fs"]})  n={mlmet["n"]}'
            ml_line = _format_metrics_line(mlmet, unit)
            if data['eq']:
                put_header = f'{data["eq"]}  n={putmet["n"]}'
                put_line = _format_metrics_line(putmet, unit)
            else:
                put_header = 'Putirka opx-only T'
                put_line = 'no equation in Thermobar 1.0.70'
            stats = (ml_header + '\n' + ml_line + '\n\n'
                     + put_header + '\n' + put_line)
            ax.text(0.02, 0.97, stats, transform=ax.transAxes,
                    ha='left', va='top', fontsize=7, color='#222222',
                    family='monospace',
                    bbox=dict(facecolor='white', edgecolor='#888888',
                              boxstyle='round,pad=0.35', alpha=0.93))

            ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

    fig.suptitle(
        'ML winner vs Putirka 2008 equation — '
        'predicted vs experimental ground truth\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190)',
        fontsize=12)
    plt.subplots_adjust(top=jgr_top(0.86), bottom=jgr_bottom(0.07),
                        left=0.07, right=0.97,
                        hspace=0.40, wspace=0.25)
    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {FIG_STEM}.pdf')

    caption = (
        'Supp. Figure 6. ML winner vs Putirka 2008 equation predictions '
        'plotted against experimental ground truth on the ExPetDB '
        'held-out test partition (opx-liq n=174, opx-only n=190). Each '
        'panel is one (track, target) cell. Colored circles = the '
        'per-cell winning ML model; gray squares = the corresponding '
        'Putirka equation evaluated through Thermobar 1.0.70 '
        '(opx-liq T = eq 28a, opx-liq P = eq 29a, opx-only P = eq 29c; '
        'opx-only T has no Putirka equivalent and shows ML only). The '
        'dashed line is the 1:1 identity (perfect prediction); the '
        'colored and gray solid lines are OLS least-squares fits of '
        'predicted on experimental for each method. Stats box reports '
        'standard ML predicted-vs-actual metrics: RMSE, R², regression '
        'slope (1.0 = unbiased through the range), and signed bias '
        '(mean residual; 0.0 = unbiased on average), with sample count '
        'where the method returned a finite prediction (Thermobar '
        'refuses to predict for samples failing its internal sanity '
        'checks; ML predicts for all rows). The figure makes the '
        'comparison the manuscript reports in aggregate per-regime form '
        '(Figures 5 and 7) visible at the per-sample level: where ML '
        'scatter is tighter against the 1:1 line than Putirka scatter, '
        'the headline RMSE win shows up here as a tighter cloud and a '
        'slope closer to 1. Source: '
        'results/classical_equivalence_regression.csv.'
    )
    (OUT_DIR / 'supp_fig_6.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {OUT_DIR}/supp_fig_6.txt')


if __name__ == '__main__':
    main()
