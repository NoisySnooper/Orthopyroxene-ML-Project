#!/usr/bin/env python3
"""Phase 2 P2.4: classical-equivalence regression.

For each of 8 cells, fit a classical-form linear model with the WINNER'S
ML test predictions as the dependent variable and classical-form features
(Putirka 2008 shape for the corresponding cell) as independent variables.
R^2 measures how well a classical functional form approximates the ML
prediction. High R^2 means the ML IS approximately the classical formula
plus a smooth residual; low R^2 means the ML is doing something
genuinely different.

Output:
  results/classical_equivalence_regression.csv
  figures/core/Core_16_fig_classical_equivalence.pdf (+ .png)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_train_test import prepare_train_test  # noqa: E402

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OUT_CSV = PROJECT_ROOT / 'results' / 'classical_equivalence_regression.csv'
FIG_STEM = PROJECT_ROOT / 'figures' / 'core' / 'Core_16_fig_classical_equivalence'


CELLS = [
    ('opx', 'opx_liq', 'T_C',    'Putirka 28a (opx-liq T)'),
    ('opx', 'opx_liq', 'P_kbar', 'Putirka 29a (opx-liq P)'),
    ('opx', 'opx_only', 'T_C',   'Brey-Kohler 1990 (opx-only T)'),
    ('opx', 'opx_only', 'P_kbar', 'Putirka 29c (opx-only P)'),
    ('cpx', 'cpx_liq', 'T_C',    'Putirka 33 (cpx-liq T)'),
    ('cpx', 'cpx_liq', 'P_kbar', 'Putirka 30 (cpx-liq P)'),
    ('cpx', 'cpx_only', 'T_C',   'Putirka 32d (cpx-only T)'),
    ('cpx', 'cpx_only', 'P_kbar', 'Putirka 32a (cpx-only P)'),
]


def classical_features_for(pipeline, track, target, feat_names, X):
    """Return (X_classical, feature_labels) built from available raw
    features. Missing features are skipped silently."""
    def col(name):
        return X[:, feat_names.index(name)] if name in feat_names else None

    features = []
    labels = []

    def add(label, series, transform=None):
        if series is None:
            return
        v = np.asarray(series, dtype=float)
        if transform == 'log':
            v = np.log(np.clip(v, 1e-6, None))
        elif transform == 'log_ratio_mg_num':
            v = np.log(np.clip(v, 1e-6, None))
        features.append(v)
        labels.append(label)

    if pipeline == 'opx' and track == 'opx_liq':
        add('liq_MgO', col('raw_liq_MgO'))
        add('liq_FeO', col('raw_liq_FeO'))
        add('liq_SiO2', col('raw_liq_SiO2'))
        add('liq_Al2O3', col('raw_liq_Al2O3'))
        add('liq_CaO', col('raw_liq_CaO'))
        add('liq_Na2O', col('raw_liq_Na2O'))
        add('liq_K2O', col('raw_liq_K2O'))
        mgn = col('liq_Mg_num')
        if mgn is not None:
            add('ln_liq_Mg_num', mgn, transform='log')
        add('Al_VI_opx', col('Al_VI'))
        add('Mg_num_opx', col('Mg_num'))
    elif pipeline == 'opx' and track == 'opx_only':
        add('Al_VI_opx', col('Al_VI'))
        add('Al_IV_opx', col('Al_IV'))
        add('MgTs', col('MgTs'))
        add('Mg_num_opx', col('Mg_num'))
        add('En_frac', col('En_frac'))
        add('Fs_frac', col('Fs_frac'))
        add('Wo_frac', col('Wo_frac'))
        add('raw_Al2O3', col('raw_Al2O3'))
        add('raw_CaO', col('raw_CaO'))
        add('raw_Cr2O3', col('raw_Cr2O3'))
        add('raw_MgO', col('raw_MgO'))
        add('raw_FeO_total', col('raw_FeO_total'))
    elif pipeline == 'cpx' and track == 'cpx_liq':
        # Use the available liq oxides + cpx cation features the cpx
        # feature builder exposes.
        for nm in feat_names:
            if nm.startswith('raw_liq_') or nm in ('Mg_num', 'Al_IV', 'Al_VI',
                                                    'Jd', 'DiHd', 'En_Fs',
                                                    'Ca_cpx', 'Na_cpx',
                                                    'Al_cpx', 'Mg_cpx',
                                                    'Fe_cpx', 'Si_cpx',
                                                    'Ti_cpx'):
                add(nm, col(nm))
    elif pipeline == 'cpx' and track == 'cpx_only':
        for nm in feat_names:
            if nm in ('Mg_num', 'Al_IV', 'Al_VI', 'Jd', 'DiHd', 'En_Fs',
                      'Ca_cpx', 'Na_cpx', 'Al_cpx', 'Mg_cpx', 'Fe_cpx',
                      'Si_cpx', 'Ti_cpx', 'Cr_cpx',
                      'raw_Al2O3', 'raw_CaO', 'raw_Cr2O3', 'raw_MgO',
                      'raw_Na2O', 'raw_SiO2', 'raw_TiO2',
                      'raw_FeO_total'):
                add(nm, col(nm))

    if not features:
        return None, []
    X_cl = np.column_stack(features)
    return X_cl, labels


def canonical_path(pipeline, model, target, track, fs):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{fs}.joblib')


def interpret(r2):
    if r2 > 0.90:
        return (f'ML prediction well-approximated by classical functional '
                f'form (R^2={r2:.2f}); ML is a flexible smoother of the '
                f'classical relationship rather than a fundamentally opaque '
                f'black box')
    if r2 >= 0.75:
        return (f'ML prediction partially tracks classical functional form '
                f'(R^2={r2:.2f}); departures encode non-linear residual '
                f'structure the classical equation cannot express')
    return (f'ML prediction diverges from classical functional form '
            f'(R^2={r2:.2f}); the ML is doing something genuinely different '
            f'from the classical equation at this task')


def bootstrap_r2(y, yhat, n_boot=500, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[b] = r2_score(y[idx], yhat[idx])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    boot = pd.read_csv(BOOT_CSV)
    rows = []
    ml_vs_cl = []

    for pipe, track, tgt, form_name in CELLS:
        winner = (boot[(boot.pipeline == pipe) & (boot.track == track)
                       & (boot.target == tgt)]
                  .sort_values('rmse_point').iloc[0])
        family = winner['model']
        fs = winner['feature_set']
        print(f'\n=== {track}/{tgt} winner {family}/{fs} ===')

        # Get ML predictions on test set (in target units)
        if family == 'TabPFN':
            tab = pd.read_csv('results/tabpfn_predictions.csv')
            m = tab[(tab.track == track) & (tab.target == tgt)
                    & (tab.seed == 42)]
            y_ml = m['y_pred'].to_numpy(dtype=float)
            y_true = m['y_true'].to_numpy(dtype=float)
            d = prepare_train_test(pipe, track, tgt, 'raw')
            feat_names = d['feat_names']
            X_te_raw = d['X_te']
        else:
            mp = canonical_path(pipe, family, tgt, track, fs)
            est = joblib.load(mp)
            d_win = prepare_train_test(pipe, track, tgt, fs)
            y_ml = np.asarray(est.predict(d_win['X_te']), dtype=float)
            y_true = d_win['y_te']
            d = prepare_train_test(pipe, track, tgt, 'raw')
            feat_names = d['feat_names']
            X_te_raw = d['X_te']

        X_cl, labels = classical_features_for(
            pipe, track, tgt, feat_names, X_te_raw)
        if X_cl is None:
            print('  NO classical features available; skip')
            continue

        # Drop rows with any NaN in classical features
        mask = np.all(np.isfinite(X_cl), axis=1) & np.isfinite(y_ml)
        if mask.sum() < 10:
            print('  too few rows after NaN drop; skip')
            continue
        Xfit = X_cl[mask]; yfit = y_ml[mask]
        reg = LinearRegression().fit(Xfit, yfit)
        yhat = reg.predict(Xfit)
        r2 = float(r2_score(yfit, yhat))
        lo, hi = bootstrap_r2(yfit, yhat, n_boot=500, random_state=42)
        print(f'  {form_name}: R^2={r2:.3f} [{lo:.3f}, {hi:.3f}] '
              f'n={mask.sum()} n_feat={X_cl.shape[1]}')

        rows.append({
            'pipeline':           pipe,
            'track':              track,
            'target':             tgt,
            'winner_family':      family,
            'winner_feature_set': fs,
            'classical_form_name': form_name,
            'n_samples':          int(mask.sum()),
            'n_coefficients':     int(X_cl.shape[1]),
            'r_squared':          r2,
            'r_squared_ci_lo':    lo,
            'r_squared_ci_hi':    hi,
            'interpretation':     interpret(r2),
        })
        ml_vs_cl.append((track, tgt, form_name, yfit, yhat, r2))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}')
    print(out[['track', 'target', 'r_squared',
               'r_squared_ci_lo', 'r_squared_ci_hi']].to_string(index=False))

    # Figure Core_16: 2x4 grid of ML vs classical-fit scatters
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (track, tgt, form_name, y, yhat, r2) in zip(axes.ravel(), ml_vs_cl):
        ax.scatter(yhat, y, s=8, alpha=0.55, color='#2b8cbe', edgecolor='none')
        lo, hi = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('classical-fit predicted')
        ax.set_ylabel('ML predicted')
        unit = '°C' if tgt == 'T_C' else 'kbar'
        ax.set_title(f'{track}/{tgt}  R²={r2:.2f}\n{form_name} ({unit})',
                     fontsize=9, loc='left')
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(ml_vs_cl):]:
        ax.set_axis_off()
    fig.suptitle('Classical-equivalence regression: Putirka-form fit '
                 'to ML predictions', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {FIG_STEM}.pdf')


if __name__ == '__main__':
    main()
