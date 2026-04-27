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

import json
import os
import sys
import textwrap
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
from scripts.figures._style import apply_pub_style  # noqa: E402
from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

apply_pub_style()

TRACK_COLOR = {
    'opx_liq':  OKABE_ITO['blue'],
    'opx_only': OKABE_ITO['orange'],
    'cpx_liq':  OKABE_ITO['green'],
    'cpx_only': OKABE_ITO['pink'],
}

TRACK_FORM_LABEL = {
    'opx_liq':  'Putirka 28a (T), Putirka 29a (P)',
    'opx_only': 'Brey-Kohler 1990 (T), Putirka 29c (P)',
    'cpx_liq':  'Putirka 33 (T), Putirka 30 (P)',
    'cpx_only': 'Putirka 32d (T), Putirka 32a (P)',
}

REF_LINE_C = '#333333'
BOX_C = OKABE_ITO['blue']

# Fixed axis limits so every panel has identical scale.
P_LIM = (-5.0, 50.0)        # kbar
T_LIM = (700.0, 1900.0)     # C
DP_LIM = (-20.0, 20.0)      # kbar
DT_LIM = (-300.0, 300.0)    # C

TITLE_WRAP = 60
XLABEL_WRAP = 42
TITLE_PAD = 8
TITLE_FONT = 8.5
LABEL_FONT = 8


def _wrap(text, width):
    return '\n'.join(textwrap.fill(line, width=width)
                     for line in text.split('\n'))

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

    # Figure Core_16: 4 rows (one per track) x 3 cols (P 1:1, T 1:1, diseq)
    # Pair up T and P per track from ml_vs_cl.
    by_track: dict = {}
    winner_by_cell: dict = {}
    for trk, tgt, form, y, yhat, r2 in ml_vs_cl:
        by_track.setdefault(trk, {})[tgt] = (form, y, yhat, r2)
    for row_rec in rows:
        winner_by_cell[(row_rec['track'], row_rec['target'])] = (
            row_rec['winner_family'], row_rec['winner_feature_set'],
            row_rec['r_squared'],
            row_rec['r_squared_ci_lo'], row_rec['r_squared_ci_hi'])

    # Conformal sigmas from nb07
    sigma_T = 50.0; sigma_P = 3.0
    qhat_path = PROJECT_ROOT / 'results' / 'nb07_conformal_qhat.json'
    if qhat_path.exists():
        with open(qhat_path) as f:
            qh = json.load(f)
        sigma_T = float(qh.get('T_C', {}).get('qhat', sigma_T))
        sigma_P = float(qh.get('P_kbar', {}).get('qhat', sigma_P))

    tracks_order = ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only']
    N = len(tracks_order)
    row_height = 3.5
    suptitle_height = 2.4
    fig_height = row_height * N + suptitle_height
    fig, axes = plt.subplots(N, 3, figsize=(18, fig_height))

    for r_idx, trk in enumerate(tracks_order):
        color = TRACK_COLOR[trk]
        form_label = TRACK_FORM_LABEL[trk]
        axP, axT, axD = axes[r_idx]

        pack_P = by_track.get(trk, {}).get('P_kbar')
        pack_T = by_track.get(trk, {}).get('T_C')

        # P 1:1 with fixed P_LIM (aspect='auto' so panel matches diseq width)
        axP.set_xlim(P_LIM); axP.set_ylim(P_LIM)
        axP.plot(P_LIM, P_LIM, ls='--', lw=1, color=REF_LINE_C,
                 alpha=0.8, zorder=1, label='1:1 identity')
        if pack_P is not None:
            form_P, yP, yhatP, _ = pack_P
            w_fam, w_fs, r2c, r2lo, r2hi = winner_by_cell[(trk, 'P_kbar')]
            n = len(yP)
            axP.scatter(yhatP, yP, s=16, alpha=0.55, color=color,
                        edgecolor='black', linewidths=0.25,
                        label=f'test set (n={n})')
            title_P = (f'({chr(65+r_idx)}1) P: {trk} winner {w_fam}/{w_fs} '
                       f'vs classical {form_P}. '
                       f'R^2 = {r2c:.2f} [{r2lo:.2f}, {r2hi:.2f}], n = {n}')
            axP.legend(fontsize=7, loc='lower right', framealpha=0.9)
        else:
            title_P = f'({chr(65+r_idx)}1) no P cell for {trk}'
        axP.set_title(_wrap(title_P, TITLE_WRAP),
                      loc='left', pad=TITLE_PAD, fontsize=TITLE_FONT)
        axP.set_xlabel(_wrap('classical-form fit predicted P (kbar)',
                             XLABEL_WRAP), fontsize=LABEL_FONT)
        axP.set_ylabel(_wrap('ML predicted P (kbar)', XLABEL_WRAP),
                       fontsize=LABEL_FONT)
        axP.tick_params(labelsize=7)
        axP.grid(True); axP.set_axisbelow(True)

        # T 1:1 with fixed T_LIM
        axT.set_xlim(T_LIM); axT.set_ylim(T_LIM)
        axT.plot(T_LIM, T_LIM, ls='--', lw=1, color=REF_LINE_C,
                 alpha=0.8, zorder=1, label='1:1 identity')
        if pack_T is not None:
            form_T, yT, yhatT, _ = pack_T
            w_fam, w_fs, r2c, r2lo, r2hi = winner_by_cell[(trk, 'T_C')]
            n = len(yT)
            axT.scatter(yhatT, yT, s=16, alpha=0.55, color=color,
                        edgecolor='black', linewidths=0.25,
                        label=f'test set (n={n})')
            title_T = (f'({chr(65+r_idx)}2) T: {trk} winner {w_fam}/{w_fs} '
                       f'vs classical {form_T}. '
                       f'R^2 = {r2c:.2f} [{r2lo:.2f}, {r2hi:.2f}], n = {n}')
            axT.legend(fontsize=7, loc='lower right', framealpha=0.9)
        else:
            title_T = f'({chr(65+r_idx)}2) no T cell for {trk}'
        axT.set_title(_wrap(title_T, TITLE_WRAP),
                      loc='left', pad=TITLE_PAD, fontsize=TITLE_FONT)
        axT.set_xlabel(_wrap('classical-form fit predicted T (C)',
                             XLABEL_WRAP), fontsize=LABEL_FONT)
        axT.set_ylabel(_wrap('ML predicted T (C)', XLABEL_WRAP),
                       fontsize=LABEL_FONT)
        axT.tick_params(labelsize=7)
        axT.grid(True); axT.set_axisbelow(True)

        # Disequilibrium: ML - classical-fit, fixed DP_LIM x DT_LIM
        axD.set_xlim(DP_LIM); axD.set_ylim(DT_LIM)
        axD.axvspan(-2 * sigma_P, 2 * sigma_P, alpha=0.08, color=BOX_C)
        axD.axhspan(-2 * sigma_T, 2 * sigma_T, alpha=0.08, color=BOX_C)
        axD.axvline(0, color='k', lw=0.6); axD.axhline(0, color='k', lw=0.6)
        if pack_P is not None and pack_T is not None:
            _, yP, yhatP, _ = pack_P
            _, yT, yhatT, _ = pack_T
            nmin = min(len(yP), len(yT))
            dP = yP[:nmin] - yhatP[:nmin]
            dT = yT[:nmin] - yhatT[:nmin]
            axD.scatter(dP, dT, s=16, alpha=0.55, color=color,
                        edgecolor='black', linewidths=0.25,
                        label=f'test set (n={nmin})')
            in_box = (np.abs(dP) < 2 * sigma_P) & (np.abs(dT) < 2 * sigma_T)
            frac = float(in_box.mean()) * 100
            title_D = (f'({chr(65+r_idx)}3) ML minus classical-fit '
                       f'residuals, {trk}: {form_label}. '
                       f'equilib. box {frac:.1f}%')
            axD.legend(fontsize=7, loc='lower right', framealpha=0.9)
        else:
            title_D = f'({chr(65+r_idx)}3) incomplete pair for {trk}'
        axD.set_title(_wrap(title_D, TITLE_WRAP),
                      loc='left', pad=TITLE_PAD, fontsize=TITLE_FONT)
        axD.set_xlabel(_wrap(f'DeltaP (kbar); box = +/- {2*sigma_P:.1f}',
                             XLABEL_WRAP), fontsize=LABEL_FONT)
        axD.set_ylabel(_wrap(f'DeltaT (C); box = +/- {2*sigma_T:.0f}',
                             XLABEL_WRAP), fontsize=LABEL_FONT)
        axD.tick_params(labelsize=7)
        axD.grid(True); axD.set_axisbelow(True)

    suptitle_y = 1.0 - 0.40 / fig_height
    fig.suptitle(
        'Core_16. Classical-equivalence regression: per-track winner ML '
        'prediction regressed on Putirka-form classical features.\n'
        'Rows: opx-liq, opx-only, cpx-liq, cpx-only. '
        'Cols: P 1:1, T 1:1, ML-minus-classical residual.\n'
        f'Axes fixed: P in {P_LIM} kbar, T in {T_LIM} C, '
        f'DeltaP in {DP_LIM} kbar, DeltaT in {DT_LIM} C.\n'
        f'Box = +/- 2 sigma_conformal (sigma_P = {sigma_P:.1f} kbar, '
        f'sigma_T = {sigma_T:.1f} C from nb07).',
        y=suptitle_y, fontsize=12, va='top')
    top_frac = 1.0 - suptitle_height / fig_height
    plt.subplots_adjust(hspace=0.55, wspace=0.22,
                        top=top_frac, bottom=0.03,
                        left=0.05, right=0.99)

    fig.savefig(f'{FIG_STEM}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {FIG_STEM}.pdf')

    caption = (
        'Figure 16. Classical-equivalence regression per track. For each of '
        'four track groups (opx-liq, opx-only, cpx-liq, cpx-only), we fit a '
        'linear regression on Putirka-form classical features to the '
        'winner ML predictions on the held-out test set. Columns: (1) P '
        '1:1 scatter of ML predicted vs classical-fit predicted in kbar, '
        '(2) T 1:1 scatter in Celsius, (3) ML-minus-classical-fit residual '
        'disequilibrium map. Each row uses a distinct Okabe-Ito color; '
        'identity line dashed gray; legend gives n. Per-panel title names '
        'the cell winner (model family / feature set) and the classical '
        'form used for the fit: opx-liq uses Putirka 28a (T) and 29a (P), '
        'opx-only uses Brey-Kohler 1990 (T) and Putirka 29c (P), cpx-liq '
        'uses Putirka 33 (T) and 30 (P), cpx-only uses Putirka 32d (T) '
        'and 32a (P). The R^2 with 500-bootstrap 95% CI measures how much '
        'of the ML prediction variance is explained by a linear combination '
        'of the classical features. The disequilibrium map overlays a +/- '
        f'2 sigma_conformal box (sigma_P = {sigma_P:.1f} kbar, sigma_T = '
        f'{sigma_T:.1f} C from nb07 calibration) and annotates the fraction '
        'of residuals inside the equilibrium tolerance. A high in-box '
        'fraction means the ML tracks the classical form within the '
        'calibrated petrological tolerance; a low fraction means the ML is '
        'doing something systematically different. Source: '
        'results/classical_equivalence_regression.csv.'
    )
    (FIG_STEM.parent / 'Core_16_txt_caption.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {FIG_STEM.parent}/Core_16_txt_caption.txt')


if __name__ == '__main__':
    main()
