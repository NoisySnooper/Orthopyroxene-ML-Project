#!/usr/bin/env python3
"""Core_14b: cpx-perspective pairing matrix (LEPR two-pyroxene corpus).

Sections:
  Section 1: cpx ML vs baseline regression models (row W; new in this revision)
  Section 2: cpx ML vs own machine learning models (rows A, I; cross-mineral, duplicated from Core_14a)
  Section 3: cpx ML vs external machine learning models (rows Q, R, S, T)
  Section 4: external ML vs external ML (row J)

Section 1 has only one row because the corpus has no real cpx-only
natural-sample predictions and no separate Putirka cpx-liq baseline
columns; cpx-liq vs Putirka 2-px eq36/39 (row W) is the only
non-proxy cpx baseline pairing currently available.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pairing._pairing_figure import (  # noqa: E402
    OUT_DIR, P_LIM, T_LIM, DP_LIM, DT_LIM, render_pairing_figure
)

ROWS_BASELINE = [
    ('W', 'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P)',
          'Putirka 2-px (eq 36 T, eq 39 P)',
          'T_ml_cpx_liq', 'T_putirka_2px',
          'P_ml_cpx_liq', 'P_putirka_2px'),
]

ROWS_OWN_ML = [
    ('A', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P)',
          'T_ml_opx_liq', 'T_ml_cpx_liq',
          'P_ml_opx_liq', 'P_ml_cpx_liq'),
    ('I', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'our ML cpx-only (ERT pwlr T, TabPFN raw P)',
          'T_ml_opx_only', 'T_ml_cpx_only',
          'P_ml_opx_only', 'P_ml_cpx_only'),
]

ROWS_EXTERNAL_ML = [
    ('Q', 'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P)',
          'Agreda-Lopez 2024 cpx-liq',
          'T_ml_cpx_liq', 'T_agreda_cpx_liq',
          'P_ml_cpx_liq', 'P_agreda_cpx_liq'),
    ('R', 'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P)',
          'Jorgenson 2022 cpx-liq (ERT)',
          'T_ml_cpx_liq', 'T_jorgenson_cpx_liq',
          'P_ml_cpx_liq', 'P_jorgenson_cpx_liq'),
    ('S', 'our ML cpx-only (ERT pwlr T, TabPFN raw P)',
          'Agreda-Lopez 2024 cpx-only',
          'T_ml_cpx_only', 'T_agreda_cpx_only',
          'P_ml_cpx_only', 'P_agreda_cpx_only'),
    ('T', 'our ML cpx-only (ERT pwlr T, TabPFN raw P)',
          'Jorgenson 2022 cpx-only (ERT)',
          'T_ml_cpx_only', 'T_jorgenson_cpx_only',
          'P_ml_cpx_only', 'P_jorgenson_cpx_only'),
]

ROWS_EXT_VS_EXT = [
    ('J', 'Agreda-Lopez 2024 cpx-only',
          'Jorgenson 2022 cpx-only (ERT)',
          'T_agreda_cpx_only', 'T_jorgenson_cpx_only',
          'P_agreda_cpx_only', 'P_jorgenson_cpx_only'),
]

SECTIONS = [
    ('Section 1: cpx ML vs baseline regression models',  ROWS_BASELINE),
    ('Section 2: cpx ML vs our own machine learning models (cross-mineral; duplicated from Core_14a)', ROWS_OWN_ML),
    ('Section 3: cpx ML vs external machine learning models', ROWS_EXTERNAL_ML),
    ('Section 4: external ML vs external ML', ROWS_EXT_VS_EXT),
]


def main():
    n_rows = sum(len(r) for _, r in SECTIONS)
    suptitle = (
        f'Core_14b. Cpx-perspective pairing matrix ({n_rows} rows, '
        'LEPR two-pyroxene corpus).\n'
        'Each row pairs Method A (y-axis) against Method B (x-axis) for '
        'P (col 1), T (col 2), disequilibrium (col 3).\n'
        f'Axes fixed: P in {P_LIM} kbar, T in {T_LIM} C, '
        f'DeltaP in {DP_LIM} kbar, DeltaT in {DT_LIM} C.\n'
        'Disequilibrium box = +/- 2 sigma_conformal from nb07; in-box '
        'percent annotated per row. RMSE of disagreement with 95% '
        'bootstrap CI shown in each panel title.'
    )
    out_stem = OUT_DIR / 'Core_14b_fig_pairing_cpx'
    sigma_T, sigma_P = render_pairing_figure(out_stem, suptitle, SECTIONS)

    caption = (
        f'Figure 14b. Cpx-perspective pairing matrix ({n_rows} rows; '
        'LEPR two-pyroxene corpus). Each row compares Method A (y-axis) '
        'against a Method B (x-axis) baseline as three panels: '
        '(col 1) pressure 1:1 scatter in kbar, (col 2) temperature '
        '1:1 scatter in Celsius, (col 3) disequilibrium map plotting '
        'DeltaP = A_P - B_P against DeltaT = A_T - B_T. Rows are '
        'grouped into four sections demarcated by header strips: '
        '(1) cpx ML vs baseline regression models (row W: our cpx-liq '
        'vs Putirka 2-px eq36/eq39; new in this revision), '
        '(2) cpx ML vs our own machine learning models (rows A and I; '
        'cross-mineral, duplicated from Core_14a), '
        '(3) cpx ML vs external machine learning models (rows Q, R, '
        'S, T), and '
        '(4) external ML vs external ML (row J: Agreda vs Jorgenson). '
        'Each row uses a distinct Okabe-Ito color; identity line is '
        'dashed gray; legend gives the per-panel n of valid pairs. '
        'The disequilibrium panel overlays a shaded +/- 2 '
        f'sigma_conformal box (sigma_P = {sigma_P:.1f} kbar, '
        f'sigma_T = {sigma_T:.1f} C from nb07 conformal calibration); '
        'panel title annotates the fraction of residuals inside the '
        'box. RMSE of disagreement with 95% bootstrap confidence '
        'intervals (n_boot = 500, seed = 42) is shown in each panel '
        'title. Section 1 has only one row because the corpus has no '
        'real cpx-only natural-sample predictions and no separate '
        'Putirka cpx-liq baseline columns; rows S, T and the '
        'cpx-only legs of J use cpx-liq data labelled as cpx-only '
        '(known proxy from compute_pairing_matrix.py). Source data: '
        'results/nb08_natural_predictions.csv and '
        'results/core11_extended_predictions.csv merged on '
        'Experiment; row statistics from '
        'results/pairing_matrix_22rows.csv.'
    )
    (OUT_DIR / 'Core_14b_txt_caption.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.pdf / .png / Core_14b_txt_caption.txt')


if __name__ == '__main__':
    main()
