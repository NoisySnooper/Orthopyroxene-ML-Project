#!/usr/bin/env python3
"""Core_14a: opx-perspective pairing matrix (LEPR two-pyroxene corpus).

Sections:
  Section 1: opx ML vs baseline regression models (rows C, G, H, O)
  Section 2: opx ML vs own machine learning models (rows A, F, I)
  Section 3: opx ML vs external machine learning models (rows B, D, E, M, N, P)

Row C (Putirka 2008 opx-liq 28a+29a) is revived: 28a T is real, 29a P is
not in the corpus so the P panel renders a 'no valid pairs' placeholder.
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
    ('C', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'Putirka 2008 opx-liq (28a T; 29a P unavailable)',
          'T_ml_opx_liq', 'T_putirka_opx_liq',
          'P_ml_opx_liq', 'P_putirka_opx_liq'),
    ('G', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Putirka 2-px (eq 36 T, eq 39 P)',
          'T_ml_opx_only', 'T_putirka_2px',
          'P_ml_opx_only', 'P_putirka_2px'),
    ('H', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Brey-Kohler 1990 two-pyroxene',
          'T_ml_opx_only', 'T_brey_kohler',
          'P_ml_opx_only', 'P_brey_kohler'),
    ('O', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Putirka 32a/32b/32c cpx-only',
          'T_ml_opx_only', 'T_putirka_cpx_only',
          'P_ml_opx_only', 'P_putirka_cpx_only'),
]

ROWS_OWN_ML = [
    ('A', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'our ML cpx-liq (ERT pwlr T, LightGBM pwlr P)',
          'T_ml_opx_liq', 'T_ml_cpx_liq',
          'P_ml_opx_liq', 'P_ml_cpx_liq'),
    ('F', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'our ML opx-only (RF pwlr T, RF pwlr P)',
          'T_ml_opx_liq', 'T_ml_opx_only',
          'P_ml_opx_liq', 'P_ml_opx_only'),
    ('I', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'our ML cpx-only (ERT pwlr T, TabPFN raw P)',
          'T_ml_opx_only', 'T_ml_cpx_only',
          'P_ml_opx_only', 'P_ml_cpx_only'),
]

ROWS_EXTERNAL_ML = [
    ('B', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'Agreda-Lopez 2024 cpx-liq',
          'T_ml_opx_liq', 'T_agreda_cpx_liq',
          'P_ml_opx_liq', 'P_agreda_cpx_liq'),
    ('D', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'Jorgenson 2022 cpx-liq (ERT)',
          'T_ml_opx_liq', 'T_jorgenson_cpx_liq',
          'P_ml_opx_liq', 'P_jorgenson_cpx_liq'),
    ('E', 'our ML opx-liq (RF pwlr T, RF alr P)',
          'Wang 2021 cpx-liq (XGB)',
          'T_ml_opx_liq', 'T_wang_cpx',
          'P_ml_opx_liq', 'P_wang_cpx'),
    ('M', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Agreda-Lopez 2024 cpx-only',
          'T_ml_opx_only', 'T_agreda_cpx_only',
          'P_ml_opx_only', 'P_agreda_cpx_only'),
    ('N', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Jorgenson 2022 cpx-only (ERT)',
          'T_ml_opx_only', 'T_jorgenson_cpx_only',
          'P_ml_opx_only', 'P_jorgenson_cpx_only'),
    ('P', 'our ML opx-only (RF pwlr T, RF pwlr P)',
          'Wang 2021 cpx-only (XGB)',
          'T_ml_opx_only', 'T_wang_cpx',
          'P_ml_opx_only', 'P_wang_cpx'),
]

SECTIONS = [
    ('Section 1: opx ML vs baseline regression models',  ROWS_BASELINE),
    ('Section 2: opx ML vs our own machine learning models', ROWS_OWN_ML),
    ('Section 3: opx ML vs external machine learning models', ROWS_EXTERNAL_ML),
]


def main():
    n_rows = sum(len(r) for _, r in SECTIONS)
    suptitle = (
        f'Core_14a. Opx-perspective pairing matrix ({n_rows} rows, '
        'LEPR two-pyroxene corpus).\n'
        'Each row pairs Method A (y-axis, opx-side) against Method B '
        '(x-axis) for P (col 1), T (col 2), disequilibrium (col 3).\n'
        f'Axes fixed: P in {P_LIM} kbar, T in {T_LIM} C, '
        f'DeltaP in {DP_LIM} kbar, DeltaT in {DT_LIM} C.\n'
        'Disequilibrium box = +/- 2 sigma_conformal from nb07; in-box '
        'percent annotated per row. RMSE of disagreement with 95% '
        'bootstrap CI shown in each panel title.'
    )
    out_stem = OUT_DIR / 'Core_14a_fig_pairing_opx'
    sigma_T, sigma_P = render_pairing_figure(out_stem, suptitle, SECTIONS)

    caption = (
        f'Figure 14a. Opx-perspective pairing matrix ({n_rows} rows; '
        'LEPR two-pyroxene corpus). Each row compares an opx-trained '
        'Method A (y-axis) against a Method B (x-axis) baseline as three '
        'panels: (col 1) pressure 1:1 scatter in kbar, (col 2) temperature '
        '1:1 scatter in Celsius, (col 3) disequilibrium map plotting '
        'DeltaP = A_P - B_P against DeltaT = A_T - B_T. Rows are grouped '
        'into three sections demarcated by header strips: '
        '(1) opx ML vs baseline regression models (rows C, G, H, O), '
        '(2) opx ML vs our own machine learning models (rows A, F, I; '
        'cross-mineral A and I are duplicated on Core_14b), and '
        '(3) opx ML vs external machine learning models (rows B, D, E, '
        'M, N, P). Each row uses a distinct Okabe-Ito color; identity '
        'line is dashed gray; legend gives the per-panel n of valid '
        'pairs. The disequilibrium panel overlays a shaded +/- 2 '
        f'sigma_conformal box (sigma_P = {sigma_P:.1f} kbar, '
        f'sigma_T = {sigma_T:.1f} C from nb07 conformal calibration); '
        'panel title annotates the fraction of residuals inside the '
        'box. RMSE of disagreement with 95% bootstrap confidence '
        'intervals (n_boot = 500, seed = 42) is shown in each panel '
        'title. Row C revives the Putirka 2008 opx-liq 28a/29a '
        'comparison: 28a T is real, but 29a P is not computed in the '
        'corpus, so the P panel renders a placeholder. Rows H and O '
        'use the Putirka 2-px eq36/eq39 columns as proxies for '
        'Brey-Kohler and Putirka cpx-only because separate Brey-Kohler '
        'and Putirka cpx-only natural-sample predictions are not in '
        'the corpus. Source data: results/nb08_natural_predictions.csv '
        'and results/core11_extended_predictions.csv merged on '
        'Experiment; row statistics from '
        'results/pairing_matrix_22rows.csv.'
    )
    (OUT_DIR / 'Core_14a_txt_caption.txt').write_text(
        caption, encoding='utf-8')
    print(f'wrote {out_stem}.pdf / .png / Core_14a_txt_caption.txt')


if __name__ == '__main__':
    main()
