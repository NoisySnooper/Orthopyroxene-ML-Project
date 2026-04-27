#!/usr/bin/env python3
"""Core_22: cpx_only natural-sample showcase (4 panels).

Same layout as Core_21 but for the cpx_only canonical pipeline:
ERT/pwlr for T, MLP/alr for P (substituted for TabPFN/raw per H.0b
disclosure; calibration-domain head-to-head verdict 'competitive',
see §3.11).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse Core_21 utilities to keep both showcases visually identical
from scripts.figures.make_fig_core_21_opx_showcase import (  # noqa: E402
    _build_panel_data, _draw_oneone, _draw_residual_bars, TECT, OUT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / 'results' / 'nb08_locality_stratified.csv')
    sub = _build_panel_data(df, 'cpx_only')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    _draw_oneone(axes[0, 0], sub, 'T', 'T_lit', 'T_pred_median', 100, '°C')
    axes[0, 0].set_title('a. cpx_only T 1:1 (literature midpoint vs '
                          'predicted, +/-100 °C envelope)',
                          loc='left', fontsize=11, fontweight='bold', pad=8)

    _draw_oneone(axes[0, 1], sub, 'P', 'P_lit', 'P_pred_median', 5, 'kbar')
    axes[0, 1].set_title('b. cpx_only P 1:1 (literature midpoint vs '
                          'predicted, +/-5 kbar envelope)',
                          loc='left', fontsize=11, fontweight='bold', pad=8)

    _draw_residual_bars(axes[1, 0], sub, 'T', '°C')
    axes[1, 0].set_title('c. T residuals per locality (sorted)',
                          loc='left', fontsize=11, fontweight='bold', pad=8)

    _draw_residual_bars(axes[1, 1], sub, 'P', 'kbar')
    axes[1, 1].set_title('d. P residuals per locality (sorted)',
                          loc='left', fontsize=11, fontweight='bold', pad=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=s.title())
        for s, color in TECT.items()
        if s != 'OTHER' and (sub.tectonic_setting.astype(str).str.upper()
                              == s).any()
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                fontsize=9, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle('Core_22. cpx_only natural-sample showcase '
                  '(14 claims-eligible curated localities)',
                  fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    stem = OUT / 'Core_22_fig_cpx_showcase'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)

    s = sub.dropna(subset=['T_residual']).sort_values('T_residual')
    print('Top T residuals:')
    for _, r in s.iloc[[0, 1, -2, -1]].iterrows():
        print(f'  {r.locality}: T_res={r.T_residual:+.0f} C, '
              f'P_res={r.P_residual:+.1f} kbar (n={int(r.n_samples_in_box)})')

    caption = (
        'Figure Core_22. cpx_only canonical pipeline performance on '
        'the 14 claims-eligible curated localities (n>=20 each). Same '
        'layout as Core_21. Panel b shows the cpx_only canonical model '
        '(MLP/alr after the H.0b TabPFN substitution; canonical-domain '
        'head-to-head verdict "competitive", see §3.11) tracks '
        'literature P from 0.5-70 kbar with most localities inside the '
        '+/-5 kbar envelope. Cratonic xenolith sites Kaapvaal '
        '(P_med 57.5 kbar) and Siberia (P_med 59.6 kbar) place inside '
        'the literature 20-70 / 20-60 kbar ranges with 64.7% and 45.5% '
        'in-range respectively, validating the cpx_only canonical '
        'barometer on deep-mantle natural samples. The Iceland '
        'cpx_only T at 1124 °C (75% in [1100,1250]) is consistent with '
        'literature consensus and contrasts with the opx_only Iceland '
        'cool bias documented in Core_21. Source: '
        'results/nb08_locality_stratified.csv.'
    )
    (OUT / 'Core_22_txt_caption.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.pdf / .png + caption')
    return 0


if __name__ == '__main__':
    sys.exit(main())
