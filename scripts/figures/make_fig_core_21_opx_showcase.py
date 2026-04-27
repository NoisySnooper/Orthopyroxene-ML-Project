#!/usr/bin/env python3
"""Core_21: opx_only natural-sample showcase (4 panels).

a. T predicted-vs-literature 1:1 across claims-eligible localities.
b. P predicted-vs-literature 1:1 across claims-eligible localities.
c. Per-locality T residuals (ML T_med - literature T midpoint), bar
   chart sorted by residual magnitude.
d. Per-locality P residuals same idea.

Reads results/nb08_locality_stratified.csv (locality summary). The
literature midpoint is (expected_T_C_low + expected_T_C_high) / 2 etc.
Marker size proportional to log(n_samples).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._model_palette import OKABE_ITO  # noqa: E402

OUT = PROJECT_ROOT / 'figures' / 'core'

TECT = {
    'CONVERGENT MARGIN':                            OKABE_ITO['vermillion'],
    'INTRAPLATE VOLCANICS':                         OKABE_ITO['pink'],
    'RIFT VOLCANICS':                               OKABE_ITO['orange'],
    'CONTINENTAL FLOOD BASALT':                     OKABE_ITO['yellow'],
    'ARCHEAN CRATON (INCLUDING GREENSTONE BELTS)':  '#999933',
    'OCEAN ISLAND':                                 OKABE_ITO['sky_blue'],
    'OCEANIC PLATEAU':                              OKABE_ITO['blue'],
    'OTHER':                                        '#888888',
}


def _color(setting: str) -> str:
    s = (setting or '').strip().upper()
    return TECT.get(s, TECT['OTHER'])


def _build_panel_data(df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    sub = df[(df.pipeline == pipeline)
             & (df.claims_eligible.astype(bool))].copy()
    sub['T_lit'] = (sub['expected_T_C_low'] + sub['expected_T_C_high']) / 2
    sub['P_lit'] = (sub['expected_P_kbar_low'] + sub['expected_P_kbar_high']) / 2
    sub['T_residual'] = sub['T_pred_median'] - sub['T_lit']
    sub['P_residual'] = sub['P_pred_median'] - sub['P_lit']
    sub['marker_size'] = 30 + 80 * np.log10(sub['n_samples_in_box'].clip(20))
    return sub


def _draw_oneone(ax, sub, axis, lit_col, pred_col, env, label):
    valid = sub.dropna(subset=[lit_col, pred_col])
    for _, r in valid.iterrows():
        ax.scatter(r[lit_col], r[pred_col], s=r.marker_size,
                    color=_color(r.tectonic_setting),
                    edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.annotate(r.locality, (r[lit_col], r[pred_col]),
                     xytext=(5, 4), textcoords='offset points',
                     fontsize=7, color='#333333')
    lo, hi = ax.get_xlim()[0], ax.get_xlim()[1]
    if axis == 'T':
        rng = (max(800, lo), min(1400, hi if hi > 1000 else 1400))
    else:
        rng = (-2, max(70, hi if hi > 30 else 70))
    ax.plot(rng, rng, ls='--', color='#444444', linewidth=1, zorder=1)
    ax.fill_between(rng, [r - env for r in rng], [r + env for r in rng],
                     color='#888888', alpha=0.12, zorder=0)
    ax.set_xlim(rng); ax.set_ylim(rng)
    ax.set_xlabel(f'Literature {axis} midpoint ({label})')
    ax.set_ylabel(f'Predicted opx_only {axis}_med ({label})')
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def _draw_residual_bars(ax, sub, target, unit):
    res_col = f'{target}_residual'
    s = sub.dropna(subset=[res_col]).sort_values(res_col)
    n = len(s)
    y = np.arange(n)
    colors = [_color(t) for t in s.tectonic_setting]
    ax.barh(y, s[res_col].values, color=colors, height=0.7,
             edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(s.locality.values, fontsize=8)
    for i, (_, r) in enumerate(s.iterrows()):
        v = r[res_col]
        x = v + (0.02 * abs(s[res_col]).max() * (1 if v >= 0 else -1))
        ax.text(x, i, f'{v:+.0f} {unit} (n={int(r.n_samples_in_box)})',
                 va='center', ha='left' if v >= 0 else 'right',
                 fontsize=7.5, color='#333333')
    ax.set_xlabel(f'{target} residual: opx_only - literature midpoint ({unit})')
    ax.grid(True, alpha=0.3, axis='x'); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / 'results' / 'nb08_locality_stratified.csv')
    sub = _build_panel_data(df, 'opx_only')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    _draw_oneone(axes[0, 0], sub, 'T', 'T_lit', 'T_pred_median', 100, '°C')
    axes[0, 0].set_title('a. opx_only T 1:1 (literature midpoint vs '
                          'predicted, +/-100 °C envelope)',
                          loc='left', fontsize=11, fontweight='bold', pad=8)

    _draw_oneone(axes[0, 1], sub, 'P', 'P_lit', 'P_pred_median', 3, 'kbar')
    axes[0, 1].set_title('b. opx_only P 1:1 (literature midpoint vs '
                          'predicted, +/-3 kbar envelope)',
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

    fig.suptitle('Core_21. opx_only natural-sample showcase '
                  '(11 claims-eligible curated localities)',
                  fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    stem = OUT / 'Core_21_fig_opx_showcase'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)

    # Print residuals to log for caption verification
    s = sub.dropna(subset=['T_residual']).sort_values('T_residual')
    print('Top T residuals:')
    for _, r in s.iloc[[0, 1, -2, -1]].iterrows():
        print(f'  {r.locality}: T_res={r.T_residual:+.0f} C, '
              f'P_res={r.P_residual:+.1f} kbar (n={int(r.n_samples_in_box)})')

    caption = (
        'Figure Core_21. opx_only canonical pipeline performance on the '
        '11 claims-eligible curated localities (n>=20 each). Panels '
        'a/b show predicted-vs-literature 1:1 for T (a) and P (b) with '
        'identity line and +/-100 °C / +/-3 kbar envelopes; markers '
        'sized by log(n) and colored by tectonic-setting on the '
        'Okabe-Ito palette. Panels c/d show per-locality residuals '
        '(ML T_med or P_med minus literature midpoint), sorted, with '
        'sample counts annotated. The opx_only canonical pair is '
        'LightGBM/alr for T and RF/pwlr for P. Spitsbergen (+240 °C) '
        'and Iceland (-244 °C) are the largest opposite-sign T '
        'residuals; Iceland is the §4.8 cool-bias finding for primitive '
        'rift settings. Cratonic xenolith sites Kaapvaal and Siberia '
        'sit at the high-T end of the literature range and show '
        'systematic warm bias on opx_only T (Siberia +174 °C); '
        'opx_only P at the cratonic localities also misses (Kaapvaal '
        'P_med 14.4 kbar inside the wide [20,70] range only at '
        '40%; Siberia P_med 28.8 kbar tracks the lower end of [20,60] '
        'at 84%). Source: results/nb08_locality_stratified.csv.'
    )
    (OUT / 'Core_21_txt_caption.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.pdf / .png + caption')
    return 0


if __name__ == '__main__':
    sys.exit(main())
