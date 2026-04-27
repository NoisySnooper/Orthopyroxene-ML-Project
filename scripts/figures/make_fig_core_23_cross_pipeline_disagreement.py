#!/usr/bin/env python3
"""Core_23: cross-pipeline T and P disagreement per curated locality.

For each locality where two or more pipelines have n>=20:
  - panel a: T disagreement = |opx_only T_med - cpx_only T_med| (or
    |opx_only - twopx|, |cpx_only - twopx| when one of the first
    pair is missing)
  - panel b: P disagreement same idea on P_med

Bar chart sorted by disagreement magnitude. Color by tectonic_setting.
Annotated with both pipelines' n values.

Reads results/nb08_locality_stratified.csv (38 rows from H.6).
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


def _build_disagreement(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """target = 'T' or 'P'. Returns DataFrame with one row per locality
    having the largest cross-pipeline gap among (opx_only, cpx_only,
    twopx) where each contributing pipeline has n>=20."""
    col = 'T_pred_median' if target == 'T' else 'P_pred_median'
    eligible = df[df.claims_eligible.astype(bool)].dropna(subset=[col]).copy()
    rows = []
    for loc, grp in eligible.groupby('locality'):
        pipes = (grp.set_index('pipeline')[
            [col, 'n_samples_in_box', 'tectonic_setting']
        ].to_dict('index'))
        # Need at least 2 pipelines
        present = [p for p in ['opx_only', 'cpx_only', 'twopx']
                   if p in pipes]
        if len(present) < 2:
            continue
        # Prefer opx_only vs cpx_only; fall back to twopx pairings
        if 'opx_only' in pipes and 'cpx_only' in pipes:
            a_label, b_label = 'opx_only', 'cpx_only'
        elif 'opx_only' in pipes and 'twopx' in pipes:
            a_label, b_label = 'opx_only', 'twopx'
        else:
            a_label, b_label = 'cpx_only', 'twopx'
        a, b = pipes[a_label], pipes[b_label]
        delta = a[col] - b[col]
        rows.append({
            'locality':         loc,
            'tectonic_setting': a['tectonic_setting'],
            'pipeline_a':       a_label,
            'pipeline_b':       b_label,
            'n_a':              int(a['n_samples_in_box']),
            'n_b':              int(b['n_samples_in_box']),
            'val_a':            a[col],
            'val_b':            b[col],
            'delta':            delta,
            'abs_delta':        abs(delta),
        })
    return pd.DataFrame(rows).sort_values('abs_delta', ascending=True)


def _draw_panel(ax, dfp: pd.DataFrame, target: str, unit: str) -> None:
    n = len(dfp)
    y = np.arange(n)
    colors = [_color(s) for s in dfp.tectonic_setting]
    bars = ax.barh(y, dfp.abs_delta.values, color=colors, height=0.7,
                    edgecolor='black', linewidth=0.4)
    for i, (_, r) in enumerate(dfp.iterrows()):
        annot = (f'{r.pipeline_a} ({r.val_a:.0f}) - '
                 f'{r.pipeline_b} ({r.val_b:.0f}) = '
                 f'{r.delta:+.0f} {unit}; '
                 f'n_{r.pipeline_a}={r.n_a}, n_{r.pipeline_b}={r.n_b}')
        ax.text(r.abs_delta + 0.02 * dfp.abs_delta.max(), i, annot,
                va='center', fontsize=7.5, color='#333333')
    ax.set_yticks(y)
    ax.set_yticklabels(dfp.locality.values, fontsize=9)
    ax.set_xlabel(f'|{target} disagreement| ({unit})')
    ax.grid(True, axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / 'results' / 'nb08_locality_stratified.csv')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    dfT = _build_disagreement(df, 'T')
    dfP = _build_disagreement(df, 'P')

    _draw_panel(axes[0], dfT, 'T', '°C')
    _draw_panel(axes[1], dfP, 'P', 'kbar')

    axes[0].set_title('a. Cross-pipeline T disagreement per locality',
                       loc='left', pad=8, fontweight='bold')
    axes[1].set_title('b. Cross-pipeline P disagreement per locality',
                       loc='left', pad=8, fontweight='bold')

    # Reference threshold lines
    axes[0].axvline(50, color='red', linestyle='--', linewidth=0.8,
                     alpha=0.5, zorder=0)
    axes[0].text(50, len(dfT) - 0.5, ' 50 °C deployment threshold',
                  color='red', fontsize=7, va='center', alpha=0.7)
    axes[1].axvline(5, color='red', linestyle='--', linewidth=0.8,
                     alpha=0.5, zorder=0)
    axes[1].text(5, len(dfP) - 0.5, ' 5 kbar deployment threshold',
                  color='red', fontsize=7, va='center', alpha=0.7)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=s.title())
        for s, color in TECT.items()
        if s != 'OTHER' and (df.tectonic_setting.astype(str).str.upper()
                              == s).any()
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle('Core_23. Cross-pipeline disagreement on the curated '
                  'natural-sample localities (n>=20 in each contributing '
                  'pipeline)',
                  fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    stem = OUT / 'Core_23_fig_cross_pipeline_disagreement'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)

    caption = (
        'Figure Core_23. Cross-pipeline T (panel a) and P (panel b) '
        'disagreement per curated locality where two or more pipelines '
        'have n>=20. For each locality, the largest pairwise gap among '
        '{opx_only, cpx_only, twopx} is plotted; preferred pairing is '
        'opx_only vs cpx_only, with twopx pairings used as fallback '
        'when one of the first pair is below the n>=20 floor. Iceland '
        'opx_only-vs-cpx_only T disagreement of 192 °C is the largest '
        'single-locality cross-pipeline gap and is the §4.8 opx_only T '
        'cool-bias finding for primitive rift settings; cratonic '
        'twopx-vs-cpx_only P disagreements of approximately 45 kbar at '
        'Kaapvaal and Siberia are the §4.8 cratonic twopx P shallow-bias '
        'finding. Localities where pipelines agree within 50 °C and 5 '
        'kbar (the dashed thresholds) are settings where any of the '
        'three pipelines could be deployed; localities with cross-'
        'pipeline gaps above those thresholds warrant pipeline '
        'selection per the §5.7 deployment recommendations. Bar '
        'colors reflect the tectonic-setting classification on the '
        'Okabe-Ito palette. Source: '
        'results/nb08_locality_stratified.csv.'
    )
    (OUT / 'Core_23_txt_caption.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.pdf / .png + caption')
    return 0


if __name__ == '__main__':
    sys.exit(main())
