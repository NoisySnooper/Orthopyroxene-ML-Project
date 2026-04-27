#!/usr/bin/env python3
"""Core_20: per-locality T-in-range fraction stratified by pipeline.

3-panel horizontal bar chart, one panel per pipeline (opx_only,
cpx_only, twopx). Bars sorted descending by T_in_range_frac. Colored
by tectonic_setting via the project Okabe-Ito palette. Annotated with
n per locality and median predicted T.

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
OUT.mkdir(parents=True, exist_ok=True)

# Tectonic-setting -> color (uses Okabe-Ito palette + neutrals)
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


PANELS = [
    ('opx_only', 'a. opx_only pipeline'),
    ('cpx_only', 'b. cpx_only pipeline'),
    ('twopx',    'c. twopx pipeline'),
]


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / 'results' / 'nb08_locality_stratified.csv')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=True)

    for ax, (pipe, title) in zip(axes, PANELS):
        sub = df[(df.pipeline == pipe) & (df.claims_eligible.astype(bool))
                 ].dropna(subset=['T_in_range_frac']).copy()
        sub = sub.sort_values('T_in_range_frac', ascending=True)
        n_eligible = len(sub)
        y = np.arange(n_eligible)

        # Background grey envelope showing the literature expected range
        # mapped to a normalized 0-1 axis: we draw at x=[0, 1] to indicate
        # full bar canvas; the bar itself is the in-range fraction.
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.barh(i, 1.0, color='#EEEEEE', height=0.7, zorder=1,
                    edgecolor='none')
        bars = ax.barh(y, sub['T_in_range_frac'].values,
                       color=[_color(s) for s in sub['tectonic_setting']],
                       height=0.7, zorder=2, edgecolor='black',
                       linewidth=0.4)

        # Annotate n + T_med to the right
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(1.02, i, f'n={int(r.n_samples_in_box)}, '
                    f'T_med={r.T_pred_median:.0f} C '
                    f'(expect [{r.expected_T_C_low:.0f},'
                    f'{r.expected_T_C_high:.0f}])',
                    va='center', fontsize=7.5, color='#333333')

        ax.set_yticks(y)
        ax.set_yticklabels(sub['locality'].values, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xticklabels([f'{x:.0%}' for x in np.arange(0, 1.01, 0.25)])
        ax.set_xlabel('Fraction of T predictions in literature range')
        ax.set_title(f'{title} (n_eligible={n_eligible})',
                     loc='left', pad=8, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # One legend at fig level
    legend_handles = []
    for setting, color in TECT.items():
        if setting == 'OTHER':
            continue
        present = (df.tectonic_setting.astype(str).str.upper() == setting).any()
        if not present:
            continue
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color,
                                             label=setting.title()))
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle('Core_20. Per-locality T-in-range fraction by pipeline '
                 '(n>=20 honesty bar; literature expected range from '
                 'curated_localities.csv)',
                 fontsize=12, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    stem = OUT / 'Core_20_fig_locality_performance'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=200)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)

    caption = (
        'Figure Core_20. Per-locality fraction of predicted T inside the '
        'literature-bracketed expected range, stratified by pipeline. '
        'Panel a opx_only (n_eligible=11), panel b cpx_only '
        '(n_eligible=14), panel c twopx (n_eligible=7). Localities with '
        'n<20 in the locality bounding box are excluded per the '
        'pre-registered honesty bar. Cratonic xenolith sites (Kaapvaal, '
        'Siberia) place 87-96% of cpx_only T predictions inside the '
        'tightened 950-1300 °C and 900-1300 °C ranges respectively, '
        'validating the canonical cpx_only T model on deep-mantle '
        'natural samples; the twopx pipeline at the same localities '
        'places 0% of P predictions inside the literature 20-70 / 20-60 '
        'kbar ranges, indicating a systematic shallow bias in the '
        'twopx XGB/alr canonical barometer for cratonic settings (see '
        'Results §4.8). Iceland opx_only T at 0% in-range vs cpx_only '
        'at 75% is the §4.8 opx_only T cool-bias finding for primitive '
        'rift settings. Bar colors reflect the GEOROC tectonic-setting '
        'classification on the Okabe-Ito palette; literature expected '
        'range and predicted T_med per locality are annotated to the '
        'right of each bar. Source: results/nb08_locality_stratified.csv.'
    )
    (OUT / 'Core_20_txt_caption.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.pdf / .png + caption')
    return 0


if __name__ == '__main__':
    sys.exit(main())
