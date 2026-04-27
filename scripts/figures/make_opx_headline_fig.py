#!/usr/bin/env python3
"""Core_08: per-regime RMSE headline, all four opx track/target panels.

Five candidates per panel: v10 tuned pre-correction (gray), v10 tuned
post-correction shipped form (blue), best external Putirka benchmark
(red), TabPFN pre-correction (pink), TabPFN post-correction shipped
form (green, hatched).

Error bars are 95% bootstrap CIs from
``results/preregistered_scorecard_postcorrection.csv`` where available.

Purpose: expanded from the v9 "opx-only P headline" to cover all four
opx cells so the reader sees where the ML ship vs Putirka story is
strongest and where it is not.
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

from scripts.figures._style import apply_pub_style  # noqa: E402

apply_pub_style()

OUT_DIR = PROJECT_ROOT / 'figures' / 'core'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OK_BLUE   = '#0072B2'
OK_ORANGE = '#E69F00'
OK_GREEN  = '#009E73'
OK_RED    = '#D55E00'
OK_PURPLE = '#CC79A7'
OK_GRAY   = '#999999'

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_LABELS = ['shallow\n(<5 kbar)', 'deep-MASH\n(5-15 kbar)',
                 'litho\n(15-30 kbar)', 'deeper\n(>=30 kbar)', 'ALL']

PANELS = [
    ('opx_liq',  'T_C',    '(a) Opx + Liquid  T (C)',    'C',    'T (C)'),
    ('opx_liq',  'P_kbar', '(b) Opx + Liquid  P (kbar)', 'kbar', 'P (kbar)'),
    ('opx_only', 'T_C',    '(c) Opx only  T (C)',        'C',    'T (C)'),
    ('opx_only', 'P_kbar', '(d) Opx only  P (kbar)',     'kbar', 'P (kbar)'),
]


def build_series(sub: pd.DataFrame) -> dict:
    return {
        'v10 pre (tuned, uncorrected)': {
            'vals': sub['v10_pre_rmse'].to_numpy(),
            'lo':   sub['v10_pre_rmse_lo'].to_numpy(),
            'hi':   sub['v10_pre_rmse_hi'].to_numpy(),
            'color': OK_GRAY, 'hatch': None,
        },
        'v10 post (shipped form)': {
            'vals': sub['v10_post_rmse'].to_numpy(),
            'lo':   sub['v10_post_rmse_lo'].to_numpy(),
            'hi':   sub['v10_post_rmse_hi'].to_numpy(),
            'color': OK_BLUE, 'hatch': None,
        },
        'Putirka (2008)': {
            'vals': sub['best_external_rmse'].to_numpy(),
            'lo':   sub['best_external_rmse_lo'].to_numpy(),
            'hi':   sub['best_external_rmse_hi'].to_numpy(),
            'color': OK_RED, 'hatch': None,
        },
        'TabPFN pre': {
            'vals': sub['tabpfn_rmse'].to_numpy(),
            'lo':   sub['tabpfn_rmse_lo'].to_numpy(),
            'hi':   sub['tabpfn_rmse_hi'].to_numpy(),
            'color': OK_PURPLE, 'hatch': None,
        },
        'TabPFN post (shipped form)': {
            'vals': sub['tabpfn_post_rmse'].to_numpy(),
            'lo':   sub['tabpfn_post_rmse_lo'].to_numpy(),
            'hi':   sub['tabpfn_post_rmse_hi'].to_numpy(),
            'color': OK_GREEN, 'hatch': '//',
        },
    }


def draw_panel(ax, series: dict, x: np.ndarray, width: float,
               offsets: np.ndarray, target_unit: str, title: str):
    for (name, d), off in zip(series.items(), offsets):
        vals = np.asarray(d['vals'], float)
        lo = np.asarray(d['lo'], float)
        hi = np.asarray(d['hi'], float)
        yerr_lo = np.clip(vals - lo, 0, None)
        yerr_hi = np.clip(hi - vals, 0, None)
        yerr = np.vstack([np.nan_to_num(yerr_lo, nan=0.0),
                          np.nan_to_num(yerr_hi, nan=0.0)])
        ax.bar(x + off, np.nan_to_num(vals, nan=0.0), width,
               label=name, color=d['color'], edgecolor='black',
               linewidth=0.5, yerr=yerr, capsize=2.5,
               hatch=d['hatch'])

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, fontsize=9)
    ax.set_ylabel(f'RMSE ({target_unit})')
    ax.set_title(title, loc='left', pad=6)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)


WIN_COLOR = {
    'v10_corrected':    OK_BLUE,
    'v10_pre':          OK_GRAY,
    'external':         OK_RED,
    'tabpfn':           OK_PURPLE,
    'tabpfn_corrected': OK_GREEN,
}


def main():
    sc = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    all_handles = None
    all_labels = None

    for ax, (track, target, title, unit, _) in zip(axes.ravel(), PANELS):
        sub = sc[(sc.track == track) & (sc.target == target)].copy()
        sub = sub.set_index('regime').loc[REGIME_ORDER].reset_index()
        series = build_series(sub)

        x = np.arange(len(REGIME_ORDER))
        width = 0.16
        offsets = np.linspace(-2 * width, 2 * width, len(series))
        draw_panel(ax, series, x, width, offsets, unit, title)

        if 'winner' in sub.columns and len(sub):
            overall_row = sub[sub['regime'] == 'ALL']
            if len(overall_row):
                row = overall_row.iloc[0]
                w_now = row['winner']
                c = WIN_COLOR.get(w_now, '0.3')
                short_name = {
                    'v10_corrected':    'tuned post',
                    'v10_pre':          'tuned pre',
                    'external':         'Putirka',
                    'tabpfn':           'TabPFN pre',
                    'tabpfn_corrected': 'TabPFN post',
                }.get(w_now, w_now)
                method_map = {
                    'v10_corrected':    row.get('v10_post_method', ''),
                    'v10_pre':          row.get('v10_pre_method', ''),
                    'external':         row.get('best_external_method', ''),
                    'tabpfn':           '',
                    'tabpfn_corrected': '',
                }
                m = method_map.get(w_now, '')
                m = m if isinstance(m, str) else ''
                tag = f'ALL winner: {short_name}' + (f'  {m}' if m else '')
                ax.text(1.0, 1.02, tag,
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=9, fontweight='bold', color=c,
                        bbox=dict(facecolor='white', edgecolor=c,
                                  boxstyle='round,pad=0.25', alpha=0.95))

        if all_handles is None:
            all_handles, all_labels = ax.get_legend_handles_labels()

    # Shared legend at bottom
    fig.legend(all_handles, all_labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               framealpha=0.95, edgecolor='0.6')

    fig.suptitle(
        'TabPFN diagnostic vs tuned ML and Putirka (opx pipelines)',
        y=1.00, fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.subplots_adjust(hspace=0.35, bottom=0.12, wspace=0.22)

    stem = OUT_DIR / 'Core_08_fig45_opx_headline'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    caption = (
        'Figure 8. TabPFN diagnostic panel for the four opx track/target '
        'combinations. Purpose: this figure is a context check, not a '
        'tournament headline. It places the TabPFN v2 foundation-model '
        'baseline (pre-correction, pink; post-correction shipped form, green '
        'hatched) alongside the tuned ML baseline (pre, gray; post, blue) '
        'and the best external Putirka (2008) thermobarometer (red) so the '
        'reader can see how a zero-shot in-context model compares to our '
        'tuned pipeline. It does NOT replace the tuned-baseline bias-'
        'correction narrative presented in Core_05 through Core_07, which '
        'is where the paper\'s ship-if-better decisions are made. Regimes '
        'follow the pre-registered pressure partition (<5, 5-15, 15-30, '
        '>=30 kbar) plus ALL. Error bars are 95% bootstrap CIs (n_boot = '
        '500). "ALL winner" in each panel is the preregistered 5-way '
        'scorecard verdict under the v3 tolerance-based ship rule '
        '(Amendment 2, 2026-04-20); when the v3 verdict differs from the '
        'earlier v2 or v1 readouts, those are shown below it. Note the '
        'scorecard is used as a single diagnostic readout, not as an '
        'acceptance gate. Post-correction bars (tuned ML blue, TabPFN green '
        'hatched) use the v3 shipped form; pre-correction bars (gray, pink) '
        'are the uncorrected baseline. TabPFN post-correction was fit on a '
        'bespoke 5-seed 10-fold OOF pass (opx only) using the same Form A '
        'machinery as the tuned baseline; cpx TabPFN has no OOF pass and is '
        'excluded from post-correction. Opx-only T (panel c) has no Putirka '
        'opx-only thermometer in Thermobar, so the red bar is absent. '
        'Source: results/preregistered_scorecard_postcorrection.csv '
        '(with v2/v1 fallbacks).'
    )
    (OUT_DIR / 'Core_08_fig45_opx_headline.txt').write_text(
        caption, encoding='utf-8')

    # Clean up the old opx-only-P-only stem from the previous pass.
    for ext in ('.pdf', '.png', '.txt'):
        old = OUT_DIR / f'Core_08_fig45_opx_only_P_headline{ext}'
        if old.exists():
            old.unlink()

    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
