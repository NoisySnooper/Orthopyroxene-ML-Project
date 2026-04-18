#!/usr/bin/env python3
"""v10 Phase G.1c-extended: publication-quality figures for the
regime-stratified RMSE benchmark (reads v10_regime_allmodels.csv).

Emits four figure variants, each in PDF + PNG (300 dpi):
  figures/v10_regime_opx.{pdf,png}          opx_liq + opx_only only
  figures/v10_regime_cpx.{pdf,png}          cpx_liq + cpx_only only
  figures/v10_regime_combined.{pdf,png}     twopx only
  figures/v10_regime_all.{pdf,png}          cpx+opx+twopx tracks stacked

Universal model excluded 2026-04-18: scope limited to cpx/opx/twopx for
the current paper; universal is a later project (see
docs/v10_universal_model_exploration.md for historical context).

Each panel shows per-regime RMSE with bootstrap 95% CIs, for:
  - v10 cell spread (grey band: min/max across all Optuna cells)
  - v10 best cell (green marker + CI)
  - Putirka (red marker + CI, best equation per regime)
  - Agreda 2024 (blue diamond + CI, cpx tracks only)
  - Jorgenson 2022 (purple square + CI, cpx tracks only)
  - Wang 2021 (orange triangle + CI, cpx_liq only)

Layout per figure: rows = (track, target) combos; columns = regime type
(P | T). Always ordered P then T, so readers compare the two binnings
side-by-side for each track/target.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import RESULTS

IN_CSV = RESULTS / 'v10_regime_allmodels.csv'
OUT_DIR = PROJECT_ROOT / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered regime orders (authoritative, registered 2026-04-17).
# The manuscript cites these four-bin labels; put them first so the
# figure reflects the pre-registered partition. The exploratory finer
# bins are available in the CSV but are NOT plotted by default.
P_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
           'lithospheric_mantle', 'deeper_mantle', 'ALL']
T_ORDER = ['T<800', '800<=T<1000', '1000<=T<1200',
           '1200<=T<1400', 'T>=1400', 'ALL']

# Exploratory order — only used for the "exploratory" figure variant.
P_ORDER_EXPLORATORY = ['P<5', '5<=P<10', '10<=P<20', '20<=P<40',
                       'P>=40', 'P<20 (Agreda range)', 'ALL']

P_LABEL_PRETTY = {
    'shallow_crustal':     'Shallow crustal\n(0\u20135 kbar)',
    'deep_crustal_MASH':   'Deep crustal / MASH\n(5\u201315 kbar)',
    'lithospheric_mantle': 'Lithospheric mantle\n(15\u201330 kbar)',
    'deeper_mantle':       'Deeper mantle\n(>30 kbar)',
    'ALL':                 'ALL',
}

TRACK_TARGETS = [
    ('opx_liq',   'T_C'),
    ('opx_liq',   'P_kbar'),
    ('opx_only',  'P_kbar'),   # opx_only has only P in practice
    ('cpx_liq',   'T_C'),
    ('cpx_liq',   'P_kbar'),
    ('cpx_only',  'T_C'),
    ('cpx_only',  'P_kbar'),
    ('twopx',     'T_C'),
    ('twopx',     'P_kbar'),
]

# For opx_only we keep only P_kbar; for others we keep both if present.
SCOPES = {
    'opx':      ['opx_liq', 'opx_only'],
    'cpx':      ['cpx_liq', 'cpx_only'],
    'combined': ['twopx'],
    'all':      ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only', 'twopx'],
}

TARGET_LABEL = {'T_C': 'T (\u00b0C)', 'P_kbar': 'P (kbar)'}

STYLE = {
    'v10_best':   {'color': '#2ca02c', 'marker': 'o', 'label': 'v10 best cell'},
    'v10_spread': {'color': '0.55',    'label': 'v10 cells (min\u2013max band)'},
    'Putirka':    {'color': '#d62728', 'marker': '^', 'label': 'Putirka 2008 (best eq)'},
    'Agreda':     {'color': '#1f77b4', 'marker': 'D', 'label': 'Agreda 2024'},
    'Jorgenson':  {'color': '#9467bd', 'marker': 's', 'label': 'Jorgenson 2022'},
    'Wang':       {'color': '#ff7f0e', 'marker': 'v', 'label': 'Wang 2021'},
}


def _err(row, lo='rmse_lo', hi='rmse_hi', m='rmse'):
    """Return (elow, ehigh) magnitudes for errorbar, handling NaNs."""
    if not np.isfinite(row[lo]) or not np.isfinite(row[hi]):
        return 0.0, 0.0
    return max(0.0, row[m] - row[lo]), max(0.0, row[hi] - row[m])


def _best_of_family(df_panel, family, regimes):
    """For a family with multiple methods, return a DataFrame with one
    row per regime picking the method with lowest RMSE in that regime."""
    sub = df_panel[df_panel.method_family == family]
    if sub.empty:
        return pd.DataFrame()
    keep_rows = []
    for r in regimes:
        rdf = sub[sub.regime == r]
        if rdf.empty:
            continue
        keep_rows.append(rdf.loc[rdf['rmse'].idxmin()])
    return pd.DataFrame(keep_rows)


def panel(ax, df_panel, regime_order, track, target, show_legend=False):
    regimes = [r for r in regime_order if (df_panel.regime == r).any()]
    if not regimes:
        ax.set_visible(False); return
    x = np.arange(len(regimes))

    # v10 spread (min/max band via fill_between)
    v10 = df_panel[df_panel.method_family == 'v10']
    if not v10.empty:
        v10_min = [v10[v10.regime == r]['rmse'].min() for r in regimes]
        v10_max = [v10[v10.regime == r]['rmse'].max() for r in regimes]
        ax.fill_between(x, v10_min, v10_max, color=STYLE['v10_spread']['color'],
                        alpha=0.22, zorder=1,
                        label=STYLE['v10_spread']['label'])

    # v10 best
    if not v10.empty:
        best = v10.loc[v10.groupby('regime')['rmse'].idxmin()].set_index('regime')
        y = [best.loc[r, 'rmse'] if r in best.index else np.nan for r in regimes]
        elo, ehi = [], []
        for r in regimes:
            if r in best.index:
                lo, hi = _err(best.loc[r])
            else:
                lo, hi = 0, 0
            elo.append(lo); ehi.append(hi)
        ax.errorbar(x, y, yerr=[elo, ehi],
                    fmt=STYLE['v10_best']['marker'],
                    color=STYLE['v10_best']['color'],
                    ecolor=STYLE['v10_best']['color'],
                    capsize=2.5, markersize=6, linewidth=1.3,
                    zorder=6, label=STYLE['v10_best']['label'])

    # Offsets for externals so markers don't overlap.
    offsets = {'Putirka': -0.18, 'Agreda': 0.18,
               'Jorgenson': 0.00, 'Wang': 0.30}
    for family in ['Putirka', 'Agreda', 'Jorgenson', 'Wang']:
        bf = _best_of_family(df_panel, family, regimes)
        if bf.empty:
            continue
        bf = bf.set_index('regime')
        y, elo, ehi = [], [], []
        for r in regimes:
            if r in bf.index:
                y.append(bf.loc[r, 'rmse'])
                lo, hi = _err(bf.loc[r]); elo.append(lo); ehi.append(hi)
            else:
                y.append(np.nan); elo.append(0); ehi.append(0)
        off = offsets.get(family, 0.0)
        ax.errorbar(x + off, y, yerr=[elo, ehi],
                    fmt=STYLE[family]['marker'],
                    color=STYLE[family]['color'],
                    ecolor=STYLE[family]['color'],
                    capsize=2.5, markersize=6, linewidth=0,
                    alpha=0.9, zorder=5, label=STYLE[family]['label'])

    # n annotation above each tick
    if not df_panel.empty:
        ymax = ax.get_ylim()[1]
        for xi, r in enumerate(regimes):
            sub = df_panel[df_panel.regime == r]
            n = int(sub['n'].iloc[0]) if not sub.empty else 0
            ax.annotate(f'n={n}', (xi, ymax * 0.98), ha='center', va='top',
                        fontsize=6.5, color='0.35', zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel(f'{TARGET_LABEL[target]} RMSE', fontsize=8)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f'{track} · {target}', fontsize=9, weight='bold')

    if show_legend:
        ax.legend(loc='upper left', fontsize=6.5, framealpha=0.92,
                  ncol=2)


def figure_for_scope(df, scope_name, tracks, out_stem):
    """Build a multi-panel figure for a given set of tracks.
    Rows = one per (track, target) that exists in the data.
    Cols = [P regime, T regime]."""
    row_specs = []
    for tt, tg in TRACK_TARGETS:
        if tt not in tracks:
            continue
        if not ((df.track == tt) & (df.target == tg)).any():
            continue
        row_specs.append((tt, tg))
    if not row_specs:
        print(f'[{scope_name}] no rows available, skipping'); return

    nrows = len(row_specs)
    fig, axes = plt.subplots(nrows, 2,
                             figsize=(10.5, 2.35 * nrows + 0.5),
                             squeeze=False)

    for i, (tt, tg) in enumerate(row_specs):
        for j, (regime_type, order) in enumerate([('P', P_ORDER), ('T', T_ORDER)]):
            ax = axes[i, j]
            sub = df[(df.track == tt) & (df.target == tg)
                     & (df.regime_type == regime_type)
                     & (df.regime.isin(order))]
            panel(ax, sub, order, tt, tg,
                  show_legend=(i == 0 and j == 0))
            if j == 0:
                ax.set_xlabel('P regime (kbar)', fontsize=8)
            else:
                ax.set_xlabel('T regime (\u00b0C)', fontsize=8)

    fig.suptitle(
        f'v10 regime-stratified RMSE — {scope_name} scope\n'
        'bootstrap 95% CIs on test split; filtered to physical range for externals',
        fontsize=10.5, weight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    pdf = OUT_DIR / f'{out_stem}.pdf'
    png = OUT_DIR / f'{out_stem}.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'wrote {pdf}')
    print(f'wrote {png}')


def main():
    if not IN_CSV.exists():
        print(f'Missing {IN_CSV}. Run v10_phase_g_regime_allmodels.py first.',
              file=sys.stderr)
        return 1
    df = pd.read_csv(IN_CSV)

    for scope_name, tracks in SCOPES.items():
        figure_for_scope(df, scope_name, tracks, f'v10_regime_{scope_name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
