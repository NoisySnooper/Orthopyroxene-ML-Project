#!/usr/bin/env python3
"""main_fig_9: ArcPL external benchmark — ML pre / ML post / Putirka.

For each opx (track, target) cell, three bars per regime on the
reconstructed ArcPL opx holdout (n=197):
  (1) ML pre-correction   (winner family, faint+hatched)
  (2) ML post-correction  (winner family, solid)
  (3) Putirka 2008        (gray)

95% CIs via 2000 paired bootstraps. Panel (c) opx-only T has no Putirka
opx-only thermometer available.

Source: results/arcpl_opx_corrected_per_regime.csv,
        results/bias_correction_shipped_v3.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._style import apply_pub_style, make_fig, add_grid, resolve_out_dir # noqa: E402
from scripts.figures._model_palette import (role_style,                   # noqa: E402
                                             family_from_method)
from scripts.figures._labels import (REGIME_TICK, TARGET_UNIT,            # noqa: E402
                                      panel_header)
from scripts.figures._legend import role_patch, add_below_legend          # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
]

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_TICKS = [REGIME_TICK.get(r, r) for r in REGIME_ORDER]


def _err(rmse, lo, hi):
    return np.vstack([np.nan_to_num(rmse - lo, nan=0.0),
                      np.nan_to_num(hi - rmse, nan=0.0)])


def _load_shipped_v3_opx() -> pd.DataFrame:
    """Return opx ship rows with a winner_v3 column, regardless of which
    canonical CSV holds them in the current results layout."""
    src = Path('results/bias_correction_shipped_v3.csv')
    if not src.exists():
        src = Path('results/bias_correction_shipped.csv')
    df = pd.read_csv(src)
    if 'winner_v3' not in df.columns and 'winner_final' in df.columns:
        df = df.rename(columns={'winner_final': 'winner_v3'})
    return df[(df.pipeline == 'opx') & (df.model != 'TabPFN')].copy()


def main():
    arc = pd.read_csv('results/arcpl_opx_corrected_per_regime.csv')
    ship_v3 = _load_shipped_v3_opx()

    fig, axes = make_fig('two_col', nrows=2, ncols=2)
    axes = axes.ravel()

    for ax, (track, target, idx) in zip(axes, PANELS):
        sub = arc[(arc['track'] == track) & (arc['target'] == target)].copy()
        piv = sub.pivot_table(
            index='regime', columns='source',
            values=['rmse', 'rmse_lo', 'rmse_hi', 'n'], aggfunc='first'
        ).reindex(REGIME_ORDER)

        x = np.arange(len(REGIME_ORDER))
        width = 0.27

        def _col(metric, source):
            try:
                return piv[metric][source].to_numpy(dtype=float)
            except KeyError:
                return np.full(len(REGIME_ORDER), np.nan, dtype=float)

        pre_r  = _col('rmse',    'ml_pre')
        pre_lo = _col('rmse_lo', 'ml_pre')
        pre_hi = _col('rmse_hi', 'ml_pre')
        post_r  = _col('rmse',    'ml_post')
        post_lo = _col('rmse_lo', 'ml_post')
        post_hi = _col('rmse_hi', 'ml_post')
        put_r  = _col('rmse',    'putirka')
        put_lo = _col('rmse_lo', 'putirka')
        put_hi = _col('rmse_hi', 'putirka')
        n_pre  = _col('n', 'ml_pre')

        ship_row = ship_v3[(ship_v3.track == track)
                           & (ship_v3.target == target)]
        family = family_from_method(
            str(ship_row.iloc[0]['model'])) if not ship_row.empty else 'RF'

        pre_color, pre_alpha, pre_hatch = role_style('pre', family)
        post_color, _, _ = role_style('post', family)
        put_color, _, _ = role_style('putirka')

        ax.bar(x - width, np.nan_to_num(pre_r), width=width,
               yerr=_err(pre_r, pre_lo, pre_hi),
               color=pre_color, alpha=pre_alpha, hatch=pre_hatch)
        ax.bar(x, np.nan_to_num(post_r), width=width,
               yerr=_err(post_r, post_lo, post_hi),
               color=post_color)
        ax.bar(x + width, np.nan_to_num(put_r), width=width,
               yerr=_err(put_r, put_lo, put_hi),
               color=put_color)

        top_of_data = np.nanmax(np.concatenate([
            np.where(np.isnan(pre_hi),  0, pre_hi),
            np.where(np.isnan(post_hi), 0, post_hi),
            np.where(np.isnan(put_hi),  0, put_hi),
        ])) if len(x) else 1.0
        if top_of_data > 0:
            ax.set_ylim(0, top_of_data * 1.10)

        if np.all(np.isnan(put_r)):
            ax.text(0.98, 1.02, 'no Putirka equivalent',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, fontstyle='italic', color='#666666')

        tick_labels = []
        for r_, base in zip(REGIME_ORDER, REGIME_TICKS):
            i = REGIME_ORDER.index(r_)
            n_txt = ''
            if pd.notna(n_pre[i]):
                n_txt = f'\n(n={int(n_pre[i])})'
            tick_labels.append(base + n_txt)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(f'RMSE ({TARGET_UNIT[target]})')

        ax.set_title(panel_header(track, target, idx))
        if not ship_row.empty:
            sr = ship_row.iloc[0]
            ax.text(0.02, 0.97, f'{sr["model"]}/{sr["feature_set"]}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=8, color='#444444')
            form = sr.get('winner_v3', 'none')
            if form and form != 'none':
                ax.text(0.98, 0.97, f'[Form {form}]',
                        transform=ax.transAxes, va='top', ha='right',
                        fontsize=9, fontweight='bold', color='#333333')
        add_grid(ax)

    handles = [
        role_patch('pre', family='RF',
                   label='ML pre-correction (winner family color)'),
        role_patch('post', family='RF',
                   label='ML post-correction (winner family color)'),
        role_patch('putirka', label='Putirka 2008'),
    ]
    add_below_legend(fig, handles, [h.get_label() for h in handles], ncol=3)

    fig.suptitle('ArcPL external benchmark: per-regime RMSE (opx, n=197)')

    stem = OUT_DIR / 'main_fig_9'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 9. External benchmark on the reconstructed ArcPL '
        'opx holdout (n=197). For each (track, target, regime) three bars: '
        'ML pre-correction (winner family color, faint and hatched), ML '
        'post-correction under the v3 ship rule (winner family color, '
        'solid), Putirka 2008 (gray). Bar color identifies the per-cell '
        'winning family on the paper-wide locked palette. Whiskers are '
        '95% percentile intervals from 2000 paired bootstraps. The '
        'shipped family/feature_set/Form is named in each panel title. '
        'Panel (c) opx-only T has no Putirka opx-only thermometer in '
        'Thermobar.'
    )
    (OUT_DIR / 'main_fig_9.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
