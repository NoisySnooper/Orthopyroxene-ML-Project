#!/usr/bin/env python3
"""main_fig_5: per-regime headline RMSE, all four opx (track, target) cells.

Three bars per regime cluster:
  * pre-correction RMSE of the per-cell shipped family (faint, hatched)
  * post-correction RMSE of the same family (solid)
  * best Putirka equation for the cell + regime (gray)

The pre and post bars track the SAME family inside each panel — the cell-
level shipped family from `bias_correction_shipped.csv`. This gives one
locked color per panel so the reader can identify the model at a glance.
The shipped (model / feature_set / form) is named in each panel header.

Sources:
  results/bias_correction_shipped.csv          (cell -> shipped family/form)
  results/bias_correction_per_seed.csv         (per-seed pre/post by regime)
  results/preregistered_scorecard_postcorrection.csv  (Putirka per regime)
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
from scripts.figures._labels import (REGIME_TICK, TARGET_UNIT,            # noqa: E402
                                      panel_header)
from scripts.figures._model_palette import role_style                     # noqa: E402
from scripts.figures._legend import role_patch, add_below_legend          # noqa: E402

apply_pub_style()

OUT_DIR = resolve_out_dir(PROJECT_ROOT)

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']
REGIME_TICKS = [REGIME_TICK.get(r, r) for r in REGIME_ORDER]

PANELS = [
    ('opx_liq',  'T_C',    'a'),
    ('opx_liq',  'P_kbar', 'b'),
    ('opx_only', 'T_C',    'c'),
    ('opx_only', 'P_kbar', 'd'),
]


def shipped_family(ship_df: pd.DataFrame, track: str, target: str
                   ) -> tuple[str, str, str]:
    """Return (model, feature_set, form) for the per-cell shipped tuned
    family, ignoring TabPFN rows so we get one consistent family color
    per panel."""
    sub = ship_df[(ship_df.pipeline == 'opx')
                  & (ship_df.track == track)
                  & (ship_df.target == target)
                  & (ship_df.model != 'TabPFN')]
    if sub.empty:
        return ('', '', '')
    r = sub.iloc[0]
    form = r.get('winner_v3', 'none')
    return (str(r['model']), str(r['feature_set']), str(form))


def shipped_per_regime(ps_df: pd.DataFrame, track: str, target: str,
                       model: str, fs: str, form: str
                       ) -> dict[str, tuple[float, float, float, float, float, float]]:
    """For the shipped (model, feature_set, form), return per-regime
    (pre_mean, pre_lo, pre_hi, post_mean, post_lo, post_hi) averaged
    across the 20 seeds. The CSV stores per-seed bootstrap CIs; we
    average them to follow the convention used by main_fig_6."""
    sub = ps_df[(ps_df.pipeline == 'opx')
                & (ps_df.track == track)
                & (ps_df.target == target)
                & (ps_df.model == model)
                & (ps_df.feature_set == fs)
                & (ps_df.form == form)]
    out: dict = {}
    for r in REGIME_ORDER:
        rg = sub[sub.regime == r]
        if rg.empty:
            out[r] = (np.nan,) * 6
            continue
        out[r] = (
            float(rg['pre_rmse'].mean()),
            float(rg['pre_lo'].mean()),
            float(rg['pre_hi'].mean()),
            float(rg['post_rmse'].mean()),
            float(rg['post_lo'].mean()),
            float(rg['post_hi'].mean()),
        )
    return out


def putirka_per_regime(sc_df: pd.DataFrame, track: str, target: str
                       ) -> dict[str, tuple[float, float, float]]:
    sub = sc_df[(sc_df.track == track) & (sc_df.target == target)]
    out: dict = {}
    for r in REGIME_ORDER:
        rg = sub[sub.regime == r]
        if rg.empty:
            out[r] = (np.nan, np.nan, np.nan)
            continue
        row = rg.iloc[0]
        out[r] = (
            float(row.get('best_external_rmse', np.nan)),
            float(row.get('best_external_rmse_lo', np.nan)),
            float(row.get('best_external_rmse_hi', np.nan)),
        )
    return out


def _err(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    yerr_lo = np.clip(vals - lo, 0, None)
    yerr_hi = np.clip(hi - vals, 0, None)
    return np.vstack([np.nan_to_num(yerr_lo, nan=0.0),
                      np.nan_to_num(yerr_hi, nan=0.0)])


def _bar(ax, x_off, val, lo, hi, role, family, width):
    if not np.isfinite(val):
        return
    color, alpha, hatch = role_style(role, family if role != 'putirka' else None)
    yerr = _err(np.array([val]), np.array([lo]), np.array([hi]))
    ax.bar(x_off, val, width, color=color, alpha=alpha, hatch=hatch,
           yerr=yerr)


def draw_panel(ax, ship_df, ps_df, sc_df,
               track: str, target: str, idx: str):
    model, fs, form = shipped_family(ship_df, track, target)
    pre_post = shipped_per_regime(ps_df, track, target, model, fs, form) \
        if model else {r: (np.nan,) * 6 for r in REGIME_ORDER}
    putirka = putirka_per_regime(sc_df, track, target)

    width = 0.27
    x = np.arange(len(REGIME_ORDER))
    pre_offsets = x - width
    post_offsets = x
    putirka_offsets = x + width

    for i, regime in enumerate(REGIME_ORDER):
        pre_m, pre_lo, pre_hi, post_m, post_lo, post_hi = pre_post[regime]
        put_r, put_lo, put_hi = putirka[regime]

        if model and np.isfinite(pre_m):
            _bar(ax, pre_offsets[i], pre_m, pre_lo, pre_hi, 'pre',
                 model, width)
        if model and np.isfinite(post_m):
            _bar(ax, post_offsets[i], post_m, post_lo, post_hi, 'post',
                 model, width)
        if np.isfinite(put_r):
            _bar(ax, putirka_offsets[i], put_r, put_lo, put_hi,
                 'putirka', None, width)

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_TICKS)
    ax.set_ylabel(f'RMSE ({TARGET_UNIT[target]})')
    ax.set_title(panel_header(track, target, idx))
    add_grid(ax)

    # Match main_fig_6 pattern: short in-panel annotation (top-left) names
    # the shipped model/feature_set; the form goes in a [Form X] tag.
    if model:
        ax.text(0.02, 0.97, f'{model}/{fs}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8, color='#444444')
        if form and form != 'none':
            ax.text(0.98, 0.97, f'[Form {form}]',
                    transform=ax.transAxes, va='top', ha='right',
                    fontsize=9, fontweight='bold', color='#333333')

    # Panel (c) opx-only T has no Putirka equivalent; flag it inline with
    # the panel header (top-right) instead of inside the plot area.
    if track == 'opx_only' and target == 'T_C':
        ax.text(0.98, 1.02, 'no Putirka equivalent',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, fontstyle='italic', color='#666666')


def main():
    ship_df = pd.read_csv('results/bias_correction_shipped.csv')
    ps_df = pd.read_csv('results/bias_correction_per_seed.csv')
    sc_df = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')

    fig, axes = make_fig('two_col', nrows=2, ncols=2)

    for ax, (track, target, idx) in zip(axes.ravel(), PANELS):
        draw_panel(ax, ship_df, ps_df, sc_df, track, target, idx)

    handles = [
        role_patch('pre', family='RF',
                   label='Pre-correction (shipped family)'),
        role_patch('post', family='RF',
                   label='Post-correction (shipped family)'),
        role_patch('putirka', label='Putirka (2008)'),
    ]
    add_below_legend(fig, handles, [h.get_label() for h in handles], ncol=3)

    fig.suptitle(
        'Per-regime RMSE: pre / post / Putirka (opx)\n'
        'ExPetDB held-out (opx-liq n=174, opx-only n=190), 20-seed'
    )

    stem = OUT_DIR / 'main_fig_5'
    fig.savefig(f'{stem}.pdf')
    fig.savefig(f'{stem}.png')

    caption = (
        'Figure 5. Per-regime test RMSE for the four opx (track, target) '
        'cells. For each pre-registered pressure regime plus the ALL '
        'aggregate, three bars: (1) pre-correction RMSE of the per-cell '
        'shipped tuned family (faint, hatched), (2) post-correction RMSE '
        'of the same family under the v3 ship rule (solid), (3) best '
        'available Putirka (2008) equation (gray). The pre and post bars '
        'track the SAME family inside each panel so the bar color reads '
        'as one identity per panel; the shipped family / feature_set / '
        'correction form is named in the panel header. Whiskers are '
        '20-seed means of per-seed bootstrap 95% CIs from '
        '`results/bias_correction_per_seed.csv`; Putirka values from '
        '`results/preregistered_scorecard_postcorrection.csv`. Cross-'
        'corpus performance on the ArcPL external holdout (n=197) is '
        'reported separately. Panel (c) opx-only T has no Putirka '
        'opx-only thermometer available in Thermobar.'
    )
    (OUT_DIR / 'main_fig_5.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {stem}.(pdf|png|txt)')


if __name__ == '__main__':
    main()
