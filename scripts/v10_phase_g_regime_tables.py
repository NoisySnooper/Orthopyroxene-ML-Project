#!/usr/bin/env python3
"""v10 Phase G.1c-extended: publication-ready tables for the
regime-stratified benchmark.

Reads  results/v10_regime_allmodels.csv
Writes tables/v10_regime_{scope}_{regime_type}_{target}.md (Markdown)
       tables/v10_regime_{scope}_{regime_type}_{target}.csv (pivot)

Each pivot table: rows = regime, columns = method (v10 best + all
externals), cell = "RMSE [lo, hi]" (bootstrap 95% CI) with bolded
per-row winner. n per regime printed alongside.

Scopes and regime_types mirror v10_phase_g_regime_figure.py:
  scopes       = {opx, cpx, combined, all}
  regime_types = {P, T}
  targets      = {T_C, P_kbar}
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

from config import RESULTS

IN_CSV = RESULTS / 'v10_regime_allmodels.csv'
OUT_DIR = PROJECT_ROOT / 'tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCOPES = {
    'opx':      ['opx_liq', 'opx_only'],
    'cpx':      ['cpx_liq', 'cpx_only'],
    'combined': ['twopx'],
    'all':      ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only', 'twopx'],
}
# Universal model excluded 2026-04-18: scope limited to cpx/opx/twopx for
# the current paper; universal is a later project.
# Pre-registered P-regime labels first (authoritative, 2026-04-17 pre-reg);
# then the exploratory finer-grained bins; then ALL.
P_ORDER = ['shallow_crustal', 'deep_crustal_MASH', 'lithospheric_mantle',
           'deeper_mantle',
           'P<5', '5<=P<10', '10<=P<20', '20<=P<40', 'P>=40',
           'P<20 (Agreda range)', 'ALL']
T_ORDER = ['T<800', '800<=T<1000', '1000<=T<1200', '1200<=T<1400',
           'T>=1400', 'ALL']


def _fmt_cell(rmse, lo, hi):
    if not np.isfinite(rmse):
        return '—'
    if np.isfinite(lo) and np.isfinite(hi):
        return f'{rmse:.2f} [{lo:.2f}, {hi:.2f}]'
    return f'{rmse:.2f}'


def _best_method_per_regime(sub, family):
    """Collapse multi-method family to one row per regime (min RMSE)."""
    s = sub[sub.method_family == family]
    if s.empty:
        return pd.DataFrame()
    best = s.loc[s.groupby('regime')['rmse'].idxmin()].copy()
    return best


def build_table(df, scope_tracks, regime_type, target, regime_order):
    """Return (markdown_str, pivot_df) for one (scope, regime_type, target)."""
    sub = df[(df.track.isin(scope_tracks)) & (df.target == target)
             & (df.regime_type == regime_type)].copy()
    if sub.empty:
        return None, None

    # Aggregate across scope tracks: take the per-track best, then average?
    # Instead, emit per-track tables stacked vertically (cleaner).
    md_parts = []
    pivot_rows = []
    for track in scope_tracks:
        tsub = sub[sub.track == track]
        if tsub.empty:
            continue
        regimes = [r for r in regime_order if (tsub.regime == r).any()]

        # Per-regime: v10 best, and best per external family.
        cols = []
        v10_best = tsub[tsub.method_family == 'v10']
        if not v10_best.empty:
            v10_best = v10_best.loc[v10_best.groupby('regime')['rmse'].idxmin()]
            cols.append(('v10 best', v10_best.set_index('regime')))
        for family in ['Putirka', 'Agreda', 'Jorgenson', 'Wang']:
            fam = _best_method_per_regime(tsub, family)
            if not fam.empty:
                cols.append((family, fam.set_index('regime')))

        # Build markdown table
        header = ['Regime', 'n'] + [name for name, _ in cols]
        header_line = '| ' + ' | '.join(header) + ' |'
        sep_line    = '| ' + ' | '.join(['---'] * len(header)) + ' |'
        lines = [f'### {track} · {target} (binned by {regime_type})',
                 '', header_line, sep_line]

        for r in regimes:
            row_n = tsub[tsub.regime == r]['n'].iloc[0]
            rmses = {}
            display = {}
            for name, tbl in cols:
                if r in tbl.index:
                    rw = tbl.loc[r]
                    rmses[name] = rw['rmse'] if np.isfinite(rw['rmse']) else np.inf
                    display[name] = _fmt_cell(rw['rmse'], rw['rmse_lo'], rw['rmse_hi'])
                    # Add method label next to v10 best / external
                    if name == 'v10 best':
                        display[name] += f' ({rw.get("method","")})'
                    else:
                        display[name] += f' ({rw.get("method","")})'
                else:
                    rmses[name] = np.inf
                    display[name] = '—'
            winner = min(rmses, key=rmses.get) if rmses else None
            cells = []
            for name, _ in cols:
                cell = display[name]
                if name == winner and np.isfinite(rmses[winner]):
                    cell = f'**{cell}**'
                cells.append(cell)
            lines.append('| ' + ' | '.join([r, str(int(row_n))] + cells) + ' |')

            pivot_rows.append({
                'track': track, 'target': target,
                'regime_type': regime_type, 'regime': r, 'n': int(row_n),
                **{f'{name}_rmse': tbl.loc[r, 'rmse']
                   if r in tbl.index else np.nan for name, tbl in cols},
                **{f'{name}_rmse_lo': tbl.loc[r, 'rmse_lo']
                   if r in tbl.index else np.nan for name, tbl in cols},
                **{f'{name}_rmse_hi': tbl.loc[r, 'rmse_hi']
                   if r in tbl.index else np.nan for name, tbl in cols},
                'winner': winner,
            })
        lines.append('')
        md_parts.append('\n'.join(lines))

    md = ('\n'.join(md_parts)).strip() + '\n'
    pivot = pd.DataFrame(pivot_rows)
    return md, pivot


def main():
    if not IN_CSV.exists():
        print(f'Missing {IN_CSV}. Run v10_phase_g_regime_allmodels.py first.',
              file=sys.stderr)
        return 1
    df = pd.read_csv(IN_CSV)

    for scope_name, tracks in SCOPES.items():
        for regime_type, order in [('P', P_ORDER), ('T', T_ORDER)]:
            for target in ['T_C', 'P_kbar']:
                md, piv = build_table(df, tracks, regime_type, target, order)
                if md is None:
                    continue
                tag = f'v10_regime_{scope_name}_{regime_type}_{target}'
                md_path  = OUT_DIR / f'{tag}.md'
                csv_path = OUT_DIR / f'{tag}.csv'
                md_path.write_text(
                    f'# Regime-stratified RMSE · scope={scope_name} · '
                    f'bin={regime_type} · target={target}\n\n'
                    f'Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.\n\n'
                    + md, encoding='utf-8')
                if piv is not None and not piv.empty:
                    piv.to_csv(csv_path, index=False)
                print(f'wrote {md_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
