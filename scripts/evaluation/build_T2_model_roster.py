#!/usr/bin/env python3
"""Phase 1: build T2 model roster with bootstrap CI columns.

Output: manuscripts/opx_2026/tables/T2_model_roster.csv and .tex

Schema:
    pipeline, track, target, winner_family, winner_feature_set,
    winner_rmse, winner_ci, runner_up_family, runner_up_rmse,
    runner_up_ci, tabpfn_rank, tabpfn_rmse, tabpfn_ci

Source: results/bootstrap_rmse_cis_all_cells.csv (all 200 rows, 25 per
cell: 8 tuned families x 3 feature sets + TabPFN). For winner / runner-up
we pick the lowest RMSE per (track, target) across ALL families x feature
sets. TabPFN rank is over the same set with family collapsed to family
name (TabPFN is a single row per cell).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OUT_CSV = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'tables' / 'T2_model_roster.csv'
OUT_TEX = OUT_CSV.with_suffix('.tex')


def fmt_ci(lo, hi):
    return f'[{lo:.2f}, {hi:.2f}]'


def main():
    df = pd.read_csv(BOOT_CSV)
    rows = []
    for (pipeline, track, target), sub in df.groupby(
            ['pipeline', 'track', 'target']):
        sub = sub.sort_values('rmse_point').reset_index(drop=True)
        winner = sub.iloc[0]
        runner = sub.iloc[1]
        tab_sub = sub[sub.model == 'TabPFN']
        tab_row = tab_sub.iloc[0] if len(tab_sub) else None
        if tab_row is not None:
            tabpfn_rank = int(sub[sub.model == 'TabPFN'].index[0]) + 1
        else:
            tabpfn_rank = None

        rows.append({
            'pipeline':            pipeline,
            'track':               track,
            'target':              target,
            'winner_family':       winner['model'],
            'winner_feature_set':  winner['feature_set'],
            'winner_rmse':         float(winner['rmse_point']),
            'winner_ci':           fmt_ci(winner['rmse_ci_lo'], winner['rmse_ci_hi']),
            'winner_ci_lo':        float(winner['rmse_ci_lo']),
            'winner_ci_hi':        float(winner['rmse_ci_hi']),
            'runner_up_family':    runner['model'],
            'runner_up_feature_set': runner['feature_set'],
            'runner_up_rmse':      float(runner['rmse_point']),
            'runner_up_ci':        fmt_ci(runner['rmse_ci_lo'], runner['rmse_ci_hi']),
            'tabpfn_rank':         tabpfn_rank,
            'tabpfn_rmse':         float(tab_row['rmse_point']) if tab_row is not None else None,
            'tabpfn_ci':           (fmt_ci(tab_row['rmse_ci_lo'], tab_row['rmse_ci_hi'])
                                    if tab_row is not None else None),
            'n_families_compared': int(len(sub)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')
    print(out.to_string(index=False))

    # LaTeX
    tex_lines = []
    tex_lines.append(r'\begin{table}[t]')
    tex_lines.append(r'\centering')
    tex_lines.append(r'\caption{Model roster with 95\% bootstrap CIs. '
                     r'Winner is lowest test RMSE across eight tuned '
                     r'families x three feature sets plus TabPFN, 25 '
                     r'candidates per cell. CI computed by bootstrap '
                     r'resampling (n\_boot=500) of test-set residuals '
                     r'at canonical seed=42; replaces seed-variance '
                     r'std as primary uncertainty estimate.}')
    tex_lines.append(r'\label{tab:T2_model_roster}')
    tex_lines.append(r'\begin{tabular}{lllllllll}')
    tex_lines.append(r'\toprule')
    tex_lines.append(r'Pipeline & Track & Target & Winner & FS & RMSE [CI] & Runner-up [CI] & TabPFN rank & TabPFN [CI] \\')
    tex_lines.append(r'\midrule')
    for _, r in out.iterrows():
        tex_lines.append(
            f'{r["pipeline"]} & {r["track"]} & {r["target"]} & '
            f'{r["winner_family"]} & {r["winner_feature_set"]} & '
            f'{r["winner_rmse"]:.2f} {r["winner_ci"]} & '
            f'{r["runner_up_family"]}/{r["runner_up_feature_set"]} '
            f'{r["runner_up_rmse"]:.2f} {r["runner_up_ci"]} & '
            f'{r["tabpfn_rank"]} & {r["tabpfn_rmse"]:.2f} {r["tabpfn_ci"]} \\\\'
        )
    tex_lines.append(r'\bottomrule')
    tex_lines.append(r'\end{tabular}')
    tex_lines.append(r'\end{table}')
    OUT_TEX.write_text('\n'.join(tex_lines), encoding='utf-8')
    print(f'wrote {OUT_TEX}')


if __name__ == '__main__':
    main()
