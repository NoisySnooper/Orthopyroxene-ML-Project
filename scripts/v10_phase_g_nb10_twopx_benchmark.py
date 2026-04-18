#!/usr/bin/env python3
"""Phase G.5: NB10 extended — twopx benchmark (fix the v9 empty-CSV regression).

The v9 repo shipped an empty `nb10_two_pyroxene_benchmark.csv`. v10 has
all the per-cell and multiseed twopx results on disk (results/v10_twopx_*)
but no consolidated benchmark file that combines them per
(target, representative_cell, summary stats).

This script produces that consolidated benchmark and an accompanying
figure/table so downstream consumers (NB09 manuscript compilation and
NBF figures) can cite a single artifact.

Scope:
  * Reads results/v10_twopx_per_cell_results.csv (single-seed baseline),
    results/v10_twopx_multiseed_summary.csv (20-seed mean/std/min/max),
    results/v10_twopx_ensemble_results.csv (stacked ensembles).
  * Picks the per-cell best (lowest test_rmse) per target and attaches
    the 20-seed spread.
  * Emits one row per (target, method_class) where method_class is one
    of v10_best_base | v10_best_ensemble.
  * Known deferral: Putirka two-pyroxene thermobarometry (eq 36/37/38/39)
    requires paired opx-cpx Thermobar calls and is not executed here;
    see logs/v10_phase_g_execution_log.md for the rationale. The deferral
    is documented in the output CSV's `notes` column.

Outputs:
    results/v10_twopx_benchmark_final.csv
    figures/fig29_twopx_benchmark.{pdf,png}
    tables/S8_9_twopx_benchmark.{md,csv}
    logs/v10_phase_g_nb10_twopx.log
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import FIGURES, LOGS, RESULTS

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_nb10_twopx.log'


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START', fh)
        per_cell = pd.read_csv(RESULTS / 'v10_twopx_per_cell_results.csv')
        multiseed = pd.read_csv(RESULTS / 'v10_twopx_multiseed_summary.csv')
        ensemble = pd.read_csv(RESULTS / 'v10_twopx_ensemble_results.csv')
        _log(f'per_cell={per_cell.shape} multiseed={multiseed.shape} '
             f'ensemble={ensemble.shape}', fh)

        rows = []
        for target in ('T_C', 'P_kbar'):
            # Best single-seed base.
            sub = per_cell[per_cell.target == target]
            best = sub.loc[sub.test_rmse.idxmin()]
            ms = multiseed[
                (multiseed.model == best.model) &
                (multiseed.target == target) &
                (multiseed.track == 'twopx') &
                (multiseed.feature_set == best.feature_set)
            ]
            rows.append({
                'target':             target,
                'method_class':       'v10_best_base',
                'model':              best.model,
                'feature_set':        best.feature_set,
                'track':              'twopx',
                'single_seed_rmse':   float(best.test_rmse),
                'single_seed_r2':     float(best.test_r2),
                'seed_rmse_mean':     float(ms['mean'].iloc[0]) if len(ms) else float('nan'),
                'seed_rmse_std':      float(ms['std'].iloc[0]) if len(ms) else float('nan'),
                'seed_rmse_min':      float(ms['min'].iloc[0]) if len(ms) else float('nan'),
                'seed_rmse_max':      float(ms['max'].iloc[0]) if len(ms) else float('nan'),
                'notes':              '20-seed spread from v10_twopx_multiseed_summary.csv',
            })

            # Best ensemble.
            esub = ensemble[ensemble.target == target]
            if len(esub):
                best_ens = esub.loc[esub.test_rmse.idxmin()]
                rows.append({
                    'target':             target,
                    'method_class':       'v10_best_ensemble',
                    'model':              best_ens.get('ensemble',
                                                        best_ens.get('method', '?')),
                    'feature_set':        best_ens.feature_set,
                    'track':              'twopx',
                    'single_seed_rmse':   float(best_ens.test_rmse),
                    'single_seed_r2':     float(best_ens.get('test_r2', float('nan'))),
                    'seed_rmse_mean':     float('nan'),
                    'seed_rmse_std':      float('nan'),
                    'seed_rmse_min':      float('nan'),
                    'seed_rmse_max':      float('nan'),
                    'notes':              'stacked ensemble, single fit per Phase E',
                })

            # External thermobarometry deferral placeholder row so downstream
            # consumers see the missing Putirka row explicitly.
            rows.append({
                'target':             target,
                'method_class':       'putirka_twopx',
                'model':              'Putirka 2008 eq36/37/38/39',
                'feature_set':        'n/a',
                'track':              'twopx',
                'single_seed_rmse':   float('nan'),
                'single_seed_r2':     float('nan'),
                'seed_rmse_mean':     float('nan'),
                'seed_rmse_std':      float('nan'),
                'seed_rmse_min':      float('nan'),
                'seed_rmse_max':      float('nan'),
                'notes':              'DEFERRED: requires paired opx-cpx '
                                      'Thermobar call, not executed in Phase G.5',
            })

        out_df = pd.DataFrame(rows)
        out_csv = RESULTS / 'v10_twopx_benchmark_final.csv'
        out_df.to_csv(out_csv, index=False)
        _log(f'wrote {out_csv} rows={len(out_df)}', fh)

        # --- Figure -----------------------------------------------------------
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        targets = ['T_C', 'P_kbar']
        for ax, target in zip(axes, targets):
            sub = out_df[(out_df.target == target) &
                         (out_df.method_class != 'putirka_twopx')]
            xs = np.arange(len(sub))
            colors = ['#0072B2' if mc == 'v10_best_base' else '#D55E00'
                      for mc in sub.method_class]
            ax.bar(xs, sub['single_seed_rmse'], color=colors,
                   edgecolor='black', linewidth=0.5)
            # seed-spread whiskers where available
            for i, (_, r) in enumerate(sub.iterrows()):
                if np.isfinite(r.seed_rmse_min):
                    ax.plot([i, i], [r.seed_rmse_min, r.seed_rmse_max],
                            'k-', linewidth=1.5)
                    ax.plot([i-0.1, i+0.1], [r.seed_rmse_min, r.seed_rmse_min],
                            'k-', linewidth=1.5)
                    ax.plot([i-0.1, i+0.1], [r.seed_rmse_max, r.seed_rmse_max],
                            'k-', linewidth=1.5)
            labels = [f'{r.method_class}\n{r.model}/{r.feature_set}'
                      for _, r in sub.iterrows()]
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
            unit = '°C' if target == 'T_C' else 'kbar'
            ax.set_ylabel(f'Test RMSE ({unit})')
            ax.set_title(f'twopx {target}', fontsize=10)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
        fig.suptitle('v10 two-pyroxene benchmark (single-seed point + '
                     '20-seed whiskers)', fontsize=11)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGURES / f'fig29_twopx_benchmark.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote figures/fig29_twopx_benchmark.{{pdf,png}}', fh)

        # --- SI table ---------------------------------------------------------
        tables_dir = PROJECT_ROOT / 'tables'
        tables_dir.mkdir(exist_ok=True)
        md_lines = ['# Table S8.9: v10 two-pyroxene (twopx) benchmark',
                    '',
                    'Best v10 base model and best stacked ensemble per target '
                    'on the twopx Citation-grouped held-out test split. '
                    '`seed_rmse_{mean,std,min,max}` are the 20-seed spread '
                    'from Phase G.1c multi-seed refit. The Putirka '
                    'two-pyroxene equations (eq36/37/38/39) are DEFERRED '
                    'because the call requires paired opx-cpx Thermobar '
                    'invocation; the row is retained with NaN so downstream '
                    'consumers flag the gap.', '']
        md_lines.append(
            '| Target | Method class | Model / feature set | Single-seed RMSE | Seed mean [min, max] | Notes |')
        md_lines.append('|---|---|---|---|---|---|')
        for _, r in out_df.iterrows():
            unit = '°C' if r.target == 'T_C' else 'kbar'
            if np.isfinite(r.single_seed_rmse):
                single = f'{r.single_seed_rmse:.2f} {unit}'
            else:
                single = 'deferred'
            if np.isfinite(r.seed_rmse_mean):
                seed = (f'{r.seed_rmse_mean:.2f} [{r.seed_rmse_min:.2f}, '
                        f'{r.seed_rmse_max:.2f}] {unit}')
            else:
                seed = '—'
            md_lines.append(
                f'| {r.target} | {r.method_class} | {r.model}/{r.feature_set} | '
                f'{single} | {seed} | {r.notes} |')
        md_path = tables_dir / 'S8_9_twopx_benchmark.md'
        csv_path = tables_dir / 'S8_9_twopx_benchmark.csv'
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')
        out_df.to_csv(csv_path, index=False)
        _log(f'wrote {md_path}', fh)
        _log(f'wrote {csv_path}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
