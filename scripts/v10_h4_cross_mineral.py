#!/usr/bin/env python3
"""H.4: Cross-mineral convergence on natural twopx pairs.

For each pair, computes pairwise differences between method families
on the same sample:

  - Our ML twopx vs Putirka twopx eq36/eq39
  - Our ML twopx T vs Putirka cpx-only eq32d (cpx-side baseline check)
  - Our ML twopx P vs Putirka opx-only eq29c (opx-side baseline check)
  - twopx ML vs averaged opx_only + cpx_only is DEFERRED (would require
    re-running opx_only and cpx_only models on the opx/cpx halves of
    each pair; documented in run log as a follow-up step).

Stratifies by Fe-Mg equilibrium flag (KD in [0.95, 1.23] per Putirka
2008) and writes results/nb08_cross_mineral_agreement.csv plus a
summary table.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.4) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main() -> int:
    _log('caveman: H.4 cross-mineral convergence start')
    df = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_twopx.csv',
        low_memory=False)
    _log(f'caveman: loaded twopx predictions n={len(df)}')

    out = pd.DataFrame({
        'sample_id': df['sample_id'],
        'tectonic_setting': df['tectonic_setting'],
        'KD_FeMg_opx_cpx': df['KD_FeMg_opx_cpx'],
        'equilibrium_flag': df['equilibrium_flag'],
    })

    pairs = [
        ('ours_vs_putirka_twopx_eq36',
         'T_canonical_twopx_T', 'T_putirka_twopx_eq36'),
        ('ours_vs_putirka_twopx_eq37',
         'T_canonical_twopx_T', 'T_putirka_twopx_eq37'),
        ('ours_vs_putirka_twopx_eq39',
         'P_canonical_twopx_P', 'P_putirka_twopx_eq39'),
        ('ours_vs_putirka_twopx_eq38',
         'P_canonical_twopx_P', 'P_putirka_twopx_eq38'),
        ('ours_T_vs_putirka_cpxonly_eq32d',
         'T_canonical_twopx_T', 'T_putirka_cpx_only_eq32d'),
        ('ours_P_vs_putirka_opxonly_eq29c',
         'P_canonical_twopx_P', 'P_putirka_opx_only_eq29c'),
    ]

    for label, ours_col, theirs_col in pairs:
        if ours_col not in df.columns or theirs_col not in df.columns:
            _log(f'caveman: skip {label} (column missing)')
            continue
        delta = df[ours_col] - df[theirs_col]
        out[f'delta_{label}'] = delta

    out.to_csv(PROJECT_ROOT / 'results' / 'nb08_cross_mineral_agreement.csv',
               index=False, encoding='utf-8')
    _log(f'caveman: wrote nb08_cross_mineral_agreement.csv n={len(out)}')

    # Summary table
    rows = []
    for col in [c for c in out.columns if c.startswith('delta_')]:
        for stratum, mask in [('all', np.ones(len(out), dtype=bool)),
                               ('equilibrium', out.equilibrium_flag.astype(bool)),
                               ('disequilibrium',
                                ~out.equilibrium_flag.astype(bool))]:
            vals = out.loc[mask, col].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                'comparison': col.replace('delta_', ''),
                'stratum': stratum,
                'n': len(vals),
                'median_delta': float(np.median(vals)),
                'mean_delta': float(vals.mean()),
                'median_abs_delta': float(np.median(np.abs(vals))),
                'rmse_disagreement': float(np.sqrt(np.mean(vals ** 2))),
            })
    summary = pd.DataFrame(rows)
    summary_path = PROJECT_ROOT / 'results' / 'nb08_cross_mineral_summary.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8')
    _log(f'caveman: wrote nb08_cross_mineral_summary.csv n={len(summary)}')

    # Print headline summary to log
    _log('caveman: top headline rows (equilibrium-only):')
    eq_only = summary[summary.stratum == 'equilibrium'].sort_values(
        'rmse_disagreement')
    for _, r in eq_only.head(6).iterrows():
        _log(f'  {r.comparison} (n={r.n}): median_abs={r.median_abs_delta:.2f}, '
             f'rmse={r.rmse_disagreement:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
