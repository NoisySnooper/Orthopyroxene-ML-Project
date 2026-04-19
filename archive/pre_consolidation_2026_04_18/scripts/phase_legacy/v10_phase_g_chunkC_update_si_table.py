#!/usr/bin/env python3
"""Phase G Chunk C: write SI Table S8.5.4 (robust per-regime claims audit).

Reads results/v10_opx_per_regime_claims_audit_robust.csv and emits:
    tables/S8_5_4_regime_claims_audit_robust_opx_liq.md
    tables/S8_5_4_regime_claims_audit_robust_opx_liq.csv

The plain Chunk B audit remains at tables/S8_5_3_regime_claims_audit_opx_liq.*
for historical reference. The robust version is strictly richer (adds seed-
axis RMSE distribution columns and a two-axis verdict) and is what the
manuscript autofill cites as canonical.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import RESULTS


def main():
    src = RESULTS / 'v10_opx_per_regime_claims_audit_robust.csv'
    if not src.exists():
        print(f'missing {src}; run v10_phase_g_chunkC_robust_audit.py first',
              file=sys.stderr)
        return 1
    df = pd.read_csv(src)

    # Display-friendly formatted frame for the Markdown rendering.
    disp = pd.DataFrame({
        'Regime':           df['regime'],
        'Target':           df['target'],
        'n':                df['n'],
        'v10 cell':         df['v10_method'],
        'v10 RMSE [95% CI]': [
            f'{r:.2f} [{lo:.2f}, {hi:.2f}]'
            for r, lo, hi in zip(df['v10_rmse'],
                                 df['v10_rmse_lo'], df['v10_rmse_hi'])
        ],
        'Seed RMSE mean [min, max]': [
            f'{m:.2f} [{lo:.2f}, {hi:.2f}]'
            for m, lo, hi in zip(df['seed_rmse_mean'],
                                 df['seed_rmse_min'], df['seed_rmse_max'])
        ],
        'Putirka eq': df['putirka_method'],
        'Putirka RMSE [95% CI]': [
            f'{r:.2f} [{lo:.2f}, {hi:.2f}]'
            for r, lo, hi in zip(df['putirka_rmse'],
                                 df['putirka_rmse_lo'],
                                 df['putirka_rmse_hi'])
        ],
        'Axis-1 (residual) non-overlap': df['axis1_nonoverlap'],
        'Axis-2 (seed) non-overlap':     df['axis2_nonoverlap'],
        'Robust verdict':                df['robust_verdict'],
    })

    out_csv = Path('tables') / 'S8_5_4_regime_claims_audit_robust_opx_liq.csv'
    out_md  = Path('tables') / 'S8_5_4_regime_claims_audit_robust_opx_liq.md'
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save the full CSV (not the display-rounded one).
    df.to_csv(out_csv, index=False)

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Table S8.5.4 . Per-regime claims audit, two-axis honesty '
                'bar (opx_liq)\n\n')
        f.write('Axis 1 (test-set sampling noise): bootstrap 95% CI on RMSE '
                'over paired (y_true, y_pred), 500 resamples, '
                'seed = SEED_BOOTSTRAP. Axis 2 (model-fit stochasticity): '
                '20-seed refit of the same cell at seeds 42..61, fixed '
                'Citation-grouped train/test split. Putirka equations have '
                'no seed axis (deterministic closed-form).\n\n')
        f.write('Verdict rule: "outperforms" requires BOTH '
                'v10_rmse_hi < putirka_rmse_lo (axis 1) '
                'AND seed_rmse_hi < putirka_rmse_lo (axis 2) '
                'AND n >= 20.\n\n')
        f.write(disp.to_markdown(index=False))
        f.write('\n')

    print(f'wrote {out_csv}')
    print(f'wrote {out_md}')
    print()
    print(disp.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
