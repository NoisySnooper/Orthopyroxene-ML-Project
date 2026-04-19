"""Add TabPFN RMSE columns to the pre-registered post-correction scorecard.

Option B C7. Runs once tabpfn_regime_rmse.csv has been regenerated with
the 20-seed protocol (count=20 per row).

Adds 3 columns (tabpfn_rmse, tabpfn_rmse_lo, tabpfn_rmse_hi) and extends
the winner logic to a 4-candidate argmin over:

    {v10_pre, v10_corrected, external, tabpfn}

If TabPFN is the argmin winner='tabpfn'. Otherwise the original winner
is preserved. Best-external is NaN for opx_only and cpx_only combos
(no external model for those tracks); argmin skips NaN automatically.

Input:
  results/preregistered_scorecard_postcorrection.csv (40, 18)
  results/tabpfn_regime_rmse.csv (40, 18)

Output:
  results/preregistered_scorecard_postcorrection.csv (40, 22)

Idempotent: if tabpfn_rmse column already present, re-computes from
current tabpfn_regime_rmse (handy after a re-run) but does NOT duplicate
the column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS = ROOT / 'results'


def pick_winner(pre: float, post: float, ext: float, tab: float) -> str:
    cand = {
        'v10_pre': pre,
        'v10_corrected': post,
        'external': ext if pd.notna(ext) else np.inf,
        'tabpfn': tab if pd.notna(tab) else np.inf,
    }
    return min(cand, key=cand.get)


def main():
    sc_path = RESULTS / 'preregistered_scorecard_postcorrection.csv'
    tf_path = RESULTS / 'tabpfn_regime_rmse.csv'

    sc = pd.read_csv(sc_path)
    tf = pd.read_csv(tf_path)

    assert len(sc) == 40, f'scorecard expected 40 rows, got {len(sc)}'
    assert len(tf) == 40, f'tabpfn_regime_rmse expected 40 rows, got {len(tf)}'

    tf_slim = tf[['track', 'target', 'regime', 'rmse', 'rmse_lo', 'rmse_hi']].rename(
        columns={
            'rmse': 'tabpfn_rmse',
            'rmse_lo': 'tabpfn_rmse_lo',
            'rmse_hi': 'tabpfn_rmse_hi',
        }
    )

    # Drop pre-existing tabpfn cols if re-running.
    for col in ('tabpfn_rmse', 'tabpfn_rmse_lo', 'tabpfn_rmse_hi'):
        if col in sc.columns:
            sc = sc.drop(columns=col)

    merged = sc.merge(tf_slim, on=['track', 'target', 'regime'], how='left')
    assert len(merged) == 40
    assert merged['tabpfn_rmse'].notna().all(), (
        'missing tabpfn_rmse rows after merge; check key alignment'
    )

    merged['winner'] = merged.apply(
        lambda r: pick_winner(
            r['v10_pre_rmse'],
            r['v10_post_rmse'],
            r['best_external_rmse'],
            r['tabpfn_rmse'],
        ),
        axis=1,
    )

    merged.to_csv(sc_path, index=False)
    winner_counts = merged['winner'].value_counts().to_dict()
    print(f'wrote {sc_path.name}: shape={merged.shape}, winners={winner_counts}')


if __name__ == '__main__':
    main()
