"""Merge 20-seed TabPFN rows into the canonical multiseed + regime CSVs.

Runs at Option B C3/C4 once the 20-seed regen has completed.

C3 (multiseed):
  results/tabpfn_multiseed_summary.csv (8 rows, count=20) + TabPFN
    -> append by pipeline into results/{opx,cpx}_multiseed_summary.csv
    -> target shapes: (96+4=100, 10) per pipeline
  results/tabpfn_multiseed_results.csv (160 rows post-regen) + TabPFN
    -> project to canonical 7 cols (drop rmse/mae/r2/n_train/n_test)
    -> append by pipeline into results/{opx,cpx}_multiseed_results.csv
    -> target shapes: (1920+80=2000, 7) per pipeline

C4 (regime):
  results/tabpfn_regime_rmse.csv (40 rows, 18-col exact schema match)
    -> append into results/regime_allmodels.csv
    -> target shape: (4769+40=4809, 18)
  NOT merged into regime_allmodels_postcorrection.csv (TabPFN is
  excluded from bias correction per C5).

Idempotent: detects existing TabPFN rows in canonical CSVs and skips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS = ROOT / 'results'


CANONICAL_MULTISEED_RESULTS_COLS = [
    'pipeline', 'model', 'target', 'track', 'feature_set', 'seed', 'test_rmse',
]


def merge_multiseed():
    tab_sum = pd.read_csv(RESULTS / 'tabpfn_multiseed_summary.csv')
    tab_res = pd.read_csv(RESULTS / 'tabpfn_multiseed_results.csv')

    # Guardrail: expect 20-seed data.
    assert (tab_sum['count'] == 20).all(), (
        f'tabpfn_multiseed_summary count != 20 per row: got '
        f'{tab_sum["count"].unique()}. Run 20-seed regen first.'
    )
    assert len(tab_res) == 160, (
        f'tabpfn_multiseed_results expected 160 rows (8 combos x 20 seeds), '
        f'got {len(tab_res)}.'
    )

    # Project to canonical 7-col schema.
    tab_res_proj = tab_res[CANONICAL_MULTISEED_RESULTS_COLS].copy()

    for pipe in ('opx', 'cpx'):
        summary_path = RESULTS / f'{pipe}_multiseed_summary.csv'
        results_path = RESULTS / f'{pipe}_multiseed_results.csv'

        canonical_sum = pd.read_csv(summary_path)
        canonical_res = pd.read_csv(results_path)

        if (canonical_sum['model'] == 'TabPFN').any():
            print(f'[{pipe}] summary already contains TabPFN rows; skipping.')
        else:
            add_sum = tab_sum[tab_sum['pipeline'] == pipe]
            out_sum = pd.concat([canonical_sum, add_sum], ignore_index=True)
            out_sum.to_csv(summary_path, index=False)
            print(f'[{pipe}] summary: {canonical_sum.shape} + {add_sum.shape} '
                  f'-> {out_sum.shape}')

        if (canonical_res['model'] == 'TabPFN').any():
            print(f'[{pipe}] results already contains TabPFN rows; skipping.')
        else:
            add_res = tab_res_proj[tab_res_proj['pipeline'] == pipe]
            out_res = pd.concat([canonical_res, add_res], ignore_index=True)
            out_res.to_csv(results_path, index=False)
            print(f'[{pipe}] results: {canonical_res.shape} + {add_res.shape} '
                  f'-> {out_res.shape}')


def merge_regime():
    tab_regime = pd.read_csv(RESULTS / 'tabpfn_regime_rmse.csv')
    assert len(tab_regime) == 40, (
        f'tabpfn_regime_rmse expected 40 rows (8 combos x 5 regimes), '
        f'got {len(tab_regime)}.'
    )

    canonical_path = RESULTS / 'regime_allmodels.csv'
    canonical = pd.read_csv(canonical_path)

    # Column-order align; source schema is an exact superset in ordering.
    tab_regime = tab_regime[canonical.columns]

    if (canonical['method_family'] == 'tabpfn').any():
        print('regime_allmodels already contains tabpfn rows; skipping.')
        return

    out = pd.concat([canonical, tab_regime], ignore_index=True)
    out.to_csv(canonical_path, index=False)
    print(f'regime_allmodels: {canonical.shape} + {tab_regime.shape} '
          f'-> {out.shape}')


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('multiseed', 'all'):
        print('=== C3: multiseed merge ===')
        merge_multiseed()
    if what in ('regime', 'all'):
        print('=== C4: regime merge ===')
        merge_regime()
    print('done.')


if __name__ == '__main__':
    main()
