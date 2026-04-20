#!/usr/bin/env python3
"""Build results/nb03_per_family_winners.json from opx_multiseed_summary.csv.

Phase 4 N1 dependency: nb08_natural_twopx needs this JSON but nb03 Phase
3.5 hasn't been executed in the post-consolidation layout. Rebuild it
from the authoritative 20-seed multiseed summary so nb08 can run.

Schema (matches archive/pre_v10_rebuild_2026_04_16/results/
nb03_per_family_winners.json):

    {
      "forest_family":  { "<track>_<target>": {...}, ... },
      "boosted_family": { "<track>_<target>": {...}, ... },
      "tiebreaker_rule": "...",
      "selection_metadata": {...}
    }

Each spec: model_name, feature_set, filename, rmse_test_mean,
rmse_test_std, r2_test_mean (set to None if unavailable), r2_test_std.

Family membership:
    forest_family  : RF, ERT
    boosted_family : XGB, GB, CatBoost, LightGBM (ML course canonical
                     boosted universe)

Within each family, pick the lowest mean test RMSE. Tie-break by std
(lower preferred) then alphabetical.

Canonical joblib filenames follow the current naming scheme:
    models/canonical/opx/base_{model}_{target}_{track}_{feature_set}.joblib
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / 'results'
MODELS = PROJECT_ROOT / 'models' / 'canonical'

SUMMARY_CSV = RESULTS / 'opx_multiseed_summary.csv'
OUT_JSON = RESULTS / 'nb03_per_family_winners.json'

COMBOS = [
    ('opx_only', 'T_C'),
    ('opx_only', 'P_kbar'),
    ('opx_liq', 'T_C'),
    ('opx_liq', 'P_kbar'),
]
FAMILIES = {
    'forest_family': ['RF', 'ERT'],
    'boosted_family': ['XGB', 'GB', 'CatBoost', 'LightGBM'],
}


def canonical_filename(pipeline: str, model: str, target: str, track: str,
                       feature_set: str) -> str:
    return f'canonical/{pipeline}/base_{model}_{target}_{track}_{feature_set}.joblib'


def _pick_winner(summary_df: pd.DataFrame, track: str, target: str,
                 family_models: list[str]) -> dict:
    sub = summary_df[(summary_df['track'] == track)
                     & (summary_df['target'] == target)
                     & (summary_df['model'].isin(family_models))].copy()
    sub = sub.sort_values(['mean', 'std', 'model']).reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f'no family rows for {track}/{target} in {family_models}')
    win = sub.iloc[0]
    filename = canonical_filename('opx', win['model'], target, track, win['feature_set'])
    # Confirm on-disk presence (filename is MODELS-relative).
    fp = PROJECT_ROOT / 'models' / filename
    if not fp.exists():
        raise FileNotFoundError(f'canonical model missing: {fp}')
    return {
        'model_name': str(win['model']),
        'feature_set': str(win['feature_set']),
        'filename': filename,
        'rmse_test_mean': float(win['mean']),
        'rmse_test_std': float(win['std']),
        'r2_test_mean': None,
        'r2_test_std': None,
    }


def main():
    df = pd.read_csv(SUMMARY_CSV)
    assert 'pipeline' in df.columns
    df = df[df['pipeline'] == 'opx'].copy()

    out = {fam: {} for fam in FAMILIES}
    for fam, members in FAMILIES.items():
        for track, target in COMBOS:
            key = f'{track}_{target}'
            out[fam][key] = _pick_winner(df, track, target, members)

    out['tiebreaker_rule'] = (
        'when means within 1 std, prefer RF over ERT (forest); prefer XGB '
        'over GB/LightGBM/CatBoost by std (boosted). Reconstructed from '
        'opx_multiseed_summary.csv by argmin(mean) then argmin(std) then '
        'alphabetical model name.'
    )
    out['selection_metadata'] = {
        'n_seeds': int(df['count'].iloc[0]),
        'source': 'opx_multiseed_summary.csv',
        'rebuilt_by': 'scripts/data_prep/build_per_family_winners.py',
        'note': (
            'Rebuilt 2026-04-20 because nb03 Phase 3.5 was not present in '
            'the post-consolidation notebook layout. Identical schema to '
            'archive/pre_v10_rebuild_2026_04_16/results/'
            'nb03_per_family_winners.json.'
        ),
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'wrote {OUT_JSON}')
    for fam in FAMILIES:
        print(f'\n{fam}:')
        for k, v in out[fam].items():
            print(f'  {k}: {v["model_name"]}/{v["feature_set"]} '
                  f'mean={v["rmse_test_mean"]:.2f} std={v["rmse_test_std"]:.2f}')


if __name__ == '__main__':
    main()
