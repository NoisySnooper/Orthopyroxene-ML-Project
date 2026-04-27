#!/usr/bin/env python3
"""Phase 2 P2.8: Gasparik (1987) agreement analysis.

Gasparik (1987) shows Al solubility in orthopyroxene coexisting with
garnet is pressure-dependent: high-P opx contains more Al_VI (octahedral
Al) and more MgTs (Mg-Tschermak) component, because the Al-Si exchange
on the tetrahedral site is entropy-unfavored at high P. A functional
opx barometer must therefore use Al-bearing features heavily.

This script probes the opx-only P_kbar winner (RF/pwlr) and the
opx-liq P_kbar winner (MLP/raw) for the fraction of the top-10
SHAP features that are Al-bearing. High Al-fraction = Gasparik-
consistent.

Output: results/gasparik_agreement.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

SHAP_CSV = PROJECT_ROOT / 'results' / 'shap_importance_winners.csv'
OUT_CSV = PROJECT_ROOT / 'results' / 'gasparik_agreement.csv'


def is_al_bearing(feature: str) -> bool:
    """Return True if the feature name references Al, MgTs (Mg-Tschermak
    = Mg-Al octahedral component), or CaTs (Ca-Tschermak)."""
    f = feature.lower()
    tokens = ('al2o3', 'al_vi', 'al_iv', 'al_cpx', 'mgts', 'cats',
              '_al2o3_', '_al2o3', 'al2o3_')
    return any(tok in f for tok in tokens)


def probe(df: pd.DataFrame, track: str, target: str, top_n: int = 10):
    sub = df[(df.track == track) & (df.target == target)].copy()
    sub = sub.sort_values('rank').head(top_n)
    al_mask = sub['feature'].apply(is_al_bearing)
    n_al = int(al_mask.sum())
    al_feats = sub.loc[al_mask, 'feature'].tolist()
    all_feats = sub['feature'].tolist()
    family = sub['model'].iloc[0]
    fs = sub['feature_set'].iloc[0]
    sum_al = float(sub.loc[al_mask, 'mean_abs_shap'].sum())
    sum_all = float(sub['mean_abs_shap'].sum())
    frac_importance = sum_al / sum_all if sum_all > 0 else float('nan')
    return {
        'track':            track,
        'target':           target,
        'winner_family':    family,
        'winner_feature_set': fs,
        'top_n':            top_n,
        'n_al_bearing':     n_al,
        'al_bearing_features': ';'.join(al_feats),
        'top_n_features':   ';'.join(all_feats),
        'al_importance_share': frac_importance,
        'pass_gasparik':    bool(n_al >= 3),
    }


def interpret(row):
    if row['pass_gasparik']:
        return (
            f'{row["n_al_bearing"]}/10 of the top-ranked SHAP features of '
            f'the {row["track"]}/{row["target"]} winner '
            f'({row["winner_family"]}/{row["winner_feature_set"]}) are '
            f'Al-bearing, carrying {row["al_importance_share"]*100:.0f}% '
            f'of the aggregated top-10 importance; the ML agrees with '
            f'Gasparik (1987) that opx Al-solubility is a functional '
            f'pressure proxy.'
        )
    return (
        f'Only {row["n_al_bearing"]}/10 of the top SHAP features of the '
        f'{row["track"]}/{row["target"]} winner are Al-bearing; the ML '
        f'extracts pressure information from features beyond Gasparik\'s '
        f'Al-solubility mechanism.'
    )


def main():
    shap = pd.read_csv(SHAP_CSV)
    rows = []
    for track, target in [('opx_only', 'P_kbar'), ('opx_liq', 'P_kbar')]:
        r = probe(shap, track, target, top_n=10)
        r['interpretation'] = interpret(r)
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')
    for _, r in out.iterrows():
        print(f'\n{r["track"]}/{r["target"]} winner '
              f'{r["winner_family"]}/{r["winner_feature_set"]}')
        print(f'  Al-bearing in top-10: {r["n_al_bearing"]}/10 '
              f'(share={r["al_importance_share"]*100:.1f}%)')
        print(f'  features: {r["al_bearing_features"]}')
        print(f'  {r["interpretation"]}')


if __name__ == '__main__':
    main()
