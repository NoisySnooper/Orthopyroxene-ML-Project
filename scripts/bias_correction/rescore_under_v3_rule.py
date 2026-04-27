#!/usr/bin/env python3
"""Generate the canonical bias-correction scorecard under the locked
ship rule registered in docs/preregistration/p_regime_preregistration.md.

Reads:
  - results/bias_correction_per_seed.csv

Writes:
  - results/bias_correction_shipped.csv
  - results/preregistered_scorecard_postcorrection.csv

Rule (Section 6 of the prereg doc): a correction form ships iff overall
RMSE improves by more than SHIP_TOL and no well-populated regime
(n >= N_MIN_FOR_VETO) degrades by more than the tolerance envelope:

    veto_tol_r = max(SHIP_TOL, T_ABS, T_REL * pre_r)

Constants below are the registered values; edits to these constants
will break test T21.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.bias_correction import ship_decision, choose_winner  # noqa: E402

# Pre-registered constants (Amendment 2).
N_MIN = 20
T_ABS_T = 10.0       # degC, temperature targets
T_ABS_P = 1.0        # kbar, pressure targets
T_REL = 0.10         # fraction of pre-regime RMSE
CANONICAL_SEED = 42


def _tol_abs_for_target(target: str) -> float:
    if target == 'T_C':
        return T_ABS_T
    if target == 'P_kbar':
        return T_ABS_P
    raise ValueError(f'unknown target {target!r}')


def _per_seed_decision(pseed_rows: pd.DataFrame, n_min: int,
                       tol_abs: float, tol_rel: float) -> dict:
    all_row = pseed_rows[pseed_rows.regime == 'ALL']
    reg_rows = pseed_rows[pseed_rows.regime != 'ALL']
    if all_row.empty or reg_rows.empty:
        return dict(ships=False, overall_delta=np.nan,
                    max_regime_degradation=np.nan,
                    reason='missing ALL/regime row',
                    low_n=json.dumps({}))
    pre_all = float(all_row['pre_rmse'].iloc[0])
    post_all = float(all_row['post_rmse'].iloc[0])
    per_regime_pre = dict(zip(reg_rows.regime, reg_rows.pre_rmse.astype(float)))
    per_regime_post = dict(zip(reg_rows.regime, reg_rows.post_rmse.astype(float)))
    per_regime_n = dict(zip(reg_rows.regime, reg_rows.n.astype(int)))
    d = ship_decision(
        form=str(pseed_rows['form'].iloc[0]),
        pre_overall=pre_all, post_overall=post_all,
        per_regime_pre=per_regime_pre,
        per_regime_post=per_regime_post,
        per_regime_n=per_regime_n,
        n_min_for_veto=n_min,
        degradation_tol_abs=tol_abs,
        degradation_tol_rel=tol_rel,
    )
    return dict(ships=bool(d.ships),
                overall_delta=float(d.overall_delta),
                max_regime_degradation=float(d.max_regime_degradation),
                reason=str(d.reason),
                low_n=json.dumps(d.low_n_degradations))


def rescore_shipped(pseed: pd.DataFrame, shipped_v1: pd.DataFrame,
                    shipped_v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cell in shipped_v1.iterrows():
        pipe = cell['pipeline']
        track = cell['track']
        target = cell['target']
        model = cell['model']
        fs = cell['feature_set']

        # TabPFN: preserve v1 verdict byte-for-byte (Amendment 2 §4).
        if model == 'TabPFN':
            rows.append({
                'pipeline': pipe, 'track': track, 'target': target,
                'model': model, 'feature_set': fs,
                'canonical_seed': cell['canonical_seed'],
                'winner_v1': cell['winner'],
                'winner_v2': cell['winner'],
                'winner_v3': cell['winner'],
                'flip_reason_v3': '',
                'ship_a_v3': cell['ship_a'],
                'ship_b_v3': cell['ship_b'],
                'n_min_for_veto': N_MIN,
                'degradation_tol_abs': _tol_abs_for_target(target),
                'degradation_tol_rel': T_REL,
                'n_seeds_done': cell['n_seeds_done'],
                'note': 'TabPFN has no OOF residuals; v1 verdict preserved',
            })
            continue

        sub = pseed[(pseed.pipeline == pipe) & (pseed.track == track)
                    & (pseed.target == target)
                    & (pseed.seed == int(cell['canonical_seed']))]
        if sub.empty:
            continue
        a = sub[sub.form == 'A']
        b = sub[sub.form == 'B']
        tol_abs = _tol_abs_for_target(target)
        dec_a = _per_seed_decision(a, n_min=N_MIN, tol_abs=tol_abs,
                                   tol_rel=T_REL) if not a.empty else None
        dec_b = _per_seed_decision(b, n_min=N_MIN, tol_abs=tol_abs,
                                   tol_rel=T_REL) if not b.empty else None

        class _D:
            def __init__(self, ships, overall_delta):
                self.ships = ships
                self.overall_delta = overall_delta
        ra = _D(dec_a['ships'], dec_a['overall_delta']) if dec_a else _D(False, 0.0)
        rb = _D(dec_b['ships'], dec_b['overall_delta']) if dec_b else _D(False, 0.0)
        winner_v3 = choose_winner(ra, rb)

        v1_winner = cell['winner']
        sv2 = shipped_v2[(shipped_v2.pipeline == pipe)
                         & (shipped_v2.track == track)
                         & (shipped_v2.target == target)
                         & (shipped_v2.model == model)]
        v2_winner = sv2['winner_v2'].iloc[0] if not sv2.empty else v1_winner

        if winner_v3 == v2_winner:
            flip_reason = ''
        else:
            if winner_v3 in ('A', 'B') and v2_winner in ('none', ''):
                flip_reason = (f'v3_promoted_{winner_v3}: '
                               f'degradation within tolerance '
                               f'max({tol_abs:g}, {T_REL:g}*pre)')
            elif winner_v3 == 'none' and v2_winner in ('A', 'B'):
                flip_reason = f'v3_demoted_from_{v2_winner}'
            else:
                flip_reason = f'v2={v2_winner}->v3={winner_v3}'

        rows.append({
            'pipeline': pipe, 'track': track, 'target': target,
            'model': model, 'feature_set': fs,
            'canonical_seed': cell['canonical_seed'],
            'winner_v1': v1_winner,
            'winner_v2': v2_winner,
            'winner_v3': winner_v3,
            'flip_reason_v3': flip_reason,
            'ship_a_v3': json.dumps(dec_a) if dec_a else '',
            'ship_b_v3': json.dumps(dec_b) if dec_b else '',
            'n_min_for_veto': N_MIN,
            'degradation_tol_abs': tol_abs,
            'degradation_tol_rel': T_REL,
            'n_seeds_done': cell['n_seeds_done'],
            'note': '',
        })
    return pd.DataFrame(rows)


def rescore_scorecard(sc_v2: pd.DataFrame, shipped_v3: pd.DataFrame,
                      pseed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sc_row in sc_v2.iterrows():
        row = sc_row.to_dict()
        pipe = 'opx' if row['track'].startswith('opx') else (
               'cpx' if row['track'].startswith('cpx') else row['track'])
        if pipe not in ('opx', 'cpx'):
            row['winner_v3'] = row.get('winner_v2', row['winner'])
            row['flip_reason_v3'] = ''
            rows.append(row)
            continue

        shipped = shipped_v3[(shipped_v3.pipeline == pipe)
                             & (shipped_v3.track == row['track'])
                             & (shipped_v3.target == row['target'])
                             & (shipped_v3.model != 'TabPFN')]
        if shipped.empty:
            row['winner_v3'] = row.get('winner_v2', row['winner'])
            row['flip_reason_v3'] = ''
            rows.append(row)
            continue
        v2_w = shipped['winner_v2'].iloc[0]
        v3_w = shipped['winner_v3'].iloc[0]

        if v2_w == v3_w:
            row['winner_v3'] = row.get('winner_v2', row['winner'])
            row['flip_reason_v3'] = ''
            rows.append(row)
            continue

        # Under v3, the shipped form is v3_w. Pull the post-RMSE from
        # per-seed at canonical seed for this regime + form.
        canonical = int(shipped['canonical_seed'].iloc[0])
        if v3_w in ('A', 'B'):
            ps = pseed[(pseed.pipeline == pipe)
                       & (pseed.track == row['track'])
                       & (pseed.target == row['target'])
                       & (pseed.form == v3_w)
                       & (pseed.seed == canonical)
                       & (pseed.regime == row['regime'])]
            if not ps.empty:
                row['v10_post_rmse'] = float(ps['post_rmse'].iloc[0])
                row['v10_post_rmse_lo'] = float(ps['post_lo'].iloc[0])
                row['v10_post_rmse_hi'] = float(ps['post_hi'].iloc[0])
                row['correction_form'] = v3_w
        elif v3_w == 'none' and v2_w in ('A', 'B'):
            row['v10_post_rmse'] = row['v10_pre_rmse']
            row['v10_post_rmse_lo'] = row['v10_pre_rmse_lo']
            row['v10_post_rmse_hi'] = row['v10_pre_rmse_hi']
            row['correction_form'] = 'none'

        candidates = {
            'v10_pre': row.get('v10_pre_rmse', np.nan),
            'v10_corrected': row.get('v10_post_rmse', np.nan),
            'external': row.get('best_external_rmse', np.nan),
            'tabpfn': row.get('tabpfn_rmse', np.nan),
            'tabpfn_corrected': row.get('tabpfn_post_rmse', np.nan),
        }
        finite = {k: v for k, v in candidates.items()
                  if isinstance(v, (int, float)) and np.isfinite(v)}
        winner_v3 = min(finite, key=finite.get) if finite else row['winner']
        row['winner_v3'] = winner_v3
        row['flip_reason_v3'] = shipped['flip_reason_v3'].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    pseed = pd.read_csv('results/bias_correction_per_seed.csv')
    shipped_v1 = pd.read_csv('results/bias_correction_shipped.csv')
    shipped_v2 = pd.read_csv('results/bias_correction_shipped_v2.csv')
    sc_v2_path = Path('results/preregistered_scorecard_postcorrection_v2.csv')
    sc_v2 = pd.read_csv(sc_v2_path)

    shipped_v3 = rescore_shipped(pseed, shipped_v1, shipped_v2)
    out_ship = Path('results/bias_correction_shipped.csv')
    shipped_v3.to_csv(out_ship, index=False)
    print(f'wrote {out_ship}: {len(shipped_v3)} rows')

    sc_v3 = rescore_scorecard(sc_v2, shipped_v3, pseed)
    out_sc = Path('results/preregistered_scorecard_postcorrection.csv')
    sc_v3.to_csv(out_sc, index=False)
    print(f'wrote {out_sc}: {len(sc_v3)} rows')

    flips = shipped_v3[shipped_v3['winner_v2'] != shipped_v3['winner_v3']]
    if len(flips):
        print('\n=== v2 -> v3 flips (shipped verdict) ===')
        for _, r in flips.iterrows():
            print(f"  {r['pipeline']}/{r['track']}/{r['target']} "
                  f"({r['model']}/{r['feature_set']}): "
                  f"{r['winner_v2']} -> {r['winner_v3']}  "
                  f"[{r['flip_reason_v3']}]")
    else:
        print('\nNo shipped-verdict flips under v3 (v2 == v3 for all cells).')

    sc_flips = sc_v3[sc_v3['winner_v2'] != sc_v3['winner_v3']]
    if len(sc_flips):
        print('\n=== v2 -> v3 flips (scorecard winner) ===')
        for _, r in sc_flips.iterrows():
            print(f"  {r['track']}/{r['target']}/{r['regime']}: "
                  f"{r['winner_v2']} -> {r['winner_v3']}  "
                  f"[{r.get('flip_reason_v3', '')}]")


if __name__ == '__main__':
    main()
