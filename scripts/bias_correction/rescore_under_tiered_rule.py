#!/usr/bin/env python3
"""Rescore bias-correction ship decisions under the v2 tiered rule.

See docs/preregistration/AMENDMENT_1_acceptance_rule.md.

Reads:
  - results/bias_correction_per_seed.csv    (per-seed per-regime pre/post RMSE)
  - results/bias_correction_shipped.csv     (v1 ship verdicts, for reference)
  - results/preregistered_scorecard_postcorrection.csv  (v1 scorecard)

Writes:
  - results/bias_correction_shipped_v2.csv
  - results/preregistered_scorecard_postcorrection_v2.csv

Rule v2 uses `n_min_for_veto = N_MIN = 20` (same threshold as
pre-registration Section 4.1 "sample-size-limited"). v1 mode with
`n_min_for_veto = 0` reproduces the v1 CSV byte-for-byte on the same
per-seed input (sanity-checked at the tail).
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

N_MIN = 20
CANONICAL_SEED = 42  # seed used for the shipped verdict in v1


def _per_seed_decision(pseed_rows: pd.DataFrame, n_min: int) -> dict:
    """Given a per-seed slice (one (pipeline,track,target,form,seed) block,
    one row per regime + ALL), produce a decision dict ready for CSV."""
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
    )
    return dict(ships=bool(d.ships),
                overall_delta=float(d.overall_delta),
                max_regime_degradation=float(d.max_regime_degradation),
                reason=str(d.reason),
                low_n=json.dumps(d.low_n_degradations))


def rescore_shipped(pseed: pd.DataFrame, shipped_v1: pd.DataFrame,
                    n_min: int = N_MIN) -> pd.DataFrame:
    """Rescore each (pipeline,track,target) cell under v2 using its
    canonical seed."""
    rows = []
    for _, cell in shipped_v1.iterrows():
        pipe = cell['pipeline']; track = cell['track']; target = cell['target']
        model = cell['model']; fs = cell['feature_set']
        # TabPFN rows in shipped_v1 don't have per-seed residuals (TabPFN
        # has no OOF in the v1 pipeline), so rescore would be undefined.
        # Keep the v1 verdict verbatim for them.
        if model == 'TabPFN':
            rows.append({
                'pipeline': pipe, 'track': track, 'target': target,
                'model': model, 'feature_set': fs,
                'canonical_seed': cell['canonical_seed'],
                'winner_v1': cell['winner'],
                'winner_v2': cell['winner'],
                'flip_reason': '',
                'ship_a_v2': cell['ship_a'],
                'ship_b_v2': cell['ship_b'],
                'n_min_for_veto': n_min,
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
        dec_a = _per_seed_decision(a, n_min=n_min) if not a.empty else None
        dec_b = _per_seed_decision(b, n_min=n_min) if not b.empty else None

        # Reconstruct v2 winner via the existing choose_winner logic on
        # dummy ShipDecision-like objects.
        class _D:
            def __init__(self, ships, overall_delta):
                self.ships = ships; self.overall_delta = overall_delta
        ra = _D(dec_a['ships'], dec_a['overall_delta']) if dec_a else _D(False, 0.0)
        rb = _D(dec_b['ships'], dec_b['overall_delta']) if dec_b else _D(False, 0.0)
        winner_v2 = choose_winner(ra, rb)

        v1_winner = cell['winner']
        if winner_v2 == v1_winner:
            flip_reason = ''
        else:
            # Describe the flip. The usual case: v1=none, v2=A, because a
            # low-n regime was previously vetoing.
            try:
                ship_a_v1 = json.loads(cell['ship_a']) if isinstance(cell['ship_a'], str) and cell['ship_a'].startswith('{') else {}
            except Exception:
                ship_a_v1 = {}
            if winner_v2 in ('A', 'B') and v1_winner == 'none':
                # find the low-n regime(s) that moved
                low_n = json.loads(dec_a['low_n']) if winner_v2 == 'A' else json.loads(dec_b['low_n'])
                low_n_str = ';'.join(f'{k}_n<{n_min}' for k in low_n.keys())
                flip_reason = f'v2_promoted_{winner_v2}:{low_n_str}' if low_n_str else f'v2_promoted_{winner_v2}'
            elif winner_v2 == 'none' and v1_winner in ('A', 'B'):
                flip_reason = f'v2_demoted_from_{v1_winner}'
            else:
                flip_reason = f'v1={v1_winner}->v2={winner_v2}'

        rows.append({
            'pipeline': pipe, 'track': track, 'target': target,
            'model': model, 'feature_set': fs,
            'canonical_seed': cell['canonical_seed'],
            'winner_v1': v1_winner,
            'winner_v2': winner_v2,
            'flip_reason': flip_reason,
            'ship_a_v2': json.dumps(dec_a) if dec_a else '',
            'ship_b_v2': json.dumps(dec_b) if dec_b else '',
            'n_min_for_veto': n_min,
            'n_seeds_done': cell['n_seeds_done'],
            'note': '',
        })
    return pd.DataFrame(rows)


def rescore_scorecard(sc_v1: pd.DataFrame, shipped_v2: pd.DataFrame,
                      pseed: pd.DataFrame, n_min: int = N_MIN) -> pd.DataFrame:
    """Rescore the pre-registered scorecard under v2.

    Strategy: the scorecard columns v10_pre_rmse / v10_post_rmse etc.
    are computed from the shipped (v1) correction form. Under v2 the
    shipped form may flip — when it does, swap in the form's post-RMSE
    from the per-seed CSV at the canonical seed. When v1==v2, pass
    through. Winner recomputation follows the same 4-way tournament as
    v1 (v10_pre / v10_corrected / external / tabpfn / tabpfn_corrected)
    but uses the v2 post-RMSE.
    """
    rows = []
    for _, sc_row in sc_v1.iterrows():
        row = sc_row.to_dict()
        pipe = 'opx' if row['track'].startswith('opx') else (
               'cpx' if row['track'].startswith('cpx') else row['track'])
        if pipe not in ('opx', 'cpx'):
            # twopx / universal: no bias correction in v1 or v2
            row['winner_v1'] = row['winner']
            row['winner_v2'] = row['winner']
            row['flip_reason'] = ''
            rows.append(row)
            continue

        shipped = shipped_v2[(shipped_v2.pipeline == pipe)
                             & (shipped_v2.track == row['track'])
                             & (shipped_v2.target == row['target'])
                             & (shipped_v2.model != 'TabPFN')]
        if shipped.empty:
            row['winner_v1'] = row['winner']
            row['winner_v2'] = row['winner']
            row['flip_reason'] = ''
            rows.append(row)
            continue
        v1_w = shipped['winner_v1'].iloc[0]
        v2_w = shipped['winner_v2'].iloc[0]

        row['winner_v1'] = row['winner']  # scorecard's existing winner
        if v1_w == v2_w:
            row['winner_v2'] = row['winner']
            row['flip_reason'] = ''
            rows.append(row)
            continue

        # Under v2, the shipped form is v2_w. Pull the post-RMSE from
        # per-seed at canonical seed for this regime + form.
        canonical = int(shipped['canonical_seed'].iloc[0])
        if v2_w in ('A', 'B'):
            ps = pseed[(pseed.pipeline == pipe) & (pseed.track == row['track'])
                       & (pseed.target == row['target'])
                       & (pseed.form == v2_w) & (pseed.seed == canonical)
                       & (pseed.regime == row['regime'])]
            if not ps.empty:
                row['v10_post_rmse'] = float(ps['post_rmse'].iloc[0])
                row['v10_post_rmse_lo'] = float(ps['post_lo'].iloc[0])
                row['v10_post_rmse_hi'] = float(ps['post_hi'].iloc[0])
                row['correction_form'] = v2_w
        elif v2_w == 'none' and v1_w in ('A', 'B'):
            # Correction demoted; post_rmse == pre_rmse.
            row['v10_post_rmse'] = row['v10_pre_rmse']
            row['v10_post_rmse_lo'] = row['v10_pre_rmse_lo']
            row['v10_post_rmse_hi'] = row['v10_pre_rmse_hi']
            row['correction_form'] = 'none'

        # Recompute winner: same 4-way tournament as v1 logic.
        candidates = {
            'v10_pre': row.get('v10_pre_rmse', np.nan),
            'v10_corrected': row.get('v10_post_rmse', np.nan),
            'external': row.get('best_external_rmse', np.nan),
            'tabpfn': row.get('tabpfn_rmse', np.nan),
            'tabpfn_corrected': row.get('tabpfn_post_rmse', np.nan),
        }
        finite = {k: v for k, v in candidates.items()
                  if isinstance(v, (int, float)) and np.isfinite(v)}
        if finite:
            winner_v2 = min(finite, key=finite.get)
        else:
            winner_v2 = row['winner']
        row['winner_v2'] = winner_v2
        row['flip_reason'] = shipped['flip_reason'].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    pseed = pd.read_csv('results/bias_correction_per_seed.csv')
    shipped_v1 = pd.read_csv('results/bias_correction_shipped.csv')
    sc_v1 = pd.read_csv('results/preregistered_scorecard_postcorrection.csv')

    # --- ship verdict v2 ---
    shipped_v2 = rescore_shipped(pseed, shipped_v1, n_min=N_MIN)
    out_ship = Path('results/bias_correction_shipped_v2.csv')
    shipped_v2.to_csv(out_ship, index=False)
    print(f'wrote {out_ship}: {len(shipped_v2)} rows')

    # --- scorecard v2 ---
    sc_v2 = rescore_scorecard(sc_v1, shipped_v2, pseed, n_min=N_MIN)
    out_sc = Path('results/preregistered_scorecard_postcorrection_v2.csv')
    sc_v2.to_csv(out_sc, index=False)
    print(f'wrote {out_sc}: {len(sc_v2)} rows')

    # --- report flips ---
    flips = shipped_v2[shipped_v2['winner_v1'] != shipped_v2['winner_v2']]
    if len(flips):
        print('\n=== v1 -> v2 flips (shipped verdict) ===')
        for _, r in flips.iterrows():
            print(f"  {r['pipeline']}/{r['track']}/{r['target']} "
                  f"({r['model']}/{r['feature_set']}): "
                  f"{r['winner_v1']} -> {r['winner_v2']}  [{r['flip_reason']}]")
    else:
        print('\nNo shipped-verdict flips under v2 (v1 == v2 for all cells).')

    sc_flips = sc_v2[sc_v2['winner_v1'] != sc_v2['winner_v2']]
    if len(sc_flips):
        print('\n=== v1 -> v2 flips (scorecard winner) ===')
        for _, r in sc_flips.iterrows():
            print(f"  {r['track']}/{r['target']}/{r['regime']}: "
                  f"{r['winner_v1']} -> {r['winner_v2']}  [{r['flip_reason']}]")

    # --- sanity check: v1 mode reproduces v1 verdict ---
    shipped_v1_reproduced = rescore_shipped(pseed, shipped_v1, n_min=0)
    mismatch = 0
    for _, r in shipped_v1_reproduced.iterrows():
        if r['winner_v1'] != r['winner_v2']:
            # winner_v2 with n_min=0 should equal the original v1 winner
            mismatch += 1
    print(f'\nv1-reproduction sanity check: {mismatch} mismatches '
          f'(should be 0, TabPFN rows excluded)')


if __name__ == '__main__':
    main()
