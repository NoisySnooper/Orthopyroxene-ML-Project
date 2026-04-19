#!/usr/bin/env python3
"""Phase G.7 D7: manuscript autofill for the bias-correction section.

Writes manuscripts/opx_2026/text/bias_correction_autofilled.md
with placeholder fields filled from the Phase G.7 D2/D3/D3b outputs:

  * Per-cell ship decision and winning form
  * Overall Delta-RMSE with 95% bootstrap CI (mean across 20 seeds)
  * Max per-regime degradation
  * Form B quantile-thresholds at canonical seed (if winner == B)
  * Edge-sensitivity swing (Form A robustness)
  * Form B CV-reseed parameter stability

No draft manuscript is modified. This script only writes the autofilled
markdown, matching the convention used for regime_results_autofilled.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import LOGS, RESULTS

OUT_MD = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'bias_correction_autofilled.md'
LOG_PATH = LOGS / 'v10_phase_g_bias_correction_manuscript.log'

TRACK_LABEL = {
    'opx_liq':  'opx-liq',
    'opx_only': 'opx-only',
    'cpx_liq':  'cpx-liq',
    'cpx_only': 'cpx-only',
}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _fmt(x, prec=2):
    if x is None:
        return '--'
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(fx):
        return '--'
    return f'{fx:.{prec}f}'


def _load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        ship = _load_csv(RESULTS / 'v10_bias_correction_shipped.csv')
        summ = _load_csv(RESULTS / 'v10_bias_correction_summary.csv')
        pseed = _load_csv(RESULTS / 'v10_bias_correction_per_seed.csv')
        edge = _load_csv(RESULTS / 'v10_bias_correction_edge_sensitivity.csv')
        stab = _load_csv(RESULTS / 'v10_bias_correction_form_b_stability.csv')

        if ship is None or summ is None or pseed is None:
            _log('ERROR: required D2 outputs missing; aborting', fh)
            return 1

        lines = []
        lines.append('# Bias-correction results (auto-filled)')
        lines.append('')
        lines.append(f'*Generated: {time.strftime("%Y-%m-%d")} (Phase G.7).'
                     f' Source tables: `results/v10_bias_correction_'
                     f'{{per_seed,summary,shipped}}.csv`,'
                     f' `results/v10_bias_correction_edge_sensitivity.csv`,'
                     f' `results/v10_bias_correction_form_b_stability.csv`.'
                     f' Figures: `figures/fig30-34_*`.*')
        lines.append('')

        lines.append('## Framing')
        lines.append('')
        lines.append('Phase G.7 quantifies whether a post-hoc bias correction'
                     ' -- fit on citation-grouped out-of-fold residuals of the'
                     ' training split and applied at inference -- reduces'
                     ' test-set error without inflating any of the four'
                     ' pre-registered P regimes. Two forms are tested:')
        lines.append('')
        lines.append('* **Form A**: per-regime ordinary least squares,'
                     ' `y_corr = a_r * y_pred + b_r` for r in'
                     ' {shallow_crustal, deep_crustal_MASH, lithospheric_mantle,'
                     ' deeper_mantle}.')
        lines.append('* **Form B**: quantile-thresholded piecewise,'
                     ' `bias(y) = s_L * (y - alpha_L)` if `y < alpha_L`,'
                     ' `0` in the middle, `s_R * (y - alpha_R)` if `y > alpha_R`,'
                     ' with alphas chosen from a quantile grid under a minimum'
                     ' middle-width constraint and a slope cap `|s| <= 2`.')
        lines.append('')
        lines.append('A correction **ships** for a cell only if, on the held-out'
                     ' test set, overall RMSE improves by more than `SHIP_TOL ='
                     ' 1e-6` and no regime RMSE degrades by more than'
                     ' `SHIP_TOL`. When both Form A and Form B ship, the larger'
                     ' overall `Delta`-RMSE wins; ties go to Form A (simpler).')
        lines.append('')

        # Per-cell summary
        lines.append('## Per-cell ship decisions (canonical seed = 42)')
        lines.append('')
        lines.append('| track | target | model / feat. | winner | '
                     '$\\Delta$RMSE (mean) | max reg. deg. | ships (maj. of 20) |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in ship.sort_values(['target', 'track']).iterrows():
            win = r['winner']
            sub = summ[
                (summ.track == r['track']) & (summ.target == r['target'])
                & (summ.form == win) & (summ.regime == 'ALL')
            ]
            if not sub.empty and win in ('A', 'B'):
                d = float(sub['delta_rmse_mean'].iloc[0])
                sub_nonall = summ[
                    (summ.track == r['track']) & (summ.target == r['target'])
                    & (summ.form == win) & (summ.regime != 'ALL')
                ]
                reg_deg = (sub_nonall['post_rmse_mean'] - sub_nonall['pre_rmse_mean']).max() \
                    if not sub_nonall.empty else np.nan
                ps = pseed[
                    (pseed.track == r['track']) & (pseed.target == r['target'])
                    & (pseed.form == win) & (pseed.regime == 'ALL')
                ]
                n_ship = int((ps['winner'] == win).sum())
                n_tot = len(ps)
                maj = 'yes' if n_ship > n_tot / 2 else 'no'
            else:
                d, reg_deg, n_ship, n_tot, maj = np.nan, np.nan, 0, 0, 'no'
            lines.append(
                f"| {TRACK_LABEL.get(r['track'], r['track'])} | {r['target']} | "
                f"{r['model']} / {r['feature_set']} | {win} | "
                f"{_fmt(d, 3)} | {_fmt(reg_deg, 3)} | {maj} ({n_ship}/{n_tot}) |")
        lines.append('')

        # Form B details (if any cell ships B)
        b_cells = ship[ship.winner == 'B']
        if not b_cells.empty:
            lines.append('## Form B parameters (canonical seed = 42) at shipping cells')
            lines.append('')
            lines.append('| track | target | alpha_L | alpha_R | s_L | s_R |')
            lines.append('|---|---|---|---|---|---|')
            for _, r in b_cells.iterrows():
                try:
                    fb = json.loads(r['form_b_params'])
                except Exception:
                    fb = {}
                lines.append(
                    f"| {TRACK_LABEL.get(r['track'], r['track'])} | {r['target']} "
                    f"| {_fmt(fb.get('alpha_L'), 3)} | {_fmt(fb.get('alpha_R'), 3)} "
                    f"| {_fmt(fb.get('s_L'), 3)} | {_fmt(fb.get('s_R'), 3)} |")
            lines.append('')

        # Edge sensitivity
        if edge is not None:
            lines.append('## Form A bin-edge sensitivity (+/-1 kbar)')
            lines.append('')
            lines.append('| track | target | $\\Delta$RMSE base | max $|\\Delta|$ swing |')
            lines.append('|---|---|---|---|')
            ecell = edge.drop_duplicates(['pipeline', 'track', 'target', 'perturbation'])
            for k, g in ecell.groupby(['pipeline', 'track', 'target']):
                base = g[g.perturbation == 'base']
                if base.empty:
                    continue
                b_delta = float(base['overall_delta'].iloc[0])
                swings = []
                for es in ('inner_m1', 'inner_p1'):
                    ss = g[g.perturbation == es]
                    if not ss.empty:
                        swings.append(abs(float(ss['overall_delta'].iloc[0]) - b_delta))
                swing = max(swings) if swings else np.nan
                lines.append(
                    f"| {TRACK_LABEL.get(k[1], k[1])} | {k[2]} "
                    f"| {_fmt(b_delta, 3)} | {_fmt(swing, 3)} |")
            lines.append('')

        # Form B stability
        if stab is not None and (stab['form_b_ok']
                                 if 'form_b_ok' in stab.columns else True).any():
            sub = stab[stab.form_b_ok] if 'form_b_ok' in stab.columns else stab
            lines.append('## Form B parameter stability across 5 CV reseeds')
            lines.append('')
            lines.append('| track | target | alpha_L mean | alpha_L std | '
                         'alpha_R mean | alpha_R std |')
            lines.append('|---|---|---|---|---|---|')
            for k, g in sub.groupby(['pipeline', 'track', 'target']):
                lines.append(
                    f"| {TRACK_LABEL.get(k[1], k[1])} | {k[2]} "
                    f"| {_fmt(g.alpha_L.mean(), 3)} "
                    f"| {_fmt(g.alpha_L.std(ddof=0), 3)} "
                    f"| {_fmt(g.alpha_R.mean(), 3)} "
                    f"| {_fmt(g.alpha_R.std(ddof=0), 3)} |")
            lines.append('')

        lines.append('## Caveats')
        lines.append('')
        lines.append('* Per-regime pre/post rows (non-ALL) feed Table S9.')
        lines.append('* `ships (maj. of 20)` counts seeds where, at the winning'
                     ' form, both the overall-improve and no-regime-degrade'
                     ' conditions hold on the held-out test split.')
        lines.append('* `GEOROC natural inference` post-correction (cell'
                     ' opx-only) is in `results/v10_natural_opx_post_correction'
                     '_inference.csv`; regime is assigned from the _predicted_'
                     ' P (no ground truth on natural samples), following the'
                     ' convention in Agreda-Lopez 2024.')
        lines.append('')

        OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        _log(f'wrote {OUT_MD}', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
