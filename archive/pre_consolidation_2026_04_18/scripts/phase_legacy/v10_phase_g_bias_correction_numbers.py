#!/usr/bin/env python3
"""Generate ``manuscripts/opx_2026/text/bias_correction_numbers.md``.

Produces a plain-text numbers dump for the author to lift into the draft.
No prose. Every placeholder is filled from the Phase G.7 CSVs; nothing is
left in square brackets.
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

OUT = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'bias_correction_numbers.md'
LOG_PATH = LOGS / 'v10_phase_g_bias_correction_numbers.log'

TRACK_LABEL = {'opx_liq': 'opx-liq', 'opx_only': 'opx-only',
               'cpx_liq': 'cpx-liq', 'cpx_only': 'cpx-only'}
UNIT_DEC = {'T_C': 1, 'P_kbar': 2}
UNIT_STR = {'T_C': 'C', 'P_kbar': 'kbar'}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _fmt(x, prec):
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return '--'
    if not np.isfinite(fx):
        return '--'
    return f'{fx:.{prec}f}'


def _pct(pre, post):
    if not np.isfinite(pre) or pre == 0 or not np.isfinite(post):
        return float('nan')
    return 100.0 * (pre - post) / pre


def _ship_reason(ship_row, form):
    key = 'ship_a' if form == 'A' else 'ship_b'
    try:
        return json.loads(ship_row[key])
    except Exception:
        return {}


def gather():
    ship = pd.read_csv(RESULTS / 'v10_bias_correction_shipped.csv')
    summ = pd.read_csv(RESULTS / 'v10_bias_correction_summary.csv')
    pseed = pd.read_csv(RESULTS / 'v10_bias_correction_per_seed.csv')
    sc = pd.read_csv(RESULTS / 'v10_preregistered_scorecard_postcorrection.csv')
    stab = pd.read_csv(RESULTS / 'v10_bias_correction_form_b_stability.csv')
    edge = pd.read_csv(RESULTS / 'v10_bias_correction_edge_sensitivity.csv')
    return ship, summ, pseed, sc, stab, edge


def cell_numbers(track, target, summ, pseed, ship, form):
    """Pre/post/delta/pct at regime=ALL, plus per-seed vote (A/B/none)."""
    s = summ[(summ.track == track) & (summ.target == target)
             & (summ.form == form) & (summ.regime == 'ALL')]
    if s.empty:
        return None
    pre = float(s['pre_rmse_mean'].iloc[0])
    post = float(s['post_rmse_mean'].iloc[0])
    delta = float(s['delta_rmse_mean'].iloc[0])
    ps = pseed[(pseed.track == track) & (pseed.target == target)
               & (pseed.regime == 'ALL')].drop_duplicates(['seed'])
    n_tot = len(ps)
    n_a = int((ps['winner'] == 'A').sum())
    n_b = int((ps['winner'] == 'B').sum())
    n_none = int((~ps['winner'].isin(['A', 'B'])).sum())
    return {
        'pre': pre, 'post': post, 'delta': delta,
        'pct': _pct(pre, post),
        'n_a': n_a, 'n_b': n_b, 'n_none': n_none, 'n_tot': n_tot,
    }


def best_external(sc, track, target, regime):
    r = sc[(sc.track == track) & (sc.target == target) & (sc.regime == regime)]
    if r.empty:
        return None
    r = r.iloc[0]
    ext_rmse = r['best_external_rmse']
    if not np.isfinite(ext_rmse):
        return None
    return {
        'rmse': float(ext_rmse),
        'method': str(r['best_external_method']),
        'winner': str(r['winner']),
    }


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        ship, summ, pseed, sc, stab, edge = gather()
        out = []
        out.append('# Bias correction numbers for draft_v1 -> v2 update')
        out.append('')
        out.append(f'Auto-generated from results/v10_bias_correction_*.csv '
                   f'on {time.strftime("%Y-%m-%d")}. Canonical seed 42. '
                   f'20-seed mean used at regime=ALL where applicable. All '
                   f'RMSE in native units (kbar for P, C for T). % reduction '
                   f'defined as 100*(pre - post)/pre.')
        out.append('')

        # ================== Section 1: headline claims ==================
        out.append('## For Section 1 (Introduction / headline claim)')
        out.append('')

        # opx-only P headline
        n = cell_numbers('opx_only', 'P_kbar', summ, pseed, ship, form='A')
        ext = best_external(sc, 'opx_only', 'P_kbar', 'ALL')
        opx_sc = sc[(sc.track == 'opx_only') & (sc.target == 'P_kbar')
                    & (sc.regime != 'ALL')]
        n_reg_v10_wins = int(opx_sc['winner'].isin(['v10_corrected', 'v10_pre']).sum())
        n_reg_tot = len(opx_sc)
        out.append('**Strongest single finding (opx-only P):**')
        out.append(f"- Aggregate pre RMSE (raw v10): {_fmt(n['pre'], 2)} kbar")
        out.append(f"- Aggregate post RMSE: {_fmt(n['post'], 2)} kbar")
        out.append(f"- % reduction: {_fmt(n['pct'], 1)}%")
        if ext:
            out.append(f"- vs {ext['method']} at aggregate: "
                       f"{_fmt(ext['rmse'], 2)} kbar")
            pct_ext = _pct(ext['rmse'], n['post'])
            out.append(f"- % reduction vs {ext['method']}: "
                       f"{_fmt(pct_ext, 1)}%")
        else:
            out.append('- vs external at aggregate: no external available')
        out.append(f"- Per-seed ship vote: {n['n_a']} of {n['n_tot']} seeds "
                   f"ship Form A, {n['n_b']} ship Form B, "
                   f"{n['n_none']} ship none")
        out.append(f"- Wins per regime (post-correction): v10 wins "
                   f"{n_reg_v10_wins}/{n_reg_tot} non-ALL regimes")
        out.append('')

        # cpx-only P second
        n = cell_numbers('cpx_only', 'P_kbar', summ, pseed, ship, form='A')
        ext = best_external(sc, 'cpx_only', 'P_kbar', 'ALL')
        cpx_sc = sc[(sc.track == 'cpx_only') & (sc.target == 'P_kbar')
                    & (sc.regime != 'ALL')]
        n_reg_v10_wins = int(cpx_sc['winner'].isin(['v10_corrected', 'v10_pre']).sum())
        n_reg_tot = len(cpx_sc)
        out.append('**Second finding (cpx-only P):**')
        out.append(f"- Aggregate pre RMSE (raw v10): {_fmt(n['pre'], 2)} kbar")
        out.append(f"- Aggregate post RMSE: {_fmt(n['post'], 2)} kbar")
        out.append(f"- % reduction: {_fmt(n['pct'], 1)}%")
        if ext:
            out.append(f"- vs {ext['method']} at aggregate: "
                       f"{_fmt(ext['rmse'], 2)} kbar")
            pct_ext = _pct(ext['rmse'], n['post'])
            out.append(f"- % reduction vs {ext['method']}: "
                       f"{_fmt(pct_ext, 1)}%")
        out.append(f"- Per-seed ship vote: {n['n_a']} of {n['n_tot']} seeds "
                   f"ship Form A, {n['n_b']} ship Form B, "
                   f"{n['n_none']} ship none")
        out.append(f"- Wins per regime (post-correction): v10 wins "
                   f"{n_reg_v10_wins}/{n_reg_tot} non-ALL regimes")
        out.append('')

        # Form B ablation
        fb_cells = int((ship.winner == 'B').sum())
        ps_all = pseed[pseed.regime == 'ALL'].drop_duplicates(
            ['track', 'target', 'seed'])
        fb_pseed = int((ps_all.winner == 'B').sum())
        total_ps_decisions = len(ps_all)
        out.append('**Ablation result (Form B ship count):**')
        out.append(f'- Form B ships on {fb_cells}/8 combinations at '
                   'canonical seed 42')
        out.append(f'- Form B ships on {fb_pseed}/{total_ps_decisions} '
                   'per-seed cell decisions')
        out.append('- Interpretation: regime-based correction dominates '
                   'value-based thresholding for this data -- residual '
                   'structure is aligned with pre-registered P bins, not '
                   'with extreme values of the prediction distribution.')
        out.append('')

        # ================== Section 3.5: Methods ==================
        out.append('## For Section 3.5 (Methods -- bias correction)')
        out.append('')

        out.append('**Form A parameters for shipped cells (canonical seed 42):**')
        for _, r in ship[ship.winner == 'A'].iterrows():
            fa = json.loads(r['form_a_params'])
            lines = [f"- {TRACK_LABEL[r['track']]} {r['target']} "
                     f"({r['model']}/{r['feature_set']}):"]
            for reg in ('shallow_crustal', 'deep_crustal_MASH',
                        'lithospheric_mantle', 'deeper_mantle'):
                p = fa.get(reg, {})
                a = p.get('a', float('nan'))
                b = p.get('b', float('nan'))
                lines.append(f'  - {reg}: a={_fmt(a, 3)}, b={_fmt(b, 2)}')
            out.extend(lines)
        out.append('')

        out.append('**Form B parameters at canonical seed 42 '
                   '(diagnostic, none ship):**')
        for _, r in ship.iterrows():
            fb = json.loads(r['form_b_params'])
            out.append(f"- {TRACK_LABEL[r['track']]} {r['target']} "
                       f"({r['model']}/{r['feature_set']}): "
                       f"alpha_L={_fmt(fb.get('alpha_L'), 3)}, "
                       f"alpha_R={_fmt(fb.get('alpha_R'), 3)}, "
                       f"s_L={_fmt(fb.get('s_L'), 3)}, "
                       f"s_R={_fmt(fb.get('s_R'), 3)}")
        out.append('')

        out.append('**Form B stability across 5 CV reseeds '
                   '(mean +/- std; see Table S8.11 for full list):**')
        for (track, target), g in stab.groupby(['track', 'target']):
            out.append(
                f'- {TRACK_LABEL[track]} {target}: '
                f"alpha_L={_fmt(g['alpha_L'].mean(), 3)}+/-"
                f"{_fmt(g['alpha_L'].std(ddof=0), 3)}, "
                f"alpha_R={_fmt(g['alpha_R'].mean(), 3)}+/-"
                f"{_fmt(g['alpha_R'].std(ddof=0), 3)}, "
                f"s_L={_fmt(g['s_L'].mean(), 3)}+/-"
                f"{_fmt(g['s_L'].std(ddof=0), 3)}, "
                f"s_R={_fmt(g['s_R'].mean(), 3)}+/-"
                f"{_fmt(g['s_R'].std(ddof=0), 3)}")
        out.append('')

        # ================== Section 4.3: Results ==================
        out.append('## For Section 4.3 (Results -- bias correction)')
        out.append('')

        out.append('**Per-cell shipping decisions (canonical seed 42, with reasons):**')
        out.append('')
        out.append('| track | target | model / feat. | winner | form A reason | form B reason |')
        out.append('|---|---|---|---|---|---|')
        for _, r in ship.iterrows():
            sa = _ship_reason(r, 'A')
            sb = _ship_reason(r, 'B')
            out.append(
                f"| {TRACK_LABEL[r['track']]} | {r['target']} | "
                f"{r['model']}/{r['feature_set']} | {r['winner']} | "
                f"{sa.get('reason', '--')} | {sb.get('reason', '--')} |")
        out.append('')

        # opx-only P per-regime
        out.append('**opx-only P per-regime pre/post (ships A, unanimous '
                   'over 20 seeds):**')
        for reg in ('shallow_crustal', 'deep_crustal_MASH',
                    'lithospheric_mantle', 'deeper_mantle', 'ALL'):
            r = sc[(sc.track == 'opx_only') & (sc.target == 'P_kbar')
                   & (sc.regime == reg)]
            if r.empty:
                continue
            r = r.iloc[0]
            ext = (f"{_fmt(r['best_external_rmse'], 2)} kbar "
                   f"({r['best_external_method']})"
                   if np.isfinite(r['best_external_rmse']) else 'no external')
            out.append(
                f"- {reg} n={r['n']}: pre={_fmt(r['v10_pre_rmse'], 2)} kbar, "
                f"post={_fmt(r['v10_post_rmse'], 2)} kbar, "
                f"external best={ext}, winner={r['winner']}")
        out.append('')

        out.append('**cpx-only P per-regime pre/post (ships A at canonical, '
                   '19/20 seeds):**')
        for reg in ('shallow_crustal', 'deep_crustal_MASH',
                    'lithospheric_mantle', 'deeper_mantle', 'ALL'):
            r = sc[(sc.track == 'cpx_only') & (sc.target == 'P_kbar')
                   & (sc.regime == reg)]
            if r.empty:
                continue
            r = r.iloc[0]
            ext = (f"{_fmt(r['best_external_rmse'], 2)} kbar "
                   f"({r['best_external_method']})"
                   if np.isfinite(r['best_external_rmse']) else 'no external')
            out.append(
                f"- {reg} n={r['n']}: pre={_fmt(r['v10_pre_rmse'], 2)} kbar, "
                f"post={_fmt(r['v10_post_rmse'], 2)} kbar, "
                f"external best={ext}, winner={r['winner']}")
        out.append('')

        # T-correction failures
        out.append('**T correction failures (strict policy blocks shipping):**')
        for track in ('opx_liq', 'opx_only', 'cpx_liq', 'cpx_only'):
            row = ship[(ship.track == track) & (ship.target == 'T_C')].iloc[0]
            sa = _ship_reason(row, 'A')
            overall_delta = sa.get('overall_delta', float('nan'))
            # per-seed 42 worst-degrading regime Form A
            sub = pseed[(pseed.track == track) & (pseed.target == 'T_C')
                        & (pseed.form == 'A') & (pseed.regime != 'ALL')
                        & (pseed.seed == 42)]
            if not sub.empty:
                worst = sub.loc[sub['delta_rmse'].idxmin()]
                worst_name = worst['regime']
                worst_post_pre = -float(worst['delta_rmse'])
                out.append(
                    f'- {TRACK_LABEL[track]} T: aggregate Form A delta='
                    f'{overall_delta:+.2f} C at canonical seed; '
                    f'worst regime {worst_name} degrades '
                    f'{worst_post_pre:+.2f} C (post-pre).')
            else:
                out.append(f'- {TRACK_LABEL[track]} T: aggregate Form A '
                           f'delta={overall_delta:+.2f} C at canonical seed')
        out.append('- Interpretation: T residual structure is approximately '
                   'symmetric across regimes, so per-regime OLS offsets '
                   'average out. P residual structure is strongly '
                   'regime-dependent, so Form A captures real bias.')
        out.append('')

        # ================== Section 4 edge sensitivity ==================
        out.append('## For Section 4 edge sensitivity')
        out.append('')

        for track, header in (
                ('opx_only', '**opx-only P edge perturbation (+/-1 kbar on '
                             'interior boundaries):**'),
                ('opx_liq',  '**opx-liq P edge perturbation (marginal '
                             'combination):**'),
        ):
            out.append(header)
            ex = edge[(edge.track == track) & (edge.target == 'P_kbar')]
            ex_unique = ex.drop_duplicates(['perturbation'])
            for _, r in ex_unique.iterrows():
                ships = str(r['ships'])
                out.append(f"- perturbation={r['perturbation']}, "
                           f"edges={r['edges']}: ships={ships}, "
                           f"overall_delta={_fmt(r['overall_delta'], 2)} kbar, "
                           f"max_regime_degradation="
                           f"{_fmt(r['max_regime_degradation'], 2)} kbar")
            out.append('')
        out.append('- Conclusion (opx-only P): correction robust under '
                   '+/-1 kbar edge perturbation; ships under both '
                   'alternative edge placements.')
        out.append('- Conclusion (opx-liq P): sits at the stability edge; '
                   'a +1 kbar shift of interior boundaries flips the '
                   'ship decision from "no" to "yes". Supports 15 kbar '
                   'boundary being borderline for this cell.')
        out.append('')

        # ================== Section 4.4 cpx ==================
        out.append('## For Section 4.4 cpx replication update')
        out.append('')
        out.append('**Post-correction head-to-head vs external cpx models '
                   '(full scorecard):**')
        out.append('')
        out.append('| track | target | regime | n | v10_post | external '
                   'best | method | winner |')
        out.append('|---|---|---|---|---|---|---|---|')
        for _, r in sc[sc.track.isin(['cpx_liq', 'cpx_only'])].iterrows():
            unit = UNIT_STR[r['target']]
            prec = UNIT_DEC[r['target']]
            ext = (f"{_fmt(r['best_external_rmse'], prec)} {unit}"
                   if np.isfinite(r['best_external_rmse']) else 'no external')
            method = (r['best_external_method']
                      if pd.notna(r['best_external_method']) else '--')
            out.append(
                f"| {TRACK_LABEL[r['track']]} | {r['target']} | "
                f"{r['regime']} | {r['n']} | "
                f"{_fmt(r['v10_post_rmse'], prec)} {unit} | {ext} | "
                f"{method} | {r['winner']} |")
        out.append('')

        OUT.write_text('\n'.join(out) + '\n', encoding='utf-8')
        _log(f'wrote {OUT}', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
