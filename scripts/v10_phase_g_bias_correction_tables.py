#!/usr/bin/env python3
"""Phase G.7 D5: Tables 4, S9, S10 for the bias-correction section.

Table 4 (main text): shipped corrections summary. One row per (track,
target) cell, columns = model / feature_set / form / overall_delta_rmse
/ max_regime_degradation / ships, so a reader can see at a glance
which cells ship a correction and which do not.

Table S9 (supplementary): per-seed detailed results from
v10_bias_correction_per_seed.csv. Carries pre_rmse / post_rmse /
delta_rmse for each cell x seed x form, for full transparency on how
often each form wins.

Table S10 (supplementary): stability / sensitivity. Combines
  * edge_sensitivity  (re-fit Form A at 4 P regime bin-edge sets)
  * form_b_stability  (Form B parameters across 5 CV reseeds)
into one compact table per cell.

Outputs (both .csv and .tex):
    manuscripts/opx_2026/tables/T4_bias_correction_shipped.{csv,tex}
    manuscripts/opx_2026/tables/S9_bias_correction_per_seed.{csv,tex}
    manuscripts/opx_2026/tables/S10_bias_correction_stability.{csv,tex}

The .tex files use the `booktabs` style (\\toprule, \\midrule,
\\bottomrule) with explicit numeric formatting. Main-text T4 uses
`\\begin{tabular}` with `tabcolsep`; S9/S10 use `longtable` so they
can span pages.
"""
from __future__ import annotations

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

TABLES_DIR = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'tables'
LOG_PATH = LOGS / 'v10_phase_g_bias_correction_tables.log'

TRACK_LABEL = {
    'opx_liq':  'opx-liq',
    'opx_only': 'opx-only',
    'cpx_liq':  'cpx-liq',
    'cpx_only': 'cpx-only',
}
TARGET_LABEL = {'T_C': 'T (C)', 'P_kbar': 'P (kbar)'}


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _escape_latex(s):
    return (str(s)
            .replace('_', r'\_')
            .replace('&', r'\&')
            .replace('%', r'\%'))


def _fmt_num(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f'{x:.{prec}f}'


def build_table_4(shipped_csv: Path, summary_csv: Path,
                  per_seed_csv: Path) -> pd.DataFrame:
    """One row per cell, giving the shipped form (or none), its
    overall delta-RMSE, the max regime degradation, and the majority
    ship count. Values are derived from the summary + per_seed CSVs;
    the shipped CSV is only used to identify the canonical winner."""
    ship = pd.read_csv(shipped_csv)
    summ = pd.read_csv(summary_csv)
    pseed = pd.read_csv(per_seed_csv)

    rows = []
    keys = ['pipeline', 'track', 'target', 'model', 'feature_set']
    cell_keys = summ[keys].drop_duplicates().to_dict('records')
    ship_map = {(r['track'], r['target']): r for _, r in ship.iterrows()}
    n_total_seeds = pseed[pseed.regime == 'ALL'].groupby(
        ['track', 'target', 'form']).size().to_dict()

    for k in cell_keys:
        tk = (k['track'], k['target'])
        s = ship_map.get(tk)
        winner = s['winner'] if s is not None else 'none'

        if winner in ('A', 'B'):
            sub = summ[
                (summ.pipeline == k['pipeline']) & (summ.track == k['track'])
                & (summ.target == k['target']) & (summ.form == winner)
            ]
            all_row = sub[sub.regime == 'ALL']
            delta_mean = float(all_row['delta_rmse_mean'].iloc[0]) \
                if not all_row.empty else np.nan
            non_all = sub[sub.regime != 'ALL']
            if not non_all.empty:
                reg_deg = (non_all['post_rmse_mean'] - non_all['pre_rmse_mean']).max()
            else:
                reg_deg = np.nan

            # Per-seed majority ship at winning form.
            ps = pseed[
                (pseed.track == k['track']) & (pseed.target == k['target'])
                & (pseed.form == winner) & (pseed.regime == 'ALL')
            ]
            n_ship = int(ps['ships'].sum()) if 'ships' in ps.columns else 0
            n_seeds = int(n_total_seeds.get((k['track'], k['target'], winner), 0))
            ships_majority = n_ship > (n_seeds / 2.0) if n_seeds > 0 else False
        else:
            delta_mean = np.nan
            reg_deg = np.nan
            n_ship = 0
            n_seeds = 0
            ships_majority = False

        rows.append({
            'track':    TRACK_LABEL.get(k['track'], k['track']),
            'target':   TARGET_LABEL.get(k['target'], k['target']),
            'model':    k['model'],
            'feature_set': k['feature_set'],
            'form':     winner,
            'overall_delta_rmse_mean': delta_mean,
            'max_regime_degradation_mean': reg_deg,
            'n_seeds_ship':  n_ship,
            'n_seeds_total': n_seeds,
            'ships_majority': ships_majority,
        })
    t4 = pd.DataFrame(rows).sort_values(['target', 'track']).reset_index(drop=True)
    return t4


def write_table_4_tex(df: pd.DataFrame, out: Path):
    lines = [
        r'\begin{table}[tbp]',
        r'\centering',
        r'\caption{Shipped bias-correction summary (Phase G.7, Table 4).'
        r' Each row is the aggregate-best (model, feature set) cell of'
        r' the multiseed summary. Form A is piecewise-linear per pre-registered'
        r' P regime; Form B is a single quantile-thresholded piecewise'
        r' correction (shared across regimes). ships\_majority=True means the'
        r' winning form reduced test-set RMSE without inflating any regime'
        r' on strictly more than half of the 20 seeds.}',
        r'\label{tab:bc_shipped}',
        r'\begin{tabular}{llllr r r l}',
        r'\toprule',
        r'track & target & model & feat.\ set & form & $\Delta$RMSE'
        r' & max reg.\ deg.\ & ships (maj.) \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{_escape_latex(r['track'])} & "
            f"{_escape_latex(r['target'])} & "
            f"{_escape_latex(r['model'])} & "
            f"{_escape_latex(r['feature_set'])} & "
            f"{_escape_latex(r['form'])} & "
            f"{_fmt_num(r['overall_delta_rmse_mean'], 3)} & "
            f"{_fmt_num(r['max_regime_degradation_mean'], 3)} & "
            f"{'yes' if r['ships_majority'] else 'no'} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_table_s9(per_seed_csv: Path) -> pd.DataFrame:
    """Per-seed detailed table. One row per (cell, seed, form). We keep
    the overall row (regime == 'ALL') for Table S9 so it's compact."""
    df = pd.read_csv(per_seed_csv)
    # Some drivers emit per-regime rows; we want the 'ALL' roll-up.
    if 'regime' in df.columns:
        df = df[df.regime == 'ALL']
    cols = ['pipeline', 'track', 'target', 'model', 'feature_set',
            'seed', 'form', 'pre_rmse', 'post_rmse', 'delta_rmse',
            'ships']
    have = [c for c in cols if c in df.columns]
    return df[have].copy().sort_values(
        ['target', 'track', 'form', 'seed']).reset_index(drop=True)


def write_longtable_tex(df: pd.DataFrame, out: Path, caption: str, label: str):
    """Generic longtable writer. Caller passes an already-ordered df."""
    n = len(df.columns)
    col_spec = 'l' * min(5, n) + 'r' * max(0, n - 5)
    lines = [
        r'\begin{longtable}{' + col_spec + '}',
        r'\caption{' + caption + r'} \label{' + label + r'} \\',
        r'\toprule',
        ' & '.join(_escape_latex(c) for c in df.columns) + r' \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        ' & '.join(_escape_latex(c) for c in df.columns) + r' \\',
        r'\midrule',
        r'\endhead',
        r'\bottomrule',
        r'\endfoot',
    ]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append(_fmt_num(v, 3))
            else:
                cells.append(_escape_latex(v))
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\end{longtable}')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_table_s10(edge_csv: Path, stab_csv: Path) -> pd.DataFrame:
    """Combine edge sensitivity + Form B stability. One compact row
    per (track, target). Edge side: Form A mean delta-RMSE across the
    three bin-edge sets (base, inner_m1, inner_p1); we report the max
    |swing| relative to base. Stability side: mean and std of
    (alpha_L, alpha_R, s_L, s_R) across 5 CV seeds."""
    edge_rows = []
    if edge_csv.exists():
        e = pd.read_csv(edge_csv)
        keys = ['pipeline', 'track', 'target']
        for k, g in e.groupby(keys):
            base = g[g.edge_set == 'base']
            if base.empty:
                continue
            b_delta = float(base['overall_delta_rmse'].mean())
            swings = []
            for es in ('inner_m1', 'inner_p1'):
                ss = g[g.edge_set == es]
                if not ss.empty:
                    swings.append(abs(float(ss['overall_delta_rmse'].mean()) - b_delta))
            edge_rows.append({
                'pipeline': k[0], 'track': k[1], 'target': k[2],
                'A_delta_base': b_delta,
                'A_delta_edge_max_swing': max(swings) if swings else np.nan,
            })
    edge_df = pd.DataFrame(edge_rows)

    stab_rows = []
    if stab_csv.exists():
        s = pd.read_csv(stab_csv)
        if 'form_b_ok' in s.columns:
            s = s[s.form_b_ok]
        keys = ['pipeline', 'track', 'target']
        for k, g in s.groupby(keys):
            stab_rows.append({
                'pipeline': k[0], 'track': k[1], 'target': k[2],
                'B_alpha_L_mean': g['alpha_L'].mean(),
                'B_alpha_L_std':  g['alpha_L'].std(ddof=0),
                'B_alpha_R_mean': g['alpha_R'].mean(),
                'B_alpha_R_std':  g['alpha_R'].std(ddof=0),
                'B_s_L_mean':     g['s_L'].mean(),
                'B_s_L_std':      g['s_L'].std(ddof=0),
                'B_s_R_mean':     g['s_R'].mean(),
                'B_s_R_std':      g['s_R'].std(ddof=0),
                'B_n_seeds':      len(g),
            })
    stab_df = pd.DataFrame(stab_rows)

    if edge_df.empty and stab_df.empty:
        return pd.DataFrame()
    if edge_df.empty:
        return stab_df
    if stab_df.empty:
        return edge_df
    return edge_df.merge(stab_df, on=['pipeline', 'track', 'target'],
                         how='outer').sort_values(['target', 'track'])


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        per_seed_csv = RESULTS / 'v10_bias_correction_per_seed.csv'
        summary_csv  = RESULTS / 'v10_bias_correction_summary.csv'
        shipped_csv  = RESULTS / 'v10_bias_correction_shipped.csv'
        edge_csv     = RESULTS / 'v10_bias_correction_edge_sensitivity.csv'
        stab_csv     = RESULTS / 'v10_bias_correction_form_b_stability.csv'

        _log('D5 tables: T4 / S9 / S10', fh)

        # Table 4
        if shipped_csv.exists() and summary_csv.exists() and per_seed_csv.exists():
            t4 = build_table_4(shipped_csv, summary_csv, per_seed_csv)
            t4.to_csv(TABLES_DIR / 'T4_bias_correction_shipped.csv', index=False)
            write_table_4_tex(t4, TABLES_DIR / 'T4_bias_correction_shipped.tex')
            _log(f'  T4 wrote {len(t4)} rows', fh)
        else:
            _log(f'  T4 skipped: missing {shipped_csv} or {summary_csv}', fh)

        # Table S9
        if per_seed_csv.exists():
            s9 = build_table_s9(per_seed_csv)
            s9.to_csv(TABLES_DIR / 'S9_bias_correction_per_seed.csv', index=False)
            write_longtable_tex(s9, TABLES_DIR / 'S9_bias_correction_per_seed.tex',
                                caption=('Per-seed detailed bias-correction results'
                                         ' (Phase G.7, Table S9). One row per (cell,'
                                         ' seed, form); regime = ALL. "ships" column'
                                         ' is the per-seed decision.'),
                                label='tab:bc_per_seed')
            _log(f'  S9 wrote {len(s9)} rows', fh)
        else:
            _log(f'  S9 skipped: missing {per_seed_csv}', fh)

        # Table S10
        s10 = build_table_s10(edge_csv, stab_csv)
        if not s10.empty:
            s10.to_csv(TABLES_DIR / 'S10_bias_correction_stability.csv', index=False)
            write_longtable_tex(s10, TABLES_DIR / 'S10_bias_correction_stability.tex',
                                caption=('Bias-correction stability and bin-edge'
                                         ' sensitivity (Phase G.7, Table S10). Form A'
                                         ' $\\Delta$RMSE is reported at the base edges'
                                         ' plus the maximum $|\\Delta|$ swing across'
                                         ' $\\pm 1$ kbar perturbations of the inner'
                                         ' edges. Form B parameters are mean / std'
                                         ' across 5 CV reseeds at fixed model seed 42.'),
                                label='tab:bc_stability')
            _log(f'  S10 wrote {len(s10)} rows', fh)
        else:
            _log(f'  S10 skipped: no edge or stability data', fh)

        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
