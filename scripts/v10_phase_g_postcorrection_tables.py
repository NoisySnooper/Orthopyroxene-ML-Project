#!/usr/bin/env python3
"""Phase G.7 D3: post-correction regime tables and scorecard.

Consumes the per-cell shipped-correction decision from the D2 driver and
applies the shipped (Form A or Form B) correction to ExPetDB test-split
predictions for each aggregate-best cell. Emits two CSVs that mirror the
pre-correction comparables:

  results/v10_regime_allmodels_postcorrection.csv
      Same schema as results/v10_regime_allmodels.csv. Adds a new
      method_family tag `v10_corrected` with one row per
      (aggregate-best-cell, regime) pair. All original rows (v10
      pre-correction, Putirka, Agreda, Jorgenson, Wang) are copied
      through unchanged so downstream tables keep their external
      comparators.

  results/v10_preregistered_scorecard_postcorrection.csv
      Head-to-head table over the pre-registered P regimes
      (shallow_crustal, deep_crustal_MASH, lithospheric_mantle,
      deeper_mantle, ALL) x 4 tracks x 2 targets. For each cell we
      report: v10 best-of-family RMSE (pre), v10 shipped-corrected RMSE
      (post), best external RMSE, winner identity.

Inputs:
    results/v10_regime_allmodels.csv
    results/v10_bias_correction/checkpoints/*_s42.pkl
    results/v10_bias_correction_shipped.csv  (optional; recomputed if
                                              missing)

For cells whose canonical winner is `none`, the v10_corrected row
copies the pre-correction numbers (so there is always a row; the
winner column shows `none` to flag the no-op).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    LOGS, P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS, RESULTS,
)
from src.data import (
    load_cpx_liq, load_cpx_only, load_opx_liq, load_opx_only, load_splits,
)
from src.evaluation import _bootstrap_stat, _rmse, assign_p_regime

LOG_PATH = LOGS / 'v10_phase_g_postcorrection_tables.log'
CKPT_DIR = RESULTS / 'v10_bias_correction' / 'checkpoints'
ALLMODELS_IN = RESULTS / 'v10_regime_allmodels.csv'
ALLMODELS_OUT = RESULTS / 'v10_regime_allmodels_postcorrection.csv'
SCORECARD_OUT = RESULTS / 'v10_preregistered_scorecard_postcorrection.csv'

CANONICAL_SEED = 42
N_BOOT = 500

LOADERS = {
    'opx_liq':  (load_opx_liq,  'opx_liq'),
    'opx_only': (load_opx_only, 'opx_only'),
    'cpx_liq':  (load_cpx_liq,  'cpx_liq'),
    'cpx_only': (load_cpx_only, 'cpx_only'),
}
TARGETS = ('T_C', 'P_kbar')


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _regime_block(y_true, y_pred, regimes, regime_labels, seed=42):
    """Compute per-regime + ALL bootstrap RMSE/MAE CIs."""
    rows = []
    for r in regime_labels + ['ALL']:
        if r == 'ALL':
            mask = np.ones(len(y_true), dtype=bool)
        else:
            mask = (regimes == r)
        n_r = int(mask.sum())
        if n_r < 5:
            rows.append(dict(regime=r, n=n_r,
                             rmse=np.nan, rmse_lo=np.nan, rmse_hi=np.nan,
                             mae=np.nan,  mae_lo=np.nan,  mae_hi=np.nan,
                             n_used=n_r))
            continue
        rmse, rmse_lo, rmse_hi = _bootstrap_stat(
            y_true[mask], y_pred[mask], _rmse,
            n_bootstrap=N_BOOT, seed=seed)

        def _mae(yt, yp):
            return float(np.mean(np.abs(yp - yt)))

        mae, mae_lo, mae_hi = _bootstrap_stat(
            y_true[mask], y_pred[mask], _mae,
            n_bootstrap=N_BOOT, seed=seed)
        rows.append(dict(regime=r, n=n_r,
                         rmse=rmse, rmse_lo=rmse_lo, rmse_hi=rmse_hi,
                         mae=mae, mae_lo=mae_lo, mae_hi=mae_hi,
                         n_used=n_r))
    return rows


def compute_postcorrection_rows(fh) -> pd.DataFrame:
    """For each aggregate-best cell with a canonical (seed=42) checkpoint
    on disk, compute v10_corrected per-regime + ALL metrics and return
    a DataFrame that matches v10_regime_allmodels.csv schema."""
    out = []
    ckpts = sorted(CKPT_DIR.glob(f'*_s{CANONICAL_SEED}.pkl'))
    for ckpt_path in ckpts:
        with open(ckpt_path, 'rb') as f:
            res = pickle.load(f)
        track = res.track
        if track not in LOADERS:
            _log(f'  skip {ckpt_path.name}: track {track} not mapped', fh)
            continue
        loader, split_key = LOADERS[track]
        df = loader()
        tr_idx, te_idx = load_splits(split_key)
        y_te = df[res.target].to_numpy(dtype=float)[te_idx]
        p_te = df['P_kbar'].to_numpy(dtype=float)[te_idx]
        regimes_te = assign_p_regime(p_te)

        y_pred_pre = np.asarray(res.y_pred_te, dtype=float)
        if res.winner == 'A':
            y_corr = np.asarray(res.y_corr_a_te, dtype=float)
        elif res.winner == 'B':
            y_corr = np.asarray(res.y_corr_b_te, dtype=float)
        else:
            y_corr = y_pred_pre  # no-op
        blocks = _regime_block(y_te, y_corr, regimes_te,
                               regime_labels=P_REGIME_LABELS, seed=42)
        edges = P_REGIME_BIN_EDGES_KBAR
        for r in blocks:
            rname = r['regime']
            if rname == 'ALL':
                lo, hi = edges[0], edges[-1]
            else:
                i = P_REGIME_LABELS.index(rname)
                lo, hi = edges[i], edges[i + 1]
            out.append({
                'pipeline':      res.pipeline,
                'track':         track,
                'target':        res.target,
                'regime_type':   'P',
                'regime':        rname,
                'regime_lo':     lo,
                'regime_hi':     hi,
                'n':             r['n'],
                'method_family': 'v10_corrected',
                'method':        f'{res.model}/{res.feature_set}',
                'method_label':  (f'v10 {res.model} {res.feature_set} '
                                  f'(corr={res.winner})'),
                'rmse':          r['rmse'],
                'rmse_lo':       r['rmse_lo'],
                'rmse_hi':       r['rmse_hi'],
                'mae':           r['mae'],
                'mae_lo':        r['mae_lo'],
                'mae_hi':        r['mae_hi'],
                'n_used':        r['n_used'],
                'correction_form': res.winner,
            })
        _log(f'  {ckpt_path.name}: winner={res.winner} '
             f'rows={len(blocks)}', fh)
    return pd.DataFrame(out)


def build_scorecard(allmodels: pd.DataFrame,
                    corrected: pd.DataFrame) -> pd.DataFrame:
    """One row per (track, target, regime) in the pre-registered set
    (LABELS + ALL). Columns:
        n, v10_pre_best (RMSE [lo,hi] of best pre-correction v10 cell),
        v10_pre_method,
        v10_post (RMSE [lo,hi] of the aggregate-best cell's shipped
                   correction; copies pre if winner=none),
        v10_post_method,
        best_external (RMSE [lo,hi] of best non-v10 method),
        best_external_method,
        winner ('v10_corrected', 'v10_pre', 'external').
    """
    # Pre-correction v10 best per (track, target, regime): min RMSE across
    # method_family == 'v10'.
    pre = allmodels[allmodels.method_family == 'v10'].copy()
    pre = (pre.sort_values('rmse')
              .groupby(['track', 'target', 'regime_type', 'regime'])
              .head(1)
              .reset_index(drop=True))
    pre = pre[pre.regime_type == 'P'].copy()

    # Externals best per (track, target, regime).
    ext = allmodels[allmodels.method_family.isin(
        ['Putirka', 'Agreda', 'Jorgenson', 'Wang'])].copy()
    ext = (ext.sort_values('rmse')
              .groupby(['track', 'target', 'regime_type', 'regime'])
              .head(1)
              .reset_index(drop=True))
    ext = ext[ext.regime_type == 'P'].copy()

    # Corrected rows are already one-per (track, target, regime); keep
    # only P-regime rows (all of them, but guard anyway).
    corr = corrected.copy() if len(corrected) else pd.DataFrame()

    keys = ['track', 'target', 'regime']
    pre_m = pre.set_index(keys)
    ext_m = ext.set_index(keys)
    corr_m = corr.set_index(keys) if len(corr) else None

    regimes_order = P_REGIME_LABELS + ['ALL']
    rows = []
    tracks = ['opx_liq', 'opx_only', 'cpx_liq', 'cpx_only']
    for track in tracks:
        for target in TARGETS:
            for rname in regimes_order:
                key = (track, target, rname)
                row = {'track': track, 'target': target, 'regime': rname}
                if key in pre_m.index:
                    p = pre_m.loc[key]
                    row.update({
                        'n': int(p['n']),
                        'v10_pre_rmse':    float(p['rmse']),
                        'v10_pre_rmse_lo': float(p['rmse_lo']),
                        'v10_pre_rmse_hi': float(p['rmse_hi']),
                        'v10_pre_method':  str(p['method']),
                    })
                else:
                    row.update({
                        'n': 0,
                        'v10_pre_rmse': np.nan,
                        'v10_pre_rmse_lo': np.nan,
                        'v10_pre_rmse_hi': np.nan,
                        'v10_pre_method': '',
                    })
                if corr_m is not None and key in corr_m.index:
                    c = corr_m.loc[key]
                    row.update({
                        'v10_post_rmse':    float(c['rmse']),
                        'v10_post_rmse_lo': float(c['rmse_lo']),
                        'v10_post_rmse_hi': float(c['rmse_hi']),
                        'v10_post_method':  str(c['method']),
                        'correction_form':  str(c['correction_form']),
                    })
                else:
                    row.update({
                        'v10_post_rmse': np.nan,
                        'v10_post_rmse_lo': np.nan,
                        'v10_post_rmse_hi': np.nan,
                        'v10_post_method': '',
                        'correction_form': 'missing',
                    })
                if key in ext_m.index:
                    e = ext_m.loc[key]
                    row.update({
                        'best_external_rmse':    float(e['rmse']),
                        'best_external_rmse_lo': float(e['rmse_lo']),
                        'best_external_rmse_hi': float(e['rmse_hi']),
                        'best_external_method':  str(e['method']),
                    })
                else:
                    row.update({
                        'best_external_rmse': np.nan,
                        'best_external_rmse_lo': np.nan,
                        'best_external_rmse_hi': np.nan,
                        'best_external_method': '',
                    })

                # Winner: min finite RMSE.
                candidates = []
                for tag, val in (('v10_pre',       row['v10_pre_rmse']),
                                 ('v10_corrected', row['v10_post_rmse']),
                                 ('external',      row['best_external_rmse'])):
                    if np.isfinite(val):
                        candidates.append((val, tag))
                row['winner'] = min(candidates)[1] if candidates else 'none'
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('D3: post-correction tables', fh)

        if not ALLMODELS_IN.exists():
            _log(f'Missing {ALLMODELS_IN}; run '
                 f'scripts/v10_phase_g_regime_allmodels.py first.', fh)
            return 1
        allmodels = pd.read_csv(ALLMODELS_IN)
        _log(f'loaded {ALLMODELS_IN}  rows={len(allmodels)}', fh)

        corrected = compute_postcorrection_rows(fh)
        if corrected.empty:
            _log('No checkpoints produced corrected rows; aborting', fh)
            return 1
        _log(f'corrected rows={len(corrected)}', fh)

        # Write allmodels_postcorrection = original + corrected rows.
        out_all = pd.concat(
            [allmodels.assign(correction_form='none'), corrected],
            ignore_index=True, sort=False)
        out_all.to_csv(ALLMODELS_OUT, index=False)
        _log(f'wrote {ALLMODELS_OUT}  rows={len(out_all)}', fh)

        # Build scorecard on pre-registered P regimes only.
        sc = build_scorecard(allmodels, corrected)
        sc.to_csv(SCORECARD_OUT, index=False)
        _log(f'wrote {SCORECARD_OUT}  rows={len(sc)}', fh)

        # Quick summary: v10_corrected wins vs pre vs external.
        _log('scorecard winner counts:', fh)
        for tag, count in sc.winner.value_counts().items():
            _log(f'  {tag}: {count}', fh)

        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
