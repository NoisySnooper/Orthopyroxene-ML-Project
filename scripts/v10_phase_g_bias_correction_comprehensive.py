#!/usr/bin/env python3
"""Phase G.7 D2: comprehensive bias-correction driver.

For every (pipeline, track, target) combination in the v10 scope
(opx-liq, opx-only, cpx-liq, cpx-only -- 8 cells total), pick the
aggregate-best (model, feature_set) from the corresponding multiseed
summary CSV, then run Form A and Form B bias correction at every seed
in config.SPLIT_SEEDS (20 seeds, 42..61).

Outputs
-------
Per-(cell, seed) checkpoint:
    results/v10_bias_correction/checkpoints/
        {pipeline}_{track}_{target}_{model}_{feature_set}_s{seed}.pkl

Final aggregated CSV:
    results/v10_bias_correction_per_seed.csv           long-form, one row
                                                       per (cell, seed,
                                                       form, regime)
    results/v10_bias_correction_shipped.csv            one row per cell
                                                       with winner,
                                                       seed-0 params
                                                       (used for
                                                       downstream D3)
    results/v10_bias_correction_summary.csv            one row per
                                                       (cell, form) with
                                                       mean +/- std of
                                                       per-regime +
                                                       overall metrics
                                                       across seeds

Log: logs/v10_phase_g_bias_correction.log
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import LOGS, RESULTS, SPLIT_SEEDS
from src.bias_correction import run_bias_correction_for_cell
from src.data import (
    load_cpx_liq, load_cpx_only, load_opx_liq, load_opx_only, load_splits,
)
from src.prepare_train_test import prepare_train_test
from src.v10_phase_c_analysis import load_best_params

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_bias_correction.log'
CKPT_DIR = RESULTS / 'v10_bias_correction' / 'checkpoints'
OUT_PER_SEED = RESULTS / 'v10_bias_correction_per_seed.csv'
OUT_SHIPPED = RESULTS / 'v10_bias_correction_shipped.csv'
OUT_SUMMARY = RESULTS / 'v10_bias_correction_summary.csv'

# Pipeline -> (summary CSV stem, best-params manifest, data loader, splits key)
PIPELINES = {
    'opx_liq':  {'pipeline': 'opx',
                 'summary': 'v10_opx_multiseed_summary.csv',
                 'manifest': 'v10_optuna_best_params_opx.json',
                 'loader': load_opx_liq, 'split_key': 'opx_liq'},
    'opx_only': {'pipeline': 'opx',
                 'summary': 'v10_opx_multiseed_summary.csv',
                 'manifest': 'v10_optuna_best_params_opx.json',
                 'loader': load_opx_only, 'split_key': 'opx_only'},
    'cpx_liq':  {'pipeline': 'cpx',
                 'summary': 'v10_cpx_multiseed_summary.csv',
                 'manifest': 'v10_optuna_best_params_cpx.json',
                 'loader': load_cpx_liq, 'split_key': 'cpx_liq'},
    'cpx_only': {'pipeline': 'cpx',
                 'summary': 'v10_cpx_multiseed_summary.csv',
                 'manifest': 'v10_optuna_best_params_cpx.json',
                 'loader': load_cpx_only, 'split_key': 'cpx_only'},
}

TARGETS = ('T_C', 'P_kbar')


# ---------------------------------------------------------------------------
# Logging + SIGINT
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _handle_sigint(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print('\n[SIGINT] Graceful shutdown requested; will finish current '
          'cell-seed and exit.', flush=True)


def _log(msg, fh=None):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


# ---------------------------------------------------------------------------
# Cell selection: aggregate-best (model, feature_set) per (track, target)
# ---------------------------------------------------------------------------

def select_best_cells() -> list[dict]:
    """One row per (track, target) with its aggregate-best (model,
    feature_set) from the multiseed summary CSV."""
    cells = []
    for track, cfg in PIPELINES.items():
        summary_csv = RESULTS / cfg['summary']
        if not summary_csv.exists():
            raise FileNotFoundError(
                f'Missing multiseed summary {summary_csv}. '
                f'Run scripts/v10_phase_g_multiseed_runner.py first.')
        df = pd.read_csv(summary_csv)
        df = df[df.track == track]
        for tg in TARGETS:
            sub = df[df.target == tg]
            if sub.empty:
                continue
            best = sub.loc[sub['mean'].idxmin()]
            cells.append({
                'track':       track,
                'target':      tg,
                'pipeline':    cfg['pipeline'],
                'model':       best['model'],
                'feature_set': best['feature_set'],
                'mean_rmse':   float(best['mean']),
                'manifest':    cfg['manifest'],
                'loader':      cfg['loader'],
                'split_key':   cfg['split_key'],
            })
    return cells


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(cell: dict, seed: int) -> Path:
    return CKPT_DIR / (f"{cell['pipeline']}_{cell['track']}_{cell['target']}_"
                       f"{cell['model']}_{cell['feature_set']}_s{seed}.pkl")


def _load_checkpoint(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_checkpoint(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(result, f)
    tmp.replace(path)  # atomic


# ---------------------------------------------------------------------------
# Run one (cell, seed) with retry-once on transient errors
# ---------------------------------------------------------------------------

class TransientError(RuntimeError):
    pass


def _run_once(cell: dict, seed: int, best_params: dict,
              X_tr, y_tr, groups_tr, X_te, y_te, p_tr, p_te):
    return run_bias_correction_for_cell(
        pipeline=cell['pipeline'],
        track=cell['track'],
        target=cell['target'],
        feature_set=cell['feature_set'],
        model_name=cell['model'],
        best_params=best_params,
        X_tr=X_tr, y_tr=y_tr, groups_tr=groups_tr,
        X_te=X_te, y_te=y_te,
        p_true_tr=p_tr, p_true_te=p_te,
        seed=int(seed),
    )


def run_cell_seed(cell: dict, seed: int, best_params: dict,
                  prepared: dict, p_tr, p_te, fh):
    """Run one (cell, seed) with retry-once on transient errors."""
    X_tr, y_tr = prepared['X_tr'], prepared['y_tr']
    X_te, y_te = prepared['X_te'], prepared['y_te']
    groups_tr = prepared['groups_tr']
    if groups_tr is None:
        raise RuntimeError(f'No Citation groups for {cell["track"]}; cannot '
                           f'run StratifiedGroupKFold OOF.')

    try:
        return _run_once(cell, seed, best_params,
                         X_tr, y_tr, groups_tr, X_te, y_te, p_tr, p_te)
    except (np.linalg.LinAlgError, MemoryError) as e:
        _log(f'  transient error ({type(e).__name__}: {e}); retrying once', fh)
        return _run_once(cell, seed, best_params,
                         X_tr, y_tr, groups_tr, X_te, y_te, p_tr, p_te)


# ---------------------------------------------------------------------------
# Flatten BiasCorrectionResult -> CSV rows
# ---------------------------------------------------------------------------

def _result_to_per_seed_rows(res) -> list[dict]:
    rows = res.to_summary_rows()
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', nargs='+', default=None,
                    choices=list(PIPELINES),
                    help='Restrict to a subset of tracks (default: all)')
    ap.add_argument('--targets', nargs='+', default=None,
                    choices=list(TARGETS),
                    help='Restrict to a subset of targets (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Override seeds (default: config.SPLIT_SEEDS)')
    ap.add_argument('--skip-done', action='store_true',
                    help='Skip (cell, seed) pairs whose checkpoint exists')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the plan (cells, seeds, checkpoint paths) '
                         'without running anything')
    ap.add_argument('--no-aggregate', action='store_true',
                    help='Skip the final aggregation step')
    args = ap.parse_args()

    seeds = list(args.seeds) if args.seeds else list(SPLIT_SEEDS)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _log('='*60, fh)
        _log(f'START dry_run={args.dry_run} skip_done={args.skip_done}', fh)

        cells = select_best_cells()
        if args.tracks:
            cells = [c for c in cells if c['track'] in args.tracks]
        if args.targets:
            cells = [c for c in cells if c['target'] in args.targets]

        _log(f'n_cells={len(cells)} n_seeds={len(seeds)}', fh)
        for c in cells:
            _log(f"  cell: {c['pipeline']}/{c['track']}/{c['target']} -> "
                 f"{c['model']}/{c['feature_set']} (mean={c['mean_rmse']:.3f})", fh)

        if args.dry_run:
            _log('DRY RUN -- not executing', fh)
            total = len(cells) * len(seeds)
            missing = 0
            for c in cells:
                for s in seeds:
                    if not _checkpoint_path(c, s).exists():
                        missing += 1
            _log(f'total (cell, seed) = {total}; missing = {missing}', fh)
            return 0

        # Prepare data once per cell, not per seed (splits are fixed).
        prepared_cache = {}
        p_cache = {}
        manifest_cache = {}

        for ci, cell in enumerate(cells, 1):
            if _STOP_REQUESTED:
                _log('SIGINT: stopping before cell start', fh)
                break
            key = (cell['track'], cell['target'])
            if key not in prepared_cache:
                prep = prepare_train_test(cell['pipeline'], cell['track'],
                                          cell['target'], cell['feature_set'])
                prepared_cache[key] = prep

                # Regime assignment needs true P_kbar at the split rows.
                df = cell['loader']()
                tr_idx, te_idx = load_splits(cell['split_key'])
                p_full = df['P_kbar'].to_numpy(dtype=float)
                p_cache[key] = (p_full[tr_idx], p_full[te_idx])

            if cell['manifest'] not in manifest_cache:
                manifest_cache[cell['manifest']] = load_best_params(
                    RESULTS / cell['manifest'])
            best = manifest_cache[cell['manifest']]
            bp_key = (cell['model'], cell['target'], cell['track'],
                      cell['feature_set'])
            if bp_key not in best:
                raise KeyError(
                    f'No best_params for {bp_key} in {cell["manifest"]}')
            best_params = best[bp_key]['best_params']

            prep = prepared_cache[key]
            p_tr, p_te = p_cache[key]

            for si, seed in enumerate(seeds, 1):
                if _STOP_REQUESTED:
                    _log('SIGINT: stopping between seeds', fh)
                    break
                ckpt = _checkpoint_path(cell, seed)
                if args.skip_done and ckpt.exists():
                    continue
                t0 = time.time()
                try:
                    res = run_cell_seed(cell, seed, best_params, prep,
                                        p_tr, p_te, fh)
                except Exception as e:
                    _log(f'[{ci}/{len(cells)}][{si}/{len(seeds)}] '
                         f'{cell["track"]}/{cell["target"]} seed={seed} '
                         f'FAIL {type(e).__name__}: {e}', fh)
                    _log(traceback.format_exc(), fh)
                    continue  # fail-closed on this seed, press on
                _save_checkpoint(ckpt, res)
                _log(f'[{ci}/{len(cells)}][{si}/{len(seeds)}] '
                     f'{cell["track"]}/{cell["target"]} seed={seed} '
                     f'winner={res.winner} '
                     f'elapsed={time.time()-t0:.1f}s', fh)

        if not args.no_aggregate and not _STOP_REQUESTED:
            aggregate(cells, seeds, fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Aggregation: read all checkpoints -> CSVs
# ---------------------------------------------------------------------------

def aggregate(cells, seeds, fh):
    _log('aggregating checkpoints -> CSVs', fh)
    per_seed_rows = []
    shipped_rows = []
    for cell in cells:
        found = []
        for s in seeds:
            p = _checkpoint_path(cell, s)
            if p.exists():
                try:
                    found.append((s, _load_checkpoint(p)))
                except Exception as e:
                    _log(f'  corrupt checkpoint {p}: {e}', fh)
        if not found:
            _log(f'  {cell["track"]}/{cell["target"]}: NO checkpoints', fh)
            continue
        for s, res in found:
            per_seed_rows.extend(_result_to_per_seed_rows(res))

        # Shipped: use seed=min(seeds) checkpoint as the canonical
        # correction params (per user spec Q4: "shipped" = seed-0 fit).
        canonical_seed = min(s for s, _ in found)
        canonical = dict(found)[canonical_seed]
        shipped_rows.append({
            'pipeline':      cell['pipeline'],
            'track':         cell['track'],
            'target':        cell['target'],
            'model':         cell['model'],
            'feature_set':   cell['feature_set'],
            'canonical_seed': canonical_seed,
            'winner':        canonical.winner,
            'ship_a':        json.dumps(canonical.ship_a),
            'ship_b':        json.dumps(canonical.ship_b),
            'form_a_params': json.dumps(canonical.form_a_params),
            'form_b_params': json.dumps(canonical.form_b_params),
            'n_seeds_done':  len(found),
        })

    if per_seed_rows:
        df = pd.DataFrame(per_seed_rows)
        df.to_csv(OUT_PER_SEED, index=False)
        _log(f'  wrote {OUT_PER_SEED}  rows={len(df)}', fh)

        # Cross-seed summary: mean / std of pre_rmse, post_rmse, delta_rmse
        # per (cell, form, regime).
        keys = ['pipeline', 'track', 'target', 'model', 'feature_set',
                'form', 'regime']
        summary = (df.groupby(keys)[['pre_rmse', 'post_rmse', 'delta_rmse']]
                     .agg(['mean', 'std', 'min', 'max', 'count'])
                     .reset_index())
        # Flatten MultiIndex columns
        summary.columns = [
            '_'.join([c for c in col if c]).strip('_')
            for col in summary.columns.values
        ]
        summary.to_csv(OUT_SUMMARY, index=False)
        _log(f'  wrote {OUT_SUMMARY}  rows={len(summary)}', fh)

    if shipped_rows:
        ship_df = pd.DataFrame(shipped_rows)
        ship_df.to_csv(OUT_SHIPPED, index=False)
        _log(f'  wrote {OUT_SHIPPED}  rows={len(ship_df)}', fh)


if __name__ == '__main__':
    sys.exit(main())
