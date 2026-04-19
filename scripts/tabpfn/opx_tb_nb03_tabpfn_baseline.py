#!/usr/bin/env python3
"""v10 nb03 supplementary: TabPFN v2 baseline.

Fits TabPFNRegressor on the 8 (pipeline, track, target) combinations using
the existing citation-grouped train/test splits. No Optuna tuning, no
feature-set sweep, no stacking, no SHAP. Foundation model on raw features
only, 20 seeds (42-61) matching the tuned-family multiseed protocol.

Outputs match schemas of v10_{pipeline}_multiseed_{results,summary}.csv and
regime_allmodels.csv so downstream notebooks can read them without
schema conversion.

Reference: Hollmann et al. 2025, Nature 637.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import LOGS, RESULTS
from src.evaluation import (
    _bootstrap_stat,
    _rmse,
    assign_p_regime,
    compute_metrics,
)
from src.prepare_train_test import prepare_train_test


COMBOS = [
    ('opx', 'opx_liq',  'T_C'),
    ('opx', 'opx_liq',  'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
    ('cpx', 'cpx_liq',  'T_C'),
    ('cpx', 'cpx_liq',  'P_kbar'),
    ('cpx', 'cpx_only', 'T_C'),
    ('cpx', 'cpx_only', 'P_kbar'),
]

# cpx gets smaller ensemble to control CPU runtime (spec option 3b).
N_ESTIMATORS = {'opx': 8, 'cpx': 4}

CKPT_DIR = RESULTS / 'tabpfn_checkpoints'
LOG_PATH = LOGS / 'v10_nb03_tabpfn.log'

OUT_RESULTS = RESULTS / 'tabpfn_multiseed_results.csv'
OUT_SUMMARY = RESULTS / 'tabpfn_multiseed_summary.csv'
OUT_REGIME  = RESULTS / 'tabpfn_regime_rmse.csv'
OUT_PRED    = RESULTS / 'tabpfn_predictions.csv'


def _log(fh, msg: str) -> None:
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _ckpt_path(pipeline, track, target, seed, suffix=''):
    name = f'{pipeline}_{track}_{target}_s{seed}{suffix}.pkl'
    return CKPT_DIR / name


def _p_kbar_true(data: dict, target: str, track: str, pipeline: str) -> np.ndarray:
    """Return true P_kbar on the TEST split (for regime assignment).

    If target == 'P_kbar', use y_te directly. Otherwise reload the frame
    and pull P_kbar from the test indices.
    """
    if target == 'P_kbar':
        return data['y_te'].astype(float)
    from src.data import (
        load_cpx_liq, load_cpx_only, load_opx_liq, load_opx_only, load_splits,
    )
    ld = {
        'opx_liq':  load_opx_liq,
        'opx_only': load_opx_only,
        'cpx_liq':  load_cpx_liq,
        'cpx_only': load_cpx_only,
    }[track]
    df = ld()
    _, te_idx = load_splits(track)
    return df.iloc[te_idx]['P_kbar'].to_numpy(dtype=float)


_LOGGED_VERSION_FOR: set[tuple[str, str, str]] = set()


def _build_regressor(n_est: int, seed: int, fh) -> tuple[object, str]:
    """Construct TabPFNRegressor pinned to v2. Returns (regressor, path_used).

    Prefers create_default_for_version(ModelVersion.V2, ...) when available
    (newer tabpfn>=2.5 dropped into <2.5 pin would still work if backported).
    Falls back to model_path='Prior-Labs/TabPFN-v2-reg', then to model_path
    ='auto' which in tabpfn>=2.0,<2.5 resolves to the v2 weights by construction.
    Aborts if the installed package advertises 2.5+.
    """
    try:
        import tabpfn  # noqa: WPS433
    except Exception as e:
        _log(fh, f'IMPORT FAILED: {e}')
        sys.exit(2)
    ver = getattr(tabpfn, '__version__', 'unknown')
    if ver == 'unknown' or not ver.startswith('2.') or ver.startswith('2.5'):
        _log(fh, f'ABORT: tabpfn version {ver!r} is not v2.x (<2.5). '
                 f'Pin requirements-tabpfn.txt to tabpfn>=2.0,<2.5.')
        sys.exit(2)

    from tabpfn import TabPFNRegressor

    path = 'create_default_for_version'
    try:
        from tabpfn.constants import ModelVersion  # type: ignore[attr-defined]
        reg = TabPFNRegressor.create_default_for_version(  # type: ignore[attr-defined]
            ModelVersion.V2,
            device='cpu', n_estimators=n_est, random_state=int(seed),
        )
    except (ImportError, AttributeError):
        # tabpfn 2.0-2.4 does not expose create_default_for_version; in that
        # range model_path='auto' is the TabPFN-v2 weight set by construction.
        # We rely on the >=2.0,<2.5 pip pin as the version enforcement.
        # ignore_pretraining_limits=True lets us fit cpx cells (>1000 train)
        # on CPU; the guard is a soft performance warning, not a correctness
        # bound.
        reg = TabPFNRegressor(
            device='cpu', n_estimators=n_est, random_state=int(seed),
            model_path='auto', ignore_pretraining_limits=True,
        )
        path = ("model_path='auto' ignore_pretraining_limits=True "
                "(package-version pinned to v2 by tabpfn>=2.0,<2.5)")
    return reg, path


def _fit_one(pipeline, track, target, seed, fh, data, p_true_te, force):
    ckpt = _ckpt_path(pipeline, track, target, seed)
    if ckpt.exists() and not force:
        try:
            with open(ckpt, 'rb') as f:
                cached = pickle.load(f)
            _log(fh, f'  [cache hit] {pipeline}/{track}/{target} s={seed}')
            return cached
        except Exception as e:
            _log(fh, f'  [cache corrupt, refitting] {e}')

    n_est = N_ESTIMATORS[pipeline]
    t0 = time.time()
    try:
        reg, construction_path = _build_regressor(n_est, seed, fh)
        key = (pipeline, track, target)
        if key not in _LOGGED_VERSION_FOR:
            import tabpfn as _tp
            _log(fh, f'  [model v2 load path] '
                     f'tabpfn=={_tp.__version__} via {construction_path}')
            _LOGGED_VERSION_FOR.add(key)
        reg.fit(data['X_tr'], data['y_tr'])
        y_hat = np.asarray(reg.predict(data['X_te']), dtype=float)
    except Exception as e:
        _log(fh, f'  [FAIL seed={seed}] {e}')
        tb = traceback.format_exc()
        _log(fh, tb)
        return {'failed': True, 'error': str(e)}

    dt = time.time() - t0
    if not np.all(np.isfinite(y_hat)):
        _log(fh, f'  [NAN PREDICTIONS seed={seed}] marking failed')
        return {'failed': True, 'error': 'nan predictions'}

    metrics = compute_metrics(data['y_te'], y_hat)
    out = {
        'failed': False,
        'seed': int(seed),
        'rmse': metrics['rmse'],
        'mae':  metrics['mae'],
        'r2':   metrics['r2'],
        'bias': metrics['bias'],
        'n':    metrics['n'],
        'n_train': int(len(data['y_tr'])),
        'n_test':  int(len(data['y_te'])),
        'y_pred_te': y_hat.astype(np.float32),
        'y_true_te': np.asarray(data['y_te'], dtype=np.float32),
        'p_true_te': np.asarray(p_true_te, dtype=np.float32),
        'n_estimators': int(n_est),
        'elapsed_s': float(dt),
    }
    with open(ckpt, 'wb') as f:
        pickle.dump(out, f)
    _log(fh, f'  [ok seed={seed}] rmse={metrics["rmse"]:.3f} '
             f'mae={metrics["mae"]:.3f} r2={metrics["r2"]:.3f} '
             f'elapsed={dt:.1f}s (n_est={n_est})')
    return out


def _regime_rows_from_seed_mean(pipeline, track, target, y_true, y_pred_mean,
                                p_true):
    """Return list of 5 regime rows (4 P regimes + ALL) in the v10 regime
    schema, using seed-mean predictions and bootstrap CIs."""
    rows = []
    regimes = assign_p_regime(p_true)
    bin_edges = [0.0, 5.0, 15.0, 30.0, 100.0]
    labels_by_idx = ['shallow_crustal', 'deep_crustal_MASH',
                     'lithospheric_mantle', 'deeper_mantle']
    # Per-regime
    for i, reg_label in enumerate(labels_by_idx):
        mask = regimes == reg_label
        n = int(mask.sum())
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if n == 0:
            rows.append(dict(
                pipeline=pipeline, track=track, target=target,
                regime_type='P', regime=reg_label, regime_lo=lo, regime_hi=hi,
                n=0, method_family='tabpfn', method='TabPFN/raw',
                method_label='TabPFN (foundation model)',
                rmse=np.nan, rmse_lo=np.nan, rmse_hi=np.nan,
                mae=np.nan, mae_lo=np.nan, mae_hi=np.nan, n_used=0,
            ))
            continue
        yt, yp = y_true[mask], y_pred_mean[mask]
        rmse, rlo, rhi = _bootstrap_stat(yt, yp, _rmse)
        mae_point = float(np.mean(np.abs(yp - yt)))
        _, mlo, mhi = _bootstrap_stat(yt, yp,
                                      lambda a, b: float(np.mean(np.abs(b - a))))
        rows.append(dict(
            pipeline=pipeline, track=track, target=target,
            regime_type='P', regime=reg_label, regime_lo=lo, regime_hi=hi,
            n=n, method_family='tabpfn', method='TabPFN/raw',
            method_label='TabPFN (foundation model)',
            rmse=rmse, rmse_lo=rlo, rmse_hi=rhi,
            mae=mae_point, mae_lo=mlo, mae_hi=mhi, n_used=n,
        ))
    # ALL row
    rmse, rlo, rhi = _bootstrap_stat(y_true, y_pred_mean, _rmse)
    mae_point = float(np.mean(np.abs(y_pred_mean - y_true)))
    _, mlo, mhi = _bootstrap_stat(y_true, y_pred_mean,
                                  lambda a, b: float(np.mean(np.abs(b - a))))
    rows.append(dict(
        pipeline=pipeline, track=track, target=target,
        regime_type='P', regime='ALL', regime_lo=0.0, regime_hi=100.0,
        n=int(len(y_true)), method_family='tabpfn', method='TabPFN/raw',
        method_label='TabPFN (foundation model)',
        rmse=rmse, rmse_lo=rlo, rmse_hi=rhi,
        mae=mae_point, mae_lo=mlo, mae_hi=mhi, n_used=int(len(y_true)),
    ))
    return rows


def run(seeds, pipelines_filter, dry_run, force):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    _log(fh, '='*72)
    _log(fh, f'TabPFN baseline run: seeds={seeds} '
             f'pipelines={pipelines_filter} dry_run={dry_run} force={force}')

    results_rows = []
    regime_rows = []
    pred_rows = []

    try:
        for pipeline, track, target in COMBOS:
            if pipelines_filter != 'all' and pipeline != pipelines_filter:
                continue
            tag = 'dryrun' if dry_run else 'full'
            pipeline_out = f'{pipeline}_dryrun' if dry_run else pipeline
            _log(fh, f'[{tag}] {pipeline}/{track}/{target}')
            try:
                data = prepare_train_test(pipeline, track, target, 'raw')
                if dry_run:
                    n = min(50, len(data['y_tr']))
                    data = {**data,
                            'X_tr': data['X_tr'][:n],
                            'y_tr': data['y_tr'][:n]}
                p_true_te = _p_kbar_true(data, target, track, pipeline)
            except Exception as e:
                _log(fh, f'  [DATA PREP FAILED] {e}')
                _log(fh, traceback.format_exc())
                continue

            seed_outs = []
            for seed in seeds:
                out = _fit_one(pipeline, track, target, seed, fh, data,
                               p_true_te, force)
                seed_outs.append(out)

            ok_seeds = [o for o in seed_outs if not o.get('failed', False)]
            if not ok_seeds:
                _log(fh, f'  [ALL SEEDS FAILED] skipping combination outputs')
                for seed in seeds:
                    results_rows.append(dict(
                        pipeline=pipeline_out, model='TabPFN', target=target,
                        track=track, feature_set='raw', seed=seed,
                        test_rmse=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan,
                        n_train=int(len(data['y_tr'])),
                        n_test=int(len(data['y_te'])),
                    ))
                continue

            for o in seed_outs:
                if o.get('failed'):
                    results_rows.append(dict(
                        pipeline=pipeline_out, model='TabPFN', target=target,
                        track=track, feature_set='raw', seed=-1,
                        test_rmse=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan,
                        n_train=int(len(data['y_tr'])),
                        n_test=int(len(data['y_te'])),
                    ))
                    continue
                results_rows.append(dict(
                    pipeline=pipeline_out, model='TabPFN', target=target,
                    track=track, feature_set='raw', seed=int(o['seed']),
                    test_rmse=float(o['rmse']),
                    rmse=float(o['rmse']), mae=float(o['mae']),
                    r2=float(o['r2']),
                    n_train=int(o['n_train']), n_test=int(o['n_test']),
                ))

            rmse_arr = np.array([o['rmse'] for o in ok_seeds], dtype=float)
            y_true = ok_seeds[0]['y_true_te'].astype(float)
            y_pred_stack = np.stack(
                [o['y_pred_te'].astype(float) for o in ok_seeds], axis=0)
            y_pred_mean = y_pred_stack.mean(axis=0)
            p_te = ok_seeds[0]['p_true_te'].astype(float)

            regime_rows.extend(_regime_rows_from_seed_mean(
                pipeline_out, track, target, y_true, y_pred_mean, p_te))

            regimes_vec = assign_p_regime(p_te)
            for o in ok_seeds:
                yp = o['y_pred_te'].astype(float)
                for i in range(len(y_true)):
                    pred_rows.append(dict(
                        pipeline=pipeline_out, track=track, target=target,
                        model='TabPFN', feature_set='raw',
                        sample_idx=int(i), seed=int(o['seed']),
                        y_true=float(y_true[i]), y_pred=float(yp[i]),
                        P_kbar_true=float(p_te[i]),
                        regime=str(regimes_vec[i]),
                    ))

            _log(fh, f'  [summary] rmse mean={rmse_arr.mean():.3f} '
                     f'std={rmse_arr.std(ddof=0):.3f} '
                     f'min={rmse_arr.min():.3f} max={rmse_arr.max():.3f}')

        results_df = pd.DataFrame(results_rows)
        if not results_df.empty:
            results_df.to_csv(OUT_RESULTS, index=False)
            _log(fh, f'wrote {OUT_RESULTS} ({len(results_df)} rows)')

            agg = (results_df
                   .dropna(subset=['rmse'])
                   .groupby(['pipeline', 'model', 'target',
                             'track', 'feature_set'])['rmse']
                   .agg(['mean', 'std', 'min', 'max', 'count'])
                   .reset_index())
            agg.to_csv(OUT_SUMMARY, index=False)
            _log(fh, f'wrote {OUT_SUMMARY} ({len(agg)} rows)')

        if regime_rows:
            pd.DataFrame(regime_rows).to_csv(OUT_REGIME, index=False)
            _log(fh, f'wrote {OUT_REGIME} ({len(regime_rows)} rows)')

        if pred_rows:
            pd.DataFrame(pred_rows).to_csv(OUT_PRED, index=False)
            _log(fh, f'wrote {OUT_PRED} ({len(pred_rows)} rows)')

        return 0
    finally:
        fh.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline', default='all',
                   choices=['all', 'opx', 'cpx'])
    p.add_argument('--dry-run', action='store_true',
                   help='fit on 50 train rows; output pipeline tagged _dryrun')
    p.add_argument('--seeds',
                   default=','.join(str(s) for s in range(42, 62)),
                   help='comma-separated seeds (default: 20 seeds 42-61)')
    p.add_argument('--force-rerun', action='store_true',
                   help='ignore existing checkpoints')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.force_rerun:
        for pipe, track, target in COMBOS:
            for seed in seeds:
                ck = _ckpt_path(pipe, track, target, seed)
                if ck.exists():
                    ck.unlink()
    return run(seeds, args.pipeline, args.dry_run, args.force_rerun)


if __name__ == '__main__':
    sys.exit(main())
