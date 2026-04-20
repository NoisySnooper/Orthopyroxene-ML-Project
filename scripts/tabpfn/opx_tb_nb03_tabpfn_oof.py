#!/usr/bin/env python3
"""Generate OOF predictions for TabPFN on the 4 opx combinations.

For each of 4 (pipeline, track, target) combinations, runs 10-fold
citation-grouped StratifiedGroupKFold CV with TabPFN v2 inference.
Writes per-sample OOF predictions to
`results/tabpfn_oof_predictions.csv` with the schema consumed by
`src.bias_correction.fit_form_a` / `fit_form_b`:

    pipeline, track, target, seed, sample_idx, fold,
    y_true, y_oof_pred, p_true, regime

Scope: opx only. 5 seeds (42-46). 10 folds. 4 combos = 200 fits,
~3h CPU wall clock. The five-seed reduction follows the standard
bias-correction protocol: OOF residuals FIT the correction; the
20-seed test-set TabPFN variance already exists.

Resumable: skips any (combo, seed) whose fold predictions are
already in the output CSV.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import LOGS, RESULTS
from src.bias_correction import stratify_labels
from src.evaluation import assign_p_regime
from src.prepare_train_test import prepare_train_test

COMBOS = [
    ('opx', 'opx_liq',  'T_C'),
    ('opx', 'opx_liq',  'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 10
N_ESTIMATORS = 8
OUT_CSV = RESULTS / 'tabpfn_oof_predictions.csv'
LOG_PATH = LOGS / 'tabpfn_oof.log'


def _log(fh, msg: str) -> None:
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _build_regressor(seed: int, fh) -> object:
    import tabpfn
    ver = getattr(tabpfn, '__version__', 'unknown')
    if ver == 'unknown' or not ver.startswith('2.') or ver.startswith('2.5'):
        _log(fh, f'ABORT: tabpfn version {ver!r} is not v2.x (<2.5).')
        sys.exit(2)
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion  # type: ignore[attr-defined]
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device='cpu', n_estimators=N_ESTIMATORS, random_state=int(seed),
        )
    except (ImportError, AttributeError):
        return TabPFNRegressor(
            device='cpu', n_estimators=N_ESTIMATORS, random_state=int(seed),
            model_path='auto', ignore_pretraining_limits=True,
        )


def _p_true_tr(track: str, target: str, y_tr: np.ndarray) -> np.ndarray:
    if target == 'P_kbar':
        return np.asarray(y_tr, dtype=float)
    from src.data import (
        load_opx_liq, load_opx_only, load_splits,
    )
    ld = {'opx_liq': load_opx_liq, 'opx_only': load_opx_only}[track]
    df = ld()
    tr_idx, _ = load_splits(track)
    return df.iloc[tr_idx]['P_kbar'].to_numpy(dtype=float)


def _already_done(csv_path: Path, pipeline: str, track: str,
                  target: str, seed: int) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    mask = ((df['pipeline'] == pipeline) & (df['track'] == track)
            & (df['target'] == target) & (df['seed'] == seed))
    return bool(mask.any())


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    header = not csv_path.exists()
    pd.DataFrame(rows).to_csv(csv_path, mode='a', header=header, index=False)


def run_one(pipeline, track, target, seed, fh) -> int:
    if _already_done(OUT_CSV, pipeline, track, target, seed):
        _log(fh, f'  [skip] {pipeline}/{track}/{target} seed={seed} already in CSV')
        return 0
    data = prepare_train_test(pipeline, track, target, 'raw')
    X = np.asarray(data['X_tr'], dtype=float)
    y = np.asarray(data['y_tr'], dtype=float)
    groups = np.asarray(data['groups_tr'])
    p_true = _p_true_tr(track, target, y)
    regimes = assign_p_regime(p_true)

    y_strat = stratify_labels(y)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=seed)

    rows = []
    t_all = time.time()
    for fold_i, (tr, va) in enumerate(sgkf.split(X, y_strat, groups=groups)):
        t0 = time.time()
        reg = _build_regressor(seed, fh)
        try:
            reg.fit(X[tr], y[tr])
            y_hat = np.asarray(reg.predict(X[va]), dtype=float)
        except Exception as e:
            _log(fh, f'  [FAIL fold {fold_i}] {e}')
            _log(fh, traceback.format_exc())
            return 0
        if not np.all(np.isfinite(y_hat)):
            _log(fh, f'  [NAN] fold {fold_i}, aborting combo')
            return 0
        for k, idx in enumerate(va):
            rows.append({
                'pipeline': pipeline, 'track': track, 'target': target,
                'seed': int(seed), 'sample_idx': int(idx),
                'fold': int(fold_i),
                'y_true': float(y[idx]),
                'y_oof_pred': float(y_hat[k]),
                'p_true': float(p_true[idx]),
                'regime': str(regimes[idx]),
            })
        dt = time.time() - t0
        _log(fh, f'  fold {fold_i}/{N_FOLDS}: train={len(tr)} val={len(va)} '
                 f'({dt:.1f}s)')
    _append_rows(OUT_CSV, rows)
    total = time.time() - t_all
    _log(fh, f'  [DONE] {pipeline}/{track}/{target} seed={seed} '
             f'wrote {len(rows)} rows ({total/60:.1f}m)')
    return len(rows)


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as fh:
        _log(fh, '=' * 70)
        _log(fh, f'tabpfn_oof: {len(COMBOS)} combos x {len(SEEDS)} seeds x '
                 f'{N_FOLDS} folds = {len(COMBOS)*len(SEEDS)*N_FOLDS} fits')
        _log(fh, f'output: {OUT_CSV}')
        t_total = time.time()
        n_fits_total = len(COMBOS) * len(SEEDS)
        n_done = 0
        for (pipeline, track, target) in COMBOS:
            for seed in SEEDS:
                _log(fh, f'[{n_done+1}/{n_fits_total}] '
                         f'{pipeline}/{track}/{target} seed={seed}')
                run_one(pipeline, track, target, seed, fh)
                n_done += 1
                elapsed = time.time() - t_total
                rem = (n_fits_total - n_done) * elapsed / max(n_done, 1)
                _log(fh, f'  elapsed {elapsed/60:.1f}m, ETA {rem/60:.1f}m')
        _log(fh, f'FINISHED total wall {(time.time()-t_total)/60:.1f}m')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
