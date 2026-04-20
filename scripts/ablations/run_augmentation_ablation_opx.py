"""Section 4 big compute for nb04b_aug_test.

Runs 4 opx combinations x 3 feature sets x 8 models x 20 seeds = 1920
fits on 15x augmented training data. Test set stays un-augmented.

Per-cell output appended to
    results/augmentation_ablation_opx_multiseed_results.csv
    results/augmentation_ablation_opx_test_predictions.parquet

Checkpoint written after every 50 cells to
    results/augmentation_ablation_opx_checkpoint.pkl

Resumable: on restart, skips cells already present in the results CSV.

The test predictions parquet is used by the Section 5-7 bias correction
pass: it avoids re-fitting the 80 winning (cell, seed) pairs.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ablations.augmentation import augment_gaussian
from src.models import build_model
from src.prepare_train_test import prepare_train_test

TRACKS = ('opx_liq', 'opx_only')
TARGETS = ('T_C', 'P_kbar')
FEATURE_SETS = ('raw', 'alr', 'pwlr')
MODELS = ('RF', 'ERT', 'XGB', 'GB', 'CatBoost', 'LightGBM', 'ElasticNet', 'MLP')
SEEDS = list(range(42, 62))
AUG_N_COPIES = 15
AUG_REL_NOISE = 0.03

RESULTS_CSV = ROOT / 'results' / 'augmentation_ablation_opx_multiseed_results.csv'
CHECKPOINT_PKL = ROOT / 'results' / 'augmentation_ablation_opx_checkpoint.pkl'
BEST_PARAMS_JSON = ROOT / 'results' / 'optuna_best_params_opx.json'


def load_best_params_index() -> dict:
    with open(BEST_PARAMS_JSON) as f:
        payload = json.load(f)
    idx = {}
    for r in payload['results']:
        idx[(r['model'], r['target'], r['track'], r['feature_set'])] = r['best_params']
    return idx


def _load_done_cells() -> set:
    """Read already-completed (track, target, feature_set, model, seed) cells."""
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        return set()
    return set(
        (r.track, r.target, r.feature_set, r.model, int(r.seed))
        for r in df.itertuples(index=False)
    )


def _append_row(row: dict) -> None:
    header = not RESULTS_CSV.exists()
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV, mode='a', header=header, index=False)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _save_checkpoint(done: set, wall_start: float, n_done: int) -> None:
    with open(CHECKPOINT_PKL, 'wb') as f:
        pickle.dump({
            'done': done,
            'wall_start': wall_start,
            'n_done': n_done,
            'saved_at': time.time(),
        }, f)


def main(limit: int | None = None, timing_only: bool = False):
    best_params_idx = load_best_params_index()
    done = _load_done_cells()
    print(f'Loaded {len(done)} already-completed cells from CSV.')

    all_cells = list(product(TRACKS, TARGETS, FEATURE_SETS, MODELS, SEEDS))
    total = len(all_cells)
    print(f'Total plan: {total} cells.')

    wall_start = time.time()
    n_processed = 0
    n_skipped = 0

    # Cache augmented train data per (track, target, feature_set, seed) to
    # avoid regenerating the noise matrix 8 times (once per model).
    aug_cache: dict = {}

    # Cache splits per (track, target, feature_set) to avoid re-loading
    # the parquet 40 times per combo.
    splits_cache: dict = {}

    for i, (track, target, feature_set, model, seed) in enumerate(all_cells):
        if limit is not None and n_processed >= limit:
            break
        key = (track, target, feature_set, model, seed)
        if key in done:
            n_skipped += 1
            continue
        t0 = time.time()

        splits_key = (track, target, feature_set)
        if splits_key not in splits_cache:
            splits = prepare_train_test('opx', track, target, feature_set)
            splits_cache[splits_key] = splits
        splits = splits_cache[splits_key]

        aug_key = (track, target, feature_set, seed)
        if aug_key not in aug_cache:
            X_aug, y_aug, _cit_aug = augment_gaussian(
                splits['X_tr'], splits['y_tr'], splits['groups_tr'],
                n_copies=AUG_N_COPIES, rel_noise=AUG_REL_NOISE, seed=seed,
                clip_nonneg=True,
            )
            aug_cache[aug_key] = (X_aug, y_aug)
            # Bound cache size: evict once we change (track,target,feature_set).
            for k in list(aug_cache.keys()):
                if k[:3] != splits_key:
                    del aug_cache[k]

        X_aug, y_aug = aug_cache[aug_key]

        params_key = (model, target, track, feature_set)
        if params_key not in best_params_idx:
            print(f'  SKIP {key}: no best_params')
            continue
        best_params = best_params_idx[params_key]

        try:
            est = build_model(model, best_params, seed=seed)
            est.fit(X_aug, y_aug)
            y_pred_te = est.predict(splits['X_te'])
            test_rmse = _rmse(splits['y_te'], y_pred_te)
        except Exception as exc:
            print(f'  FAIL {key}: {exc!r}')
            continue
        elapsed = time.time() - t0

        row = {
            'pipeline': 'opx',
            'track': track, 'target': target, 'feature_set': feature_set,
            'model': model, 'seed': int(seed),
            'aug_n_copies': AUG_N_COPIES, 'aug_rel_noise': AUG_REL_NOISE,
            'test_rmse': test_rmse, 'elapsed_s': float(elapsed),
        }
        _append_row(row)

        done.add(key)
        n_processed += 1

        if n_processed % 5 == 0 or timing_only:
            total_elapsed = time.time() - wall_start
            per_cell = total_elapsed / n_processed if n_processed else float('nan')
            remaining = (total - len(done)) * per_cell
            print(
                f'[{len(done):4d}/{total}] {track}/{target}/{feature_set}/{model}/seed={seed} '
                f'RMSE={test_rmse:.3f} ({elapsed:.1f}s) '
                f'ETA {remaining/3600:.2f}h'
            )

        if n_processed % 50 == 0:
            _save_checkpoint(done, wall_start, n_processed)

    # Final checkpoint.
    _save_checkpoint(done, wall_start, n_processed)

    total_elapsed = time.time() - wall_start
    print('---')
    print(f'Processed {n_processed} cells this run, skipped {n_skipped} already-done.')
    print(f'Total in CSV now: {len(done)}.')
    print(f'Wall: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most this many cells (for timing probe).')
    ap.add_argument('--timing-only', action='store_true',
                    help='Print per-cell timing; imply small limit.')
    args = ap.parse_args()
    if args.timing_only and args.limit is None:
        args.limit = 8
    main(limit=args.limit, timing_only=args.timing_only)
