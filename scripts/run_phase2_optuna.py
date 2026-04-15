"""Standalone overnight runner for the Phase 3.3b Optuna sweep.

Runs the 48-study TPE sweep without launching Jupyter. Resumable: if a
`study_{...}.joblib` already exists, the study is loaded and skipped
(TPE studies are stateful; we re-use the saved study as the source of
truth rather than reopening and appending trials, matching the
tune-once-then-freeze discipline documented in docs/optuna_strategy.md).

Usage (from repo root, with the project .venv active):

    python scripts/run_phase2_optuna.py                  # full sweep
    python scripts/run_phase2_optuna.py --dry-run        # list studies, no fit
    python scripts/run_phase2_optuna.py --only RF --only XGB

Produces the same artifacts as NB03 Phase 3.3b:
  - results/optuna_studies/study_*.joblib  (48 files at full completion)
  - results/nb03_optuna_best_params.json
  - results/nb03_optuna_best_params_partial.json (every 10 studies)
  - results/nb03_hyperparameter_search.csv

Run NB03 Phase 3.4 onward after this completes. The notebook cell will
pick the frozen_params_store up from results/nb03_optuna_best_params.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from config import (  # noqa: E402
    DATA_PROC, DATA_SPLITS, RESULTS, LOGS,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_PER_TRIAL, OPTUNA_N_JOBS_INNER,
    OPTUNA_SEED, OPTUNA_STUDIES_DIR, OPTUNA_BEST_PARAMS_FILE,
    OPTUNA_BEST_PARAMS_PARTIAL_FILE, OPTUNA_MODELS, OPTUNA_TARGETS,
    OPTUNA_TRACKS, OPTUNA_FEATURE_SETS,
)
from src.features import build_feature_matrix  # noqa: E402
from src.optuna_search import optuna_search  # noqa: E402


def _setup_logger():
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / 'run_phase2_optuna.log'
    logger = logging.getLogger('run_phase2_optuna')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(sh)
    return logger, log_path


def _load_training_frames():
    # Reproduce NB03's Phase 3.3b outer split: GroupShuffleSplit by Citation
    # with seed OPTUNA_SEED on each track's cleaned parquet.
    df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
    df_opx = pd.read_parquet(DATA_PROC / 'opx_clean_opx_only.parquet')
    return df_liq, df_opx


def _outer_split(df_track: pd.DataFrame, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_pos, te_pos = next(gss.split(df_track,
                                    groups=df_track['Citation'].values))
    return df_track.iloc[tr_pos].copy(), df_track.iloc[te_pos].copy()


def main(only_models=None, dry_run=False, n_trials=None):
    logger, log_path = _setup_logger()
    OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    n_trials = OPTUNA_N_TRIALS if n_trials is None else int(n_trials)
    only_models = tuple(only_models) if only_models else OPTUNA_MODELS

    logger.info('=' * 72)
    logger.info('run_phase2_optuna: n_trials=%d, models=%s, dry_run=%s',
                n_trials, only_models, dry_run)

    df_liq, df_opx = _load_training_frames()
    track_frames = {'opx_liq': (df_liq, True), 'opx_only': (df_opx, False)}

    frozen_params_store = {}
    search_records = []
    study_counter = 0
    t0 = time.time()

    total = len(OPTUNA_TRACKS) * len(OPTUNA_FEATURE_SETS) * len(only_models) * len(OPTUNA_TARGETS)
    logger.info('Planned studies: %d', total)

    for track in OPTUNA_TRACKS:
        df_track, use_liq = track_frames[track]
        df_train, _df_test = _outer_split(df_track, OPTUNA_SEED)
        groups_tr = df_train['Citation'].values
        for feat in OPTUNA_FEATURE_SETS:
            X_tr, _ = build_feature_matrix(df_train, feat, use_liq)
            for model_name in only_models:
                for target in OPTUNA_TARGETS:
                    study_name = f'{model_name}|{target}|{track}|{feat}'
                    study_path = OPTUNA_STUDIES_DIR / (
                        f'study_{model_name}_{target}_{track}_{feat}.joblib'
                    )
                    if dry_run:
                        logger.info('DRY %s -> %s', study_name, study_path.name)
                        continue

                    # Resume: if the study file exists, skip re-running.
                    if study_path.exists():
                        try:
                            study = joblib.load(study_path)
                            frozen_key = (track, feat, model_name, target)
                            frozen_params_store[frozen_key] = dict(study.best_params)
                            logger.info('SKIP existing %s (best_value=%.3f)',
                                        study_name, float(study.best_value))
                            study_counter += 1
                            continue
                        except Exception as e:
                            logger.warning('Could not reload %s: %s - rerunning',
                                           study_path.name, e)

                    y_tr = df_train[target].values
                    tstart = time.time()
                    try:
                        out = optuna_search(
                            model_name, X_tr, y_tr, groups_tr,
                            n_trials=n_trials,
                            seed=OPTUNA_SEED,
                            timeout_per_trial=OPTUNA_TIMEOUT_PER_TRIAL,
                            n_jobs_inner=OPTUNA_N_JOBS_INNER,
                            study_save_path=study_path,
                            study_name=study_name,
                        )
                        frozen_key = (track, feat, model_name, target)
                        frozen_params_store[frozen_key] = out['best_params']
                        study = out['study']
                        completed = [t for t in study.trials if t.state.name == 'COMPLETE']
                        pruned   = [t for t in study.trials if t.state.name == 'PRUNED']
                        failed   = [t for t in study.trials if t.state.name == 'FAIL']
                        search_records.append({
                            'track': track, 'feature_set': feat,
                            'model_name': model_name, 'target': target,
                            'best_rmse_cv': out['best_score'],
                            'best_trial':   out['best_trial'],
                            'n_trials_completed': len(completed),
                            'n_trials_pruned':    len(pruned),
                            'n_trials_failed':    len(failed),
                            'best_params':        str(out['best_params']),
                            'elapsed_s':          round(time.time() - tstart, 1),
                        })
                        logger.info('OK %s rmse=%.3f pruned=%d failed=%d (%.0fs)',
                                    study_name, out['best_score'],
                                    len(pruned), len(failed),
                                    time.time() - tstart)
                    except Exception as e:
                        logger.exception('FAIL %s: %s: %s',
                                         study_name, type(e).__name__, e)

                    study_counter += 1
                    if study_counter % 10 == 0:
                        _partial = {'||'.join(k): v for k, v in frozen_params_store.items()}
                        with open(RESULTS / OPTUNA_BEST_PARAMS_PARTIAL_FILE, 'w') as f:
                            json.dump(_partial, f, default=str)
                        logger.info('CHECKPOINT %d/%d studies done, partial saved',
                                    study_counter, total)

    if dry_run:
        logger.info('Dry run done. %d studies listed.', total)
        return

    # Final persistence.
    frozen_flat = {'||'.join(k): {kk: (vv.item() if isinstance(vv, np.generic) else vv)
                                   for kk, vv in v.items()}
                   for k, v in frozen_params_store.items()}
    with open(RESULTS / OPTUNA_BEST_PARAMS_FILE, 'w') as f:
        json.dump(frozen_flat, f, default=str)
    if search_records:
        pd.DataFrame(search_records).to_csv(
            RESULTS / 'nb03_hyperparameter_search.csv', index=False)

    elapsed_h = (time.time() - t0) / 3600
    logger.info('Complete: %d studies in %.2fh. Log: %s',
                len(frozen_params_store), elapsed_h, log_path)
    print(f'Phase 2 Optuna sweep complete: {len(frozen_params_store)} studies '
          f'in {elapsed_h:.2f}h. See {log_path}.')


def _parse_args():
    p = argparse.ArgumentParser(description='Phase 2 Optuna overnight runner')
    p.add_argument('--only', action='append', default=[],
                   help='restrict to one or more models (RF, ERT, XGB, GB); repeatable')
    p.add_argument('--dry-run', action='store_true',
                   help='list planned studies and exit')
    p.add_argument('--n-trials', type=int, default=None,
                   help='override OPTUNA_N_TRIALS (default from config)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(only_models=args.only, dry_run=args.dry_run, n_trials=args.n_trials)
