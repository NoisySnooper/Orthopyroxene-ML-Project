#!/usr/bin/env python3
"""Phase G.7 A5: Form B parameter stability across CV reseeds.

For each cell (the aggregate-best (model, feature_set) per (track,
target)), hold the model seed fixed at 42 and re-run the 10-fold
StratifiedGroupKFold OOF under 5 different CV random_states. Re-fit
Form B on each OOF and record (alpha_L, alpha_R, a_L, a_R, s_L, s_R).

This probes whether Form B breakpoint quantiles and tail slopes are
stable to the choice of fold boundaries. Small variance (e.g. SD of
alpha_L < 0.05 in quantile units) -> Form B is trustworthy; large
variance -> the correction is an artefact of one particular split.

Output:
    results/v10_bias_correction_form_b_stability.csv

One row per (cell, cv_seed) with all Form B parameters plus the
training-OOF MSE at the fitted parameters. A follow-up row per cell
with mean / std across CV seeds helps readers gauge stability at a
glance.
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
from src.bias_correction import fit_form_b, oof_predict
from src.prepare_train_test import prepare_train_test
from src.opx_tb_analysis import load_best_params

LOG_PATH = LOGS / 'v10_phase_g_form_b_stability.log'
OUT_CSV = RESULTS / 'v10_bias_correction_form_b_stability.csv'

MODEL_SEED = 42
CV_SEEDS = [42, 43, 44, 45, 46]

# Pipeline -> (summary CSV, manifest)
PIPELINE_META = {
    'opx_liq':  ('v10_opx_multiseed_summary.csv',
                 'v10_optuna_best_params_opx.json',  'opx'),
    'opx_only': ('v10_opx_multiseed_summary.csv',
                 'v10_optuna_best_params_opx.json',  'opx'),
    'cpx_liq':  ('v10_cpx_multiseed_summary.csv',
                 'v10_optuna_best_params_cpx.json',  'cpx'),
    'cpx_only': ('v10_cpx_multiseed_summary.csv',
                 'v10_optuna_best_params_cpx.json',  'cpx'),
}
TARGETS = ('T_C', 'P_kbar')


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def select_best_cells():
    cells = []
    for track, (summary_csv, manifest, pipeline) in PIPELINE_META.items():
        df = pd.read_csv(RESULTS / summary_csv)
        df = df[df.track == track]
        for tg in TARGETS:
            sub = df[df.target == tg]
            if sub.empty:
                continue
            best = sub.loc[sub['mean'].idxmin()]
            cells.append({
                'pipeline': pipeline, 'track': track, 'target': tg,
                'model': best['model'], 'feature_set': best['feature_set'],
                'manifest': manifest,
            })
    return cells


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'A5 Form-B stability: cv_seeds={CV_SEEDS} model_seed={MODEL_SEED}', fh)
        cells = select_best_cells()
        _log(f'n_cells={len(cells)}', fh)

        rows = []
        manifest_cache = {}

        for ci, cell in enumerate(cells, 1):
            if cell['manifest'] not in manifest_cache:
                manifest_cache[cell['manifest']] = load_best_params(
                    RESULTS / cell['manifest'])
            best = manifest_cache[cell['manifest']]
            bp_key = (cell['model'], cell['target'], cell['track'],
                      cell['feature_set'])
            best_params = best[bp_key]['best_params']

            prep = prepare_train_test(cell['pipeline'], cell['track'],
                                      cell['target'], cell['feature_set'])
            X_tr, y_tr = prep['X_tr'], prep['y_tr']
            groups_tr = prep['groups_tr']

            _log(f"[{ci}/{len(cells)}] {cell['track']}/{cell['target']} "
                 f"({cell['model']}/{cell['feature_set']})", fh)

            for cv_seed in CV_SEEDS:
                t0 = time.time()
                oof = oof_predict(cell['model'], best_params,
                                  X_tr, y_tr, groups_tr,
                                  seed=MODEL_SEED, n_folds=10, cv_seed=cv_seed)
                mask = np.isfinite(oof)
                pb = fit_form_b(y_tr[mask], oof[mask])
                row = {
                    'pipeline': cell['pipeline'], 'track': cell['track'],
                    'target': cell['target'], 'model': cell['model'],
                    'feature_set': cell['feature_set'],
                    'model_seed': MODEL_SEED, 'cv_seed': cv_seed,
                    'elapsed_s': round(time.time() - t0, 2),
                }
                if pb is not None:
                    row.update(pb.to_dict())
                    row['form_b_ok'] = True
                else:
                    row.update({
                        'alpha_L': np.nan, 'alpha_R': np.nan,
                        'a_L': np.nan, 'a_R': np.nan,
                        's_L': np.nan, 's_R': np.nan, 'oof_mse': np.nan,
                        'form_b_ok': False,
                    })
                rows.append(row)
                _log(f'  cv_seed={cv_seed}: '
                     + (f'alpha=({row["alpha_L"]:.2f},{row["alpha_R"]:.2f}) '
                        f's=({row["s_L"]:+.3f},{row["s_R"]:+.3f}) '
                        f'oof_mse={row["oof_mse"]:.3f}'
                        if pb is not None else 'no valid fit')
                     + f' elapsed={row["elapsed_s"]}s', fh)

        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV}  rows={len(df)}', fh)

        # Per-cell summary
        keys = ['pipeline', 'track', 'target', 'model', 'feature_set']
        stats = (df[df.form_b_ok]
                 .groupby(keys)[['alpha_L', 'alpha_R', 's_L', 's_R',
                                 'a_L', 'a_R', 'oof_mse']]
                 .agg(['mean', 'std']))
        _log('stability summary (mean / std across CV seeds):', fh)
        for line in stats.to_string().split('\n'):
            _log('  ' + line, fh)

        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
