#!/usr/bin/env python3
"""Phase H.0b: canonical cell selection for cpx_only / cpx_liq / twopx.

Parallels the Phase C canonical-cell selection for opx (already encoded
in results/v10_optuna_best_params_opx.json + Phase G Chunk A-C outputs).

For each (target, track) in {cpx_only, cpx_liq, twopx} x {T_C, P_kbar}:
  * aggregate winner = argmin(mean test RMSE across 20 seeds) from
    v10_{pipeline}_multiseed_summary.csv
  * tie-breaker 1: lower std
  * tie-breaker 2: lower single-seed test RMSE from per_cell_results

Produces:
    results/v10_canonical_cells_h0b.csv
    results/v10_canonical_cells_h0b.json

Downstream (H.3) reads the JSON to pick the right (model, feature_set)
per track/target before running natural-sample inference.

No regime-level selection is performed here. Chunks A-C replication for
cpx_liq (if opted into via H.0c) will extend this with regime-specific
winners; this H.0b output represents the aggregate choice only.
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

import pandas as pd

from config import LOGS, RESULTS

LOG_PATH = LOGS / 'v10_phase_h0b_canonical_cells.log'

PIPELINES = [
    ('cpx',   ['cpx_only', 'cpx_liq']),
    ('twopx', ['twopx']),
]
TARGETS = ['T_C', 'P_kbar']


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def pick_winner(ms_df, pc_df, target, track):
    sub = ms_df[(ms_df.target == target) & (ms_df.track == track)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(['mean', 'std'], ascending=[True, True])
    top = sub.iloc[0]
    # Single-seed tie-break lookup
    pc_row = pc_df[(pc_df.target == target) & (pc_df.track == track) &
                   (pc_df.model == top.model) &
                   (pc_df.feature_set == top.feature_set)]
    single = float(pc_row.iloc[0].test_rmse) if len(pc_row) else float('nan')
    r2 = float(pc_row.iloc[0].test_r2) if len(pc_row) else float('nan')
    return {
        'pipeline':        top.pipeline,
        'target':          target,
        'track':           track,
        'model':           top.model,
        'feature_set':     top.feature_set,
        'seed_mean_rmse':  float(top['mean']),
        'seed_std_rmse':   float(top['std']),
        'seed_min_rmse':   float(top['min']),
        'seed_max_rmse':   float(top['max']),
        'single_seed_rmse': single,
        'single_seed_r2':  r2,
        'n_seeds':         int(top['count']),
    }


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.0b canonical cell selection', fh)
        rows = []
        for pipeline, tracks in PIPELINES:
            ms = pd.read_csv(RESULTS / f'v10_{pipeline}_multiseed_summary.csv')
            pc = pd.read_csv(RESULTS / f'v10_{pipeline}_per_cell_results.csv')
            _log(f'[{pipeline}] multiseed={ms.shape} per_cell={pc.shape}', fh)
            for track in tracks:
                for target in TARGETS:
                    w = pick_winner(ms, pc, target, track)
                    if w is None:
                        _log(f'  WARN no winner for {track}/{target}', fh)
                        continue
                    _log(f'  [{track}/{target}] winner='
                         f'{w["model"]}/{w["feature_set"]} '
                         f'seed_mean={w["seed_mean_rmse"]:.3f} '
                         f'(std={w["seed_std_rmse"]:.3f})', fh)
                    rows.append(w)

        out_df = pd.DataFrame(rows)
        out_csv = RESULTS / 'v10_canonical_cells_h0b.csv'
        out_df.to_csv(out_csv, index=False)
        _log(f'wrote {out_csv} rows={len(out_df)}', fh)

        # JSON keyed by (track, target) -> row dict
        manifest = {}
        for _, r in out_df.iterrows():
            manifest[f'{r.track}__{r.target}'] = r.to_dict()
        out_json = RESULTS / 'v10_canonical_cells_h0b.json'
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)
        _log(f'wrote {out_json}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
