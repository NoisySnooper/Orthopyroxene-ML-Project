#!/usr/bin/env python3
"""Phase G Chunk C probe: per-seed per-regime RMSE for the opx_liq canonical
base models.

Why this exists:
    The Chunk B claims audit (bootstrap-on-residuals CI, single seed) quantifies
    test-set sampling noise only. It does NOT quantify model-fit stochasticity
    across seeds. The existing 20-seed multi-seed CSV
    (results/v10_opx_multiseed_results.csv) logs aggregate test RMSE per
    (cell, seed) but NOT predictions or per-regime breakdowns, so it cannot
    be reused to build per-regime seed-axis CIs. This probe refits ONLY the
    two opx_liq canonical cells at all 20 SPLIT_SEEDS (42..61), saves every
    per-sample prediction, and emits per-seed per-regime RMSE. Chunk C's
    robust audit then requires non-overlap on BOTH axes before issuing an
    "outperforms Putirka" verdict.

    Canonical cells (opx_liq, per the Chunk B fig24/fig25 configuration):
      * ElasticNet / raw / T_C
      * MLP        / raw / P_kbar

Outputs:
    results/v10_chunkC_perseed_predictions.csv   long, one row per
                                                 (seed, target, sample_idx)
    results/v10_chunkC_perseed_regime_rmse.csv   long, one row per
                                                 (seed, target, regime)
    results/v10_chunkC_perseed_aggregate_rmse.csv
                                                 one row per (seed, target),
                                                 cross-check vs the existing
                                                 multi-seed summary.

Log: logs/v10_phase_g_chunkC_probe.log

Scope note: deliberately narrow to opx_liq canonical bases. Extending to
opx_only / cpx / twopx / universal, or to non-canonical cells, is a
follow-up. Honesty-bar verdicts only apply to the claims actually made in
the opx manuscript; Chunk C therefore scopes to those.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import LOGS, RESULTS, SPLIT_SEEDS
from src.data import load_opx_liq, load_splits
from src.evaluation import assign_p_regime
from src.models import build_model
from src.opx_tb_analysis import load_best_params, prepare_train_test

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_chunkC_probe.log'
MANIFEST = RESULTS / 'v10_optuna_best_params_opx.json'

CANONICAL_CELLS = [
    # (model, target, track, feature_set)
    #
    # Cells to probe = union over (regime, target) of the per-regime-best cell
    # selected by nb04's regime benchmark (results/v10_opx_per_regime_benchmark.csv).
    # The Chunk B audit's only "outperforms Putirka" verdict is
    # shallow_crustal / P_kbar / ElasticNet/raw. The other cells below are
    # probed so the robust audit can report the seed-axis spread for EVERY
    # row of the existing claims audit, not only the sole winning row.
    #
    # T_C:
    ('ERT',        'T_C',    'opx_liq', 'pwlr'),  # shallow_crustal best
    ('ElasticNet', 'T_C',    'opx_liq', 'raw'),   # deep, litho best
    ('ElasticNet', 'T_C',    'opx_liq', 'pwlr'),  # deeper_mantle best (n<20)
    #
    # P_kbar:
    ('ElasticNet', 'P_kbar', 'opx_liq', 'raw'),   # shallow_crustal best  <-- headline claim
    ('MLP',        'P_kbar', 'opx_liq', 'raw'),   # deep best
    ('CatBoost',   'P_kbar', 'opx_liq', 'raw'),   # litho best
    ('MLP',        'P_kbar', 'opx_liq', 'alr'),   # deeper_mantle best (n<20)
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def load_test_p_kbar(track='opx_liq'):
    """Return the P_kbar vector for the test split of `track`.

    Needed to assign regimes even when the target is T_C; regime labels are
    always derived from the sample's true pressure, not its target value.
    """
    df = load_opx_liq() if track == 'opx_liq' else None
    if df is None:
        raise ValueError(f'unsupported track: {track}')
    _, te_idx = load_splits(track)
    return df.iloc[te_idx]['P_kbar'].to_numpy(dtype=float)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'a', encoding='utf-8')
    try:
        _log(f'START seeds={SPLIT_SEEDS} cells={CANONICAL_CELLS}', fh)

        best = load_best_params(MANIFEST)
        p_true_te = load_test_p_kbar('opx_liq')
        regimes_te = assign_p_regime(p_true_te)
        n_te = len(p_true_te)
        _log(f'opx_liq test n={n_te}. Regime histogram: '
             f'{pd.Series(regimes_te).value_counts().to_dict()}', fh)

        pred_rows = []
        regime_rows = []
        agg_rows = []

        for (model, target, track, feature_set) in CANONICAL_CELLS:
            key = (model, target, track, feature_set)
            if key not in best:
                raise RuntimeError(f'no best_params for {key} in {MANIFEST}')
            best_params = best[key]['best_params']

            splits = prepare_train_test(track, target, feature_set)
            X_tr, y_tr = splits['X_tr'], splits['y_tr']
            X_te, y_te = splits['X_te'], splits['y_te']
            assert len(y_te) == n_te, (len(y_te), n_te)

            _log(f'[{model}/{feature_set}/{target}/{track}] X_tr={X_tr.shape} '
                 f'X_te={X_te.shape}', fh)

            for s in SPLIT_SEEDS:
                t0 = time.time()
                est = build_model(model, best_params, seed=int(s))
                est.fit(X_tr, y_tr)
                y_hat = est.predict(X_te)

                # Sample-level rows
                for i in range(n_te):
                    pred_rows.append({
                        'seed':        int(s),
                        'model':       model,
                        'target':      target,
                        'track':       track,
                        'feature_set': feature_set,
                        'sample_idx':  int(i),
                        'P_kbar_true': float(p_true_te[i]),
                        'regime':      regimes_te[i],
                        'y_true':      float(y_te[i]),
                        'y_pred':      float(y_hat[i]),
                    })

                # Per-regime RMSE
                for regime in np.unique(regimes_te):
                    mask = (regimes_te == regime)
                    n_r = int(mask.sum())
                    if n_r == 0:
                        continue
                    rmse_r = float(np.sqrt(mean_squared_error(
                        y_te[mask], y_hat[mask])))
                    regime_rows.append({
                        'seed':        int(s),
                        'model':       model,
                        'target':      target,
                        'track':       track,
                        'feature_set': feature_set,
                        'regime':      regime,
                        'n':           n_r,
                        'rmse':        rmse_r,
                    })

                # Aggregate (cross-check vs existing multi-seed summary)
                rmse_all = float(np.sqrt(mean_squared_error(y_te, y_hat)))
                agg_rows.append({
                    'seed':        int(s),
                    'model':       model,
                    'target':      target,
                    'track':       track,
                    'feature_set': feature_set,
                    'n':           n_te,
                    'rmse':        rmse_all,
                })
                _log(f'  seed={s} rmse={rmse_all:.3f}  '
                     f'elapsed={time.time()-t0:.1f}s', fh)

        # Write outputs.
        pred_df = pd.DataFrame(pred_rows)
        regime_df = pd.DataFrame(regime_rows)
        agg_df = pd.DataFrame(agg_rows)

        out_pred = RESULTS / 'v10_chunkC_perseed_predictions.csv'
        out_reg  = RESULTS / 'v10_chunkC_perseed_regime_rmse.csv'
        out_agg  = RESULTS / 'v10_chunkC_perseed_aggregate_rmse.csv'
        pred_df.to_csv(out_pred, index=False)
        regime_df.to_csv(out_reg, index=False)
        agg_df.to_csv(out_agg, index=False)

        _log(f'wrote {out_pred} rows={len(pred_df)}', fh)
        _log(f'wrote {out_reg} rows={len(regime_df)}', fh)
        _log(f'wrote {out_agg} rows={len(agg_df)}', fh)

        # Sanity check vs existing aggregate multi-seed summary.
        try:
            summary = pd.read_csv(RESULTS / 'v10_opx_multiseed_summary.csv')
            for (model, target, track, feature_set) in CANONICAL_CELLS:
                mine = agg_df[(agg_df.model == model) &
                              (agg_df.target == target) &
                              (agg_df.track == track) &
                              (agg_df.feature_set == feature_set)]
                ref = summary[(summary.model == model) &
                              (summary.target == target) &
                              (summary.track == track) &
                              (summary.feature_set == feature_set)]
                if len(ref) == 1:
                    delta = float(mine.rmse.mean()) - float(ref['mean'].iloc[0])
                    _log(f'SANITY {model}/{feature_set}/{target}: '
                         f'chunkC mean={mine.rmse.mean():.4f} '
                         f'ref mean={float(ref["mean"].iloc[0]):.4f} '
                         f'delta={delta:+.4f}', fh)
        except Exception as e:
            _log(f'SANITY check skipped: {e}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
