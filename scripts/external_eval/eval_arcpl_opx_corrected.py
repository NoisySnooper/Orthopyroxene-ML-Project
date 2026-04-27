#!/usr/bin/env python3
"""ArcPL opx-liq external evaluation: v3 shipped ML (pre + post) vs Putirka.

Applies the v3-shipped bias-corrected ML pipeline and the Putirka 2008
thermobarometers to the reconstructed opx-liq ArcPL holdout
(`src.external.arcpl_opx.load_arcpl_opx_liq`, n ~= 197), and emits a
per-regime RMSE table. The ML side uses the canonical seed-42
checkpoints in `results/bias_correction/checkpoints/`, applies Form A or
Form B depending on the v3 winner, and falls back to identity when v3
did not promote a shipped form. The Putirka side uses
`predict_putirka_opx_liq` / `predict_putirka_opx_only` from
`src.external_models` (eq28a for opx-liq T, eq29a for opx-liq P, eq29c
for opx-only P; opx-only T has no Putirka calibration and is emitted as
NaN).

Output:
  results/arcpl_opx_corrected_per_regime.csv

Schema:
  track, target, regime, n, source, rmse, rmse_lo, rmse_hi
  source in {'ml_pre', 'ml_post', 'putirka'}

Bootstrap: 2000 paired resamples at SEED_BOOTSTRAP, percentile 95% CI.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config import SEED_BOOTSTRAP  # noqa: E402
from src.bias_correction import (  # noqa: E402
    FormBParams, apply_form_a, apply_form_b,
)
from src.evaluation import assign_p_regime  # noqa: E402
from src.external.arcpl_opx import (  # noqa: E402
    load_arcpl_opx_liq, to_thermobar_schema,
)
from src.external_models import (  # noqa: E402
    predict_putirka_opx_liq, predict_putirka_opx_only,
)
from src.features import build_feature_matrix  # noqa: E402

OUT_CSV = PROJECT_ROOT / 'results' / 'arcpl_opx_corrected_per_regime.csv'

REGIME_ORDER = ['shallow_crustal', 'deep_crustal_MASH',
                'lithospheric_mantle', 'deeper_mantle', 'ALL']

CELLS = [
    # (track, target, unit, use_liq_for_ml, putirka_fn_name)
    ('opx_liq',  'T_C',    'C',    True,  'opx_liq'),
    ('opx_liq',  'P_kbar', 'kbar', True,  'opx_liq'),
    ('opx_only', 'T_C',    'C',    False, None),          # no Putirka
    ('opx_only', 'P_kbar', 'kbar', False, 'opx_only'),
]


def _load_shipped_v3() -> pd.DataFrame:
    df = pd.read_csv('results/bias_correction_shipped_v3.csv')
    return df[(df.pipeline == 'opx') & (df.model != 'TabPFN')].copy()


def _load_checkpoint(track: str, target: str, model: str,
                     feature_set: str, seed: int = 42):
    path = (PROJECT_ROOT / 'results' / 'bias_correction' / 'checkpoints' /
            f'opx_{track}_{target}_{model}_{feature_set}_s{seed}.pkl')
    return joblib.load(path)


def _load_base_model(track: str, target: str, model: str, feature_set: str):
    path = (PROJECT_ROOT / 'models' / 'canonical' / 'opx' /
            f'base_{model}_{target}_{track}_{feature_set}.joblib')
    return joblib.load(path)


def _bootstrap_rmse(y: np.ndarray, yhat: np.ndarray,
                    *, n_boot: int = 2000,
                    seed: int = SEED_BOOTSTRAP) -> tuple[float, float, float, int]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    n = int(mask.sum())
    if n < 3:
        return (np.nan, np.nan, np.nan, n)
    yt = y[mask]
    yp = yhat[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2, axis=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return (rmse, float(lo), float(hi), n)


def _regime_masks(regimes: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for r in REGIME_ORDER[:-1]:
        out[r] = (regimes == r)
    out['ALL'] = np.ones(len(regimes), dtype=bool)
    return out


def _record(rows: list[dict], track: str, target: str, regime: str,
            source: str, y: np.ndarray, yhat: np.ndarray,
            mask: np.ndarray):
    rmse, lo, hi, n = _bootstrap_rmse(y[mask], yhat[mask])
    rows.append(dict(
        track=track, target=target, regime=regime, source=source,
        n=n, rmse=rmse, rmse_lo=lo, rmse_hi=hi,
    ))


def _ml_pre_post(ship_row: pd.Series, arcpl_df: pd.DataFrame,
                 track: str, target: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (y_pre, y_post, winner_v3) for one opx cell on ArcPL."""
    model = ship_row['model']
    feature_set = ship_row['feature_set']
    winner_v3 = ship_row['winner_v3']

    use_liq = (track == 'opx_liq')
    X, _ = build_feature_matrix(arcpl_df, feature_set, use_liq=use_liq)
    base = _load_base_model(track, target, model, feature_set)
    y_pre = np.asarray(base.predict(X), dtype=float)

    ck = _load_checkpoint(track, target, model, feature_set, seed=42)
    regimes_te = assign_p_regime(arcpl_df['P_kbar'].to_numpy())
    if winner_v3 == 'A':
        params_a = {r: (v['a'], v['b']) for r, v in ck.form_a_params.items()}
        y_post = apply_form_a(y_pre, regimes_te, params_a)
    elif winner_v3 == 'B' and ck.form_b_params is not None:
        fb = FormBParams(**ck.form_b_params)
        y_post = apply_form_b(y_pre, fb)
    else:
        y_post = y_pre.copy()
    return y_pre, y_post, winner_v3


def main():
    arcpl = load_arcpl_opx_liq()
    arcpl_tb = to_thermobar_schema(arcpl)

    ship_v3 = _load_shipped_v3()

    rows: list[dict] = []
    regimes = assign_p_regime(arcpl['P_kbar'].to_numpy())
    masks = _regime_masks(regimes)
    y_T = arcpl['T_C'].to_numpy(dtype=float)
    y_P = arcpl['P_kbar'].to_numpy(dtype=float)

    # For Putirka opx-only P we need T_K; use observed T.
    T_K_obs = y_T + 273.15

    for track, target, _unit, _use_liq, putirka_fn in CELLS:
        ship_row = ship_v3[(ship_v3.track == track)
                           & (ship_v3.target == target)]
        if ship_row.empty:
            print(f'[warn] no v3 ship row for {track}/{target}; skipping')
            continue
        ship_row = ship_row.iloc[0]

        y_true = y_T if target == 'T_C' else y_P
        y_pre, y_post, winner = _ml_pre_post(
            ship_row, arcpl, track, target)

        # Putirka
        if putirka_fn == 'opx_liq':
            y_put = predict_putirka_opx_liq(
                arcpl_tb, target=('T' if target == 'T_C' else 'P'),
                P_kbar=(y_P if target == 'T_C' else None),
                T_K=(T_K_obs if target == 'P_kbar' else None))
        elif putirka_fn == 'opx_only':
            y_put = predict_putirka_opx_only(
                arcpl_tb, target=('T' if target == 'T_C' else 'P'),
                T_K=(T_K_obs if target == 'P_kbar' else None))
        else:
            y_put = np.full(len(arcpl), np.nan, dtype=float)

        for regime, mask in masks.items():
            _record(rows, track, target, regime, 'ml_pre',   y_true, y_pre,  mask)
            _record(rows, track, target, regime, 'ml_post',  y_true, y_post, mask)
            _record(rows, track, target, regime, 'putirka',  y_true, y_put,  mask)

        print(f'{track}/{target}: winner_v3={winner}, '
              f'pre RMSE={_bootstrap_rmse(y_true, y_pre)[0]:.2f}, '
              f'post RMSE={_bootstrap_rmse(y_true, y_post)[0]:.2f}, '
              f'putirka RMSE={_bootstrap_rmse(y_true, y_put)[0]}')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}  ({len(out)} rows)')

    meta = {
        'source': 'src.external.arcpl_opx.load_arcpl_opx_liq',
        'n_arcpl': int(len(arcpl)),
        'v3_shipped_cells': ship_v3[[
            'track', 'target', 'model', 'feature_set', 'winner_v3'
        ]].to_dict(orient='records'),
        'seed_bootstrap': SEED_BOOTSTRAP,
        'n_boot': 2000,
    }
    (OUT_CSV.with_suffix('.json')).write_text(
        json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
