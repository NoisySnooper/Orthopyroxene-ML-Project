#!/usr/bin/env python3
"""Phase H.3a follow-on: descriptive per-regime summary of the 53k opx_only
natural predictions.

Uses the pre-registered P-regime bins from config.py (0/5/15/30/100 kbar,
labels shallow_crustal / deep_crustal_MASH / lithospheric_mantle /
deeper_mantle). Right-open bin edges per pre-registration.

IMPORTANT: descriptive only. Per Phase G collision 4, no per-regime RMSE
claims on natural samples (that's calibration-domain only). Regime
assignment here is for visualization and narrative context, not quantitative
error reporting.
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

from config import (RESULTS, LOGS, P_REGIME_BIN_EDGES_KBAR,
                    P_REGIME_LABELS, P_REGIME_MIN_N_FOR_CLAIMS)

PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
OUT_CSV = RESULTS / 'v10_phase_h3a_regime_summary.csv'
LOG_PATH = LOGS / 'v10_phase_h3a_regime_summary.log'


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def assign_regime(p_kbar):
    # Right-open bins: edges [0, 5, 15, 30, 100] -> labels 0..3
    p = np.asarray(p_kbar, dtype=float)
    idx = np.digitize(p, P_REGIME_BIN_EDGES_KBAR, right=False) - 1
    idx = np.clip(idx, 0, len(P_REGIME_LABELS) - 1)
    return np.array(P_REGIME_LABELS)[idx]


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.3a regime summary', fh)
        pred = pd.read_csv(PRED_CSV, low_memory=False)
        _log(f'loaded {len(pred)} natural opx predictions', fh)

        pred['regime'] = assign_regime(pred['pred_P_kbar_opx_only'])
        _log('regime labels assigned (descriptive; no RMSE claims)', fh)

        grp = (pred.groupby('regime')
                   .agg(n=('pred_T_C_opx_only', 'size'),
                        T_mean=('pred_T_C_opx_only', 'mean'),
                        T_std=('pred_T_C_opx_only', 'std'),
                        T_p10=('pred_T_C_opx_only',
                               lambda s: s.quantile(0.10)),
                        T_p90=('pred_T_C_opx_only',
                               lambda s: s.quantile(0.90)),
                        P_mean=('pred_P_kbar_opx_only', 'mean'),
                        P_std=('pred_P_kbar_opx_only', 'std'),
                        P_p10=('pred_P_kbar_opx_only',
                               lambda s: s.quantile(0.10)),
                        P_p90=('pred_P_kbar_opx_only',
                               lambda s: s.quantile(0.90)),
                        ood_pct=('ood_flag',
                                 lambda s: 100 * s.mean()))
                   .reindex(P_REGIME_LABELS))

        grp['meets_n_floor'] = grp['n'] >= P_REGIME_MIN_N_FOR_CLAIMS
        grp.to_csv(OUT_CSV)
        _log(f'wrote {OUT_CSV}', fh)

        _log('\n' + grp.round(2).to_string(), fh)

        tec_x_regime = pd.crosstab(pred['TECTONIC SETTING'], pred['regime'])
        tec_x_regime = tec_x_regime.reindex(columns=P_REGIME_LABELS,
                                             fill_value=0)
        _log('\n\ntectonic setting x regime crosstab:\n'
             + tec_x_regime.to_string(), fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
