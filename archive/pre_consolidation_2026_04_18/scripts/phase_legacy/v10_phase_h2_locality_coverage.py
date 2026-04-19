#!/usr/bin/env python3
"""Phase H.2 diagnostic: count natural opx samples inside each curated-
locality bounding box, and report opx_only predicted T/P summaries.

Drives the n>=20 floor decision for per-locality RMSE claims (Phase H.6).
Localities with fewer than 20 samples will be pooled by tectonic setting
instead.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import DATA_NATURAL, RESULTS, LOGS

LOC_CSV = DATA_NATURAL / 'curated_localities.csv'
PRED_CSV = RESULTS / 'v10_natural_opx_opx_only_predictions.csv'
OUT_CSV = RESULTS / 'v10_phase_h2_locality_coverage.csv'
LOG_PATH = LOGS / 'v10_phase_h2_locality_coverage.log'


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START H.2 locality coverage diagnostic', fh)
        loc = pd.read_csv(LOC_CSV)
        pred = pd.read_csv(PRED_CSV, low_memory=False)
        _log(f'{len(loc)} localities; {len(pred)} natural opx predictions', fh)

        rows = []
        for _, r in loc.iterrows():
            mask = (pred['lat'].between(r.lat_min, r.lat_max)
                    & pred['lon'].between(r.lon_min, r.lon_max))
            sub = pred[mask]
            n = len(sub)
            tec_match = ((sub['TECTONIC SETTING'].fillna('') ==
                         r.tectonic_setting).sum() if n else 0)
            row = {
                'locality': r.locality,
                'tectonic_setting_expected': r.tectonic_setting,
                'n_in_bbox': n,
                'n_tectonic_matched': tec_match,
                'clears_n20_floor': n >= 20,
                'pred_T_C_mean': sub['pred_T_C_opx_only'].mean() if n else None,
                'pred_T_C_std': sub['pred_T_C_opx_only'].std() if n else None,
                'pred_P_kbar_mean': sub['pred_P_kbar_opx_only'].mean() if n else None,
                'pred_P_kbar_std': sub['pred_P_kbar_opx_only'].std() if n else None,
                'ood_pct': 100 * sub['ood_flag'].mean() if n else None,
                'expected_T_C_low': r.expected_T_C_low,
                'expected_T_C_high': r.expected_T_C_high,
                'expected_P_kbar_low': r.expected_P_kbar_low,
                'expected_P_kbar_high': r.expected_P_kbar_high,
            }
            rows.append(row)

        out = pd.DataFrame(rows)
        out.to_csv(OUT_CSV, index=False)
        _log(f'wrote {OUT_CSV}', fh)

        n_cleared = out.clears_n20_floor.sum()
        _log(f'localities clearing n>=20 floor: {n_cleared}/{len(out)}', fh)
        show = out[['locality', 'n_in_bbox', 'n_tectonic_matched',
                    'clears_n20_floor', 'pred_T_C_mean',
                    'pred_P_kbar_mean', 'ood_pct']].copy()
        _log('\n' + show.to_string(index=False), fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
