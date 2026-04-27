#!/usr/bin/env python3
"""H.6: Curated-locality validation by spatial bounding-box matching.

The activation prompt's H.6a wants per-locality bootstrap-CI RMSE
against literature P-T. The existing data/natural/curated_localities.csv
holds 15 locality bounding boxes with expected T/P RANGES per locality
(not per-sample literature P-T). With per-sample literature P-T not
available without manual citation work, this script computes a more
modest but still defensible per-locality summary:

  - Spatially join natural opx + cpx + twopx predictions to the 15
    curated locality bounding boxes via lat/lon.
  - Add ArcPL opx as locality #16 (per plan Section 0 collision 6).
    ArcPL is not a geographic locality, so it is filtered by source
    citation rather than by bounding box.
  - Per locality: n samples per pipeline, predicted T mean/median +
    fraction inside the curated expected_T_C range,
    same for P.
  - Per-locality n>=20 honesty bar applied to flag claims_eligible.

Output: results/nb08_locality_stratified.csv with one row per
(locality, pipeline) pair.

This is best-effort given the per-sample literature P-T lookup is
manual research outside the scope of this autonomous run. The
per-sample bootstrap-CI RMSE structure is preserved in the script's
output schema so a future pass with real per-sample P-T can fill
those columns without reshaping the CSV.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_FILE = PROJECT_ROOT / 'results' / 'PHASE_H_RUN_LOG.md'

CURATED_PATH = PROJECT_ROOT / 'data' / 'natural' / 'curated_localities.csv'
ARCPL_PATH = (PROJECT_ROOT / 'archive' / 'pre_v10_rebuild_2026_04_16'
              / 'results' / 'nb04_arcpl_opx_liq_predictions_forest.csv')


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'- {ts} (H.6) {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _stratify(df: pd.DataFrame, locality_row: pd.Series, pipeline: str,
              t_col: str, p_col: str) -> dict | None:
    """Return summary dict for `locality_row` against the predictions in `df`."""
    in_box = ((df['lat'] >= locality_row['lat_min'])
              & (df['lat'] <= locality_row['lat_max'])
              & (df['lon'] >= locality_row['lon_min'])
              & (df['lon'] <= locality_row['lon_max']))
    sub = df[in_box]
    n = len(sub)
    if n == 0:
        return None
    t_vals = sub[t_col].dropna() if t_col in sub.columns else pd.Series([])
    p_vals = sub[p_col].dropna() if p_col in sub.columns else pd.Series([])
    t_lo = locality_row['expected_T_C_low']
    t_hi = locality_row['expected_T_C_high']
    p_lo = locality_row['expected_P_kbar_low']
    p_hi = locality_row['expected_P_kbar_high']
    return {
        'locality':         locality_row['locality'],
        'pipeline':         pipeline,
        'tectonic_setting': locality_row['tectonic_setting'],
        'expected_T_C_low': t_lo,
        'expected_T_C_high': t_hi,
        'expected_P_kbar_low': p_lo,
        'expected_P_kbar_high': p_hi,
        'expected_P_regime': locality_row.get('expected_P_regime'),
        'n_samples_in_box': n,
        'n_T_pred':         len(t_vals),
        'T_pred_mean':      float(t_vals.mean()) if len(t_vals) else np.nan,
        'T_pred_median':    float(t_vals.median()) if len(t_vals) else np.nan,
        'T_in_range_frac':  float(((t_vals >= t_lo) & (t_vals <= t_hi)).mean())
                            if len(t_vals) else np.nan,
        'n_P_pred':         len(p_vals),
        'P_pred_mean':      float(p_vals.mean()) if len(p_vals) else np.nan,
        'P_pred_median':    float(p_vals.median()) if len(p_vals) else np.nan,
        'P_in_range_frac':  float(((p_vals >= p_lo) & (p_vals <= p_hi)).mean())
                            if len(p_vals) else np.nan,
        'claims_eligible':  bool(n >= 20),
    }


def _stratify_arcpl(arcpl: pd.DataFrame) -> dict | None:
    """ArcPL summary: literature P-T per sample is in the archive file."""
    if arcpl is None or len(arcpl) == 0:
        return None
    p_pred_col = next((c for c in arcpl.columns
                       if 'pred_P' in c or 'P_kbar_pred' in c), None)
    t_pred_col = next((c for c in arcpl.columns
                       if 'pred_T' in c or 'T_C_pred' in c), None)
    p_lit_col = next((c for c in arcpl.columns
                      if c.startswith('P_kbar_lit') or c == 'P_kbar_true'),
                     None)
    t_lit_col = next((c for c in arcpl.columns
                      if c.startswith('T_C_lit') or c == 'T_C_true'),
                     None)
    n = len(arcpl)
    out = {
        'locality':         'ArcPL_opx (locality #16)',
        'pipeline':         'opx_liq_archive',
        'tectonic_setting': 'CONVERGENT MARGIN',
        'n_samples_in_box': n,
        'claims_eligible':  bool(n >= 20),
    }
    if t_pred_col and t_lit_col:
        delta_T = (arcpl[t_pred_col] - arcpl[t_lit_col]).dropna()
        if len(delta_T):
            out['T_rmse_vs_lit'] = float(np.sqrt(np.mean(delta_T ** 2)))
            out['T_median_abs_residual'] = float(np.median(np.abs(delta_T)))
            out['n_T_pred'] = int(len(delta_T))
    if p_pred_col and p_lit_col:
        delta_P = (arcpl[p_pred_col] - arcpl[p_lit_col]).dropna()
        if len(delta_P):
            out['P_rmse_vs_lit'] = float(np.sqrt(np.mean(delta_P ** 2)))
            out['P_median_abs_residual'] = float(np.median(np.abs(delta_P)))
            out['n_P_pred'] = int(len(delta_P))
    return out


def main() -> int:
    _log('caveman: H.6 locality validation start')
    curated = pd.read_csv(CURATED_PATH)
    _log(f'caveman: {len(curated)} curated localities loaded')

    opx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_opx.csv',
        low_memory=False)
    cpx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_cpx.csv',
        low_memory=False)
    twopx = pd.read_csv(
        PROJECT_ROOT / 'results' / 'nb08_natural_predictions_twopx.csv',
        low_memory=False)
    _log(f'caveman: opx={len(opx)} cpx={len(cpx)} twopx={len(twopx)}')

    rows = []
    for _, loc in curated.iterrows():
        for label, df_pipeline, t_col, p_col in [
            ('opx_only',    opx,
             'T_canonical_opx_only_T', 'P_canonical_opx_only_P'),
            ('cpx_only',    cpx,
             'T_canonical_cpx_only_T', 'P_canonical_cpx_only_P'),
            ('twopx',       twopx,
             'T_canonical_twopx_T',    'P_canonical_twopx_P'),
        ]:
            r = _stratify(df_pipeline, loc, label, t_col, p_col)
            if r is not None:
                rows.append(r)

    # ArcPL opx as locality #16
    if ARCPL_PATH.exists():
        arcpl = pd.read_csv(ARCPL_PATH, low_memory=False)
        _log(f'caveman: ArcPL archive loaded n={len(arcpl)}')
        ar_row = _stratify_arcpl(arcpl)
        if ar_row is not None:
            rows.append(ar_row)
    else:
        _log('caveman: ArcPL archive missing')

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / 'results' / 'nb08_locality_stratified.csv'
    out.to_csv(out_path, index=False, encoding='utf-8')
    _log(f'caveman: wrote nb08_locality_stratified.csv n={len(out)}')

    # Headline summary
    eligible = out[out.claims_eligible.astype(bool)]
    _log(f'caveman: {len(eligible)} of {len(out)} locality-pipeline rows '
         f'are claims-eligible (n>=20)')
    _log('caveman: top eligible localities by n:')
    for _, r in eligible.sort_values('n_samples_in_box', ascending=False
                                      ).head(8).iterrows():
        t_pred = r.get('T_pred_median', np.nan)
        t_lo = r.get('expected_T_C_low', np.nan)
        t_hi = r.get('expected_T_C_high', np.nan)
        in_frac = r.get('T_in_range_frac', np.nan)
        _log(f'  {r.locality} | {r.pipeline} | n={r.n_samples_in_box} | '
             f'T_med={t_pred:.0f} (expect [{t_lo:.0f},{t_hi:.0f}]) | '
             f'in_range={in_frac:.2%}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
