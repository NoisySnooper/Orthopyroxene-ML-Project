#!/usr/bin/env python3
"""Phase G Chunk A + B + C self-audit.

Superset of the Chunk B self-audit. Adds items covering:
    C1. Chunk C probe CSVs (predictions, per-regime RMSE, aggregate).
    C2. Chunk C aggregate matches the existing multi-seed summary to within
        1e-6 kbar / deg C.
    C3. The robust audit CSV exists and has the expected shape (one row per
        (regime, target) of the Chunk B benchmark CSV).
    C4. The single axis-1 'outperforms' row from Chunk B is still present in
        the robust audit.
    C5. At least one row is classified as 'v10 outperforms Putirka (robust)'
        (Chunk C's post-condition: the pre-registered claim survives).
    C6. S8.5.4 robust table artifact present (md + csv) and non-empty.
    C7. T15 log has the latest row sourced from 'robust' (the v2 pass
        condition).
    C8. Manuscript autofill references S8.5.4 and the robust audit CSV.

Emits logs/v10_phase_g_chunkC_selfaudit.md and exits 0 if all items pass,
1 otherwise. Does NOT re-execute training or audits; pure artifact +
consistency check.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    FIGURES, RESULTS,
    P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS,
    P_REGIME_REGISTERED_DATE, P_REGIME_MIN_N_FOR_CLAIMS,
)


checks = []


def ok(name: str, cond: bool, detail: str = ''):
    checks.append({'name': name, 'pass': bool(cond), 'detail': detail})


# ---------------------------------------------------------------------------
# Chunk B carry-overs (abbreviated; the full Chunk B self-audit still runs
# separately in scripts/v10_phase_g_chunkB_selfaudit.py, so here we only
# re-verify the rows whose semantics Chunk C modifies).
# ---------------------------------------------------------------------------

# 1. Pre-registration document present.
_doc = PROJECT_ROOT / 'docs' / 'v10_p_regime_preregistration.md'
ok('1. pre-registration doc exists', _doc.exists(), _doc.as_posix())

# 2. Config constants sane.
ok('2. P_REGIME_BIN_EDGES_KBAR == [0, 5, 15, 30, 100]',
   P_REGIME_BIN_EDGES_KBAR == [0.0, 5.0, 15.0, 30.0, 100.0],
   str(P_REGIME_BIN_EDGES_KBAR))
ok('3. honesty-bar floor n >= 20',
   P_REGIME_MIN_N_FOR_CLAIMS == 20, str(P_REGIME_MIN_N_FOR_CLAIMS))

# 4. Axis-1 claims audit (Chunk B) still on disk.
_chunkB_aud = RESULTS / 'v10_opx_per_regime_claims_audit.csv'
_chunkB_df  = pd.read_csv(_chunkB_aud) if _chunkB_aud.exists() else pd.DataFrame()
ok('4. Chunk B claims audit CSV present (axis-1 only)',
   _chunkB_aud.exists() and len(_chunkB_df) == 8,
   f'rows={len(_chunkB_df)}')

# ---------------------------------------------------------------------------
# Chunk C items.
# ---------------------------------------------------------------------------

# C1. Probe CSVs.
_pred = RESULTS / 'v10_chunkC_perseed_predictions.csv'
_regime = RESULTS / 'v10_chunkC_perseed_regime_rmse.csv'
_agg = RESULTS / 'v10_chunkC_perseed_aggregate_rmse.csv'
_pred_df = pd.read_csv(_pred) if _pred.exists() else pd.DataFrame()
_regime_df = pd.read_csv(_regime) if _regime.exists() else pd.DataFrame()
_agg_df = pd.read_csv(_agg) if _agg.exists() else pd.DataFrame()
ok('C1a. probe predictions CSV non-empty',
   _pred.exists() and len(_pred_df) > 0, f'rows={len(_pred_df)}')
ok('C1b. probe per-regime RMSE CSV non-empty',
   _regime.exists() and len(_regime_df) > 0, f'rows={len(_regime_df)}')
ok('C1c. probe aggregate RMSE CSV non-empty',
   _agg.exists() and len(_agg_df) > 0, f'rows={len(_agg_df)}')

# C2. Aggregate matches existing multi-seed summary.
_ms = RESULTS / 'v10_opx_multiseed_summary.csv'
if _ms.exists() and not _agg_df.empty:
    ms_df = pd.read_csv(_ms)
    mismatches = []
    grp = (_agg_df.groupby(['model', 'target', 'track', 'feature_set'])['rmse']
                  .mean().reset_index())
    for _, r in grp.iterrows():
        mm = ms_df[(ms_df.model == r.model) & (ms_df.target == r.target) &
                   (ms_df.track == r.track) &
                   (ms_df.feature_set == r.feature_set)]
        if len(mm) != 1:
            mismatches.append(f'no match for {r.model}/{r.feature_set}/{r.target}')
        else:
            delta = float(r.rmse) - float(mm['mean'].iloc[0])
            if abs(delta) > 1e-6:
                mismatches.append(
                    f'{r.model}/{r.feature_set}/{r.target}: delta={delta:+.6f}')
    ok('C2. probe aggregate matches existing multi-seed summary (tol 1e-6)',
       len(mismatches) == 0,
       '; '.join(mismatches) or 'all cells match')
else:
    ok('C2. probe aggregate matches existing multi-seed summary', False,
       'one of the CSVs is missing')

# C3. Robust audit CSV shape.
_robust = RESULTS / 'v10_opx_per_regime_claims_audit_robust.csv'
_rob_df = pd.read_csv(_robust) if _robust.exists() else pd.DataFrame()
ok('C3. robust audit CSV has 8 rows (4 regimes x 2 targets)',
   _robust.exists() and len(_rob_df) == 8,
   f'rows={len(_rob_df)}')

required_cols = {
    'regime', 'target', 'n', 'v10_method',
    'v10_rmse', 'v10_rmse_lo', 'v10_rmse_hi',
    'putirka_method', 'putirka_rmse',
    'putirka_rmse_lo', 'putirka_rmse_hi',
    'seed_rmse_mean', 'seed_rmse_std', 'seed_rmse_lo', 'seed_rmse_hi',
    'seed_rmse_min', 'seed_rmse_max', 'seed_is_deterministic',
    'axis1_nonoverlap', 'axis2_nonoverlap',
    'robust_verdict', 'robust_outperforms',
}
ok('C3b. robust audit CSV has all required columns',
   required_cols.issubset(set(_rob_df.columns)) if not _rob_df.empty else False,
   str(sorted(required_cols - set(_rob_df.columns))
       if not _rob_df.empty else 'empty'))

# C4. The single Chunk B outperforms row is still present in robust audit.
_chunkB_hits = _chunkB_df[_chunkB_df.get('verdict', '') ==
                          'v10 outperforms Putirka'] if not _chunkB_df.empty else pd.DataFrame()
_both_present = True
_match_detail = []
if not _chunkB_hits.empty and not _rob_df.empty:
    for _, h in _chunkB_hits.iterrows():
        mm = _rob_df[(_rob_df.regime == h['regime']) &
                     (_rob_df.target == h['target'])]
        if len(mm) != 1:
            _both_present = False
            _match_detail.append(f'{h["regime"]}/{h["target"]}: missing in robust')
    ok('C4. every Chunk B outperforms row is represented in robust audit',
       _both_present,
       '; '.join(_match_detail) or f'{len(_chunkB_hits)} row(s) matched')
else:
    ok('C4. every Chunk B outperforms row is represented in robust audit',
       False, 'one side empty')

# C5. At least one robust outperforms row.
_robust_hits = _rob_df[_rob_df.get('robust_outperforms', False)
                       .astype(bool)] if not _rob_df.empty else pd.DataFrame()
ok('C5. at least one robust outperforms Putirka row in audit',
   not _robust_hits.empty,
   f'n_rows={len(_robust_hits)}: '
   f'{sorted(_robust_hits[["regime","target"]].apply(tuple, axis=1).tolist()) if not _robust_hits.empty else []}')

# C6. S8.5.4 artifact.
_t_md = PROJECT_ROOT / 'tables' / 'S8_5_4_regime_claims_audit_robust_opx_liq.md'
_t_csv = PROJECT_ROOT / 'tables' / 'S8_5_4_regime_claims_audit_robust_opx_liq.csv'
ok('C6a. S8.5.4 markdown exists and non-trivial (>= 500 bytes)',
   _t_md.exists() and _t_md.stat().st_size >= 500,
   f'bytes={_t_md.stat().st_size if _t_md.exists() else 0}')
ok('C6b. S8.5.4 csv exists and non-trivial (>= 500 bytes)',
   _t_csv.exists() and _t_csv.stat().st_size >= 500,
   f'bytes={_t_csv.stat().st_size if _t_csv.exists() else 0}')

# C7. T15 log row sourced from robust.
_log_csv = RESULTS / 'v10_nb03_test_log.csv'
if _log_csv.exists():
    _log = pd.read_csv(_log_csv)
    _t15 = _log[_log.test_id == 'T15']
    if len(_t15):
        _row = _t15.iloc[-1]
        try:
            _det = json.loads(_row['details'])
        except Exception:
            _det = {}
        _src = _det.get('source', '')
        _passed = str(_row['passed']).lower() == 'true'
        ok('C7a. latest T15 log row sourced from robust audit',
           _src == 'robust', f'source={_src}')
        ok('C7b. latest T15 log row passed',
           _passed, f'passed={_row["passed"]}')
    else:
        ok('C7a. latest T15 log row sourced from robust audit', False,
           'no T15 rows in log')
        ok('C7b. latest T15 log row passed', False,
           'no T15 rows in log')
else:
    ok('C7a. latest T15 log row sourced from robust audit', False,
       'log CSV missing')
    ok('C7b. latest T15 log row passed', False,
       'log CSV missing')

# C8. Manuscript autofill references robust audit artifacts.
_auto = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'regime_results_autofilled.md'
if _auto.exists():
    text = _auto.read_text(encoding='utf-8')
    ref_s854 = 'S8_5_4' in text or 'S8.5.4' in text
    ref_robust_csv = 'v10_opx_per_regime_claims_audit_robust.csv' in text
    ref_two_axis = 'two-axis' in text.lower() or 'axis 1' in text.lower()
    ok('C8a. autofill references Table S8.5.4',
       ref_s854, f'found={ref_s854}')
    ok('C8b. autofill references results/v10_opx_per_regime_claims_audit_robust.csv',
       ref_robust_csv, f'found={ref_robust_csv}')
    ok('C8c. autofill describes the two-axis honesty bar',
       ref_two_axis, f'found={ref_two_axis}')
else:
    ok('C8a. autofill references Table S8.5.4', False, 'file missing')
    ok('C8b. autofill references results/v10_opx_per_regime_claims_audit_robust.csv',
       False, 'file missing')
    ok('C8c. autofill describes the two-axis honesty bar', False,
       'file missing')

# ---------------------------------------------------------------------------
# Write markdown summary.
# ---------------------------------------------------------------------------

out_path = PROJECT_ROOT / 'logs' / 'v10_phase_g_chunkC_selfaudit.md'
out_path.parent.mkdir(exist_ok=True)
passed = sum(1 for c in checks if c['pass'])
total = len(checks)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('# Phase G Chunk C self-audit\n\n')
    f.write(f'Summary: {passed}/{total} checks pass.\n\n')
    f.write('| # | Check | Pass | Detail |\n')
    f.write('|---|---|---|---|\n')
    for i, c in enumerate(checks, 1):
        f.write(f'| {i} | {c["name"]} | '
                f'{"PASS" if c["pass"] else "FAIL"} | {c["detail"]} |\n')
print(f'wrote {out_path}: {passed}/{total} pass')

sys.exit(0 if passed == total else 1)
