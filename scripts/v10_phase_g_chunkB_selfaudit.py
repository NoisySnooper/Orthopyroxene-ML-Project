#!/usr/bin/env python3
"""Phase G Chunk A + B self-audit (16 items).

Each item either PASS or FAIL. Emits a markdown report and exits 0 if all
items pass, 1 otherwise. The intent is to catch regressions before committing
Chunk B deliverables.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import (
    RESULTS, FIGURES,
    P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS,
    P_REGIME_REGISTERED_DATE, P_REGIME_MIN_N_FOR_CLAIMS,
)


checks = []


def ok(name: str, cond: bool, detail: str = ''):
    checks.append({'name': name, 'pass': bool(cond), 'detail': detail})


# 1. Pre-registration document exists.
_doc = PROJECT_ROOT / 'docs' / 'v10_p_regime_preregistration.md'
ok('1. pre-registration doc exists', _doc.exists(), _doc.as_posix())

# 2. Config constants present and sane.
ok('2. P_REGIME_BIN_EDGES_KBAR == [0, 5, 15, 30, 100]',
   P_REGIME_BIN_EDGES_KBAR == [0.0, 5.0, 15.0, 30.0, 100.0],
   str(P_REGIME_BIN_EDGES_KBAR))
ok('3. four pre-registered labels present',
   P_REGIME_LABELS == ['shallow_crustal', 'deep_crustal_MASH',
                        'lithospheric_mantle', 'deeper_mantle'],
   str(P_REGIME_LABELS))
ok('4. registration date = 2026-04-17',
   P_REGIME_REGISTERED_DATE == '2026-04-17', P_REGIME_REGISTERED_DATE)
ok('5. honesty-bar floor n >= 20',
   P_REGIME_MIN_N_FOR_CLAIMS == 20, str(P_REGIME_MIN_N_FOR_CLAIMS))

# 6. evaluation.py exposes the three helpers.
try:
    from src.evaluation import (assign_p_regime, compute_per_regime_metrics,
                                 per_regime_benchmark)
    _h = callable(assign_p_regime) and callable(compute_per_regime_metrics) \
         and callable(per_regime_benchmark)
except Exception as e:
    _h = False
ok('6. src.evaluation regime helpers importable', _h)

# 7. allmodels CSV has all six tracks.
_all = RESULTS / 'v10_regime_allmodels.csv'
tracks_needed = {'opx_liq', 'opx_only', 'cpx_liq', 'cpx_only', 'twopx', 'universal'}
_all_df = pd.read_csv(_all) if _all.exists() else pd.DataFrame()
ok('7. v10_regime_allmodels.csv has all 6 tracks',
   _all.exists() and tracks_needed.issubset(set(_all_df.track.unique()))
   if not _all_df.empty else False,
   f'tracks={sorted(_all_df.track.unique()) if not _all_df.empty else []}')

# 8. Pre-reg labels and exploratory bins both present.
_lbl = set(_all_df.regime.unique()) if not _all_df.empty else set()
ok('8. all four pre-registered labels in allmodels CSV',
   set(P_REGIME_LABELS).issubset(_lbl), sorted(_lbl & set(P_REGIME_LABELS)))
_explor = {'P<5', '5<=P<10', '10<=P<20', '20<=P<40', 'P>=40'}
ok('9. exploratory finer bins retained in allmodels CSV',
   _explor.issubset(_lbl), sorted(_lbl & _explor))

# 10. Bootstrap CI columns present.
_ci_cols = {'rmse_lo', 'rmse_hi', 'mae_lo', 'mae_hi'}
ok('10. bootstrap CI columns present',
   _ci_cols.issubset(set(_all_df.columns)) if not _all_df.empty else False,
   str(sorted(set(_all_df.columns) & _ci_cols)))

# 11. opx per-regime benchmark CSV exists and has 8 rows (4 regimes x 2 targets).
_hl = RESULTS / 'v10_opx_per_regime_benchmark.csv'
_hl_df = pd.read_csv(_hl) if _hl.exists() else pd.DataFrame()
ok('11. opx per-regime benchmark CSV has 8 rows (4 bins x 2 targets)',
   _hl.exists() and len(_hl_df) == 8, f'rows={len(_hl_df)}')

# 12. Claims audit CSV exists and honesty-bar logic consistent.
_aud = RESULTS / 'v10_opx_per_regime_claims_audit.csv'
_aud_df = pd.read_csv(_aud) if _aud.exists() else pd.DataFrame()
_hb_consistent = True
_hb_detail = []
if not _aud_df.empty:
    for _, r in _aud_df.iterrows():
        if r['verdict'] == 'v10 outperforms Putirka':
            # must be non-overlapping and n >= 20
            if bool(r['ci_overlap']) or bool(r['sample_size_limited']):
                _hb_consistent = False
                _hb_detail.append(f"{r.regime}/{r.target}: verdict fails honesty bar")
ok('12. honesty-bar verdict rule enforced in claims audit',
   _hb_consistent and not _aud_df.empty, '; '.join(_hb_detail) or 'all rows OK')

# 13. Fig 24 and Fig 25 exist (both PDF+PNG).
for f in ['fig24_per_regime_rmse_opx_liq.pdf',
          'fig24_per_regime_rmse_opx_liq.png',
          'fig25_per_regime_residual_violins_opx_liq.pdf',
          'fig25_per_regime_residual_violins_opx_liq.png']:
    p = FIGURES / f
    ok(f'13. figure artifact {f}', p.exists() and p.stat().st_size > 1000,
       f'bytes={p.stat().st_size if p.exists() else 0}')

# 14. SI Tables S8.5.1-3 exist in tables/ (both md + csv).
for f in ['S8_5_1_regime_rmse_opx_liq.md', 'S8_5_1_regime_rmse_opx_liq.csv',
          'S8_5_2_regime_rmse_ci_opx_liq.md', 'S8_5_2_regime_rmse_ci_opx_liq.csv',
          'S8_5_3_regime_claims_audit_opx_liq.md',
          'S8_5_3_regime_claims_audit_opx_liq.csv']:
    p = PROJECT_ROOT / 'tables' / f
    ok(f'14. SI table artifact {f}', p.exists() and p.stat().st_size > 100,
       f'bytes={p.stat().st_size if p.exists() else 0}')

# 15. Manuscript autofill exists.
_auto = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'regime_results_autofilled.md'
ok('15. regime_results_autofilled.md exists',
   _auto.exists() and _auto.stat().st_size > 500,
   f'bytes={_auto.stat().st_size if _auto.exists() else 0}')

# 16. T15 logged and consistent with audit.
_log = RESULTS / 'v10_nb03_test_log.csv'
if _log.exists():
    _ldf = pd.read_csv(_log)
    _t15 = _ldf[_ldf.test_id == 'T15']
    if len(_t15):
        _t15_row = _t15.iloc[-1]
        _t15_pass = str(_t15_row['passed']).lower() == 'true'
        # Honest consistency: if audit has a P_kbar verdict "v10 outperforms Putirka",
        # T15 must be pass; otherwise fail.
        _audit_has_hit = (
            not _aud_df.empty
            and any((_aud_df.target == 'P_kbar')
                    & (_aud_df.verdict == 'v10 outperforms Putirka'))
        )
        consistent = _t15_pass == _audit_has_hit
        ok('16. T15 log row consistent with claims audit',
           consistent,
           f't15_pass={_t15_pass} audit_hit={_audit_has_hit}')
    else:
        ok('16. T15 log row present', False, 'no T15 rows in log')
else:
    ok('16. T15 log row present', False, 'log CSV missing')

# Write markdown summary.
out_path = PROJECT_ROOT / 'logs' / 'v10_phase_g_chunkB_selfaudit.md'
out_path.parent.mkdir(exist_ok=True)
passed = sum(1 for c in checks if c['pass'])
total = len(checks)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('# Phase G Chunk A + B self-audit\n\n')
    f.write(f'Summary: {passed}/{total} checks pass.\n\n')
    f.write('| # | Check | Pass | Detail |\n')
    f.write('|---|---|---|---|\n')
    for i, c in enumerate(checks, 1):
        f.write(f'| {i} | {c["name"]} | '
                f'{"PASS" if c["pass"] else "FAIL"} | {c["detail"]} |\n')
print(f'wrote {out_path}: {passed}/{total} pass')

sys.exit(0 if passed == total else 1)
