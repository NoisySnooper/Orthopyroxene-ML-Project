#!/usr/bin/env python3
"""Phase G full self-audit covering G.2 - G.6 deliverables.

Does NOT re-execute training; pure artifact + shape consistency check.
Chunks A/B/C already have their own self-audits
(v10_phase_g_chunkB_selfaudit.py, v10_phase_g_chunkC_selfaudit.py); this
one covers the later sub-phases that Chunk C did not touch:

  G.2  NB05 generalization rebuild (opx-liq only)
  G.3  NB06 SHAP rebuild (opx-liq canonical cells)
  G.4  NB07 bias correction rebuild (opx-liq piecewise P)
  G.5  NB10 twopx benchmark rebuild (consolidated)
  G.6  Figure audit (fig24-fig29 PDF+PNG+TXT)

Emits logs/v10_phase_g_full_selfaudit.md and exits 0 if all pass, 1
otherwise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import FIGURES, RESULTS

checks = []


def ok(name: str, cond: bool, detail: str = ''):
    checks.append({'name': name, 'pass': bool(cond), 'detail': detail})


def _exists_nonempty(path: Path, min_bytes: int = 100) -> tuple[bool, str]:
    if not path.exists():
        return False, 'missing'
    sz = path.stat().st_size
    if sz < min_bytes:
        return False, f'{sz}B < {min_bytes}B'
    return True, f'{sz}B'


# ---------------------------------------------------------------------------
# G.2 NB05 generalization
# ---------------------------------------------------------------------------
_gen_csv = RESULTS / 'v10_opx_liq_generalization.csv'
_gen_df = pd.read_csv(_gen_csv) if _gen_csv.exists() else pd.DataFrame()
ok('G.2a. generalization CSV exists',
   _gen_csv.exists(), _gen_csv.as_posix())
ok('G.2b. generalization has >= 12 rows (3 cells x 4 strategies)',
   len(_gen_df) >= 12, f'rows={len(_gen_df)}')
if not _gen_df.empty:
    required = {'strategy', 'model', 'target', 'track', 'feature_set',
                'rmse', 'rmse_lo', 'rmse_hi', 'mae', 'n', 'n_folds'}
    ok('G.2c. generalization has required columns',
       required.issubset(set(_gen_df.columns)),
       sorted(required - set(_gen_df.columns)))
    strategies = set(_gen_df.strategy.unique())
    expected = {'LOSO', 'ClusterKFold', 'TargetBinKFold', 'LeaveOneRegionOut'}
    ok('G.2d. all 4 strategies present',
       expected.issubset(strategies),
       str(sorted(expected - strategies)))
else:
    ok('G.2c. generalization has required columns', False, 'empty CSV')
    ok('G.2d. all 4 strategies present', False, 'empty CSV')

_gen_pred = RESULTS / 'v10_opx_liq_generalization_predictions.csv'
exists, det = _exists_nonempty(_gen_pred, 10_000)
ok('G.2e. generalization predictions CSV non-trivial', exists, det)

# ---------------------------------------------------------------------------
# G.3 NB06 SHAP
# ---------------------------------------------------------------------------
_shap_imp = RESULTS / 'v10_opx_liq_shap_importance.csv'
_shap_df = pd.read_csv(_shap_imp) if _shap_imp.exists() else pd.DataFrame()
ok('G.3a. SHAP importance CSV exists',
   _shap_imp.exists(), _shap_imp.as_posix())
ok('G.3b. SHAP importance has >= 5 unique cells',
   len(_shap_df['cell'].unique()) >= 5 if not _shap_df.empty else False,
   f'cells={sorted(_shap_df["cell"].unique()) if not _shap_df.empty else []}')

_shap_npz = RESULTS / 'v10_opx_liq_shap_values.npz'
exists, det = _exists_nonempty(_shap_npz, 1_000)
ok('G.3c. SHAP values npz non-trivial', exists, det)

_t87_md = PROJECT_ROOT / 'tables' / 'S8_7_shap_top_features_opx_liq.md'
_t87_csv = PROJECT_ROOT / 'tables' / 'S8_7_shap_top_features_opx_liq.csv'
ok('G.3d. Table S8.7 markdown exists',
   _t87_md.exists(),
   str(_t87_md.stat().st_size) if _t87_md.exists() else 'missing')
ok('G.3e. Table S8.7 csv exists',
   _t87_csv.exists(),
   str(_t87_csv.stat().st_size) if _t87_csv.exists() else 'missing')

# ---------------------------------------------------------------------------
# G.4 NB07 bias correction
# ---------------------------------------------------------------------------
_bc_csv = RESULTS / 'v10_opx_liq_bias_correction.csv'
_bc_df = pd.read_csv(_bc_csv) if _bc_csv.exists() else pd.DataFrame()
ok('G.4a. bias-correction CSV exists',
   _bc_csv.exists(), _bc_csv.as_posix())
if not _bc_df.empty:
    required = {'model', 'target', 'track', 'feature_set', 'regime', 'n',
                'pre_rmse', 'post_rmse', 'improvement'}
    ok('G.4b. bias-correction has required columns',
       required.issubset(set(_bc_df.columns)),
       sorted(required - set(_bc_df.columns)))
    all_row = _bc_df[_bc_df.regime == 'ALL']
    ok('G.4c. bias-correction has ALL-regime rows',
       not all_row.empty, f'n_all={len(all_row)}')
    # Known: MLP P_kbar and ElasticNet P_kbar ALL should improve
    mlp_p = all_row[(all_row.model == 'MLP') & (all_row.target == 'P_kbar')]
    en_p = all_row[(all_row.model == 'ElasticNet') & (all_row.target == 'P_kbar')]
    ok('G.4d. MLP P_kbar ALL improves (improvement > 0)',
       (not mlp_p.empty) and float(mlp_p.improvement.iloc[0]) > 0,
       f'improvement={mlp_p.improvement.iloc[0] if not mlp_p.empty else "n/a"}')
    ok('G.4e. ElasticNet P_kbar ALL improves (improvement > 0)',
       (not en_p.empty) and float(en_p.improvement.iloc[0]) > 0,
       f'improvement={en_p.improvement.iloc[0] if not en_p.empty else "n/a"}')
else:
    ok('G.4b. bias-correction has required columns', False, 'empty CSV')
    ok('G.4c. bias-correction has ALL-regime rows', False, 'empty CSV')
    ok('G.4d. MLP P_kbar ALL improves (improvement > 0)', False, 'empty CSV')
    ok('G.4e. ElasticNet P_kbar ALL improves (improvement > 0)', False, 'empty CSV')

_bc_params = RESULTS / 'v10_opx_liq_bias_correction_params.csv'
exists, det = _exists_nonempty(_bc_params, 500)
ok('G.4f. bias-correction params CSV non-trivial', exists, det)

# ---------------------------------------------------------------------------
# G.5 NB10 twopx benchmark
# ---------------------------------------------------------------------------
_tpx = RESULTS / 'v10_twopx_benchmark_final.csv'
_tpx_df = pd.read_csv(_tpx) if _tpx.exists() else pd.DataFrame()
ok('G.5a. twopx benchmark CSV exists',
   _tpx.exists(), _tpx.as_posix())
ok('G.5b. twopx benchmark has exactly 6 rows (2 targets x 3 method_class)',
   len(_tpx_df) == 6, f'rows={len(_tpx_df)}')
if not _tpx_df.empty:
    method_classes = set(_tpx_df.method_class.unique())
    expected_mc = {'v10_best_base', 'v10_best_ensemble', 'putirka_twopx'}
    ok('G.5c. twopx benchmark has all 3 method_classes',
       expected_mc.issubset(method_classes),
       str(sorted(expected_mc - method_classes)))
    putirka = _tpx_df[_tpx_df.method_class == 'putirka_twopx']
    ok('G.5d. Putirka row is NaN (deferred)',
       len(putirka) == 2 and putirka['single_seed_rmse'].isna().all(),
       f'n_putirka={len(putirka)}, rmse_nan={putirka["single_seed_rmse"].isna().all() if len(putirka) else False}')
else:
    ok('G.5c. twopx benchmark has all 3 method_classes', False, 'empty CSV')
    ok('G.5d. Putirka row is NaN (deferred)', False, 'empty CSV')

_t89_md = PROJECT_ROOT / 'tables' / 'S8_9_twopx_benchmark.md'
exists, det = _exists_nonempty(_t89_md, 300)
ok('G.5e. Table S8.9 markdown non-trivial', exists, det)

# ---------------------------------------------------------------------------
# G.6 Figure audit
# ---------------------------------------------------------------------------
_audit_csv = RESULTS / 'v10_phase_g_figure_audit.csv'
_audit_df = pd.read_csv(_audit_csv) if _audit_csv.exists() else pd.DataFrame()
ok('G.6a. figure audit CSV exists',
   _audit_csv.exists(), _audit_csv.as_posix())
ok('G.6b. figure audit has 6 rows (fig24-fig29)',
   len(_audit_df) == 6, f'rows={len(_audit_df)}')
if not _audit_df.empty:
    n_fail = int((_audit_df.status == 'FAIL').sum())
    ok('G.6c. all figure audit rows PASS',
       n_fail == 0, f'n_fail={n_fail}')
else:
    ok('G.6c. all figure audit rows PASS', False, 'empty CSV')

# Figure files directly
for stem in ['fig24_per_regime_rmse_opx_liq',
             'fig25_per_regime_residual_violins_opx_liq',
             'fig26_generalization_opx_liq',
             'fig27_shap_summary_opx_liq',
             'fig28_bias_correction_opx_liq',
             'fig29_twopx_benchmark']:
    pdf = FIGURES / f'{stem}.pdf'
    png = FIGURES / f'{stem}.png'
    txt = FIGURES / f'{stem}.txt'
    ok(f'G.6d. {stem} pdf+png+txt',
       pdf.exists() and png.exists() and txt.exists(),
       f'pdf={pdf.exists()}, png={png.exists()}, txt={txt.exists()}')

# ---------------------------------------------------------------------------
# Write markdown summary.
# ---------------------------------------------------------------------------
out_path = PROJECT_ROOT / 'logs' / 'v10_phase_g_full_selfaudit.md'
out_path.parent.mkdir(exist_ok=True)
passed = sum(1 for c in checks if c['pass'])
total = len(checks)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('# Phase G full self-audit (G.2 - G.6)\n\n')
    f.write(f'Summary: {passed}/{total} checks pass.\n\n')
    f.write('| # | Check | Pass | Detail |\n')
    f.write('|---|---|---|---|\n')
    for i, c in enumerate(checks, 1):
        f.write(f'| {i} | {c["name"]} | '
                f'{"PASS" if c["pass"] else "FAIL"} | {c["detail"]} |\n')
print(f'wrote {out_path}: {passed}/{total} pass')

sys.exit(0 if passed == total else 1)
