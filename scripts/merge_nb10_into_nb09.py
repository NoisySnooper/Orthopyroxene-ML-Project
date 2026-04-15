"""Merge NB10 (extended analyses) into NB09 (manuscript compilation).

Inserts the NB10 computation cells between NB09's imports cell and the first
Table-building cell so the artifact CSVs are freshly written before the
Tables 8-11 summary cell reads them. Idempotent: cells tagged with the
v6-nb10-merge- prefix are removed before re-insertion.
"""
from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / 'notebooks' / 'nb10_extended_analyses.ipynb'
DST_PATH = ROOT / 'notebooks' / 'nb09_manuscript_compilation.ipynb'
TAG = 'v6-nb10-merge-'

# We will copy a subset of NB10 cell source (indices 3 through 13), prefixed
# by a new section header and a "setup" block that adds the extra imports
# NB09 doesnt already have, plus loads the canonical models + ArcPL table.

SECTION_HEADER = """## Phase 9R.EXT: Extended analyses (absorbed from former NB10)

**Purpose:** Run the downstream analyses that used to live in NB10 so NB09 is
self-contained - one compute-and-compile notebook rather than two. The
artifact CSVs (`nb10_*.csv`, `model_IsolationForest_opx_liq.joblib`,
`model_RF_*_opx_liq_H2O.joblib`) are still named with the `nb10_` prefix so
every existing manuscript reference keeps resolving.

Analyses: 10.1 two-pyroxene Thermobar baseline; 10.2 H2O residual dependence;
10.2b H2O-engineered retrain; 10.3 IQR uncertainty; 10.4 analytical-noise MC;
10.5 IsolationForest OOD filter; 10.6 OOD-paradox method comparison."""

SETUP_BLOCK = """# Phase 9R.EXT setup: extra imports + canonical model/data loads for the
# absorbed NB10 analyses. NB09's earlier imports supply ROOT, DATA_PROC,
# DATA_SPLITS, MODELS, FIGURES, RESULTS, WIN_FEAT, numpy as np, pandas as pd.

from src.features import build_feature_matrix
from src.models import predict_median, predict_iqr
from src.evaluation import compute_metrics as metrics
from src.plot_style import canonical_model_filename, apply_style
apply_style()

import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

feat_fn = lambda df, use_liq: build_feature_matrix(df, WIN_FEAT, use_liq=use_liq)
model_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))
qrf_T_path = MODELS / 'model_QRF_T_C_opx_liq.joblib'
qrf_P_path = MODELS / 'model_QRF_P_kbar_opx_liq.joblib'
qrf_T = joblib.load(qrf_T_path) if qrf_T_path.exists() else None
qrf_P = joblib.load(qrf_P_path) if qrf_P_path.exists() else None

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()
X_train, feat_names = feat_fn(df_train, use_liq=True)
X_test,  _          = feat_fn(df_test,  use_liq=True)

arcpl_path = RESULTS / 'nb04b_arcpl_predictions.csv'
if not arcpl_path.exists():
    raise FileNotFoundError(f'Missing {arcpl_path}. Run nb04b first.')
df_arcpl = pd.read_csv(arcpl_path)
print(f'Ext setup OK: train n={len(df_train)} test n={len(df_test)} ArcPL n={len(df_arcpl)}')
"""

FOOTER_VERIFY = """# Phase 9R.EXT verification: confirm every absorbed-NB10 artifact was written.
_expected = [
    'nb10_two_pyroxene_benchmark.csv', 'nb10_h2o_dependence.csv',
    'nb10_h2o_engineered_arcpl.csv',   'nb10_h2o_engineered_test_rmse.csv',
    'nb10_iqr_uncertainty.csv',        'nb10_analytical_uncertainty.csv',
    'nb10_mc_per_sample.csv',          'nb10_ood_isoforest.csv',
    'nb10_arcpl_ood_scores.csv',       'nb10_ood_paradox_methods.csv',
    'nb10_ood_scores_all_methods.csv',
]
_missing = [f for f in _expected if not (RESULTS / f).exists()]
assert not _missing, f'Phase 9R.EXT missing: {_missing}'
assert (MODELS / 'model_IsolationForest_opx_liq.joblib').exists()
assert (MODELS / 'model_RF_T_C_opx_liq_H2O.joblib').exists()
assert (MODELS / 'model_RF_P_kbar_opx_liq_H2O.joblib').exists()
print('=== Phase 9R.EXT extended analyses COMPLETE ===')
"""


def main():
    nb_src = nbformat.read(str(SRC_PATH), as_version=4)
    nb_dst = nbformat.read(str(DST_PATH), as_version=4)

    # Strip any previously merged cells so reruns replace cleanly.
    nb_dst.cells = [c for c in nb_dst.cells
                    if not str(c.get('id', '')).startswith(TAG)]

    # Locate insertion index: right after NB09's imports cell.
    anchor = None
    for i, c in enumerate(nb_dst.cells):
        if str(c.get('id', '')) == 'cell-001-aeec7cfd':
            anchor = i
            break
    if anchor is None:
        raise RuntimeError("anchor cell 'cell-001-aeec7cfd' not found in NB09")
    insert_at = anchor + 1

    # Build the cells to insert.
    cells_to_insert: list = []

    def _add(suffix, kind, src):
        cid = TAG + suffix
        if kind == 'markdown':
            c = new_markdown_cell(src)
        else:
            c = new_code_cell(src)
        c['id'] = cid
        cells_to_insert.append(c)

    _add('header', 'markdown', SECTION_HEADER)
    _add('setup',  'code',     SETUP_BLOCK)

    # Copy NB10 cells 5-13 (computation + v5 additions). These use df_train/
    # df_test/X_train/X_test/model_T/model_P/feat_fn/df_arcpl provided by
    # SETUP_BLOCK above, so the chain is preserved.
    NB10_SLICE = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    for j, src_idx in enumerate(NB10_SLICE):
        src_cell = nb_src.cells[src_idx]
        _add(f'nb10-{src_idx:02d}', src_cell.cell_type, src_cell.source)

    _add('verify', 'code', FOOTER_VERIFY)

    nb_dst.cells = (nb_dst.cells[:insert_at]
                    + cells_to_insert
                    + nb_dst.cells[insert_at:])

    nbformat.write(nb_dst, str(DST_PATH))
    print(f'inserted {len(cells_to_insert)} cells (tag "{TAG}") into {DST_PATH.name} '
          f'at index {insert_at} (after cell-001-aeec7cfd)')


if __name__ == '__main__':
    main()
