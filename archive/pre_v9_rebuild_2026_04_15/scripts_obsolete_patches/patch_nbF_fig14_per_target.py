"""Patch nbF_figures fig 14 block to build per-target feature matrices.

Same fix as cell 6: opx_liq T_C forest uses alr (23) and P_kbar uses raw (25),
so `feat_fn` (WIN_FEAT = alr) alone cannot feed the P model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nbF_figures.ipynb"

OLD = """df_test_liq = df_liq.loc[np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')].copy()
X_liq_t, _ = feat_fn(df_test_liq, use_liq=True)
m_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest', RESULTS))
m_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))

iqr_T = predict_iqr(m_T, X_liq_t)[2] - predict_iqr(m_T, X_liq_t)[0]
iqr_P = predict_iqr(m_P, X_liq_t)[2] - predict_iqr(m_P, X_liq_t)[0]"""

NEW = """df_test_liq = df_liq.loc[np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')].copy()
# Per-target feature sets (forest T=alr, forest P=raw for opx_liq).
from src.data import canonical_model_spec
_spec_T14 = canonical_model_spec('T_C',    'opx_liq', 'forest', RESULTS)
_spec_P14 = canonical_model_spec('P_kbar', 'opx_liq', 'forest', RESULTS)
X_liq_t_T, _ = build_feature_matrix(df_test_liq, _spec_T14['feature_set'], use_liq=True)
X_liq_t_P, _ = build_feature_matrix(df_test_liq, _spec_P14['feature_set'], use_liq=True)
m_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest', RESULTS))
m_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))

iqr_T = predict_iqr(m_T, X_liq_t_T)[2] - predict_iqr(m_T, X_liq_t_T)[0]
iqr_P = predict_iqr(m_P, X_liq_t_P)[2] - predict_iqr(m_P, X_liq_t_P)[0]"""


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[12]
    if NEW in cell.source:
        print('fig 14 block: already patched.')
        return 0
    if OLD not in cell.source:
        print('fig 14 block: anchor not found.')
        return 1
    cell.source = cell.source.replace(OLD, NEW, 1)
    cell.outputs = []
    cell.execution_count = None
    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print('fig 14 block: patched.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
