"""Patch nbF_figures cell 6 to use per-(track, target) canonical feature sets.

Forest winners differ across combos (alr/raw/pwlr), so `feat_fn` built from
WIN_FEAT alone cannot feed all four models. Build X per combo via
canonical_model_spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nbF_figures.ipynb"

CELL6_OLD_BLOCK = """X_liq_te, _ = feat_fn(df_te_liq, use_liq=True)
X_opx_te, _ = feat_fn(df_te_opx, use_liq=False)

models = {
    ('opx_only', 'P_kbar'): joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_only', 'forest', RESULTS)),
    ('opx_only', 'T_C'):    joblib.load(MODELS / canonical_model_filename('T_C', 'opx_only', 'forest', RESULTS)),
    ('opx_liq',  'P_kbar'): joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest',  RESULTS)),
    ('opx_liq',  'T_C'):    joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest',  RESULTS)),
}"""

CELL6_NEW_BLOCK = """# Per-(track, target) canonical feature matrices. Forest winners differ
# across combos (alr/raw/pwlr), so we cannot reuse a single feat_fn.
from src.data import canonical_model_spec
_combos = [('opx_only', 'P_kbar', False), ('opx_only', 'T_C', False),
           ('opx_liq',  'P_kbar', True),  ('opx_liq',  'T_C',  True)]
X_te = {}
for _tr, _tg, _ul in _combos:
    _spec = canonical_model_spec(_tg, _tr, 'forest', RESULTS)
    _df = df_te_liq if _tr == 'opx_liq' else df_te_opx
    X_te[(_tr, _tg)], _ = build_feature_matrix(_df, _spec['feature_set'], use_liq=_ul)

# Legacy aliases so downstream code that still references X_liq_te / X_opx_te
# keeps working for target-agnostic uses (not used by the pred_vs_obs blocks
# below, which switch to X_te[(...)]).
X_liq_te = X_te[('opx_liq', 'T_C')]
X_opx_te = X_te[('opx_only', 'T_C')]

models = {
    ('opx_only', 'P_kbar'): joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_only', 'forest', RESULTS)),
    ('opx_only', 'T_C'):    joblib.load(MODELS / canonical_model_filename('T_C', 'opx_only', 'forest', RESULTS)),
    ('opx_liq',  'P_kbar'): joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest',  RESULTS)),
    ('opx_liq',  'T_C'):    joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest',  RESULTS)),
}"""

CELL6_PREDICT_REPLACEMENTS = [
    ("pred = predict_median(models[('opx_only', 'P_kbar')], X_opx_te)",
     "pred = predict_median(models[('opx_only', 'P_kbar')], X_te[('opx_only', 'P_kbar')])"),
    ("pred = predict_median(models[('opx_liq', 'P_kbar')], X_liq_te)",
     "pred = predict_median(models[('opx_liq', 'P_kbar')], X_te[('opx_liq', 'P_kbar')])"),
    ("pred = predict_median(models[('opx_only', 'T_C')], X_opx_te)",
     "pred = predict_median(models[('opx_only', 'T_C')], X_te[('opx_only', 'T_C')])"),
    ("pred = predict_median(models[('opx_liq', 'T_C')], X_liq_te)",
     "pred = predict_median(models[('opx_liq', 'T_C')], X_te[('opx_liq', 'T_C')])"),
]


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[6]
    changed = 0

    if CELL6_NEW_BLOCK in cell.source:
        print('cell 6 X block: already patched.')
    elif CELL6_OLD_BLOCK not in cell.source:
        print('cell 6 X block: anchor not found.')
        return 1
    else:
        cell.source = cell.source.replace(CELL6_OLD_BLOCK, CELL6_NEW_BLOCK, 1)
        changed += 1
        print('cell 6 X block: patched.')

    for old, new in CELL6_PREDICT_REPLACEMENTS:
        if new in cell.source:
            continue
        if old not in cell.source:
            print(f'  predict anchor missing: {old[:70]}')
            return 1
        cell.source = cell.source.replace(old, new, 1)
        changed += 1

    if changed:
        cell.outputs = []
        cell.execution_count = None
        nbformat.validate(nb)
        nbformat.write(nb, str(NB))
        print(f'nbF cell 6 updated ({changed} change(s)).')
    else:
        print('nbF cell 6: already fully patched.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
