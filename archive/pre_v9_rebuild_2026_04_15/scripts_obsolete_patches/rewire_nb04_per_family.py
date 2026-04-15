"""Rewire nb04 and nb04b to load the per-family canonical joblibs (v7 Part H).

Before v7, both notebooks loaded `model_RF_{target}_opx_liq_{feat}.joblib`
using `get_or_train_rf`. Those files are no longer produced by nb03 (the
rebuild writes 8 per-family canonicals: `model_{target}_{track}_{family}.joblib`).

Changes applied:
- nb04 cell 4: replace the `get_or_train_rf` block with a forest-family
  canonical load via `canonical_model_path` / `canonical_model_spec`.
- nb04b cell 9: same treatment for forest, then add a parallel boosted-family
  load writing `T_pred_boosted` / `P_pred_boosted` columns.
- nb04b cell 11: append per-family prediction CSV exports
  (`nb04b_arcpl_predictions_forest.csv`, `..._boosted.csv`).

Idempotent via sentinel markers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"

MARKER_NB04_C4 = "# v7 Part H: forest-family canonical load for fair comparison"
MARKER_NB04B_C9 = "# v7 Part H: per-family canonical load (forest + boosted) for external validation"
MARKER_NB04B_C11 = "# v7 Part H: per-family prediction CSVs"

NB04_C4_NEW = '''# v7 Part H: forest-family canonical load for fair comparison
# Earlier versions selected the RF model per feature set directly from the
# multi-seed summary. v7 consolidates that choice into the per-family
# winner JSON written by nb03 Phase 3.5; we now load the canonical forest
# joblib (RF or ERT depending on the tiebreaker outcome).
from src.data import canonical_model_path, canonical_model_spec

spec_T = canonical_model_spec('T_C', 'opx_liq', 'forest', RESULTS)
spec_P = canonical_model_spec('P_kbar', 'opx_liq', 'forest', RESULTS)
fs_T = spec_T['feature_set']
fs_P = spec_P['feature_set']
print(f"Forest-family opx-liq: T uses {spec_T['model_name']}/{fs_T}, "
      f"P uses {spec_P['model_name']}/{fs_P}")

model_T_ml = joblib.load(canonical_model_path('T_C', 'opx_liq', 'forest',
                                              MODELS, RESULTS))
model_P_ml = joblib.load(canonical_model_path('P_kbar', 'opx_liq', 'forest',
                                              MODELS, RESULTS))
'''

NB04B_C9_NEW = '''# v7 Part H: per-family canonical load (forest + boosted) for external validation
# nb04b now produces predictions from BOTH canonical families so the manuscript
# can report a two-family row per benchmark scope.
from src.data import canonical_model_path, canonical_model_spec

spec_T = canonical_model_spec('T_C', 'opx_liq', 'forest', RESULTS)
spec_P = canonical_model_spec('P_kbar', 'opx_liq', 'forest', RESULTS)
fs_T, fs_P = spec_T['feature_set'], spec_P['feature_set']
print(f"Forest-family opx-liq: T uses {spec_T['model_name']}/{fs_T}, "
      f"P uses {spec_P['model_name']}/{fs_P}")

model_T = joblib.load(canonical_model_path('T_C', 'opx_liq', 'forest',
                                           MODELS, RESULTS))
model_P = joblib.load(canonical_model_path('P_kbar', 'opx_liq', 'forest',
                                           MODELS, RESULTS))

X_T, _ = build_feature_matrix(arcpl, fs_T, use_liq=True)
X_P, _ = build_feature_matrix(arcpl, fs_P, use_liq=True)
arcpl['T_pred'] = model_T.predict(X_T)
arcpl['P_pred'] = model_P.predict(X_P)

# Boosted family parallel predictions for the two-family manuscript table.
spec_T_b = canonical_model_spec('T_C', 'opx_liq', 'boosted', RESULTS)
spec_P_b = canonical_model_spec('P_kbar', 'opx_liq', 'boosted', RESULTS)
fs_T_b, fs_P_b = spec_T_b['feature_set'], spec_P_b['feature_set']
print(f"Boosted-family opx-liq: T uses {spec_T_b['model_name']}/{fs_T_b}, "
      f"P uses {spec_P_b['model_name']}/{fs_P_b}")

model_T_b = joblib.load(canonical_model_path('T_C', 'opx_liq', 'boosted',
                                             MODELS, RESULTS))
model_P_b = joblib.load(canonical_model_path('P_kbar', 'opx_liq', 'boosted',
                                             MODELS, RESULTS))
X_T_b, _ = build_feature_matrix(arcpl, fs_T_b, use_liq=True)
X_P_b, _ = build_feature_matrix(arcpl, fs_P_b, use_liq=True)
arcpl['T_pred_boosted'] = model_T_b.predict(X_T_b)
arcpl['P_pred_boosted'] = model_P_b.predict(X_P_b)
'''

NB04B_C11_APPEND = '''

# v7 Part H: per-family prediction CSVs
arcpl_forest = arcpl.drop(columns=['_key', 'T_pred_boosted', 'P_pred_boosted'],
                          errors='ignore').copy()
arcpl_forest.to_csv(RESULTS / 'nb04b_arcpl_predictions_forest.csv', index=False)

arcpl_boosted = arcpl.drop(columns=['_key', 'T_pred', 'P_pred'],
                           errors='ignore').copy()
arcpl_boosted = arcpl_boosted.rename(columns={
    'T_pred_boosted': 'T_pred',
    'P_pred_boosted': 'P_pred',
})
arcpl_boosted.to_csv(RESULTS / 'nb04b_arcpl_predictions_boosted.csv', index=False)
print('Per-family prediction CSVs written (forest, boosted).')
'''


def _read(stem: str):
    path = NBDIR / f"{stem}.ipynb"
    return path, nbformat.read(str(path), as_version=4)


def _write(path: Path, nb):
    nbformat.write(nb, str(path))


def patch_nb04() -> int:
    path, nb = _read("nb04_putirka_benchmark")
    cell = nb.cells[4]
    if MARKER_NB04_C4 in cell.source:
        return 0
    cell.source = NB04_C4_NEW
    _write(path, nb)
    return 1


def patch_nb04b() -> dict:
    path, nb = _read("nb04b_lepr_arcpl_validation")
    n9 = n11 = 0
    cell9 = nb.cells[9]
    if MARKER_NB04B_C9 not in cell9.source:
        cell9.source = NB04B_C9_NEW
        n9 = 1
    cell11 = nb.cells[11]
    if MARKER_NB04B_C11 not in cell11.source:
        cell11.source = cell11.source.rstrip() + NB04B_C11_APPEND
        n11 = 1
    if n9 or n11:
        _write(path, nb)
    return {"cell9": n9, "cell11": n11}


def main() -> int:
    print(f"nb04 cell 4: {patch_nb04()}")
    print(f"nb04b: {patch_nb04b()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
