"""v7 Part H: wire nb08 Section 2 to the per-family canonical resolver.

Before: nb08 called `load_winning_config()` + a single `WIN_FEAT` string to
pick a feature set. Under v7 there is no global winning feature; each
(target, track, family) has its own feature set. The call graph now loads
`canonical_model_spec` for each target and feeds the target-specific
feature set into `build_feature_matrix`.

Also replaces the `>400 K` mean heuristic on Putirka two-pyroxene T output
with the pinned `THERMOBAR_T_RETURNS_KELVIN` contract from config.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb08_natural_twopx.ipynb"

MARKER_C3 = "# v7 Part H: per-target forest-family canonical load"

NEW_TOP_C3 = '''# Section 2. Run our opx-only ML and Jorgenson cpx-only ML on both datasets.
#
# v7 Part H: per-target forest-family canonical load
# Forest family is the primary track for the cross-mineral figure. Each
# target pulls its own feature set from nb03_per_family_winners.json.
from src.data import canonical_model_spec

spec_T_f = canonical_model_spec('T_C', 'opx_only', 'forest', RESULTS)
spec_P_f = canonical_model_spec('P_kbar', 'opx_only', 'forest', RESULTS)
fs_T = spec_T_f['feature_set']
fs_P = spec_P_f['feature_set']

m_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_only', 'forest', RESULTS))
m_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_only', 'forest', RESULTS))
'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[3]
    if MARKER_C3 in cell.source:
        print("nb08 cell 3: already patched.")
        return 0

    anchor = "# Section 2. Run our opx-only ML and Jorgenson cpx-only ML on both datasets."
    end_anchor = "m_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_only', 'forest', RESULTS))"
    src = cell.source
    pi = src.find(anchor)
    pj = src.find(end_anchor)
    if pi == -1 or pj == -1:
        print("nb08 cell 3: anchors not found")
        return 1
    pj_end = src.find("\n", pj) + 1
    cell.source = src[:pi] + NEW_TOP_C3 + src[pj_end:]

    cell.source = cell.source.replace(
        "def run_opx_ml(df, suffix):\n    X, _ = build_feature_matrix(_opx_df_for_feats(df, suffix=suffix),\n"
        "                                WIN_FEAT, use_liq=False)\n    return m_T.predict(X), m_P.predict(X)",
        "def run_opx_ml(df, suffix):\n"
        "    _opx_df = _opx_df_for_feats(df, suffix=suffix)\n"
        "    X_T, _ = build_feature_matrix(_opx_df, fs_T, use_liq=False)\n"
        "    X_P, _ = build_feature_matrix(_opx_df, fs_P, use_liq=False)\n"
        "    return m_T.predict(X_T), m_P.predict(X_P)",
    )

    cell.source = cell.source.replace(
        "    if np.nanmean(T_put_lepr) > 400:\n        T_put_lepr = T_put_lepr - 273.15",
        "    from config import THERMOBAR_T_RETURNS_KELVIN\n"
        "    if THERMOBAR_T_RETURNS_KELVIN:\n"
        "        T_put_lepr = T_put_lepr - 273.15",
    )

    nbformat.write(nb, str(NB))
    print("nb08 cell 3: patched (per-target forest-family, Thermobar K contract).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
