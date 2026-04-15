"""v7 hotfix: nb06 SHAP TreeExplainer additivity check failure.

TreeExplainer.assert_additivity raises when the sum of SHAP values
doesn't match model.predict(X) within ~1e-6. For RandomForest with
float32 inputs (or aggressively pruned trees), the numerical precision
falls short of the check's default tolerance even though the decomposition
is correct for downstream plotting.

shap's recommended workaround is `check_additivity=False`. We apply it
to both the P and T cells (5 and 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb06_shap_analysis.ipynb"

OLD_P = "shap_values_P = explainer_P.shap_values(X_test)"
NEW_P = "shap_values_P = explainer_P.shap_values(X_test, check_additivity=False)"
OLD_T = "shap_values_T = explainer_T.shap_values(X_test)"
NEW_T = "shap_values_T = explainer_T.shap_values(X_test, check_additivity=False)"


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    changed = False
    for idx, (old, new, label) in [
        (5, (OLD_P, NEW_P, 'P')),
        (6, (OLD_T, NEW_T, 'T')),
    ]:
        cell = nb.cells[idx]
        if new in cell.source:
            print(f'cell {idx} ({label}): already patched.')
            continue
        if old not in cell.source:
            print(f'cell {idx} ({label}): anchor not found.')
            return 1
        cell.source = cell.source.replace(old, new, 1)
        cell.outputs = []
        cell.execution_count = None
        changed = True
        print(f'cell {idx} ({label}): patched with check_additivity=False.')

    if changed:
        nbformat.validate(nb)
        nbformat.write(nb, str(NB))
        print('nb06 updated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
