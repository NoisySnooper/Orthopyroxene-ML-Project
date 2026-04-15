"""v7 Part H: update nb04 three-way benchmark to show BOTH families as rows.

Cell 24: "Ours opx-liq" row becomes "Ours opx-liq forest" + "Ours opx-liq
boosted", read from `nb04b_arcpl_predictions_{family}.csv`.

Cell 26: COLORS dict gets forest/boosted keys driven by config.FAMILY_COLORS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"

MARKER_C24 = "# 6. Ours opx-liq (v7 two-family: forest + boosted)"
MARKER_C26 = "# v7 Part H: family-colored COLORS dict"

NEW_OURS_BLOCK = '''

# 6. Ours opx-liq (v7 two-family: forest + boosted)
for _family in ['forest', 'boosted']:
    try:
        ours_df = pd.read_csv(
            RESULTS / f'nb04b_arcpl_predictions_{_family}.csv'
        )
        if 'Experiment' in ours_df.columns:
            _o = m.merge(ours_df[['Experiment', 'T_pred', 'P_pred']],
                         on='Experiment', how='left')
            preds[f'Ours opx-liq {_family}'] = (
                _o['T_pred'].values, _o['P_pred'].values
            )
        else:
            preds[f'Ours opx-liq {_family}'] = (
                ours_df['T_pred'].values[:len(m)],
                ours_df['P_pred'].values[:len(m)],
            )
    except Exception as e:
        print(f'Ours opx-liq {_family} skipped ({e})')

print('\\nMethods available:', list(preds.keys()))
'''

NEW_COLORS_BLOCK = '''# v7 Part H: family-colored COLORS dict
from config import FAMILY_COLORS

COLORS = {
    'Ours opx-liq forest':    FAMILY_COLORS['forest'],
    'Ours opx-liq boosted':   FAMILY_COLORS['boosted'],
    'Agreda-Lopez cpx-liq':   FAMILY_COLORS['external_cpx'],
    'Agreda-Lopez cpx-only':  OKABE_ITO['yellow'],
    'Jorgenson cpx-only':     OKABE_ITO['green'],
    'Wang 2021 cpx-liq':      OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':   FAMILY_COLORS['putirka'],
}'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    changed = False

    cell24 = nb.cells[24]
    if MARKER_C24 not in cell24.source:
        # Strip the entire legacy "6. Ours opx-liq RF" try-except block and the
        # trailing 'print' line, replacing with the two-family block.
        src = cell24.source
        anchor = "# 6. Ours opx-liq RF"
        pi = src.find(anchor)
        if pi == -1:
            print("  nb04 cell 24: legacy Ours anchor not found; leaving untouched")
        else:
            cell24.source = src[:pi].rstrip() + NEW_OURS_BLOCK
            changed = True

    cell26 = nb.cells[26]
    if MARKER_C26 not in cell26.source:
        anchor = "COLORS = {"
        pi = cell26.source.find(anchor)
        pi_end = cell26.source.find("}", pi) + 1
        if pi != -1 and pi_end > pi:
            cell26.source = (
                cell26.source[:pi]
                + NEW_COLORS_BLOCK
                + cell26.source[pi_end:]
            )
            changed = True

    if changed:
        nbformat.write(nb, str(NB))
        print("nb04 three-way two-family patch applied.")
    else:
        print("nb04 three-way two-family patch: already applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
