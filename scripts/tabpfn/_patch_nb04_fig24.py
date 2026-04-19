"""Targeted patch: add TabPFN errorbar series + subtitle note to nb04 cell 22.

One-shot helper. Preserves the original cell's escape-sequence encoding
by using textual search-replace on the `source` strings rather than
re-authoring the cell from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent.parent / 'notebooks' / 'nb04_regime_benchmark.ipynb'
CELL_IDX = 22

ADD_TAB_DF_BLOCK = (
    "    put = sub[sub.method_family == 'Putirka']\n",
    "    put = sub[sub.method_family == 'Putirka']\n"
    "    tab = sub[sub.method_family == 'tabpfn']\n",
)

ADD_TAB_BEST_BLOCK = (
    "    put_best = (put.loc[put.groupby('regime')['rmse'].idxmin()]\n"
    "                   .set_index('regime') if not put.empty else None)\n",
    "    put_best = (put.loc[put.groupby('regime')['rmse'].idxmin()]\n"
    "                   .set_index('regime') if not put.empty else None)\n"
    "    tab_best = (tab.loc[tab.groupby('regime')['rmse'].idxmin()]\n"
    "                   .set_index('regime') if not tab.empty else None)\n",
)

# Shift v10 offset -0.12 -> -0.2 and Putirka +0.12 -> +0.2 to make room.
SHIFT_V10 = (
    "    ax.errorbar(x - 0.12, y, yerr=[elo, ehi], fmt='o', color='#2ca02c',\n",
    "    ax.errorbar(x - 0.2, y, yerr=[elo, ehi], fmt='o', color='#2ca02c',\n",
)
SHIFT_PUT = (
    "        ax.errorbar(x + 0.12, y2, yerr=[elo2, ehi2], fmt='^', color='#d62728',\n",
    "        ax.errorbar(x + 0.2, y2, yerr=[elo2, ehi2], fmt='^', color='#d62728',\n",
)

TAB_ERRBAR_BLOCK = """
    # TabPFN (middle, 9th BASE_ORDER family, Okabe-Ito green).
    if tab_best is not None:
        y3, elo3, ehi3 = [], [], []
        for r in regimes:
            if r in tab_best.index:
                row = tab_best.loc[r]
                y3.append(row.rmse); l, h = _err(row)
                elo3.append(l); ehi3.append(h)
            else:
                y3.append(np.nan); elo3.append(0); ehi3.append(0)
        ax.errorbar(x, y3, yerr=[elo3, ehi3], fmt='s', color='#009E73',
                    ecolor='#009E73', capsize=3, markersize=6.5, linewidth=0,
                    label='TabPFN v2 (9th family)', zorder=5.5)

"""

# Subtitle update: append TabPFN 20-seed note.
SUBTITLE_PATCH = (
    "    'Per-regime RMSE, pre-registered bins (reg. 2026-04-17)\\n'\n"
    "    'bootstrap 95% CIs; shaded bins are sample-size-limited (n < 20)',\n",
    "    'Per-regime RMSE, pre-registered bins (reg. 2026-04-17)\\n'\n"
    "    'bootstrap 95% CIs; shaded bins are sample-size-limited (n < 20); '\n"
    "    'TabPFN via 20-seed ensemble (42-61)',\n",
)


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cell = nb['cells'][CELL_IDX]
    src = ''.join(cell['source'])

    # Idempotent: skip if tabpfn already present.
    if 'tabpfn' in src.lower():
        print(f'cell {CELL_IDX} already contains TabPFN block; no-op')
        return

    replacements = [
        ADD_TAB_DF_BLOCK,
        ADD_TAB_BEST_BLOCK,
        SHIFT_V10,
        SHIFT_PUT,
        SUBTITLE_PATCH,
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(f'anchor not found in cell {CELL_IDX}: {old!r}')
        src = src.replace(old, new, 1)

    # Insert TabPFN errorbar block just before the Putirka block.
    put_anchor = "    # Putirka best equation"
    if put_anchor not in src:
        raise RuntimeError('Putirka anchor not found')
    # Also shift to "rightmost" in the Putirka label line.
    src = src.replace(
        put_anchor,
        TAB_ERRBAR_BLOCK.lstrip('\n') + '    # Putirka best equation (rightmost)',
        1,
    )
    # Clean up the old comment ending
    src = src.replace(
        '    # Putirka best equation (rightmost).\n',
        '    # Putirka best equation (rightmost).\n',
        1,
    )

    cell['source'] = src.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(f'patched {NB_PATH} cell {CELL_IDX}; {len(cell["source"])} source lines')


if __name__ == '__main__':
    main()
