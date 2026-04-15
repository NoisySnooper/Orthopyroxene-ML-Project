"""Apply the residual v7 Part K phase-naming fixes identified by
verify_phase_naming.py.

Fixes:
- nb02: normalize `=== NB02 (v6) COMPLETE ===` -> `=== NB02 complete ===`.
- nb03: drop the stray `=== Phase 3.8 model-family ceiling COMPLETE ===`
  line (the notebook's canonical marker is elsewhere).
- nb08: append `print('=== NB08 complete ===')` to the last code cell.
- nb09: rename misnumbered H2 phases:
    `## Phase 10.2b:` -> `## Phase 9.2b:`
    `## Phase 10.6:`  -> `## Phase 9.6:`
  and drop the stray `=== Phase 9.EXT extended analyses COMPLETE ===`
  intermediate marker.

Idempotent: re-running produces no additional changes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"


def _read(stem: str):
    path = NBDIR / f"{stem}.ipynb"
    return path, nbformat.read(str(path), as_version=4)


def _write(path: Path, nb):
    nbformat.write(nb, str(path))


def fix_nb02() -> int:
    path, nb = _read("nb02_eda_pca")
    changed = 0
    for cell in nb.cells:
        if "NB02 (v6) COMPLETE" in cell.source:
            cell.source = cell.source.replace(
                "=== NB02 (v6) COMPLETE ===", "=== NB02 complete ==="
            )
            changed += 1
    if changed:
        _write(path, nb)
    return changed


def fix_nb03() -> int:
    path, nb = _read("nb03_baseline_models")
    changed = 0
    for cell in nb.cells:
        if "=== Phase 3.8 model-family ceiling COMPLETE ===" in cell.source:
            new_src = re.sub(
                r"\s*print\(['\"]=== Phase 3\.8 model-family ceiling COMPLETE ===['\"]\)\s*\n?",
                "\n",
                cell.source,
            )
            if new_src != cell.source:
                cell.source = new_src
                changed += 1
    if changed:
        _write(path, nb)
    return changed


def fix_nb08() -> int:
    path, nb = _read("nb08_natural_twopx")
    for cell in nb.cells:
        if "=== NB08 complete ===" in cell.source:
            return 0
    last_code = None
    for cell in nb.cells:
        if cell.cell_type == "code":
            last_code = cell
    if last_code is None:
        return 0
    addition = "\nprint('=== NB08 complete ===')\n"
    if not last_code.source.endswith("\n"):
        last_code.source += "\n"
    last_code.source += addition
    _write(path, nb)
    return 1


def fix_nb09() -> int:
    path, nb = _read("nb09_manuscript_compilation")
    changed = 0
    for cell in nb.cells:
        before = cell.source
        cell.source = cell.source.replace(
            "## Phase 10.2b:", "## Phase 9.2b:"
        )
        cell.source = cell.source.replace(
            "## Phase 10.6:", "## Phase 9.6:"
        )
        cell.source = re.sub(
            r"\s*print\(['\"]=== Phase 9\.EXT extended analyses COMPLETE ===['\"]\)\s*\n?",
            "\n",
            cell.source,
        )
        if cell.source != before:
            changed += 1
    if changed:
        _write(path, nb)
    return changed


def main() -> int:
    results = {
        "nb02_complete_case": fix_nb02(),
        "nb03_phase38_marker_drop": fix_nb03(),
        "nb08_add_complete_marker": fix_nb08(),
        "nb09_phase_renumber_and_marker_drop": fix_nb09(),
    }
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
