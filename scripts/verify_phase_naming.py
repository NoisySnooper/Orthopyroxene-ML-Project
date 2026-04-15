"""Verify v7 Part K phase-naming canonical scheme.

Checks every live notebook for:

1. No `Phase \\d+R\\.` (the old v6 R-suffix).
2. No stray `=== PHASE ... COMPLETE ===` or `=== Phase ... complete ===`
   markers. The only allowed completion marker phrasing is
   `=== NBxx complete ===` with matching case.
3. Exactly one `=== NBxx complete ===` marker per notebook.
4. H2 markdown `## Phase X.Y:` headers: the numeric prefix must match the
   notebook's number (or `4b` for nb04b, or `EXT` for nb09 absorbed-phase
   headers).
5. No pre-v7 filename references in markdown/code text:
   `nb03_winning_config.json` -> `nb03_per_family_winners.json`.

Exits 0 on zero violations, 1 otherwise. Used by run_all_v7.py as a gate
before the 20-seed rerun.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"

LIVE_NBS = [
    "nb01_data_cleaning",
    "nb02_eda_pca",
    "nb03_baseline_models",
    "nb04_putirka_benchmark",
    "nb04b_lepr_arcpl_validation",
    "nb05_loso_validation",
    "nb06_shap_analysis",
    "nb07_bias_correction",
    "nb08_natural_twopx",
    "nb09_manuscript_compilation",
    "nbF_figures",
]

# Map notebook stem -> expected canonical label for completion marker
# and expected numeric prefix for H2 phase headers.
NB_LABEL = {
    "nb01_data_cleaning": ("NB01", ["1"]),
    "nb02_eda_pca": ("NB02", ["2"]),
    "nb03_baseline_models": ("NB03", ["3"]),
    "nb04_putirka_benchmark": ("NB04", ["4"]),
    "nb04b_lepr_arcpl_validation": ("NB04b", ["4b"]),
    "nb05_loso_validation": ("NB05", ["5"]),
    "nb06_shap_analysis": ("NB06", ["6"]),
    "nb07_bias_correction": ("NB07", ["7"]),
    "nb08_natural_twopx": ("NB08", ["8"]),
    "nb09_manuscript_compilation": ("NB09", ["9", "EXT"]),
    "nbF_figures": ("NBF", ["F"]),
}

R_SUFFIX_PAT = re.compile(r"Phase \d+R\.")
H2_PHASE_PAT = re.compile(r"^## Phase ([\w]+)\.([\w]+):", re.MULTILINE)
H2_PHASE_NO_COLON = re.compile(r"^## Phase ([\w]+)\.([\w]+)(?!:)", re.MULTILINE)
COMPLETION_PAT = re.compile(r"===\s*(NB[\w]+|PHASE[^=]*|Phase[^=]*)\s+(complete|COMPLETE)\s*===", re.IGNORECASE)
CANONICAL_COMPLETION = re.compile(r"=== (NB[\w]+) complete ===")
OLD_FILENAME_PAT = re.compile(r"nb03_winning_config\.json")


def _scan_nb(path: Path, label: str, allowed_prefixes: list) -> list:
    violations = []
    nb = nbformat.read(str(path), as_version=4)
    canonical_marker_count = 0

    for idx, cell in enumerate(nb.cells):
        src = cell.source

        for m in R_SUFFIX_PAT.finditer(src):
            violations.append(
                f"{path.name} cell {idx}: R-suffix '{m.group(0)}' should be stripped (v7 Part K)"
            )

        for m in OLD_FILENAME_PAT.finditer(src):
            violations.append(
                f"{path.name} cell {idx}: stale filename 'nb03_winning_config.json' "
                f"should be 'nb03_per_family_winners.json' (v7 Part C/H2)"
            )

        for m in H2_PHASE_PAT.finditer(src):
            prefix = m.group(1)
            if prefix not in allowed_prefixes:
                violations.append(
                    f"{path.name} cell {idx}: H2 'Phase {prefix}.{m.group(2)}' has wrong "
                    f"notebook prefix (expected one of {allowed_prefixes})"
                )

        for m in COMPLETION_PAT.finditer(src):
            full = m.group(0)
            if CANONICAL_COMPLETION.match(full):
                tok = CANONICAL_COMPLETION.match(full).group(1)
                if tok == label:
                    canonical_marker_count += 1
                else:
                    violations.append(
                        f"{path.name} cell {idx}: completion marker references '{tok}', "
                        f"expected '{label}' ({full})"
                    )
            else:
                violations.append(
                    f"{path.name} cell {idx}: non-canonical completion marker '{full}' "
                    f"-- v7 requires only '=== {label} complete ===' (case-exact)"
                )

    if canonical_marker_count == 0:
        violations.append(
            f"{path.name}: missing canonical completion marker '=== {label} complete ==='"
        )
    elif canonical_marker_count > 1:
        violations.append(
            f"{path.name}: found {canonical_marker_count} canonical completion markers, "
            f"expected exactly one"
        )
    return violations


def main() -> int:
    all_violations = []
    for stem in LIVE_NBS:
        path = NBDIR / f"{stem}.ipynb"
        if not path.exists():
            all_violations.append(f"{stem}.ipynb: notebook not found")
            continue
        label, allowed = NB_LABEL[stem]
        all_violations.extend(_scan_nb(path, label, allowed))

    if all_violations:
        print(f"verify_phase_naming: {len(all_violations)} violation(s) found\n")
        for v in all_violations:
            print(f"  - {v}")
        print(f"\nTotal: {len(all_violations)} violation(s). v7 requires zero.")
        return 1
    print(f"verify_phase_naming: OK ({len(LIVE_NBS)} notebooks, 0 violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
