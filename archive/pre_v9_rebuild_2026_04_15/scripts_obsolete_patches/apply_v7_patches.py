"""Apply the text-level v7 patches across live notebooks and scripts.

This script is idempotent: running it twice produces no additional changes.
It covers the "easy" v7 edits:

- Part C/L1: replace hardcoded `np.random.default_rng(42)` with
  `np.random.default_rng(SEED_BOOTSTRAP)` and add the import.
- Part C/H2: fix `nb03_winning_config.json` typo in nbF.
- Part C/M7: add ownership comment at the head of nb09 nb10_* CSV
  producer cells.
- Part E/M6: rename Gridded-PT -> TargetBinKFold in nb05 markdown and
  code (purely a display rename; does not touch filenames).
- Part H (partial): migrate canonical_model_filename call sites in
  nb06/nb07/nb09/nbF/nb04b to the new (target, track, family) signature.
  Defaults to `family="forest"` so the existing single-family output is
  preserved; the family-split extensions in nb04b/nb06/nb07/nb09 are
  applied by subsequent cells those notebooks pick up via
  `load_per_family_winners`.
- Part K: phase-naming standardization pass (partial; see
  apply_v7_phase_naming.py for the full pass).

Does NOT cover:
- nb03 rebuild (see rebuild_nb03_v7.py).
- Part F markdown title inserts (see apply_v7_titles.py).
- Complex two-family figure/CSV reorganization.

Run:
    .venv\\Scripts\\python.exe scripts/apply_v7_patches.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"


def read_nb(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(str(path), as_version=4)


def write_nb(nb: nbformat.NotebookNode, path: Path) -> None:
    nbformat.write(nb, str(path))


def replace_in_all_code_cells(nb, find: str, replace: str) -> int:
    """Replace verbatim text in every code cell. Returns #cells modified."""
    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if find in cell.source:
            cell.source = cell.source.replace(find, replace)
            n += 1
    return n


def replace_in_all_cells(nb, find: str, replace: str) -> int:
    """Replace verbatim text in every cell (code + markdown)."""
    n = 0
    for cell in nb.cells:
        if find in cell.source:
            cell.source = cell.source.replace(find, replace)
            n += 1
    return n


def regex_replace_all_cells(nb, pattern: str, replace: str) -> int:
    n = 0
    for cell in nb.cells:
        new_src, count = re.subn(pattern, replace, cell.source)
        if count:
            cell.source = new_src
            n += count
    return n


def ensure_config_import(nb, symbol: str) -> int:
    """Add `symbol` to the first `from config import (...)` block that does
    not already include it. No-op if the symbol is already imported or no
    block exists. Returns #cells modified."""
    patt_open = re.compile(r"from config import \(")
    patt_close = re.compile(r"\)\s*$", re.MULTILINE)
    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "from config import" not in cell.source:
            continue
        if symbol in cell.source and re.search(rf"\b{re.escape(symbol)}\b", cell.source):
            # already imported
            continue
        src = cell.source
        m = patt_open.search(src)
        if not m:
            # single-line form: `from config import A, B`
            line_pat = re.compile(r"^(from config import\s+)([^\n()]+)$", re.MULTILINE)
            def _extend(match):
                return match.group(1) + match.group(2).rstrip() + f", {symbol}"
            new_src, c = line_pat.subn(_extend, src, count=1)
            if c:
                cell.source = new_src
                n += 1
            continue
        # multi-line form: inject before the closing ')'
        end = patt_close.search(src, m.end())
        if not end:
            continue
        injection_point = end.start()
        new_src = src[:injection_point] + f"    {symbol},\n" + src[injection_point:]
        # Avoid duplicate inject if a line already ends with symbol
        if not re.search(rf"\b{re.escape(symbol)}\b", src):
            cell.source = new_src
            n += 1
    return n


def patch_nb04(path: Path) -> dict:
    nb = read_nb(path)
    changes = {
        "import_seed_bootstrap": ensure_config_import(nb, "SEED_BOOTSTRAP"),
        "rng_bootstrap": (
            replace_in_all_code_cells(nb, "np.random.default_rng(42)",
                                      "np.random.default_rng(SEED_BOOTSTRAP)")
        ),
    }
    if any(changes.values()):
        write_nb(nb, path)
    return changes


def patch_nb04b(path: Path) -> dict:
    nb = read_nb(path)
    changes = {
        "import_seed_bootstrap": ensure_config_import(nb, "SEED_BOOTSTRAP"),
        "rng_bootstrap": (
            replace_in_all_code_cells(nb, "np.random.default_rng(42)",
                                      "np.random.default_rng(SEED_BOOTSTRAP)")
        ),
    }
    if any(changes.values()):
        write_nb(nb, path)
    return changes


def patch_nbF_winning_config_typo(path: Path) -> dict:
    nb = read_nb(path)
    before_config = replace_in_all_code_cells(
        nb,
        "nb03_winning_config.json",
        "nb03_per_family_winners.json",
    )
    # Also update any caption/text references
    before_config += replace_in_all_cells(
        nb,
        "nb03_winning_config.json",
        "nb03_per_family_winners.json",
    ) - before_config
    # nbF typo fix
    changes = {"typo_fix": before_config}
    if before_config:
        write_nb(nb, path)
    return changes


def patch_nb05_target_binkfold(path: Path) -> dict:
    """Rename 'Gridded-PT' display label to 'TargetBinKFold' in nb05 without
    touching CSV filenames. M6 in the audit."""
    nb = read_nb(path)
    count = 0
    count += replace_in_all_cells(nb, "Gridded-PT", "TargetBinKFold")
    count += replace_in_all_cells(nb, "gridded-PT", "TargetBinKFold")
    count += replace_in_all_cells(nb, "Gridded PT", "TargetBinKFold")
    if count:
        write_nb(nb, path)
    return {"gridded_pt_renames": count}


def _migrate_canonical_call(source: str) -> tuple[str, int]:
    """Rewrite legacy call `canonical_model_filename('RF', 'P_kbar', 'opx_liq', ...)`
    into the v7 `canonical_model_filename('P_kbar', 'opx_liq', 'forest', ...)`.
    Heuristic: RF/ERT map to 'forest'; GB/XGB map to 'boosted'. Preserves
    extra args (results_dir) and any kw args."""
    pattern = re.compile(
        r"canonical_model_filename\(\s*"
        r"(?P<mq>['\"])(?P<model>RF|ERT|GB|XGB)(?P=mq)\s*,\s*"
        r"(?P<tq>['\"])(?P<target>T_C|P_kbar)(?P=tq)\s*,\s*"
        r"(?P<trq>['\"])(?P<track>opx_only|opx_liq)(?P=trq)"
        r"(?P<rest>[^)]*)\)"
    )
    counts = 0
    def _repl(m: re.Match) -> str:
        nonlocal counts
        counts += 1
        model = m.group("model")
        family = "forest" if model in ("RF", "ERT") else "boosted"
        target = m.group("target")
        track = m.group("track")
        rest = m.group("rest")
        return (f"canonical_model_filename('{target}', '{track}', '{family}'{rest})")
    new_source = pattern.sub(_repl, source)
    return new_source, counts


def patch_canonical_call_sites(path: Path) -> dict:
    nb = read_nb(path)
    total = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "canonical_model_filename" not in cell.source:
            continue
        new_src, n = _migrate_canonical_call(cell.source)
        if n:
            cell.source = new_src
            total += n
    if total:
        write_nb(nb, path)
    return {"canonical_call_migrations": total}


def patch_script_canonical_calls(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    new_src, n = _migrate_canonical_call(src)
    if n:
        path.write_text(new_src, encoding="utf-8")
    return {"canonical_call_migrations": n}


def patch_nb09_nb10_ownership(path: Path) -> dict:
    """Add a one-line ownership note above the first cell that writes any
    nb10_*.csv file. M7 in the audit."""
    nb = read_nb(path)
    note = ("# NOTE: The nb10_* filename prefix below is retained for "
            "manuscript stability. These files are produced here by NB09 "
            "Phase 9.2-9.4 after the v6 archival of NB10.\n")
    modified = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "nb10_" not in cell.source:
            continue
        if "The nb10_* filename prefix below is retained" in cell.source:
            continue
        if re.search(r"\.to_csv\([^)]*nb10_", cell.source) or \
                re.search(r"nb10_\w+\.csv", cell.source):
            cell.source = note + cell.source
            modified += 1
    if modified:
        write_nb(nb, path)
    return {"nb10_ownership_comments": modified}


def patch_phase_naming(path: Path, nb_label: str) -> dict:
    """Apply the simpler Part K string replacements: drop 'R' suffix,
    normalize completion markers. Structural inserts live in apply_v7_phase_naming.py."""
    nb = read_nb(path)
    total = 0
    # Drop 'R' suffix from Phase NR.M -> Phase N.M
    total += regex_replace_all_cells(
        nb, r"Phase (\d+)R\.", r"Phase \1."
    )
    # Old '=== PHASE ... COMPLETE ===' markers -> '=== NBxx complete ==='
    marker = f"=== {nb_label} complete ==="
    total += regex_replace_all_cells(
        nb, r"=== PHASE [^=]*COMPLETE ===", marker
    )
    total += regex_replace_all_cells(
        nb, r"=== Phase [^=]*complete ===", marker
    )
    if total:
        write_nb(nb, path)
    return {"phase_name_fixes": total}


MAIN_PATCHES = [
    ("nb04_putirka_benchmark", patch_nb04, "NB04"),
    ("nb04b_lepr_arcpl_validation", patch_nb04b, "NB04b"),
    ("nbF_figures", patch_nbF_winning_config_typo, "NBF"),
    ("nb05_loso_validation", patch_nb05_target_binkfold, "NB05"),
    ("nb09_manuscript_compilation", patch_nb09_nb10_ownership, "NB09"),
]

CALL_SITE_NBS = [
    ("nb06_shap_analysis", "NB06"),
    ("nb07_bias_correction", "NB07"),
    ("nb09_manuscript_compilation", "NB09"),
    ("nbF_figures", "NBF"),
    ("nb04b_lepr_arcpl_validation", "NB04b"),
    ("nb08_natural_twopx", "NB08"),
]

PHASE_NAME_NBS = [
    ("nb01_data_cleaning", "NB01"),
    ("nb02_eda_pca", "NB02"),
    ("nb03_baseline_models", "NB03"),
    ("nb04_putirka_benchmark", "NB04"),
    ("nb04b_lepr_arcpl_validation", "NB04b"),
    ("nb05_loso_validation", "NB05"),
    ("nb06_shap_analysis", "NB06"),
    ("nb07_bias_correction", "NB07"),
    ("nb08_natural_twopx", "NB08"),
    ("nb09_manuscript_compilation", "NB09"),
    ("nbF_figures", "NBF"),
]


def main() -> int:
    print("v7 patches: applying text-level edits to notebooks and scripts\n")

    for nb_stem, fn, _ in MAIN_PATCHES:
        path = NBDIR / f"{nb_stem}.ipynb"
        if not path.exists():
            print(f"  SKIP {nb_stem}: not found")
            continue
        result = fn(path)
        summary = ", ".join(f"{k}={v}" for k, v in result.items())
        print(f"  {nb_stem}: {summary}")

    print("\nMigrating canonical_model_filename call sites...")
    for nb_stem, _ in CALL_SITE_NBS:
        path = NBDIR / f"{nb_stem}.ipynb"
        if not path.exists():
            print(f"  SKIP {nb_stem}: not found")
            continue
        result = patch_canonical_call_sites(path)
        print(f"  {nb_stem}: {result}")

    script_candidates = [
        ROOT / "scripts" / "build_nb04_three_way.py",
        ROOT / "scripts" / "merge_nb10_into_nb09.py",
    ]
    for sp in script_candidates:
        if sp.exists():
            result = patch_script_canonical_calls(sp)
            print(f"  script {sp.name}: {result}")

    # build_nb04_three_way.py: add SEED_BOOTSTRAP import if missing and
    # replace bare default_rng(42). Already handled as a prior pass.
    bntw = ROOT / "scripts" / "build_nb04_three_way.py"
    if bntw.exists():
        src = bntw.read_text(encoding="utf-8")
        orig = src
        if "SEED_BOOTSTRAP" not in src:
            src = src.replace(
                "from config import LEPR_XLSX, RESULTS, FIGURES, MODELS",
                "from config import LEPR_XLSX, RESULTS, FIGURES, MODELS, SEED_BOOTSTRAP",
            )
        src = src.replace(
            "np.random.default_rng(42)",
            "np.random.default_rng(SEED_BOOTSTRAP)",
        )
        if src != orig:
            bntw.write_text(src, encoding="utf-8")
            print(f"  script build_nb04_three_way.py: SEED_BOOTSTRAP applied")

    print("\nApplying Part K phase-naming fixes...")
    for nb_stem, label in PHASE_NAME_NBS:
        path = NBDIR / f"{nb_stem}.ipynb"
        if not path.exists():
            print(f"  SKIP {nb_stem}: not found")
            continue
        result = patch_phase_naming(path, label)
        print(f"  {nb_stem}: {result}")

    print("\nv7 patches complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
