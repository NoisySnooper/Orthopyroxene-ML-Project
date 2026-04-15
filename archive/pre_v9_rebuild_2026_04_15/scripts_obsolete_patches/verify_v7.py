"""v7 pre-rerun static verifier.

Checks the code-level v7 contract before launching run_all_v7.py:

A. config.py constants are in place (N_SPLIT_REPS=20, SPLIT_SEEDS len 20,
   MODEL_FAMILIES, FAMILY_COLORS, CANONICAL_FIGURES, SEED_BOOTSTRAP,
   THERMOBAR_PINNED_VERSION, THERMOBAR_T_RETURNS_KELVIN).

B. src/data.py exports v7 canonical-model API
   (canonical_model_spec, canonical_model_filename, canonical_model_path,
   load_canonical_model, load_per_family_winners).

C. src/external_models.py has been hardened: `ast` is imported, no bare
   `eval(` in the source, `_check_thermobar_version` is present.

D. src/evaluation.py exports `resolve_columns` and `qcut_with_warning`.

E. Live notebooks don't import `load_winning_config` call sites that
   still reference `global_feature_set` (hard dependency on legacy JSON).

F. Delegates phase-naming check to verify_phase_naming.py.

Exits 0 on success, 1 on any failure. Intended as the gate called
explicitly before `run_all_v7.py` launches the 20-seed rerun.
"""
from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fail(label: str, msg: str, errors: list) -> None:
    errors.append(f"[{label}] {msg}")


def check_config(errors: list) -> None:
    import config
    required = {
        'N_SPLIT_REPS': 20,
        'SEED_BOOTSTRAP': None,
        'THERMOBAR_PINNED_VERSION': None,
        'THERMOBAR_T_RETURNS_KELVIN': None,
    }
    for name, expected in required.items():
        if not hasattr(config, name):
            _fail("A/config", f"missing {name}", errors)
        elif expected is not None and getattr(config, name) != expected:
            _fail(
                "A/config",
                f"{name} = {getattr(config, name)!r}, expected {expected!r}",
                errors,
            )

    split_seeds = getattr(config, 'SPLIT_SEEDS', [])
    if len(split_seeds) != 20:
        _fail("A/config", f"SPLIT_SEEDS length {len(split_seeds)}, expected 20", errors)

    fams = getattr(config, 'MODEL_FAMILIES', {})
    if set(fams.keys()) != {'forest', 'boosted'}:
        _fail("A/config", f"MODEL_FAMILIES keys {sorted(fams.keys())}", errors)

    fc = getattr(config, 'FAMILY_COLORS', {})
    for k in ('forest', 'boosted', 'external_cpx', 'putirka'):
        if k not in fc:
            _fail("A/config", f"FAMILY_COLORS missing {k}", errors)

    cfigs = getattr(config, 'CANONICAL_FIGURES', [])
    if len(cfigs) < 10:
        _fail("A/config", f"CANONICAL_FIGURES has {len(cfigs)} entries, expected >= 10", errors)


def check_data_module(errors: list) -> None:
    try:
        m = importlib.import_module('src.data')
    except Exception as e:
        _fail("B/src.data", f"import failed: {e}", errors)
        return
    for sym in (
        'canonical_model_spec', 'canonical_model_filename',
        'canonical_model_path', 'load_canonical_model',
        'load_per_family_winners', 'PER_FAMILY_WINNERS_FILE',
        'VALID_FAMILIES', 'VALID_TARGETS', 'VALID_TRACKS',
    ):
        if not hasattr(m, sym):
            _fail("B/src.data", f"missing export {sym}", errors)

    fam = getattr(m, 'VALID_FAMILIES', ())
    if tuple(fam) != ('forest', 'boosted'):
        _fail("B/src.data", f"VALID_FAMILIES={fam}", errors)


def check_external_models(errors: list) -> None:
    path = ROOT / 'src' / 'external_models.py'
    if not path.exists():
        _fail("C/external_models", "file not found", errors)
        return
    src = path.read_text(encoding='utf-8')
    if 'import ast' not in src:
        _fail("C/external_models", "missing `import ast`", errors)
    if re.search(r'(?<!literal_)\beval\(', src):
        _fail("C/external_models", "stray eval(...) call", errors)
    if '_check_thermobar_version' not in src:
        _fail("C/external_models", "missing _check_thermobar_version", errors)
    if 'THERMOBAR_T_RETURNS_KELVIN' not in src:
        _fail("C/external_models", "K->C conversion not pinned to config flag", errors)


def check_evaluation(errors: list) -> None:
    try:
        m = importlib.import_module('src.evaluation')
    except Exception as e:
        _fail("D/src.evaluation", f"import failed: {e}", errors)
        return
    for sym in ('resolve_columns', 'qcut_with_warning', 'COLUMN_ALIASES'):
        if not hasattr(m, sym):
            _fail("D/src.evaluation", f"missing {sym}", errors)


def check_notebook_legacy_refs(errors: list) -> None:
    import nbformat
    live = [
        'nb03_baseline_models',
        'nb04_putirka_benchmark',
        'nb04b_lepr_arcpl_validation',
        'nb06_shap_analysis',
        'nb07_bias_correction',
        'nb08_natural_twopx',
        'nb09_manuscript_compilation',
        'nbF_figures',
    ]
    for stem in live:
        path = ROOT / 'notebooks' / f'{stem}.ipynb'
        if not path.exists():
            continue
        nb = nbformat.read(str(path), as_version=4)
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            src = cell.source
            if "global_feature_set" in src:
                _fail(
                    "E/notebooks",
                    f"{stem} cell {idx} still references global_feature_set "
                    f"(pre-v7 single-feature-set assumption)",
                    errors,
                )
            if "nb03_winning_configurations.json" in src:
                _fail(
                    "E/notebooks",
                    f"{stem} cell {idx} references legacy "
                    f"nb03_winning_configurations.json",
                    errors,
                )


def check_phase_naming(errors: list) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'verify_phase_naming.py')],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        _fail(
            "F/phase_naming",
            "verify_phase_naming.py failed:\n" + proc.stdout + proc.stderr,
            errors,
        )


def main() -> int:
    errors: list = []
    check_config(errors)
    check_data_module(errors)
    check_external_models(errors)
    check_evaluation(errors)
    check_notebook_legacy_refs(errors)
    check_phase_naming(errors)

    if errors:
        print(f"verify_v7: {len(errors)} issue(s) found\n")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("verify_v7: OK (all v7 static preconditions satisfied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
