"""Smoke test: extract each notebook's import cells and execute them in
isolation to confirm nothing is broken before attempting a full run.

Writes per-notebook stub scripts to `build_smoke/` and runs each.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / 'notebooks'
VENV_PY = ROOT / '.venv' / 'Scripts' / 'python.exe'
BUILD = ROOT / 'build_smoke'
BUILD.mkdir(exist_ok=True)

TARGETS = [
    'nb01_data_cleaning', 'nb02_eda_pca', 'nb03_baseline_models',
    'nb04_putirka_benchmark', 'nb04b_lepr_arcpl_validation',
    'nb05_loso_validation', 'nb06_shap_analysis', 'nb07_bias_correction',
    'nb08_natural_twopx', 'nb09_manuscript_compilation', 'nbF_figures',
]


def is_imports_cell(src: str) -> bool:
    return ('import ' in src or 'from src' in src
            or 'from config' in src or 'sys.path' in src)


for name in TARGETS:
    nb = nbformat.read(str(NBDIR / f'{name}.ipynb'), as_version=4)
    imports_srcs = []
    for c in nb.cells:
        if c.cell_type != 'code':
            continue
        if is_imports_cell(c.source) and len(c.source) < 4000:
            imports_srcs.append(c.source)
        if len(imports_srcs) >= 3:
            break

    combined = '\n'.join(imports_srcs)
    stub = BUILD / f'{name}_imports.py'
    stub.write_text(
        f"import os\nos.chdir(r\"{NBDIR}\")\n" + combined,
        encoding='utf-8',
    )
    r = subprocess.run(
        [str(VENV_PY), str(stub)],
        capture_output=True, text=True, timeout=60,
    )
    status = 'OK' if r.returncode == 0 else 'FAIL'
    print(f'{name}: {status}')
    if r.returncode != 0:
        err = r.stderr.strip().split('\n')[-5:]
        for ln in err:
            print(f'    {ln}')
