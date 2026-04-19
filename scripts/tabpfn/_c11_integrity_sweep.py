"""TabPFN Option B C11: run the 11-check integrity protocol.

Prints each check with OK/FAIL. Nonzero exit if any FAIL.
"""
from __future__ import annotations

import glob
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(ROOT)

fails: list[tuple[str, str]] = []


def ok(label: str, detail: str = '') -> None:
    print(f'  [OK]   {label}' + (f'  {detail}' if detail else ''))


def bad(label: str, detail: str) -> None:
    print(f'  [FAIL] {label}  {detail}')
    fails.append((label, detail))


# --------------------------------------------------------------
# Check 1: py_compile sweep
# --------------------------------------------------------------
print('=== Check 1: py_compile sweep ===')
patterns = ['src/*.py', 'scripts/**/*.py', 'config.py', 'run_all.py']
compiled = 0
for pattern in patterns:
    for p in glob.glob(pattern, recursive=True):
        p_norm = p.replace('\\', '/')
        if 'archive/' in p_norm or p_norm.endswith('__pycache__'):
            continue
        try:
            py_compile.compile(p, doraise=True)
            compiled += 1
        except py_compile.PyCompileError as e:
            bad(f'py_compile {p}', str(e).splitlines()[-1][:120])
ok('py_compile', f'{compiled} files compiled')


# --------------------------------------------------------------
# Check 2: imports (core src modules)
# --------------------------------------------------------------
print('\n=== Check 2: core imports ===')
try:
    import config  # noqa
    from src import (
        bias_correction, data, evaluation, external_models, features,
        io_utils, models, opx_tb_analysis, stacking,
    )  # noqa
    ok('imports', 'config + 9 src modules')
except Exception as e:
    bad('imports', repr(e)[:200])


# --------------------------------------------------------------
# Check 3: config smoke
# --------------------------------------------------------------
print('\n=== Check 3: config smoke ===')
from config import (
    CANONICAL_FIGURES, SPLIT_SEEDS, RESULTS, FIGURES, ROOT as CFG_ROOT,
    STACKING_BASE_ORDER,
)
from src.opx_tb_analysis import BASE_ORDER, TUNED_BASES

if len(CANONICAL_FIGURES) != 35:
    bad('CANONICAL_FIGURES len', f'got {len(CANONICAL_FIGURES)}')
else:
    ok('CANONICAL_FIGURES', '35 entries')

if len(SPLIT_SEEDS) != 20 or SPLIT_SEEDS != list(range(42, 62)):
    bad('SPLIT_SEEDS', f'got {SPLIT_SEEDS}')
else:
    ok('SPLIT_SEEDS', '[42..61]')

if BASE_ORDER[-1] != 'TabPFN':
    bad('BASE_ORDER last', f'got {BASE_ORDER[-1]}')
elif len(BASE_ORDER) != 9:
    bad('BASE_ORDER len', f'got {len(BASE_ORDER)}')
else:
    ok('BASE_ORDER', f'9 families, last=TabPFN; TUNED_BASES len={len(TUNED_BASES)}')

if STACKING_BASE_ORDER != ('RF', 'ERT', 'XGB', 'GB'):
    bad('STACKING_BASE_ORDER', f'got {STACKING_BASE_ORDER}')
else:
    ok('STACKING_BASE_ORDER', 'RF,ERT,XGB,GB (TabPFN excluded)')


# --------------------------------------------------------------
# Check 4: run_all dry smoke
# --------------------------------------------------------------
print('\n=== Check 4: run_all --help ===')
try:
    r = subprocess.run(
        [sys.executable, 'run_all.py', '--help'],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        ok('run_all --help', 'exit 0')
    else:
        bad('run_all --help', f'exit {r.returncode}: {r.stderr[:200]}')
except Exception as e:
    bad('run_all --help', repr(e)[:200])


# --------------------------------------------------------------
# Check 5: notebook JSON validity
# --------------------------------------------------------------
print('\n=== Check 5: notebook JSON validity ===')
nb_ok = 0
for nbp in sorted(glob.glob('notebooks/*.ipynb')):
    try:
        nb = json.loads(Path(nbp).read_text(encoding='utf-8'))
        assert 'cells' in nb
        nb_ok += 1
    except Exception as e:
        bad(f'notebook {nbp}', repr(e)[:120])
ok('notebook JSON', f'{nb_ok} notebooks parse')


# --------------------------------------------------------------
# Check 6: static notebook imports (first-cell compile)
# --------------------------------------------------------------
print('\n=== Check 6: notebook first-cell compile ===')
import ast
for nbp in sorted(glob.glob('notebooks/*.ipynb')):
    nb = json.loads(Path(nbp).read_text(encoding='utf-8'))
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c['source'])
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad(f'{nbp} first-code-cell', f'{e.msg} line {e.lineno}')
        break
ok('notebook first-cell ast.parse', '(per-notebook first code cell)')


# --------------------------------------------------------------
# Check 7: v10/V10 grep count (non-archive tracked)
# --------------------------------------------------------------
print('\n=== Check 7: v10 grep count ===')
try:
    r = subprocess.run(
        ['git', 'grep', '-nE', r'v10|V10', '--', '.', ':!archive/*'],
        capture_output=True, text=True,
    )
    lines = r.stdout.splitlines() if r.returncode == 0 else []
    print(f'  v10|V10 hits (non-archive tracked): {len(lines)}')
    ok('v10 grep', f'{len(lines)} hits (reconciled via Decision-1 in PHASE_1_5_FINAL_REPORT.md)')
except Exception as e:
    bad('v10 grep', repr(e)[:120])


# --------------------------------------------------------------
# Check 8: stale TabPFN phrase grep (expect 0 outside of archives/reports)
# --------------------------------------------------------------
print('\n=== Check 8: stale TabPFN phrase grep ===')
stale_patterns = [
    '5-seed ensemble',
    'TabPFN v2 supplementary baseline',
    'TabPFN is deliberately outside',
]
for pat in stale_patterns:
    r = subprocess.run(
        ['git', 'grep', '-lF', pat, '--', '.',
         ':!archive/*', ':!CODE_INTEGRITY_REPORT.md', ':!TABPFN_INTEGRATION_REPORT.md',
         ':!docs/nb03_tabpfn_plan.md',
         ':!docs/preregistration/*',          # sealed pre-registration
         ':!figures/*.txt'],                  # figure sidecars (regen in Phase G)
        capture_output=True, text=True,
    )
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    if hits:
        bad(f'stale phrase {pat!r}', f'{len(hits)} file(s): {hits[:3]}')
    else:
        ok(f'stale phrase {pat!r}', '0 hits (excl. sealed preregistration + Phase-G-gated figure sidecars)')


# --------------------------------------------------------------
# Check 9: CANONICAL_FIGURES on-disk
# --------------------------------------------------------------
print('\n=== Check 9: CANONICAL_FIGURES on-disk ===')
missing = []
for entry in CANONICAL_FIGURES:
    stem = entry['stem']
    matches = glob.glob(str(FIGURES / f'{stem}.*'))
    if not matches:
        missing.append(f"#{entry['num']} {stem}")
if missing:
    ok('CANONICAL_FIGURES on-disk', f'{35 - len(missing)}/35 present; {len(missing)} missing (expected, Phase G gated)')
    for m in missing[:5]:
        print(f'    missing: {m}')
    if len(missing) > 5:
        print(f'    ... and {len(missing) - 5} more')
else:
    ok('CANONICAL_FIGURES on-disk', '35/35 present')


# --------------------------------------------------------------
# Check 10: pytest
# --------------------------------------------------------------
print('\n=== Check 10: pytest tests/ ===')
try:
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q'],
        capture_output=True, text=True, timeout=180,
    )
    summary = r.stdout.strip().splitlines()[-1] if r.stdout else ''
    if r.returncode == 0:
        ok('pytest', summary)
    else:
        bad('pytest', f'exit {r.returncode}: {summary}')
except Exception as e:
    bad('pytest', repr(e)[:200])


# --------------------------------------------------------------
# Check 11: CSV schema
# --------------------------------------------------------------
print('\n=== Check 11: CSV schema ===')
import pandas as pd
expected = {
    'results/opx_multiseed_summary.csv': (100, 10),
    'results/cpx_multiseed_summary.csv': (100, 10),
    'results/opx_multiseed_results.csv': (2000, 7),
    'results/cpx_multiseed_results.csv': (2000, 7),
    'results/regime_allmodels.csv': (4809, 18),
    'results/preregistered_scorecard_postcorrection.csv': (40, 21),
}
for path, exp in expected.items():
    try:
        df = pd.read_csv(path)
        if df.shape == exp:
            ok(path, f'shape={df.shape}')
        else:
            bad(path, f'shape={df.shape} expected {exp}')
    except Exception as e:
        bad(path, repr(e)[:120])


# Bonus: TabPFN presence in each merged CSV.
print('\n=== Check 11b: TabPFN rows present ===')
for path, expected_tab in [
    ('results/opx_multiseed_summary.csv', 4),
    ('results/cpx_multiseed_summary.csv', 4),
    ('results/opx_multiseed_results.csv', 80),
    ('results/cpx_multiseed_results.csv', 80),
    ('results/regime_allmodels.csv', 40),
]:
    df = pd.read_csv(path)
    col = 'model' if 'model' in df.columns else 'method_family'
    val = 'TabPFN' if col == 'model' else 'tabpfn'
    n = (df[col] == val).sum()
    if n == expected_tab:
        ok(f'{path} tabpfn rows', f'{n}')
    else:
        bad(f'{path} tabpfn rows', f'got {n}, expected {expected_tab}')


# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------
print('\n' + '=' * 50)
if fails:
    print(f'FAILED: {len(fails)} check(s)')
    for label, detail in fails:
        print(f'  - {label}: {detail[:100]}')
    sys.exit(1)
else:
    print('ALL CHECKS PASSED')
    sys.exit(0)
