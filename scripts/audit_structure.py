"""Structural audit for the opx ML thermobarometer pipeline.

Read-only static checks. Fails loudly if any invariant is violated.
Run:
    python scripts/audit_structure.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / 'notebooks'
SRC = ROOT / 'src'

PASSES: list[str] = []
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    (PASSES if cond else FAILS).append(msg)


# 1. src/ modules present
for name in ['__init__', 'features', 'models', 'data', 'evaluation',
             'geotherm', 'io_utils', 'plot_style']:
    check((SRC / f'{name}.py').exists(), f'src/{name}.py exists')

# 2. expected notebooks present, deprecated ones gone
expected = [
    'nb01_data_cleaning', 'nb02_eda_pca', 'nb03_baseline_models',
    'nb04_putirka_benchmark', 'nb04b_lepr_arcpl_validation',
    'nb05_loso_validation', 'nb06_shap_analysis', 'nb07_bias_correction',
    'nb08_natural_samples', 'nb09_manuscript_compilation',
    'nb10_extended_analyses', 'nbF_figures',
]
for n in expected:
    check((NBDIR / f'{n}.ipynb').exists(), f'{n}.ipynb present')
for n in ['nb03b_n_aug_sensitivity', 'nb03c_phase3_benchmark',
         'nb06b_shap_robustness']:
    check(not (NBDIR / f'{n}.ipynb').exists(), f'{n}.ipynb removed')

# 3. no duplicate feature/prediction defs in notebooks
forbidden_defs = [
    r'def\s+make_pwlr_features',
    r'def\s+make_alr_features',
    r'def\s+predict_median',
    r'def\s+predict_iqr',
    r'def\s+hasterok_chapman_geotherm',
]
for nbp in NBDIR.glob('*.ipynb'):
    nb = nbformat.read(str(nbp), as_version=4)
    text = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    hits = [p for p in forbidden_defs if re.search(p, text)]
    check(not hits, f'{nbp.name}: no duplicate src/ defs (hits={hits})')

# 4. no `from figure_style` import remains
for nbp in NBDIR.glob('*.ipynb'):
    nb = nbformat.read(str(nbp), as_version=4)
    text = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    check('from figure_style' not in text and 'import figure_style' not in text,
          f'{nbp.name}: no figure_style imports')

# 5. nb02 writes clustered parquet, not overwriting core
nb02 = nbformat.read(str(NBDIR / 'nb02_eda_pca.ipynb'), as_version=4)
nb02_text = '\n'.join(c.source for c in nb02.cells if c.cell_type == 'code')
check('opx_clean_core_with_clusters.parquet' in nb02_text,
      'nb02 writes opx_clean_core_with_clusters.parquet')
check('to_parquet(DATA_PROC / "opx_clean_core.parquet"' not in nb02_text
      and "to_parquet(DATA_PROC / 'opx_clean_core.parquet'" not in nb02_text,
      'nb02 does not overwrite opx_clean_core.parquet')

# 6. nb09 does not rewrite table5_shap_importance.csv
nb09 = nbformat.read(str(NBDIR / 'nb09_manuscript_compilation.ipynb'),
                    as_version=4)
nb09_text = '\n'.join(c.source for c in nb09.cells if c.cell_type == 'code')
check('to_csv' not in nb09_text.lower().split('table5_shap_importance')[0][-200:]
      if 'table5_shap_importance' in nb09_text else True,
      'nb09 only reads table5_shap_importance.csv (does not rewrite)')

# 7. nb06 robustness appendix merged
nb06 = nbformat.read(str(NBDIR / 'nb06_shap_analysis.ipynb'), as_version=4)
nb06_text = '\n'.join(c.source for c in nb06.cells)
checks_06 = ['y-randomization', 'ablation', 'perfect']
hit_06 = [t for t in checks_06 if t.lower() in nb06_text.lower()]
check(len(hit_06) == 3, f'nb06 robustness appendix present (found {hit_06})')

# 8. nb03 n_aug sensitivity appendix merged
nb03 = nbformat.read(str(NBDIR / 'nb03_baseline_models.ipynb'), as_version=4)
nb03_text = '\n'.join(c.source for c in nb03.cells)
check('n_aug' in nb03_text.lower() or 'N_AUG' in nb03_text,
      'nb03 n_aug sensitivity appendix present')

# 9. nbF wires hasterok_chapman_geotherm
nbF = nbformat.read(str(NBDIR / 'nbF_figures.ipynb'), as_version=4)
nbF_text = '\n'.join(c.source for c in nbF.cells if c.cell_type == 'code')
check('hasterok_chapman_geotherm' in nbF_text,
      'nbF imports and uses hasterok_chapman_geotherm')

# 10. docs present
check((ROOT / 'README.md').exists(), 'README.md present')
check((ROOT / 'PROJECT_OVERVIEW.md').exists(), 'PROJECT_OVERVIEW.md present')

# 11. config constants defined
config_text = (ROOT / 'config.py').read_text(encoding='utf-8')
for const in ['FEATURE_METHODS', 'N_AUG_DEFAULT', 'QRF_QUANTILES',
              'GEOTHERM_Q_S_AVERAGE', 'OPX_CORE_CLUSTERED_FILE',
              'WINNING_CONFIG_FILE']:
    check(const in config_text, f'config.{const} defined')


print('== PASS ==')
for p in PASSES:
    print(f'  + {p}')
if FAILS:
    print('== FAIL ==')
    for f in FAILS:
        print(f'  - {f}')
print(f'\n{len(PASSES)} pass / {len(FAILS)} fail')
raise SystemExit(0 if not FAILS else 1)
