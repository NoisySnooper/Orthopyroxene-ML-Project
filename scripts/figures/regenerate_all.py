#!/usr/bin/env python3
"""Regenerate every kept figure builder.

Honors `JGR_PRINT=1` to redirect output to `figures/opx_only/jgr/` and
shrink figsize to AGU 2-column print width (6.89 in).

Usage:
    # Preview (default):
    python scripts/figures/regenerate_all.py
    # JGR submission size:
    JGR_PRINT=1 python scripts/figures/regenerate_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

BUILDERS = [
    'scripts/figures/make_fig_dataset_map.py',
    'scripts/figures/make_fig_methods_flowchart.py',
    'scripts/figures/make_fig_opx_heatmap.py',
    'scripts/figures/make_fig_opx_regime_families.py',
    'scripts/figures/make_opx_headline_fig.py',
    'scripts/figures/make_fig_bias_correction_opx.py',
    'scripts/figures/make_fig_scorecard_delta_opx.py',
    'scripts/figures/make_fig_shap_winners.py',
    'scripts/figures/make_fig_arcpl_bias_corrected_vs_putirka.py',
    'scripts/figures/make_fig_dataset_map_holdout.py',
    'scripts/figures/make_fig_opx_overall_families.py',
    'scripts/figures/make_fig_bias_residuals_opx.py',
    'scripts/figures/make_fig_pca_coverage.py',
    'scripts/interpretability/make_fig_feature_concordance.py',
    'scripts/interpretability/classical_equivalence_regression.py',
    'scripts/interpretability/partial_dependence_across_families.py',
    'scripts/interpretability/surrogate_decision_trees.py',
]


def main() -> int:
    mode = 'JGR' if os.environ.get('JGR_PRINT') == '1' else 'preview'
    print(f'Regenerating {len(BUILDERS)} figures in {mode} mode')
    failed = []
    for b in BUILDERS:
        rc = subprocess.call([sys.executable, b], cwd=str(PROJECT_ROOT))
        status = 'ok' if rc == 0 else f'FAIL ({rc})'
        print(f'  [{status}] {b}')
        if rc != 0:
            failed.append(b)
    if failed:
        print(f'\n{len(failed)} builder(s) failed:')
        for b in failed:
            print(f'  {b}')
        return 1
    print(f'\nAll {len(BUILDERS)} builders completed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
