#!/usr/bin/env python3
"""Fill the auto-fill blocks in manuscripts/opx_2026/text/tabpfn_paragraph.md.

Reads:
    results/tabpfn_head_to_head.csv
    results/tabpfn_multiseed_summary.csv

Rewrites the [AUTO-FILL block ...] placeholders in-place while preserving
framing prose around them. Verdict selection for Paragraph 3 is determined
from the count of cells where verdict == 'tabpfn_wins', 'v10_wins',
or 'competitive'.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import RESULTS

PARAGRAPH_PATH = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'tabpfn_paragraph.md'


def _fmt(m, prec):
    if m is None or pd.isna(m):
        return '--'
    return f'{float(m):.{prec}f}'


def _fmt_pair(m, s, prec):
    ms = _fmt(m, prec)
    if s is None or pd.isna(s) or s == 0:
        return ms
    return f'{ms} +/- {_fmt(s, prec)}'


def build_para2(h2h: pd.DataFrame) -> str:
    lines = []
    lines.append('Table A summarizes the head-to-head. Reading across the '
                 'eight cells:\n')
    lines.append('')
    lines.append('| track | target | v10 best | v10 RMSE | TabPFN RMSE | '
                 'external best | verdict |')
    lines.append('|---|---|---|---|---|---|---|')
    def _s(v, default='--'):
        return default if v is None or pd.isna(v) else str(v)
    for _, r in h2h.iterrows():
        prec = 2 if r['target'] == 'P_kbar' else 1
        unit = 'kbar' if r['target'] == 'P_kbar' else 'C'
        ext_m = _s(r.get('external_best_method'))
        ext_r = _fmt(r.get('external_best_rmse'), prec)
        ext_cell = f"{ext_m} ({ext_r} {unit})" if ext_m != '--' else '--'
        lines.append(
            f"| {r['track']} | {r['target']} | "
            f"{_s(r.get('tuned_best_model'))}/{_s(r.get('tuned_best_fs'))} | "
            f"{_fmt_pair(r.get('tuned_best_rmse'), r.get('tuned_best_std'), prec)} {unit} | "
            f"{_fmt_pair(r.get('tabpfn_rmse'), r.get('tabpfn_std'), prec)} {unit} | "
            f"{ext_cell} | "
            f"{_s(r.get('verdict'))} |"
        )
    return '\n'.join(lines)


def build_para3(h2h: pd.DataFrame) -> str:
    counts = h2h['verdict'].value_counts().to_dict()
    tab_wins = counts.get('tabpfn_wins', 0)
    v10_wins = counts.get('v10_wins',   0)
    comp     = counts.get('competitive', 0)
    total    = int(len(h2h))

    header = (f'Across the {total} cells, TabPFN wins {tab_wins}, '
              f'v10 wins {v10_wins}, {comp} are competitive within '
              f'0.5 standard errors.')

    if tab_wins >= 4:
        body = (
            'Foundation-model pretraining has closed the gap with bespoke '
            'domain tuning on this class of small-data igneous-petrology '
            'regression. Our contribution in that case is the pre-registered '
            'regime-stratified evaluation framework and the Form A bias '
            'correction (Section 4.3), both of which are model-agnostic and '
            'apply to TabPFN as cleanly as to the v10 tuned ensemble.'
        )
    elif v10_wins >= 6:
        body = (
            'Domain-specific feature engineering and hyperparameter tuning '
            'still matter at n between 600 and 2400 for experimental '
            'petrology. TabPFN\'s synthetic-data priors cover generic tabular '
            'nonlinearities well but may not capture silicate melt '
            'compositional manifolds, where feature sparsity and '
            'citation-group clustering create structure that the pretraining '
            'distribution did not sample. This is consistent with the '
            'ordinary-pattern-versus-specialty-data distinction drawn in the '
            'TabPFN paper\'s Discussion.'
        )
    else:
        body = (
            'TabPFN validates the v10 pipeline\'s competitiveness. A '
            'pretrained foundation model reaches the same accuracy envelope '
            'without bespoke hyperparameter tuning, which is a useful lower '
            'bound on what a well-engineered ML pipeline on this data set '
            'should achieve. The paper\'s primary contribution remains the '
            'regime-stratified evaluation and bias correction, both of which '
            'sit above the choice of base predictor.'
        )
    return header + ' ' + body


def main() -> int:
    h2h_path = RESULTS / 'tabpfn_head_to_head.csv'
    if not h2h_path.exists():
        print(f'[error] {h2h_path} not found; run nb04 head-to-head cell first',
              file=sys.stderr)
        return 2
    h2h = pd.read_csv(h2h_path)

    text = PARAGRAPH_PATH.read_text(encoding='utf-8')

    # Replace Paragraph 2 auto-fill block.
    p2_sentinel_start = '[AUTO-FILL block'
    p3_sentinel_start = '[AUTO-FILL chooses'
    s2 = text.index(p2_sentinel_start)
    e2 = text.index(']', s2) + 1
    text = text[:s2] + build_para2(h2h) + text[e2:]

    # Replace Paragraph 3 auto-fill block.
    s3 = text.index(p3_sentinel_start)
    # Find the closing bracket for this block; naive: last ']' in the file
    # before "## Caveats" heading.
    caveats_i = text.index('## Caveats', s3)
    e3 = text.rindex(']', s3, caveats_i) + 1
    text = text[:s3] + build_para3(h2h) + text[e3:]

    PARAGRAPH_PATH.write_text(text, encoding='utf-8')
    print(f'wrote {PARAGRAPH_PATH} ({len(h2h)} cells, '
          f'counts={h2h["verdict"].value_counts().to_dict()})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
