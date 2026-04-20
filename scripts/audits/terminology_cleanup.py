#!/usr/bin/env python3
"""Phase 3: replace internal coinages with standard terminology.

Substitutions in manuscript prose + caption sidecars (NOT CSV data
columns, NOT code, NOT .ipynb metadata, NOT log files):

    'v10 best'        -> 'best tuned model'
    'v10 RMSE'        -> 'tuned RMSE'
    'v10_pre'         -> 'pre-correction' (in prose rows / tables only)
    'v10_corrected'   -> 'post-correction'
    'v10 '            -> 'the tuned ML baseline ' (generic prose cases)
    'opx-tb'          -> 'the opx ML thermobarometer'
    'opx_tb'          -> 'opx ML thermobarometer'

CSV column names and JSON keys are left untouched (they're machine
contracts). Only human-facing text is changed.

Writes TERMINOLOGY_CLEANUP_REPORT.md summarizing every replacement.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Order matters: longer/more-specific patterns first so they don't get
# partially rewritten by later generic patterns.
SUBSTITUTIONS = [
    (r'\bv10 best\b',       'best tuned model'),
    (r'\bv10 RMSE\b',       'tuned RMSE'),
    (r'\bv10_pre\b',        'pre-correction'),
    (r'\bv10_corrected\b',  'post-correction'),
    (r'\bv10_post\b',       'post-correction'),
    (r'\braw v10\b',        'raw (uncorrected) tuned baseline'),
    (r'\bv10 pre\b',        'pre-correction'),
    (r'\bv10 post\b',       'post-correction'),
    (r'\bv10\b',            'the tuned ML baseline'),
    (r'\bopx-tb\b',         'the opx ML thermobarometer'),
    (r'\bopx_tb\b',         'opx ML thermobarometer'),
]

# Target directories / file globs.
TARGETS = [
    (PROJECT_ROOT / 'manuscripts', '*.md'),
    (PROJECT_ROOT / 'figures', '*.txt'),
]

REPORT_PATH = PROJECT_ROOT / 'TERMINOLOGY_CLEANUP_REPORT.md'


@dataclass
class Hit:
    path: Path
    pattern: str
    replacement: str
    count: int


def apply_subs(text: str) -> tuple[str, list[tuple[str, str, int]]]:
    records = []
    for pat, rep in SUBSTITUTIONS:
        new_text, n = re.subn(pat, rep, text)
        if n:
            records.append((pat, rep, n))
            text = new_text
    return text, records


def main() -> int:
    hits: list[Hit] = []
    for root, glob in TARGETS:
        if not root.exists():
            continue
        for p in root.rglob(glob):
            try:
                orig = p.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            new, records = apply_subs(orig)
            if records:
                p.write_text(new, encoding='utf-8')
                for pat, rep, n in records:
                    hits.append(Hit(p, pat, rep, n))

    # Write report.
    lines = [
        '# Terminology cleanup report',
        '',
        f'Generated 2026-04-20 by `scripts/audits/terminology_cleanup.py`.',
        '',
        '## Replacement rules',
        '',
    ]
    for pat, rep in SUBSTITUTIONS:
        lines.append(f'- `{pat}` -> `{rep}`')
    lines.extend([
        '',
        '## Targets',
        '',
    ])
    for root, glob in TARGETS:
        lines.append(f'- `{root.relative_to(PROJECT_ROOT)}/**/{glob}`')
    lines.extend([
        '',
        '## Files modified',
        '',
    ])
    by_file: dict[Path, list[Hit]] = {}
    for h in hits:
        by_file.setdefault(h.path, []).append(h)
    if not by_file:
        lines.append('(no replacements needed)')
    for path, grp in sorted(by_file.items()):
        rel = path.relative_to(PROJECT_ROOT)
        lines.append(f'### `{rel}`')
        lines.append('')
        for h in grp:
            lines.append(f'- `{h.pattern}` -> `{h.replacement}` (x{h.count})')
        lines.append('')

    lines.extend([
        '## Intentionally NOT changed',
        '',
        '- CSV column names (e.g., `v10_pre_rmse`, `v10_post_rmse` in '
        '`results/preregistered_scorecard_postcorrection.csv`). These are '
        'machine contracts consumed by code; renaming would break every '
        'downstream script.',
        '- JSON keys in `results/bias_correction_shipped.csv` `form_a_params` '
        'and `form_b_params` fields.',
        '- Log files under `logs/` (archival of what was run).',
        '- Notebook code cells (code identifier names follow CSV columns).',
        '- Script filenames (`opx_tb_nb03_*.py`) — these are stable path '
        'contracts; renaming would orphan log cross-references.',
    ])

    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {REPORT_PATH}')
    print(f'files modified: {len(by_file)}')
    print(f'total replacements: {sum(h.count for h in hits)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
