#!/usr/bin/env python3
"""Phase 3b: two-pass terminology cleanup (compound + cosmetic).

Two-pass design fixes the regex-collision frankenwords produced by the
original single-pass sweep. Pass 1 applies compound-aware replacements
with longest patterns first; Pass 2 applies cosmetic fixups to catch
anything that slipped through (double articles, double-hyphens, etc.).

Scope (human-facing text only):
    manuscripts/**/*.md
    figures/**/*.txt
    deliverables/lee_package_20260420/figures/**/*.txt

Intentionally NOT touched:
    - CSV column names (e.g. v10_pre_rmse, v10_post_rmse). Machine contracts.
    - JSON keys.
    - Notebook code cells.
    - Log files.
    - Script filenames (opx_tb_nb03_*.py).
    - The phrase 'ship-if-better' itself (per user preference: clearer).

Writes TERMINOLOGY_CLEANUP_REPORT_V2.md.
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

# ---------------------------------------------------------------------------
# Pass 1: compound patterns (longest-first, collision-avoiding)
# ---------------------------------------------------------------------------
COMPOUND_SUBSTITUTIONS = [
    # v10 + hyphen-sensitive neighbors (fixes the collision frankenwords)
    (r'\bv10 tuned pre\b',           'pre-correction tuned baseline'),
    (r'\bv10 tuned post\b',          'post-correction tuned baseline'),
    (r'\bv10_post[-\s]correction\b', 'post-correction'),
    (r'\bv10 post[-\s]correction\b', 'post-correction'),
    (r'\bv10 pre[-\s]registered\b',  'pre-registered'),
    (r'\bv10 pre[-\s]correction\b',  'pre-correction'),

    # Canonical v10 forms (specific first)
    (r'\bv10 best\b',                'best tuned model'),
    (r'\bv10 RMSE\b',                'tuned RMSE'),
    (r'\bv10_pre\b',                 'pre-correction'),
    (r'\bv10_corrected\b',           'post-correction'),
    (r'\bv10_post\b',                'post-correction'),
    (r'\braw v10\b',                 'raw (uncorrected) tuned baseline'),
    (r'\bv10 pre\b',                 'pre-correction'),
    (r'\bv10 post\b',                'post-correction'),
    (r'\bv10 pipeline\b',            'the tuned ML pipeline'),
    (r'\bv10\b',                     'the tuned ML baseline'),

    # v9 handling
    (r'\bv9 finding\b',              'earlier finding'),
    (r'\bv9 empty\b',                'earlier empty'),
    (r'\bv9\b',                      'the previous iteration'),

    # Internal phase labels -> neutral prose
    (r'\bPhase[- ]?G\.?\d+[a-z]?\b', 'the final phase'),
    (r'\bPhase[- ]?G\b',             'the final phase'),
    (r'\bPhase[- ]?\d+\.?\d*[a-z]?\b', 'the relevant phase'),
    (r'\bphase g\.?\d+[a-z]?\b',     'the final phase'),

    # Coinages -> standard terminology
    # (intentionally NOT substituting 'ship-if-better' itself -- user finds it clearer)
    (r'\btwo-axis honesty bar\b',    'dual robustness check'),
    (r'\bno-degradation rule\b',     'no-regime-worsens requirement'),
    (r'\b"?ships A/B"?\b',           'accepts Form A or Form B'),
    (r'\b"?ships A"?\b',             'accepts Form A'),
    (r'\b"?ships B"?\b',             'accepts Form B'),

    # opx-tb compounds
    (r'\bopx-tb\b',                  'the opx ML thermobarometer'),
    (r'\bopx_tb\b',                  'opx ML thermobarometer'),
]

# ---------------------------------------------------------------------------
# Pass 2: cosmetic fixups (catch Pass-1 collisions, double articles)
# ---------------------------------------------------------------------------
COSMETIC_SUBSTITUTIONS = [
    # Double-article bugs from Pass 1
    (r'\bthe the tuned ML baseline\b',  'the tuned ML baseline'),
    (r'\bthe the tuned\b',              'the tuned'),
    (r'\bthe the final phase\b',        'the final phase'),
    (r'\bthe the relevant phase\b',     'the relevant phase'),
    (r'\bthe the previous iteration\b', 'the previous iteration'),
    (r'\bthe the\b',                    'the'),

    # Franken-word fallbacks (compounded correction suffix)
    (r'\bpost-correction-correction\b', 'post-correction'),
    (r'\bpre-correction-correction\b',  'pre-correction'),
    (r'\bpre-correction-registered\b',  'pre-registered'),
    (r'\btuned ML baseline tuned pre-correction\b',  'pre-correction tuned baseline'),
    (r'\btuned ML baseline tuned post-correction\b', 'post-correction tuned baseline'),
    (r'\btuned ML baseline tuned (pre|post)-correction\b', r'\1-correction tuned baseline'),
]

TARGETS = [
    (PROJECT_ROOT / 'manuscripts', '*.md'),
    (PROJECT_ROOT / 'figures', '*.txt'),
    (PROJECT_ROOT / 'deliverables' / 'lee_package_20260420' / 'figures', '*.txt'),
]

REPORT_PATH = PROJECT_ROOT / 'TERMINOLOGY_CLEANUP_REPORT_V2.md'


@dataclass
class Hit:
    path: Path
    pattern: str
    replacement: str
    count: int
    pass_num: int


def apply_subs(text: str, subs: list[tuple[str, str]]
               ) -> tuple[str, list[tuple[str, str, int]]]:
    records = []
    for pat, rep in subs:
        new_text, n = re.subn(pat, rep, text)
        if n:
            records.append((pat, rep, n))
            text = new_text
    return text, records


def main() -> int:
    hits: list[Hit] = []
    files_touched = 0
    for root, glob in TARGETS:
        if not root.exists():
            continue
        for p in root.rglob(glob):
            try:
                orig = p.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            # Pass 1
            after1, recs1 = apply_subs(orig, COMPOUND_SUBSTITUTIONS)
            # Pass 2 (run on Pass-1 output even if Pass 1 was empty -- catches stale frankenwords)
            after2, recs2 = apply_subs(after1, COSMETIC_SUBSTITUTIONS)
            if after2 != orig:
                p.write_text(after2, encoding='utf-8')
                files_touched += 1
                for pat, rep, n in recs1:
                    hits.append(Hit(p, pat, rep, n, 1))
                for pat, rep, n in recs2:
                    hits.append(Hit(p, pat, rep, n, 2))

    # Write report
    lines = [
        '# Terminology cleanup report v2 (two-pass)',
        '',
        'Generated 2026-04-20 by `scripts/audits/terminology_cleanup.py`.',
        '',
        '## Pass 1 rules (compound, longest-first)',
        '',
    ]
    for pat, rep in COMPOUND_SUBSTITUTIONS:
        lines.append(f'- `{pat}` -> `{rep}`')
    lines.extend([
        '',
        '## Pass 2 rules (cosmetic fixups)',
        '',
    ])
    for pat, rep in COSMETIC_SUBSTITUTIONS:
        lines.append(f'- `{pat}` -> `{rep}`')
    lines.extend([
        '',
        '## Targets',
        '',
    ])
    for root, glob in TARGETS:
        try:
            rel = root.relative_to(PROJECT_ROOT)
            lines.append(f'- `{rel}/**/{glob}`')
        except ValueError:
            lines.append(f'- `{root}/**/{glob}`')
    lines.extend([
        '',
        '## Files modified',
        '',
        f'Total: {files_touched} files, {sum(h.count for h in hits)} replacements.',
        '',
    ])
    by_file: dict[Path, list[Hit]] = {}
    for h in hits:
        by_file.setdefault(h.path, []).append(h)
    if not by_file:
        lines.append('(no replacements needed)')
    for path, grp in sorted(by_file.items()):
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = path
        lines.append(f'### `{rel}`')
        lines.append('')
        for h in grp:
            lines.append(f'- [pass {h.pass_num}] `{h.pattern}` -> `{h.replacement}` (x{h.count})')
        lines.append('')

    lines.extend([
        '## Intentionally NOT changed',
        '',
        '- CSV column names (e.g., `v10_pre_rmse`, `v10_post_rmse`). Machine contracts.',
        '- JSON keys in `form_a_params` / `form_b_params` fields.',
        '- Log files under `logs/` (archival of what was run).',
        '- Notebook code cells (identifiers follow CSV columns).',
        '- Script filenames (`opx_tb_nb03_*.py`).',
        '- The phrase `ship-if-better` itself (user preference: clearer).',
    ])

    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {REPORT_PATH}')
    print(f'files modified: {files_touched}')
    print(f'total replacements: {sum(h.count for h in hits)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
