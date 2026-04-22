#!/usr/bin/env python3
"""Phase 3: build a Word doc from the draft manuscript sections.

Concatenates `manuscripts/opx_2026/text/draft/sections/*.md` in order
and renders to `manuscripts/opx_2026/arxiv_submission/manuscript.docx`.

Renders markdown headings (##, ###, ####), paragraphs, **bold**,
*italics*, `inline code`, fenced code blocks, and bullet lists. Complex
content (tables, images) is not rendered — this doc is a copy-editing
substrate, not a publication artifact.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_LINE_SPACING

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

SECTIONS_DIR = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'text' / 'draft' / 'sections'
DOCX_DIR = PROJECT_ROOT / 'manuscripts' / 'opx_2026' / 'arxiv_submission'
DOCX_OUT = DOCX_DIR / 'manuscript.docx'

ORDER = [
    '00_abstract.md',
    '00a_front_matter.md',
    '01_introduction.md',
    '02_data.md',
    '03_methods.md',
    '04_results.md',
    '05_discussion.md',
    '06_conclusions.md',
    '07_data_and_code.md',
    '08_acknowledgments_and_credit.md',
    '09_cover_letter.md',
]


def add_inline(paragraph, text: str):
    """Render **bold**, *italic*, `code` inline in a paragraph."""
    parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text)
    for p in parts:
        if not p:
            continue
        if p.startswith('**') and p.endswith('**'):
            r = paragraph.add_run(p[2:-2]); r.bold = True
        elif p.startswith('*') and p.endswith('*'):
            r = paragraph.add_run(p[1:-1]); r.italic = True
        elif p.startswith('`') and p.endswith('`'):
            r = paragraph.add_run(p[1:-1])
            r.font.name = 'Consolas'; r.font.size = Pt(10)
        else:
            paragraph.add_run(p)


def render_markdown(doc: Document, md: str):
    lines = md.split('\n')
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                p = doc.add_paragraph()
                r = p.add_run('\n'.join(code_buf))
                r.font.name = 'Consolas'; r.font.size = Pt(9)
            i += 1
            continue
        if in_code:
            code_buf.append(ln); i += 1; continue
        if ln.startswith('#### '):
            h = doc.add_heading(ln[5:], level=4)
            i += 1; continue
        if ln.startswith('### '):
            h = doc.add_heading(ln[4:], level=3)
            i += 1; continue
        if ln.startswith('## '):
            h = doc.add_heading(ln[3:], level=2)
            i += 1; continue
        if ln.startswith('# '):
            h = doc.add_heading(ln[2:], level=1)
            i += 1; continue
        if ln.startswith('- ') or ln.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            add_inline(p, ln[2:])
            i += 1; continue
        if re.match(r'^\d+\.\s', ln):
            p = doc.add_paragraph(style='List Number')
            add_inline(p, re.sub(r'^\d+\.\s+', '', ln))
            i += 1; continue
        if ln.strip() == '':
            i += 1; continue
        # collect a paragraph across continuation lines
        buf = [ln]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not (
            lines[j].startswith('#') or lines[j].startswith('- ') or
            lines[j].startswith('* ') or re.match(r'^\d+\.\s', lines[j])
            or lines[j].startswith('```')
        ):
            buf.append(lines[j]); j += 1
        p = doc.add_paragraph()
        add_inline(p, ' '.join(buf))
        i = j


def main():
    DOCX_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    # base style
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE

    for fname in ORDER:
        fp = SECTIONS_DIR / fname
        if not fp.exists():
            print(f'MISSING: {fp}'); continue
        print(f'  + {fname}')
        md = fp.read_text(encoding='utf-8')
        render_markdown(doc, md)
        doc.add_page_break()

    doc.save(DOCX_OUT)
    print(f'wrote {DOCX_OUT}')


if __name__ == '__main__':
    main()
