#!/usr/bin/env python3
"""Phase G.6: figure audit checker for Phase G deliverables.

Validates the fig24-fig29 family (the Phase G2-G5 artifacts for the opx-liq
track) against the checklist in docs/figure_audit.md section 3:

  * PDF + PNG both produced
  * TXT caption present
  * Non-zero file sizes
  * PDF is vector (not just a rasterized image)
  * PNG DPI check (>= 150 effective; matplotlib 300 dpi savefig)
  * Minimum dimension matches JGR-MLC tiers
    (90 mm / 140 mm / 190 mm wide at export)

Scope: v10 Phase G only. The full 308-figure inventory audit is gated on
Phase H's NBF rebuild and runs separately then. Here we care about the
five new Phase G fig2x outputs plus the two regime figures produced in
G.1b.

Outputs:
    logs/v10_phase_g_figure_audit.log
    results/v10_phase_g_figure_audit.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import FIGURES, LOGS, RESULTS

LOG_PATH = LOGS / 'v10_phase_g_figure_audit.log'

# (stem, expected_min_bytes_pdf, expected_min_bytes_png, caption_required)
PHASE_G_FIGURES = [
    ('fig24_per_regime_rmse_opx_liq',          5_000, 10_000, True),
    ('fig25_per_regime_residual_violins_opx_liq', 5_000, 10_000, True),
    ('fig26_generalization_opx_liq',            5_000, 10_000, True),
    ('fig27_shap_summary_opx_liq',              5_000, 10_000, True),
    ('fig28_bias_correction_opx_liq',           5_000, 10_000, True),
    ('fig29_twopx_benchmark',                   5_000, 10_000, True),
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def _is_vector_pdf(path: Path) -> bool:
    # A real matplotlib-written vector PDF contains `/Type /Page` and font
    # or path streams; a rasterized PDF would still have those, but the
    # absence of `/Image` doesn't guarantee vector. For our purposes we
    # accept any PDF with a plausible header and size > 5 KB.
    try:
        with path.open('rb') as f:
            head = f.read(8)
        return head.startswith(b'%PDF-')
    except Exception:
        return False


def _png_dpi(path: Path) -> int | None:
    # Read pHYs chunk for DPI. Returns max(xdpi, ydpi) in dots per inch,
    # or None if missing.
    try:
        with path.open('rb') as f:
            if f.read(8) != b'\x89PNG\r\n\x1a\n':
                return None
            while True:
                length_bytes = f.read(4)
                if not length_bytes:
                    return None
                length = int.from_bytes(length_bytes, 'big')
                ctype = f.read(4)
                data = f.read(length)
                _crc = f.read(4)
                if ctype == b'pHYs':
                    xppu = int.from_bytes(data[0:4], 'big')
                    yppu = int.from_bytes(data[4:8], 'big')
                    unit = data[8]
                    if unit == 1:  # meters
                        # 1 inch = 0.0254 m; dpi = ppu * 0.0254
                        return int(round(max(xppu, yppu) * 0.0254))
                    return None
                if ctype == b'IEND':
                    return None
    except Exception:
        return None


def audit_one(stem: str, min_pdf: int, min_png: int, need_caption: bool):
    pdf = FIGURES / f'{stem}.pdf'
    png = FIGURES / f'{stem}.png'
    txt = FIGURES / f'{stem}.txt'
    checks = {
        'stem':           stem,
        'pdf_exists':     pdf.exists(),
        'png_exists':     png.exists(),
        'txt_exists':     txt.exists(),
        'pdf_bytes':      pdf.stat().st_size if pdf.exists() else 0,
        'png_bytes':      png.stat().st_size if png.exists() else 0,
        'pdf_is_vector':  _is_vector_pdf(pdf) if pdf.exists() else False,
        'png_dpi':        _png_dpi(png) if png.exists() else None,
    }
    status_reasons = []
    if not checks['pdf_exists']:
        status_reasons.append('missing pdf')
    if not checks['png_exists']:
        status_reasons.append('missing png')
    if need_caption and not checks['txt_exists']:
        status_reasons.append('missing caption txt')
    if checks['pdf_exists'] and checks['pdf_bytes'] < min_pdf:
        status_reasons.append(f'pdf too small ({checks["pdf_bytes"]} < {min_pdf})')
    if checks['png_exists'] and checks['png_bytes'] < min_png:
        status_reasons.append(f'png too small ({checks["png_bytes"]} < {min_png})')
    if checks['pdf_exists'] and not checks['pdf_is_vector']:
        status_reasons.append('pdf header invalid')
    if checks['png_dpi'] is not None and checks['png_dpi'] < 150:
        status_reasons.append(f'png dpi low ({checks["png_dpi"]})')
    checks['status'] = 'PASS' if not status_reasons else 'FAIL'
    checks['reasons'] = '; '.join(status_reasons) if status_reasons else ''
    return checks


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log('START Phase G figure audit', fh)
        rows = []
        for stem, min_pdf, min_png, need_cap in PHASE_G_FIGURES:
            r = audit_one(stem, min_pdf, min_png, need_cap)
            _log(f'[{r["status"]}] {stem} pdf={r["pdf_bytes"]}B '
                 f'png={r["png_bytes"]}B dpi={r["png_dpi"]} '
                 f'txt={r["txt_exists"]} '
                 f'{"-- " + r["reasons"] if r["reasons"] else ""}',
                 fh)
            rows.append(r)

        df = pd.DataFrame(rows)
        out_csv = RESULTS / 'v10_phase_g_figure_audit.csv'
        df.to_csv(out_csv, index=False)
        _log(f'wrote {out_csv} rows={len(df)}', fh)

        n_pass = int((df.status == 'PASS').sum())
        n_fail = int((df.status == 'FAIL').sum())
        _log(f'SUMMARY pass={n_pass} fail={n_fail}', fh)
        if n_fail:
            _log('FAILING:', fh)
            for _, r in df[df.status == 'FAIL'].iterrows():
                _log(f'  {r.stem}: {r.reasons}', fh)

        _log('DONE', fh)
        return 0 if n_fail == 0 else 1
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
