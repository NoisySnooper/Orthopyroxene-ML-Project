#!/usr/bin/env python3
"""Compare GEOROC opx and cpx precompiled file schemas.

Read-only diagnostic. Produces:
  * Columns present in opx but not cpx
  * Columns present in cpx but not opx
  * Shared columns (safe to share ingest logic across)

Run this after Phase H.1b retrieves the cpx file. Until then, the cpx
file does not exist and this script will exit with an informative
error message (not a crash).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_NATURAL

OPX = DATA_NATURAL / '2024-12-SGFTFN_ORTHOPYROXENES.csv'
CPX = DATA_NATURAL / '2024-12-SGFTFN_CLINOPYROXENES.csv'


def read_header(path):
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist. Run Phase H.1b (pygeoroc download) '
            'to retrieve the GEOROC CPX precompiled file first.')
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            return set(pd.read_csv(path, nrows=0, encoding=enc).columns), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f'Cannot decode {path}')


def main():
    opx_cols, opx_enc = read_header(OPX)
    cpx_cols, cpx_enc = read_header(CPX)

    only_opx = sorted(opx_cols - cpx_cols)
    only_cpx = sorted(cpx_cols - opx_cols)
    shared = sorted(opx_cols & cpx_cols)

    print(f'OPX encoding: {opx_enc}, columns: {len(opx_cols)}')
    print(f'CPX encoding: {cpx_enc}, columns: {len(cpx_cols)}')
    print(f'Shared: {len(shared)}')
    print()
    print(f'Only in OPX ({len(only_opx)}):')
    for c in only_opx:
        print(f'  {c}')
    print()
    print(f'Only in CPX ({len(only_cpx)}):')
    for c in only_cpx:
        print(f'  {c}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
