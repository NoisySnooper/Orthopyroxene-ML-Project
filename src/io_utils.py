"""Shared IO helpers for figures, tables, and progress reporting.

`save_figure` and `save_table` centralize the write-to-disk conventions:
PNG plus PDF at 300 dpi alongside an optional caption `.txt`, and CSV
with a leading index if one is present. `with_progress` wraps an
iterable in tqdm using the project's house style.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from config import FIGURES, RESULTS


def save_figure(fig, name, dir=None, dpi=300, caption=''):
    """Write `fig` to `figures/<name>.png` and `.pdf`. If `caption` is
    non-empty, also write `<name>.txt` with the caption text."""
    target = Path(dir) if dir is not None else Path(FIGURES)
    stem = target / name
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + '.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
    if caption:
        with open(str(stem) + '.txt', 'w', encoding='utf-8') as fh:
            fh.write(caption.strip() + '\n')


def save_table(df, name, dir=None, index=False):
    """Write a DataFrame to `results/<name>.csv`."""
    target = Path(dir) if dir is not None else Path(RESULTS)
    target.mkdir(parents=True, exist_ok=True)
    if not name.endswith('.csv'):
        name = name + '.csv'
    df.to_csv(target / name, index=index)


def with_progress(iterable, desc='', position=0, leave=True, total=None):
    """Wrap `iterable` in tqdm with a consistent house style. For nested
    loops pass `position` and `leave=False` to the inner call."""
    return tqdm(iterable, desc=desc, position=position, leave=leave,
                total=total, ncols=88)
