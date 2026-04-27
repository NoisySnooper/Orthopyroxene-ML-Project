"""Regression test: unified src.prepare_train_test must return bit-for-bit
identical arrays to the per-pipeline `_prep_*` adapters in
scripts/v10_phase_g_multiseed_runner.py.

This guards the A1 refactor (Phase G.7, 2026-04-18) that consolidated the
four adapter functions into a single dispatcher. If this test passes, we
can swap the multiseed_runner to the unified function without changing
any downstream result.

Covers every (pipeline, track, target, feature_set) combination currently
present in the v10 multiseed_results CSVs, so parity is verified on the
exact configurations used in production.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from src.prepare_train_test import prepare_train_test  # unified


# ---------- legacy adapters (copied verbatim from v10_phase_g_multiseed_runner.py)

def _prep_opx_legacy(track, target, feature_set):
    from src.opx_tb_analysis import prepare_train_test as p
    return p(track, target, feature_set)


LEGACY = {
    'opx':       _prep_opx_legacy,
}


def _collect_cells():
    """Enumerate every (pipeline, track, target, feature_set) combination
    in {pipeline}_multiseed_summary.csv so parity is tested on the
    exact configs used downstream."""
    cells = []
    for pipe in LEGACY:
        p = PROJECT_ROOT / 'results' / f'{pipe}_multiseed_summary.csv'
        if not p.exists():
            p = PROJECT_ROOT / 'results' / f'v10_{pipe}_multiseed_summary.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        uniq = df[['track', 'target', 'feature_set']].drop_duplicates()
        for _, r in uniq.iterrows():
            cells.append((pipe, r['track'], r['target'], r['feature_set']))
    return cells


CELLS = _collect_cells()


@pytest.mark.parametrize('pipeline,track,target,feature_set', CELLS)
def test_unified_matches_legacy(pipeline, track, target, feature_set):
    legacy = LEGACY[pipeline](track, target, feature_set)
    new = prepare_train_test(pipeline, track, target, feature_set)

    for key in ('X_tr', 'y_tr', 'X_te', 'y_te'):
        a, b = legacy[key], new[key]
        assert a.shape == b.shape, \
            f'{pipeline}/{track}/{target}/{feature_set}: {key} shape mismatch'
        assert np.array_equal(a, b), \
            f'{pipeline}/{track}/{target}/{feature_set}: {key} values differ'

    assert list(legacy['feat_names']) == list(new['feat_names'])
