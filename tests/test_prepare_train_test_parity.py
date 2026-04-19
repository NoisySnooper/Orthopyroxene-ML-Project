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
    from src.v10_phase_c_analysis import prepare_train_test as p
    return p(track, target, feature_set)


def _prep_cpx_legacy(track, target, feature_set):
    from src.cpx_features import build_cpx_feature_matrix
    from src.data import load_cpx_liq, load_cpx_only, load_splits
    df = load_cpx_liq() if track == 'cpx_liq' else load_cpx_only()
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    use_liq = (track == 'cpx_liq')
    X_tr, feat_names = build_cpx_feature_matrix(df_tr, feature_set, use_liq=use_liq)
    X_te, _ = build_cpx_feature_matrix(df_te, feature_set, use_liq=use_liq)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te,
            'feat_names': feat_names}


def _prep_twopx_legacy(track, target, feature_set):
    from src.twopx_features import build_twopx_feature_matrix
    from src.data import load_twopx, load_splits
    df = load_twopx()
    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = build_twopx_feature_matrix(df_tr, feature_set)
    X_te, _ = build_twopx_feature_matrix(df_te, feature_set)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te,
            'feat_names': feat_names}


def _prep_universal_legacy(track, target, feature_set):
    from src.universal_features import build_universal_matrix
    from src.data import load_universal, load_splits
    df = load_universal()
    tr_idx, te_idx = load_splits('universal')
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = build_universal_matrix(df_tr)
    X_te, _ = build_universal_matrix(df_te)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    return {'X_tr': np.asarray(X_tr, float), 'y_tr': y_tr,
            'X_te': np.asarray(X_te, float), 'y_te': y_te,
            'feat_names': feat_names}


LEGACY = {
    'opx':       _prep_opx_legacy,
    'cpx':       _prep_cpx_legacy,
    'twopx':     _prep_twopx_legacy,
    'universal': _prep_universal_legacy,
}


def _collect_cells():
    """Enumerate every (pipeline, track, target, feature_set) combination
    in v10_{pipeline}_multiseed_summary.csv so parity is tested on the
    exact configs used downstream."""
    cells = []
    for pipe in LEGACY:
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
