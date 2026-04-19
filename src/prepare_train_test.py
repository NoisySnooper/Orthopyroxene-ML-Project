"""Unified train/test preparation dispatcher for v10 pipelines.

Factored out 2026-04-18 as part of Phase G.7 (bias correction) to give
every pipeline a single entry point with an identical return schema.

Return schema (consistent across pipelines):
    {
        'X_tr': np.ndarray (n_train, n_features),
        'y_tr': np.ndarray (n_train,),
        'groups_tr': np.ndarray or None,   # Citation strings for grouped CV
        'X_te': np.ndarray (n_test, n_features),
        'y_te': np.ndarray (n_test,),
        'groups_te': np.ndarray or None,
        'feat_names': list[str],
    }

The opx branch delegates to `src.v10_phase_c_analysis.prepare_train_test`
to preserve bit-for-bit parity with existing opx-pipeline results. cpx,
twopx, and universal branches mirror the per-pipeline adapters used in
`scripts/v10_phase_g_multiseed_runner.py` (`_prep_cpx`, `_prep_twopx`,
`_prep_universal`) but additionally pull Citation groups when present.
"""
from __future__ import annotations

import numpy as np

VALID_PIPELINES = ('opx', 'cpx', 'twopx', 'universal')


def prepare_train_test(pipeline: str, track: str, target: str,
                       feature_set: str) -> dict:
    """Load data, apply feature engineering, and return the canonical
    train/test dict for a v10 pipeline.

    Parameters
    ----------
    pipeline : {'opx', 'cpx', 'twopx', 'universal'}
    track    : pipeline-specific track label, e.g. 'opx_liq', 'cpx_only',
               'twopx', 'universal'
    target   : {'T_C', 'P_kbar'}
    feature_set : feature transform, e.g. 'raw', 'alr', 'pwlr',
                  'twopx_components', 'universal_raw'
    """
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f'Unknown pipeline: {pipeline!r}; '
                         f'valid={VALID_PIPELINES}')

    if pipeline == 'opx':
        from src.v10_phase_c_analysis import prepare_train_test as _opx_prep
        return _opx_prep(track, target, feature_set)

    from src.data import load_splits
    if pipeline == 'cpx':
        from src.cpx_features import build_cpx_feature_matrix
        from src.data import load_cpx_liq, load_cpx_only
        df = load_cpx_liq() if track == 'cpx_liq' else load_cpx_only()
        use_liq = (track == 'cpx_liq')
        def _build(dff):
            return build_cpx_feature_matrix(dff, feature_set, use_liq=use_liq)
    elif pipeline == 'twopx':
        from src.twopx_features import build_twopx_feature_matrix
        from src.data import load_twopx
        df = load_twopx()
        def _build(dff):
            return build_twopx_feature_matrix(dff, feature_set)
    else:  # universal
        from src.universal_features import build_universal_matrix
        from src.data import load_universal
        df = load_universal()
        def _build(dff):
            return build_universal_matrix(dff)

    tr_idx, te_idx = load_splits(track)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    X_tr, feat_names = _build(df_tr)
    X_te, _ = _build(df_te)
    y_tr = df_tr[target].to_numpy(dtype=float)
    y_te = df_te[target].to_numpy(dtype=float)
    groups_tr = (df_tr['Citation'].to_numpy() if 'Citation' in df_tr.columns
                 else None)
    groups_te = (df_te['Citation'].to_numpy() if 'Citation' in df_te.columns
                 else None)
    return {
        'X_tr': np.asarray(X_tr, dtype=float),
        'y_tr': y_tr,
        'groups_tr': groups_tr,
        'X_te': np.asarray(X_te, dtype=float),
        'y_te': y_te,
        'groups_te': groups_te,
        'feat_names': feat_names,
    }
