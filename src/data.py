"""Loaders for processed data, splits, and canonical model bookkeeping.

Paths come from `config.py` so notebooks never hardcode filesystem
locations. The core parquet files are written by NB01 (unclustered) and
NB02 (with chemical cluster). Downstream notebooks that need cluster
labels call `load_opx_core_clustered`; others use `load_opx_core`.

v7 canonical-model API (per-family, 8 models):
    canonical_model_filename(target, track, family, results_dir=None)
    canonical_model_spec(target, track, family, results_dir=None)
    load_canonical_model(target, track, family, models_dir=None, results_dir=None)

Legacy signature kept behind `canonical_model_filename_legacy` until the
20-seed rerun lands and every caller is migrated.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_PROC, DATA_SPLITS, MODELS, RESULTS,
)

OPX_CORE_FILE = 'opx_clean_core.parquet'
OPX_CORE_CLUSTERED_FILE = 'opx_clean_core_with_clusters.parquet'
OPX_FULL_FILE = 'opx_clean_full.parquet'
OPX_LIQ_FILE = 'opx_clean_opx_liq.parquet'
OPX_ONLY_FILE = 'opx_clean_opx_only.parquet'

WINNING_CONFIG_FILE = 'nb03_winning_configurations.json'
PER_FAMILY_WINNERS_FILE = 'nb03_per_family_winners.json'

VALID_FAMILIES = ('forest', 'boosted')
VALID_TARGETS = ('T_C', 'P_kbar')
VALID_TRACKS = ('opx_only', 'opx_liq')


def load_opx_core():
    return pd.read_parquet(DATA_PROC / OPX_CORE_FILE)


def load_opx_core_clustered():
    path = DATA_PROC / OPX_CORE_CLUSTERED_FILE
    if not path.exists():
        raise FileNotFoundError(
            f'Clustered core not found at {path}. Run NB02 first.'
        )
    return pd.read_parquet(path)


def load_opx_full():
    return pd.read_parquet(DATA_PROC / OPX_FULL_FILE)


def load_opx_liq():
    return pd.read_parquet(DATA_PROC / OPX_LIQ_FILE)


def load_opx_only():
    return pd.read_parquet(DATA_PROC / OPX_ONLY_FILE)


def load_splits(track):
    if track == 'opx_liq':
        tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
        te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
    elif track in ('opx', 'opx_only'):
        tr = np.load(DATA_SPLITS / 'train_indices_opx.npy')
        te = np.load(DATA_SPLITS / 'test_indices_opx.npy')
    else:
        raise ValueError(f'unknown track: {track!r}')
    return tr, te


def _resolve_results_dir(results_dir):
    return Path(results_dir) if results_dir is not None else Path(RESULTS)


def load_winning_config(results_dir=None):
    """Legacy (pre-v7) single-global-feature-set winners JSON."""
    rdir = _resolve_results_dir(results_dir)
    config_path = rdir / WINNING_CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f'NB03 winning config not found at {config_path}. Run NB03 first.'
        )
    with open(config_path) as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _load_per_family_winners_cached(results_dir_str: str) -> dict:
    path = Path(results_dir_str) / PER_FAMILY_WINNERS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run nb03 Phase 3.5 to generate it. '
            'This file is the single source of truth for v7 canonical model selection.'
        )
    with open(path) as f:
        return json.load(f)


def load_per_family_winners(results_dir=None) -> dict:
    """Load and cache the per-family winners JSON. The resolver call sites
    (`canonical_model_filename`, `canonical_model_spec`, `load_canonical_model`)
    already use this internally; call directly when a notebook needs the raw
    dictionary (e.g., to iterate both families)."""
    rdir = _resolve_results_dir(results_dir)
    return _load_per_family_winners_cached(str(rdir))


def _validate_task(target, track, family):
    if family not in VALID_FAMILIES:
        raise ValueError(f"family must be one of {VALID_FAMILIES}, got {family!r}")
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    if track not in VALID_TRACKS:
        raise ValueError(f"track must be one of {VALID_TRACKS}, got {track!r}")


def canonical_model_spec(target, track, family, results_dir=None) -> dict:
    """Return the spec dict (model_name, feature_set, filename, rmse_*) for a
    given (target, track, family)."""
    _validate_task(target, track, family)
    winners = load_per_family_winners(results_dir)
    key = f"{track}_{target}"
    fam_block = winners.get(f"{family}_family", {})
    spec = fam_block.get(key)
    if spec is None:
        raise KeyError(f"No winner recorded for {family}/{key}")
    return spec


def canonical_model_filename(*args, **kwargs):
    """v7 canonical-model filename resolver.

    Signature:  canonical_model_filename(target, track, family, results_dir=None)

    Returns the on-disk filename (not a full path) for the canonical joblib,
    matching the `filename` field in `nb03_per_family_winners.json`.
    """
    if len(args) >= 1 and args[0] in ('RF', 'ERT', 'GB', 'XGB'):
        # Caller passed the legacy (model_name, target, track[, results_dir]) order.
        warnings.warn(
            "canonical_model_filename(model_name, target, track, ...) is the legacy v6 "
            "signature. Switch to canonical_model_filename(target, track, family, ...) "
            "and read per-family winners from nb03_per_family_winners.json.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical_model_filename_legacy(*args, **kwargs)

    if args and 'target' not in kwargs:
        target, *rest = args
    else:
        target = kwargs.get('target')
        rest = []
    if rest and 'track' not in kwargs:
        track, *rest = rest
    else:
        track = kwargs.get('track')
    if rest and 'family' not in kwargs:
        family, *rest = rest
    else:
        family = kwargs.get('family')
    if rest:
        results_dir = rest[0]
    else:
        results_dir = kwargs.get('results_dir')

    spec = canonical_model_spec(target, track, family, results_dir)
    return spec['filename']


def canonical_model_path(target, track, family, models_dir=None, results_dir=None) -> Path:
    """Return the absolute Path to the canonical joblib. Raises if missing."""
    _validate_task(target, track, family)
    mdir = Path(models_dir) if models_dir is not None else Path(MODELS)
    fname = canonical_model_filename(target, track, family, results_dir)
    candidates = [mdir / fname, mdir / 'canonical' / fname,
                  _resolve_results_dir(results_dir) / fname]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Canonical joblib {fname} not found in any of: "
        + ", ".join(str(c) for c in candidates)
    )


def load_canonical_model(target, track, family, models_dir=None, results_dir=None):
    """Load the joblib for a (target, track, family) canonical model."""
    return joblib.load(canonical_model_path(target, track, family, models_dir, results_dir))


def canonical_model_filename_legacy(model_name, target, track, results_dir=None):
    """Pre-v7 filename convention: assumes the single global feature set from
    `nb03_winning_configurations.json`. Retained for migration; do not use in
    new code."""
    config = load_winning_config(results_dir)
    win_feat = config.get('global_feature_set', 'pwlr')
    return f'model_{model_name}_{target}_{track}_{win_feat}.joblib'
