"""Gaussian-noise data augmentation for opx experimental petrology data.

Matches Agreda-Lopez et al. 2024 protocol: 15x oversampling with 3%
relative Gaussian noise on each input feature. Citation grouping is
preserved across augmented rows so citation-grouped cross-validation
still holds (no augmented sample ever crosses into a fold that does not
contain its parent's citation).

Used in notebooks/nb04b_aug_test.ipynb for the augmentation sensitivity
analysis reported in Section 5.3 of the opx manuscript.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def augment_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    citations: Optional[np.ndarray] = None,
    n_copies: int = 15,
    rel_noise: float = 0.03,
    seed: int = 42,
    clip_nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate augmented samples via Gaussian relative-noise injection.

    For each original sample (X_i, y_i), produce `n_copies` augmented
    copies of the form

        X_aug = X_i * (1 + epsilon),  epsilon ~ N(0, rel_noise**2)
        y_aug = y_i

    The originals are prepended to the output (rows [0, n) of the
    returned arrays), so total length is `(1 + n_copies) * n`. With
    `n_copies=15` this gives a 16x multiplier. We follow the literal
    reading of Agreda-Lopez's "15x augmentation" as 15 noisy copies on
    top of the originals; downstream consumers can subsample the first
    `n` rows to get the un-augmented baseline for free.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix. Entries should be non-negative composition-like
        quantities; the noise is multiplicative and `clip_nonneg`
        guarantees the outputs stay valid.
    y : np.ndarray, shape (n,)
        Targets. Unchanged during augmentation.
    citations : np.ndarray, shape (n,), optional
        Citation labels. Preserved across all copies so augmented rows
        inherit their parent's group for citation-grouped CV.
    n_copies : int, default 15
        Number of noisy copies per original.
    rel_noise : float, default 0.03
        Relative standard deviation of the multiplicative noise (3%).
    seed : int, default 42
        Seed for the noise generator.
    clip_nonneg : bool, default True
        Clip augmented features to be non-negative after noise.

    Returns
    -------
    X_aug : np.ndarray, shape ((1 + n_copies) * n, d)
    y_aug : np.ndarray, shape ((1 + n_copies) * n,)
    citations_aug : np.ndarray or None
    """
    if n_copies < 0:
        raise ValueError(f'n_copies must be >= 0, got {n_copies}')
    if rel_noise < 0:
        raise ValueError(f'rel_noise must be >= 0, got {rel_noise}')
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must be 2-D, got shape {X.shape}')
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f'y shape {y.shape} incompatible with X shape {X.shape}')
    if citations is not None:
        citations = np.asarray(citations)
        if citations.shape[0] != X.shape[0]:
            raise ValueError(
                f'citations shape {citations.shape} incompatible with '
                f'X shape {X.shape}')

    rng = np.random.default_rng(seed)
    n, _ = X.shape

    X_blocks: List[np.ndarray] = [X.copy()]
    y_blocks: List[np.ndarray] = [y.copy()]
    cit_blocks: Optional[List[np.ndarray]] = (
        [citations.copy()] if citations is not None else None)

    for _ in range(n_copies):
        noise = rng.normal(0.0, rel_noise, size=X.shape)
        X_copy = X * (1.0 + noise)
        if clip_nonneg:
            X_copy = np.clip(X_copy, 0.0, None)
        X_blocks.append(X_copy)
        y_blocks.append(y.copy())
        if cit_blocks is not None:
            cit_blocks.append(citations.copy())

    X_aug = np.vstack(X_blocks)
    y_aug = np.concatenate(y_blocks)
    cit_aug = np.concatenate(cit_blocks) if cit_blocks is not None else None
    return X_aug, y_aug, cit_aug


def augment_train_only(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    citations_tr: Optional[np.ndarray],
    n_copies: int = 15,
    rel_noise: float = 0.03,
    seed: int = 42,
    clip_nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Thin wrapper documenting the "train only" augmentation policy.

    Test augmentation would be circular (reducing variance on inflated
    data). This helper exists to make the policy visible at the call
    site in nb04b.
    """
    return augment_gaussian(
        X_tr, y_tr, citations_tr,
        n_copies=n_copies, rel_noise=rel_noise, seed=seed,
        clip_nonneg=clip_nonneg,
    )


def make_augmented_cv_splits(
    citations: np.ndarray,
    augmented_citations: np.ndarray,
    base_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Translate non-augmented CV fold indices to augmented data.

    For each fold in `base_splits` (train_idx, test_idx on original
    data), produce an augmented fold where augmented rows inherit their
    parent's fold membership via the citation label. Test (validation)
    indices stay at original-row indices only -- augmented rows are
    never in the held-out fold.

    Parameters
    ----------
    citations : np.ndarray, shape (n,)
        Citation labels on the original data.
    augmented_citations : np.ndarray, shape ((1+n_copies)*n,)
        Citation labels on the augmented stack (output of
        `augment_gaussian`).
    base_splits : list of (train_idx, val_idx)
        CV splits computed on the original data.

    Returns
    -------
    augmented_splits : list of (train_idx_aug, val_idx)
    """
    citations = np.asarray(citations)
    augmented_citations = np.asarray(augmented_citations)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in base_splits:
        train_cits = set(map(_hashable, citations[train_idx].tolist()))
        mask_aug_train = np.array(
            [_hashable(c) in train_cits for c in augmented_citations.tolist()])
        train_idx_aug = np.flatnonzero(mask_aug_train)
        out.append((train_idx_aug, np.asarray(val_idx)))
    return out


def _hashable(x):
    """Coerce numpy scalars to a Python hashable for set membership."""
    try:
        return x.item() if hasattr(x, 'item') else x
    except Exception:
        return x


def oof_predict_augmented(
    model_name: str,
    best_params: dict,
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    citations_orig: np.ndarray,
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    citations_aug: np.ndarray,
    seed: int = 42,
    n_folds: int = 10,
) -> np.ndarray:
    """10-fold citation-grouped OOF predictions on augmented training data.

    Splits are computed on the *original* data so we can score OOF only
    at original-row indices (augmented rows never appear in a held-out
    fold). Training rows per fold include the augmented copies of
    citations that belong to the training-side partition.

    Returns an (n_original,) array of OOF predictions.
    """
    # Local imports to avoid circular-import risk during module load.
    from sklearn.model_selection import StratifiedGroupKFold

    from src.evaluation import stratify_labels
    from src.models import build_model

    X_orig = np.asarray(X_orig, dtype=float)
    y_orig = np.asarray(y_orig, dtype=float)
    citations_orig = np.asarray(citations_orig)
    X_aug = np.asarray(X_aug, dtype=float)
    y_aug = np.asarray(y_aug, dtype=float)
    citations_aug = np.asarray(citations_aug)

    n_orig = X_orig.shape[0]
    y_strat = stratify_labels(y_orig)
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.full(n_orig, fill_value=np.nan, dtype=float)

    for train_orig_idx, val_orig_idx in sgkf.split(
            X_orig, y_strat, groups=citations_orig):
        train_cits = set(_hashable(c) for c in citations_orig[train_orig_idx].tolist())
        mask_aug_train = np.array(
            [_hashable(c) in train_cits for c in citations_aug.tolist()])
        X_tr_fold = X_aug[mask_aug_train]
        y_tr_fold = y_aug[mask_aug_train]
        est = build_model(model_name, best_params, seed=seed)
        est.fit(X_tr_fold, y_tr_fold)
        oof[val_orig_idx] = est.predict(X_orig[val_orig_idx])

    if np.isnan(oof).any():
        raise RuntimeError(
            'oof_predict_augmented: OOF has unvisited original rows')
    return oof


def noise_profile(
    X: np.ndarray,
    X_aug: np.ndarray,
    n_original: int,
    n_copies: int,
) -> dict:
    """Empirically measure the relative-noise std across augmented rows.

    Diagnostic helper for unit tests and reviewer scrutiny: re-derives
    the effective per-feature relative standard deviation from the
    augmented block and returns it alongside the specified `rel_noise`
    so reviewers can audit that `rel_noise=0.03` really means 3% RSD on
    the feature column.

    Returns a dict with keys:
        'per_feature_rel_std': np.ndarray (d,) -- mean over copies of
            std((X_aug_copy / X_original - 1), axis=0) when originals
            are non-zero.
        'overall_rel_std': scalar -- per-feature RSDs averaged.
    """
    X = np.asarray(X, dtype=float)
    X_aug = np.asarray(X_aug, dtype=float)
    n, d = X.shape
    per_copy_rel_std = []
    for c in range(1, n_copies + 1):
        block = X_aug[c * n:(c + 1) * n]
        safe = np.abs(X) > 0
        rel = np.where(safe, (block - X) / np.where(safe, X, 1.0), np.nan)
        with np.errstate(all='ignore'):
            per_copy_rel_std.append(np.nanstd(rel, axis=0))
    per_copy_arr = np.vstack(per_copy_rel_std)
    per_feature_rel_std = np.nanmean(per_copy_arr, axis=0)
    return {
        'per_feature_rel_std': per_feature_rel_std,
        'overall_rel_std': float(np.nanmean(per_feature_rel_std)),
    }
