"""Unit tests for src/ablations/augmentation.py.

Four contract tests:
    (a) augment_gaussian produces (1 + n_copies) * len(X) samples
    (b) citation groups are preserved across augmented rows
    (c) augmented samples from citation c never appear in a fold that
        does not contain original citation c samples
    (d) noise magnitude matches spec (per-feature rel std within 10% of
        target 0.03, empirically measured over copies)
"""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.model_selection import GroupKFold

from src.ablations.augmentation import (
    augment_gaussian,
    make_augmented_cv_splits,
    noise_profile,
)


def _toy_data(n: int = 40, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Emulate oxide wt%: roughly-positive features with variable scale.
    X = rng.uniform(1.0, 50.0, size=(n, d))
    y = rng.uniform(800.0, 1400.0, size=n)
    # 5 citation groups, 8 rows each.
    citations = np.array([f'Paper_{i % 5}' for i in range(n)])
    return X, y, citations


def test_a_augmented_shape():
    X, y, cit = _toy_data(n=40, d=5)
    n_copies = 15
    X_a, y_a, c_a = augment_gaussian(
        X, y, cit, n_copies=n_copies, rel_noise=0.03, seed=123)
    expected = (1 + n_copies) * len(X)
    assert X_a.shape == (expected, X.shape[1])
    assert y_a.shape == (expected,)
    assert c_a is not None and c_a.shape == (expected,)
    # The first n rows are the originals (byte-identical).
    assert np.array_equal(X_a[:len(X)], X)
    assert np.array_equal(y_a[:len(X)], y)


def test_b_citation_preserved_across_copies():
    X, y, cit = _toy_data(n=40, d=5)
    n_copies = 15
    X_a, _, c_a = augment_gaussian(
        X, y, cit, n_copies=n_copies, rel_noise=0.03, seed=7)
    n = len(X)
    # Each copy block [c*n, (c+1)*n) must have identical citations to
    # the original block [0, n).
    for c in range(n_copies + 1):
        block = c_a[c * n:(c + 1) * n]
        assert np.array_equal(block, cit), (
            f'copy block {c} citations do not match original')


def test_c_augmented_rows_respect_fold_membership():
    X, y, cit = _toy_data(n=40, d=5)
    X_a, _, c_a = augment_gaussian(
        X, y, cit, n_copies=5, rel_noise=0.03, seed=11)
    base_splits = list(GroupKFold(n_splits=5).split(X, y, groups=cit))
    aug_splits = make_augmented_cv_splits(cit, c_a, base_splits)
    for (train_idx_aug, val_idx), (train_idx_orig, _) in zip(aug_splits, base_splits):
        train_cits_orig = set(cit[train_idx_orig].tolist())
        val_cits_orig = set(cit[val_idx].tolist())
        # Every augmented training row's citation must be in the
        # original-train citation set.
        aug_train_cits = set(c_a[train_idx_aug].tolist())
        assert aug_train_cits.issubset(train_cits_orig), (
            f'Augmented train has citations {aug_train_cits - train_cits_orig} '
            f'that leaked from val')
        # And in particular, no val citation should appear in train.
        leak = aug_train_cits & val_cits_orig
        assert not leak, f'leak: {leak}'
        # Val indices are left untouched (still on original data only).
        assert set(val_idx.tolist()).issubset(set(range(len(X))))


def test_d_noise_magnitude_matches_spec():
    X, y, cit = _toy_data(n=200, d=8, seed=42)
    n_copies = 15
    target_rel = 0.03
    X_a, _, _ = augment_gaussian(
        X, y, cit, n_copies=n_copies, rel_noise=target_rel, seed=42)
    prof = noise_profile(X, X_a, n_original=len(X), n_copies=n_copies)
    # Per-feature empirical relative std should be within ~10% of 0.03.
    per_feat = prof['per_feature_rel_std']
    assert np.all(np.isfinite(per_feat))
    max_rel_err = np.max(np.abs(per_feat - target_rel) / target_rel)
    assert max_rel_err < 0.12, (
        f'per-feature RSD deviates too much: {per_feat} (target {target_rel})')
    overall = prof['overall_rel_std']
    assert abs(overall - target_rel) / target_rel < 0.05, (
        f'overall RSD {overall:.4f} differs from target {target_rel}')


def test_e_rejects_bad_inputs():
    X, y, cit = _toy_data(n=10, d=3)
    with pytest.raises(ValueError):
        augment_gaussian(X, y, cit, n_copies=-1)
    with pytest.raises(ValueError):
        augment_gaussian(X, y, cit, rel_noise=-0.1)
    with pytest.raises(ValueError):
        augment_gaussian(X, y[:5], cit)  # y length mismatch
    with pytest.raises(ValueError):
        augment_gaussian(X, y, cit[:5])  # cit length mismatch


def test_f_clip_nonneg_after_noise():
    # Features close to zero should not produce negative augmented
    # values when clip_nonneg=True.
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 0.01, size=(50, 4))  # near-zero oxide trace
    y = rng.uniform(500, 1500, size=50)
    X_a, _, _ = augment_gaussian(
        X, y, citations=None, n_copies=10, rel_noise=0.5, seed=0,
        clip_nonneg=True)
    assert (X_a >= 0).all()


def test_g_zero_copies_returns_originals_only():
    X, y, cit = _toy_data(n=10, d=3)
    X_a, y_a, c_a = augment_gaussian(X, y, cit, n_copies=0)
    assert X_a.shape == X.shape
    assert np.array_equal(X_a, X)
    assert np.array_equal(y_a, y)
    assert np.array_equal(c_a, cit)
