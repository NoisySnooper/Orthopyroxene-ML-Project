#!/usr/bin/env python3
"""D1.6 smoke test: TabPFN install + basic fit on 50 rows of opx_liq.

Run from the tabpfn venv:
    .venv-tabpfn/Scripts/python.exe scripts/tabpfn/opx_tb_nb03_test_tabpfn_smoke.py

Exit codes:
    0 = import + fit + finite RMSE within sane bounds (<1000 kbar)
    2 = import failure
    3 = fit or predict failure
    4 = assertion failure (non-finite or absurd RMSE)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import tabpfn
    from tabpfn import TabPFNRegressor
except Exception as e:
    print(f'IMPORT FAILED: {e}', file=sys.stderr)
    sys.exit(2)

from src.prepare_train_test import prepare_train_test


def _build_v2_regressor(n_est=4, seed=42):
    """Pin construction to TabPFN v2. See baseline script docstring."""
    path = 'create_default_for_version'
    try:
        from tabpfn.constants import ModelVersion  # type: ignore[attr-defined]
        reg = TabPFNRegressor.create_default_for_version(  # type: ignore[attr-defined]
            ModelVersion.V2, device='cpu', n_estimators=n_est,
            random_state=seed,
        )
    except (ImportError, AttributeError):
        # tabpfn 2.0-2.4 does not expose create_default_for_version; in that
        # range model_path='auto' is the TabPFN-v2 weight set by construction.
        reg = TabPFNRegressor(device='cpu', n_estimators=n_est,
                              random_state=seed, model_path='auto')
        path = "model_path='auto' (package-version pinned to v2 by tabpfn>=2.0,<2.5)"
    return reg, path


def main() -> int:
    t0 = time.time()
    ver = getattr(tabpfn, '__version__', 'unknown')
    print(f'tabpfn package version: {ver}')
    if ver == 'unknown' or not ver.startswith('2.') or ver.startswith('2.5'):
        print(f'ASSERT FAILED: expected tabpfn 2.x (<2.5), got {ver!r}. '
              f'Check requirements-tabpfn.txt pin.', file=sys.stderr)
        return 4
    data = prepare_train_test('opx', 'opx_liq', 'P_kbar', 'raw')
    X_tr, y_tr = data['X_tr'][:50], data['y_tr'][:50]
    X_te, y_te = data['X_te'][:50], data['y_te'][:50]
    print(f'loaded: train={X_tr.shape} test={X_te.shape} feats={len(data["feat_names"])}')
    try:
        reg, construction_path = _build_v2_regressor(n_est=4, seed=42)
        print(f'construction path: {construction_path}')
        reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
    except Exception as e:
        print(f'FIT/PREDICT FAILED: {e}', file=sys.stderr)
        return 3
    rmse = float(np.sqrt(np.mean((y_hat - y_te) ** 2)))
    dt = time.time() - t0
    print(f'smoke RMSE={rmse:.3f} kbar (elapsed {dt:.1f}s)')
    if not np.isfinite(rmse) or rmse > 1000:
        print(f'ASSERTION FAILED: rmse={rmse}', file=sys.stderr)
        return 4
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
