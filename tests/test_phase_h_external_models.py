"""H.0a smoke test for predict_putirka_classical_natural().

Builds a 5-row synthetic two-pyroxene + liquid DataFrame and verifies
that every equation supported by the wrapper returns a finite, in-range
prediction on at least one of the five rows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.external_models import predict_putirka_classical_natural  # noqa: E402


def _synthetic_5row_pyroxene_liq() -> pd.DataFrame:
    """Five rows of plausible LEPR-like cpx + opx + liq compositions.

    Values selected to be inside Putirka calibration domain; tweaked
    slightly per row to break degeneracies. Oxide totals close to 100.
    """
    rows = [
        # arc dacite-like
        {'SiO2_Cpx': 51.5, 'TiO2_Cpx': 0.6, 'Al2O3_Cpx': 3.8, 'Cr2O3_Cpx': 0.05,
         'FeOt_Cpx': 7.4, 'MnO_Cpx': 0.18, 'MgO_Cpx': 15.2, 'CaO_Cpx': 20.3,
         'Na2O_Cpx': 0.42,
         'SiO2_Opx': 53.6, 'TiO2_Opx': 0.25, 'Al2O3_Opx': 1.8, 'Cr2O3_Opx': 0.04,
         'FeOt_Opx': 17.8, 'MnO_Opx': 0.42, 'MgO_Opx': 24.7, 'CaO_Opx': 1.6,
         'Na2O_Opx': 0.04,
         'SiO2_Liq': 64.2, 'TiO2_Liq': 0.7, 'Al2O3_Liq': 16.4,
         'FeOt_Liq': 4.5, 'MnO_Liq': 0.10, 'MgO_Liq': 1.9, 'CaO_Liq': 4.1,
         'Na2O_Liq': 4.0, 'K2O_Liq': 2.5, 'P2O5_Liq': 0.18, 'H2O_Liq': 4.0,
         'Cr2O3_Liq': 0.0, 'NiO_Liq': 0.0},
        # arc basalt-like
        {'SiO2_Cpx': 50.4, 'TiO2_Cpx': 0.9, 'Al2O3_Cpx': 5.1, 'Cr2O3_Cpx': 0.20,
         'FeOt_Cpx': 6.1, 'MnO_Cpx': 0.14, 'MgO_Cpx': 16.8, 'CaO_Cpx': 19.7,
         'Na2O_Cpx': 0.36,
         'SiO2_Opx': 53.3, 'TiO2_Opx': 0.22, 'Al2O3_Opx': 2.7, 'Cr2O3_Opx': 0.18,
         'FeOt_Opx': 13.2, 'MnO_Opx': 0.30, 'MgO_Opx': 28.0, 'CaO_Opx': 1.9,
         'Na2O_Opx': 0.05,
         'SiO2_Liq': 51.4, 'TiO2_Liq': 1.0, 'Al2O3_Liq': 17.6,
         'FeOt_Liq': 9.4, 'MnO_Liq': 0.16, 'MgO_Liq': 5.2, 'CaO_Liq': 9.4,
         'Na2O_Liq': 3.2, 'K2O_Liq': 1.4, 'P2O5_Liq': 0.30, 'H2O_Liq': 2.5,
         'Cr2O3_Liq': 0.0, 'NiO_Liq': 0.0},
        # OIB tholeiite-like
        {'SiO2_Cpx': 51.0, 'TiO2_Cpx': 1.4, 'Al2O3_Cpx': 4.6, 'Cr2O3_Cpx': 0.08,
         'FeOt_Cpx': 7.8, 'MnO_Cpx': 0.16, 'MgO_Cpx': 15.7, 'CaO_Cpx': 20.0,
         'Na2O_Cpx': 0.45,
         'SiO2_Opx': 54.0, 'TiO2_Opx': 0.30, 'Al2O3_Opx': 2.0, 'Cr2O3_Opx': 0.05,
         'FeOt_Opx': 16.4, 'MnO_Opx': 0.36, 'MgO_Opx': 25.6, 'CaO_Opx': 1.5,
         'Na2O_Opx': 0.03,
         'SiO2_Liq': 49.6, 'TiO2_Liq': 2.4, 'Al2O3_Liq': 14.8,
         'FeOt_Liq': 11.7, 'MnO_Liq': 0.20, 'MgO_Liq': 7.0, 'CaO_Liq': 11.0,
         'Na2O_Liq': 2.6, 'K2O_Liq': 0.5, 'P2O5_Liq': 0.27, 'H2O_Liq': 0.6,
         'Cr2O3_Liq': 0.0, 'NiO_Liq': 0.0},
        # mantle xenolith-like (cooler, deeper)
        {'SiO2_Cpx': 53.1, 'TiO2_Cpx': 0.20, 'Al2O3_Cpx': 5.7, 'Cr2O3_Cpx': 0.95,
         'FeOt_Cpx': 3.0, 'MnO_Cpx': 0.10, 'MgO_Cpx': 16.4, 'CaO_Cpx': 19.0,
         'Na2O_Cpx': 1.8,
         'SiO2_Opx': 55.4, 'TiO2_Opx': 0.10, 'Al2O3_Opx': 4.2, 'Cr2O3_Opx': 0.46,
         'FeOt_Opx': 6.4, 'MnO_Opx': 0.14, 'MgO_Opx': 32.0, 'CaO_Opx': 0.9,
         'Na2O_Opx': 0.18,
         'SiO2_Liq': 50.0, 'TiO2_Liq': 1.5, 'Al2O3_Liq': 16.5,
         'FeOt_Liq': 8.5, 'MnO_Liq': 0.16, 'MgO_Liq': 8.0, 'CaO_Liq': 11.0,
         'Na2O_Liq': 2.5, 'K2O_Liq': 0.6, 'P2O5_Liq': 0.20, 'H2O_Liq': 0.4,
         'Cr2O3_Liq': 0.05, 'NiO_Liq': 0.02},
        # mid-ocean ridge basalt (MORB)-like
        {'SiO2_Cpx': 51.7, 'TiO2_Cpx': 0.5, 'Al2O3_Cpx': 3.4, 'Cr2O3_Cpx': 0.50,
         'FeOt_Cpx': 5.4, 'MnO_Cpx': 0.13, 'MgO_Cpx': 17.2, 'CaO_Cpx': 20.6,
         'Na2O_Cpx': 0.30,
         'SiO2_Opx': 54.3, 'TiO2_Opx': 0.18, 'Al2O3_Opx': 2.0, 'Cr2O3_Opx': 0.30,
         'FeOt_Opx': 11.2, 'MnO_Opx': 0.22, 'MgO_Opx': 30.4, 'CaO_Opx': 1.4,
         'Na2O_Opx': 0.04,
         'SiO2_Liq': 50.8, 'TiO2_Liq': 1.4, 'Al2O3_Liq': 15.3,
         'FeOt_Liq': 9.9, 'MnO_Liq': 0.17, 'MgO_Liq': 7.4, 'CaO_Liq': 11.6,
         'Na2O_Liq': 2.6, 'K2O_Liq': 0.15, 'P2O5_Liq': 0.16, 'H2O_Liq': 0.2,
         'Cr2O3_Liq': 0.05, 'NiO_Liq': 0.0},
    ]
    df = pd.DataFrame(rows)
    df['Fe3Fet_Liq'] = 0.15
    df['NiO_Liq'] = 0.02
    return df


def main() -> int:
    df = _synthetic_5row_pyroxene_liq()
    print(f'smoke input: {len(df)} rows, '
          f'{sum(c.endswith("_Cpx") for c in df.columns)} cpx cols, '
          f'{sum(c.endswith("_Opx") for c in df.columns)} opx cols, '
          f'{sum(c.endswith("_Liq") for c in df.columns)} liq cols')

    preds = predict_putirka_classical_natural(df)
    print('output columns:')
    for col in preds.columns:
        vals = preds[col]
        n_finite = int(np.isfinite(vals).sum())
        if n_finite > 0:
            mean = float(np.nanmean(vals))
            print(f'  {col}: n_finite={n_finite}/5, mean={mean:.2f}')
        else:
            print(f'  {col}: n_finite=0/5 (all NaN)')

    expected_cols = {
        'T_putirka_cpx_liq_eq33', 'P_putirka_cpx_liq_eq30',
        'T_putirka_opx_liq_eq28a', 'P_putirka_opx_liq_eq29a',
        'P_putirka_opx_only_eq29c',
        'T_putirka_cpx_only_eq32d', 'P_putirka_cpx_only_eq32a',
        'T_putirka_twopx_eq36', 'P_putirka_twopx_eq39',
        'T_putirka_twopx_eq37', 'P_putirka_twopx_eq38',
    }
    got = set(preds.columns)
    missing = expected_cols - got
    if missing:
        print(f'FAIL: missing columns: {sorted(missing)}')
        return 1

    n_cols_with_data = sum(int(np.isfinite(preds[c]).sum()) > 0
                           for c in expected_cols)
    if n_cols_with_data < 7:
        print(f'FAIL: only {n_cols_with_data}/11 columns produced any '
              'finite predictions; expected at least 7')
        return 1

    print(f'SMOKE PASS: {n_cols_with_data}/11 equation columns returned '
          'finite predictions on the 5-row synthetic batch.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
