"""Hasterok and Chapman (2011) layered conductive continental geotherm.

Reference: Hasterok, D., and Chapman, D.S. (2011) Heat production and
geotherms for the continental lithosphere. Earth and Planetary Science
Letters, 307, 59-70.

Layer parameters come from Table 2 of the reference continental model:
a two-layer upper crust, a lower crust, and a lithospheric mantle layer
extending to 300 km. Surface heat flow q_s sets the boundary condition.

`hasterok_chapman_geotherm(q_s_mW_m2, z_km_array)` returns T (degrees C)
and lithostatic P (kbar) at each requested depth.
"""
from __future__ import annotations

import numpy as np


LAYERS = [
    # (z_top_km, z_bot_km, A_W_per_m3, k_W_per_m_K, rho_kg_per_m3)
    (0.0,  16.0,  1.30e-6, 2.5, 2800.0),
    (16.0, 24.0,  0.40e-6, 2.5, 2800.0),
    (24.0, 40.0,  0.40e-6, 2.5, 2800.0),
    (40.0, 300.0, 0.02e-6, 3.0, 3300.0),
]

T_SURFACE_C = 10.0
GRAVITY_M_S2 = 9.81


def hasterok_chapman_geotherm(q_s_mW_m2, z_km_array):
    """Return (T_C_array, P_kbar_array) at depths `z_km_array`.

    `q_s_mW_m2` is the surface heat flow in mW/m^2. Typical values:
    cratonic ~ 40, average continental ~ 60, hot ~ 80.
    """
    z_arr = np.asarray(z_km_array, dtype=float)
    q_s = q_s_mW_m2 * 1e-3
    T_out = np.zeros_like(z_arr)
    P_out = np.zeros_like(z_arr)
    for idx, z_km in enumerate(z_arr):
        T_top, q_top, P_top = T_SURFACE_C, q_s, 0.0
        for (z0, z1, A, k, rho) in LAYERS:
            if z_km < z0:
                break
            dz_full = (z1 - z0) * 1000.0
            if z_km <= z1:
                dz = (z_km - z0) * 1000.0
                T_out[idx] = T_top + (q_top * dz - 0.5 * A * dz ** 2) / k
                P_out[idx] = P_top + rho * GRAVITY_M_S2 * dz * 1e-8
                break
            T_top = T_top + (q_top * dz_full - 0.5 * A * dz_full ** 2) / k
            q_top = q_top - A * dz_full
            P_top = P_top + rho * GRAVITY_M_S2 * dz_full * 1e-8
    return T_out, P_out
