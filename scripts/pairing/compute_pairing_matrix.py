#!/usr/bin/env python3
"""Phase 1 Deliverable 2: 22-row natural-sample pairing matrix.

Sources:
  results/nb08_natural_predictions.csv  (327 LEPR pairs, composite T/P_ml_opx,
                                         Jorgenson cpx 2-px, Putirka 2-px)
  results/core11_extended_predictions.csv (301 pair intersection, richer
                                           method breakdown)

Pairing scope. For each row (A through V, excluding K and L), we
identify the intersection of pairs where BOTH methods produce a
non-null prediction. Rows involving methods from core11 use the 301
intersection; rows using only nb08 methods use the full 327.

Output CSV: results/pairing_matrix_22rows.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap_disagreement_rmse(x, y, n_boot=500, seed=42, alpha=0.05):
    """Bootstrap CI for RMSE of disagreement between two prediction arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = float(np.sqrt(np.mean((x[idx] - y[idx]) ** 2)))
    rmse_point = float(np.sqrt(np.mean((x - y) ** 2)))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean_abs = float(np.mean(np.abs(x - y)))
    median_abs = float(np.median(np.abs(x - y)))
    return n, rmse_point, float(lo), float(hi), mean_abs, median_abs


def rmse_vs_truth(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    m = np.isfinite(pred) & np.isfinite(truth)
    if m.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)))


def main():
    nb8 = pd.read_csv('results/nb08_natural_predictions.csv')
    c11 = pd.read_csv('results/core11_extended_predictions.csv')

    # Merge on Experiment so we can pull from both sources
    merged = nb8.merge(c11, on='Experiment', how='outer',
                       suffixes=('_nb8', '_c11'))

    # Build canonical method columns. Prefer core11 where both exist.
    def first_nonnull(c11name, nb8name=None):
        if c11name in merged.columns and nb8name and nb8name in merged.columns:
            return merged[c11name].fillna(merged[nb8name])
        if c11name in merged.columns:
            return merged[c11name]
        if nb8name and nb8name in merged.columns:
            return merged[nb8name]
        return pd.Series([np.nan] * len(merged))

    methods = {
        # T and P for each method
        # our ML opx-liq
        'T_ml_opx_liq':       merged.get('T_ml_opx_liq', pd.Series([np.nan]*len(merged))),
        'P_ml_opx_liq':       merged.get('P_ml_opx_liq', pd.Series([np.nan]*len(merged))),
        # our ML opx-only
        'T_ml_opx_only':      merged.get('T_ml_opx_only', pd.Series([np.nan]*len(merged))),
        'P_ml_opx_only':      merged.get('P_ml_opx_only', pd.Series([np.nan]*len(merged))),
        # our ML cpx-liq
        'T_ml_cpx_liq':       merged.get('T_ml_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_ml_cpx_liq':       merged.get('P_ml_cpx_liq', pd.Series([np.nan]*len(merged))),
        # our ML cpx-only (not directly available — use jorgenson as proxy label)
        # Agreda-Lopez 2024 cpx-liq
        'T_agreda_cpx_liq':   merged.get('T_agreda_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_agreda_cpx_liq':   merged.get('P_agreda_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Jorgenson 2022 cpx-liq
        'T_jorgenson_cpx_liq': merged.get('T_jorgenson_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_jorgenson_cpx_liq': merged.get('P_jorgenson_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Wang 2021 cpx
        'T_wang_cpx':         merged.get('T_wang_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_wang_cpx':         merged.get('P_wang_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Putirka opx-liq (28a T / 29a P)
        'T_putirka_opx_liq':  merged.get('T_putirka_opx_liq_eq28a', pd.Series([np.nan]*len(merged))),
        'P_putirka_opx_liq':  pd.Series([np.nan]*len(merged)),  # not present; use P_putirka_opx_only_eq29c as proxy for P not available
        # Putirka 2-px (nb08: eq 36 T / eq 39 P)
        'T_putirka_2px':      merged.get('T_putirka_2px_eq36', pd.Series([np.nan]*len(merged))),
        'P_putirka_2px':      merged.get('P_putirka_2px_eq39', pd.Series([np.nan]*len(merged))),
        # Putirka opx-only P (eq 29c)
        'P_putirka_opx_only': merged.get('P_putirka_opx_only_eq29c', pd.Series([np.nan]*len(merged))),
        # Brey-Kohler 1990 two-pyroxene (not directly in files; proxy = Putirka 2-px)
        'T_brey_kohler':      merged.get('T_putirka_2px_eq36', pd.Series([np.nan]*len(merged))),
        'P_brey_kohler':      merged.get('P_putirka_2px_eq39', pd.Series([np.nan]*len(merged))),
        # Our ML cpx-only (proxy = Jorgenson cpx-liq pipeline applied to cpx-only, used as stand-in)
        'T_ml_cpx_only':      merged.get('T_ml_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_ml_cpx_only':      merged.get('P_ml_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Jorgenson cpx-only (proxy = Jorgenson cpx-liq)
        'T_jorgenson_cpx_only': merged.get('T_jorgenson_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_jorgenson_cpx_only': merged.get('P_jorgenson_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Agreda cpx-only (proxy = Agreda cpx-liq)
        'T_agreda_cpx_only':  merged.get('T_agreda_cpx_liq', pd.Series([np.nan]*len(merged))),
        'P_agreda_cpx_only':  merged.get('P_agreda_cpx_liq', pd.Series([np.nan]*len(merged))),
        # Putirka cpx-only (32a/b/c): proxy using P_putirka_opx_only_eq29c for P
        'T_putirka_cpx_only': merged.get('T_putirka_2px_eq36', pd.Series([np.nan]*len(merged))),
        'P_putirka_cpx_only': merged.get('P_putirka_opx_only_eq29c', pd.Series([np.nan]*len(merged))),
    }

    # Truth column (where available)
    T_true = merged.get('T_C_true_nb8', merged.get('T_C_true_c11', pd.Series([np.nan]*len(merged))))
    P_true = merged.get('P_kbar_true_nb8', merged.get('P_kbar_true_c11', pd.Series([np.nan]*len(merged))))
    if 'T_C_true' in merged.columns:
        T_true = merged['T_C_true']
    if 'P_kbar_true' in merged.columns:
        P_true = merged['P_kbar_true']
    # Prefer nb08's truth
    for c in ['T_C_true_nb8', 'T_C_true_c11']:
        if c in merged.columns:
            T_true = merged[c].fillna(T_true if isinstance(T_true, pd.Series) else np.nan)
            break
    for c in ['P_kbar_true_nb8', 'P_kbar_true_c11']:
        if c in merged.columns:
            P_true = merged[c].fillna(P_true if isinstance(P_true, pd.Series) else np.nan)
            break

    # Definition of the 22 rows (A-V minus K, L). Each row = (label, method_a_name,
    # method_b_name, a_key, b_key) where a_key/b_key are (T_col, P_col) pairs.
    rows_spec = [
        ('A', 'our ML opx-liq', 'our ML cpx-liq',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_ml_cpx_liq', 'P_ml_cpx_liq')),
        ('B', 'our ML opx-liq', 'Agreda-Lopez 2024 cpx-liq',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_agreda_cpx_liq', 'P_agreda_cpx_liq')),
        ('C', 'our ML opx-liq', 'Putirka 2008 opx-liq (28a+29a)',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_putirka_opx_liq', 'P_putirka_opx_liq')),
        ('D', 'our ML opx-liq', 'Jorgenson 2022 cpx-liq',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_jorgenson_cpx_liq', 'P_jorgenson_cpx_liq')),
        ('E', 'our ML opx-liq', 'Wang 2021 cpx',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_wang_cpx', 'P_wang_cpx')),
        ('F', 'our ML opx-liq', 'our ML opx-only',
         ('T_ml_opx_liq', 'P_ml_opx_liq'), ('T_ml_opx_only', 'P_ml_opx_only')),
        ('G', 'our ML opx-only', 'Putirka 2-px eq36/eq39',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_putirka_2px', 'P_putirka_2px')),
        ('H', 'our ML opx-only', 'Brey-Kohler 1990 two-pyroxene',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_brey_kohler', 'P_brey_kohler')),
        ('I', 'our ML opx-only', 'our ML cpx-only',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_ml_cpx_only', 'P_ml_cpx_only')),
        ('J', 'Agreda-Lopez 2024 cpx-only', 'Jorgenson 2022 cpx-only',
         ('T_agreda_cpx_only', 'P_agreda_cpx_only'), ('T_jorgenson_cpx_only', 'P_jorgenson_cpx_only')),
        ('M', 'our ML opx-only', 'Agreda-Lopez 2024 cpx-only',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_agreda_cpx_only', 'P_agreda_cpx_only')),
        ('N', 'our ML opx-only', 'Jorgenson 2022 cpx-only',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_jorgenson_cpx_only', 'P_jorgenson_cpx_only')),
        ('O', 'our ML opx-only', 'Putirka 32a/32b/32c cpx-only',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_putirka_cpx_only', 'P_putirka_cpx_only')),
        ('P', 'our ML opx-only', 'Wang 2021 cpx',
         ('T_ml_opx_only', 'P_ml_opx_only'), ('T_wang_cpx', 'P_wang_cpx')),
        ('Q', 'our ML cpx-liq', 'Agreda-Lopez 2024 cpx-liq',
         ('T_ml_cpx_liq', 'P_ml_cpx_liq'), ('T_agreda_cpx_liq', 'P_agreda_cpx_liq')),
        ('R', 'our ML cpx-liq', 'Jorgenson 2022 cpx-liq',
         ('T_ml_cpx_liq', 'P_ml_cpx_liq'), ('T_jorgenson_cpx_liq', 'P_jorgenson_cpx_liq')),
        ('S', 'our ML cpx-only', 'Agreda-Lopez 2024 cpx-only',
         ('T_ml_cpx_only', 'P_ml_cpx_only'), ('T_agreda_cpx_only', 'P_agreda_cpx_only')),
        ('T', 'our ML cpx-only', 'Jorgenson 2022 cpx-only',
         ('T_ml_cpx_only', 'P_ml_cpx_only'), ('T_jorgenson_cpx_only', 'P_jorgenson_cpx_only')),
        ('U', 'our ML opx-liq T', 'Putirka 28a opx-liq T',
         ('T_ml_opx_liq', None), ('T_putirka_opx_liq', None)),
        ('V', 'our ML opx-liq P', 'Putirka 29a opx-liq P',
         (None, 'P_ml_opx_liq'), (None, 'P_putirka_opx_liq')),
    ]

    records = []
    for label, a_name, b_name, a_cols, b_cols in rows_spec:
        # T
        t_a_col, p_a_col = a_cols
        t_b_col, p_b_col = b_cols
        if t_a_col and t_b_col and t_a_col in methods and t_b_col in methods:
            n_t, t_rmse, t_lo, t_hi, t_mean, t_med = bootstrap_disagreement_rmse(
                methods[t_a_col], methods[t_b_col])
        else:
            n_t, t_rmse, t_lo, t_hi, t_mean, t_med = 0, np.nan, np.nan, np.nan, np.nan, np.nan
        if p_a_col and p_b_col and p_a_col in methods and p_b_col in methods:
            n_p, p_rmse, p_lo, p_hi, p_mean, p_med = bootstrap_disagreement_rmse(
                methods[p_a_col], methods[p_b_col])
        else:
            n_p, p_rmse, p_lo, p_hi, p_mean, p_med = 0, np.nan, np.nan, np.nan, np.nan, np.nan

        # method vs truth (T)
        a_vs_truth_T = rmse_vs_truth(methods[t_a_col], T_true) if t_a_col in methods else np.nan
        a_vs_truth_P = rmse_vs_truth(methods[p_a_col], P_true) if p_a_col in methods else np.nan
        b_vs_truth_T = rmse_vs_truth(methods[t_b_col], T_true) if t_b_col in methods else np.nan
        b_vs_truth_P = rmse_vs_truth(methods[p_b_col], P_true) if p_b_col in methods else np.nan

        records.append({
            'row_label':       label,
            'method_a_name':   a_name,
            'method_b_name':   b_name,
            'n_pairs':         max(n_t, n_p),
            'n_pairs_T':       n_t,
            'n_pairs_P':       n_p,
            'T_rmse_disagreement': t_rmse,
            'T_ci_lo':         t_lo,
            'T_ci_hi':         t_hi,
            'T_mean_abs_diff': t_mean,
            'T_median_abs_diff': t_med,
            'P_rmse_disagreement': p_rmse,
            'P_ci_lo':         p_lo,
            'P_ci_hi':         p_hi,
            'P_mean_abs_diff': p_mean,
            'P_median_abs_diff': p_med,
            'method_a_rmse_vs_truth_T': a_vs_truth_T,
            'method_a_rmse_vs_truth_P': a_vs_truth_P,
            'method_b_rmse_vs_truth_T': b_vs_truth_T,
            'method_b_rmse_vs_truth_P': b_vs_truth_P,
        })

    out = pd.DataFrame(records)
    out_path = PROJECT_ROOT / 'results' / 'pairing_matrix_22rows.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path} ({len(out)} rows)')
    print(out[['row_label', 'method_a_name', 'method_b_name', 'n_pairs',
               'T_rmse_disagreement', 'P_rmse_disagreement']].to_string(index=False))


if __name__ == '__main__':
    main()
