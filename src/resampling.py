"""P-T grid tempered resampling for imbalanced regression training data.

Equal-range P-T grid, uniform target over occupied cells, tempered by
averaging with the empirical per-cell count. See
`docs/resampling_strategy.md` for design justification.

Functions:
- `compute_pt_grid_bins`: range-based equal-width bin edges.
- `assign_pt_cells`: per-row (p_cell, t_cell) labels.
- `tempered_resample`: produce resampled df + diagnostics dict. The
  diagnostics dict has key `actions` (DataFrame of per-cell action log),
  `summary` (high-level counts), `bin_edges` (p_edges, t_edges).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pt_grid_bins(df: pd.DataFrame,
                         n_p_bins: int = 5,
                         n_t_bins: int = 5,
                         p_col: str = 'P_kbar',
                         t_col: str = 'T_C'):
    """Return (p_edges, t_edges) as 1-D numpy arrays of length n_bins+1."""
    p_min, p_max = float(df[p_col].min()), float(df[p_col].max())
    t_min, t_max = float(df[t_col].min()), float(df[t_col].max())
    # Nudge the upper edge by eps so the max value falls inside the last bin
    # under the default right=False convention of pandas.cut/np.digitize.
    eps_p = max(abs(p_max), 1.0) * 1e-9
    eps_t = max(abs(t_max), 1.0) * 1e-9
    p_edges = np.linspace(p_min, p_max + eps_p, n_p_bins + 1)
    t_edges = np.linspace(t_min, t_max + eps_t, n_t_bins + 1)
    return p_edges, t_edges


def assign_pt_cells(df: pd.DataFrame,
                    p_edges: np.ndarray,
                    t_edges: np.ndarray,
                    p_col: str = 'P_kbar',
                    t_col: str = 'T_C'):
    """Return two int arrays (p_cell, t_cell) in [0, n_bins-1]."""
    p_cell = np.clip(np.digitize(df[p_col].values, p_edges, right=False) - 1,
                     0, len(p_edges) - 2)
    t_cell = np.clip(np.digitize(df[t_col].values, t_edges, right=False) - 1,
                     0, len(t_edges) - 2)
    return p_cell, t_cell


def tempered_resample(df: pd.DataFrame,
                      target_col_p: str = 'P_kbar',
                      target_col_t: str = 'T_C',
                      n_p_bins: int = 5,
                      n_t_bins: int = 5,
                      seed: int = 42):
    """Tempered-target resampling on a P-T grid.

    Returns (resampled_df, diagnostics):
      diagnostics['bin_edges']: (p_edges, t_edges)
      diagnostics['actions']:   per-cell DataFrame with cols
                                 [p_cell, t_cell, current, target, action]
      diagnostics['summary']:   dict with n_in, n_out, n_occupied_cells,
                                 n_cells_grown, n_cells_shrunk, n_cells_held
    """
    if len(df) == 0:
        raise ValueError('tempered_resample: input dataframe is empty.')

    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)

    p_edges, t_edges = compute_pt_grid_bins(
        df, n_p_bins=n_p_bins, n_t_bins=n_t_bins,
        p_col=target_col_p, t_col=target_col_t,
    )
    p_cell, t_cell = assign_pt_cells(
        df, p_edges, t_edges, p_col=target_col_p, t_col=target_col_t,
    )
    df = df.assign(_p_cell=p_cell, _t_cell=t_cell)

    # Per-cell current counts (occupied only).
    cell_groups = df.groupby(['_p_cell', '_t_cell']).indices
    occupied = list(cell_groups.keys())
    n_occupied = len(occupied)
    n_total = len(df)
    uniform = n_total / n_occupied  # per-cell uniform target

    action_rows = []
    out_frames = []
    n_grown = n_shrunk = n_held = 0
    for (pc, tc), idx in cell_groups.items():
        current = len(idx)
        target = int(round((current + uniform) / 2.0))
        if target <= 0:
            target = 1
        if target > current:
            extra = target - current
            draws = rng.choice(idx, size=extra, replace=True)
            out_frames.append(df.iloc[np.concatenate([idx, draws])])
            action = 'oversample'
            n_grown += 1
        elif target < current:
            keep = rng.choice(idx, size=target, replace=False)
            out_frames.append(df.iloc[keep])
            action = 'subsample'
            n_shrunk += 1
        else:
            out_frames.append(df.iloc[idx])
            action = 'keep'
            n_held += 1
        action_rows.append({
            'p_cell': int(pc), 't_cell': int(tc),
            'current': int(current), 'target': int(target),
            'action': action,
        })

    resampled = pd.concat(out_frames, axis=0, ignore_index=True)
    resampled = resampled.drop(columns=['_p_cell', '_t_cell'])

    actions_df = pd.DataFrame(action_rows).sort_values(['p_cell', 't_cell']).reset_index(drop=True)
    summary = {
        'n_in': int(n_total),
        'n_out': int(len(resampled)),
        'n_occupied_cells': int(n_occupied),
        'n_cells_grown': int(n_grown),
        'n_cells_shrunk': int(n_shrunk),
        'n_cells_held': int(n_held),
        'uniform_per_cell': float(uniform),
        'seed': int(seed),
    }
    diagnostics = {
        'bin_edges': (p_edges, t_edges),
        'actions': actions_df,
        'summary': summary,
    }
    return resampled, diagnostics
