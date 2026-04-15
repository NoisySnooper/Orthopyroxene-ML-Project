"""v8-fix: restore the nb04b-cell19 headline figure (ML vs Putirka
6-panel comparison) that was lost in the v7 consolidation.

Appends a new code cell directly after nb04 Part 3 (cell 26). The cell
reuses `_p3_arcpl` (the 197-row ArcPL scope) and `_p3_pred_store`
(forest/boosted ML predictions) already defined in cell 26, then runs
Putirka 2008 eq28a/29a on the same rows via Thermobar to identify the
fair subset. Writes figures/fig_nb04_putirka_vs_ml_arcpl.{png,pdf}.

Idempotent via the sentinel 'fig_nb04_putirka_vs_ml_arcpl'.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"

SENTINEL = "fig_nb04_putirka_vs_ml_arcpl"

CELL_SRC = r'''# v8-fix: Restore nb04b cell-19 headline figure -- ML vs Putirka
# 3x2 panel comparison on ArcPL opx-liq scope (forest family).
# Reuses _p3_arcpl and _p3_pred_store from Part 3 above.
import matplotlib.pyplot as _pvm_plt
import numpy as _pvm_np
import pandas as _pvm_pd
import Thermobar as _pvm_tb
from src.plot_style import apply_style as _pvm_apply_style

_pvm_apply_style()

_pvm_df = _p3_arcpl.reset_index(drop=True).copy()
_pvm_yT = _pvm_df['T_C'].values
_pvm_yP = _pvm_df['P_kbar'].values
_pvm_n_full = len(_pvm_df)

# ---- ML predictions (forest family, headline canonical model) ------------
assert 'forest' in _p3_pred_store, 'forest predictions missing from _p3_pred_store'
_pvm_ml = _p3_pred_store['forest'].reset_index(drop=True)
_pvm_ml_T = _pvm_ml['T_pred'].values
_pvm_ml_P = _pvm_ml['P_pred'].values

# ---- Putirka 28a/29a iterative on the same rows (working recipe from
#      nb04 cell 32 encyclopedia; NOT the broken one from cell 21) -------
_pvm_opx_cols = [c for c in _pvm_df.columns if c.endswith('_Opx')]
_pvm_liq_cols = [c for c in _pvm_df.columns if c.endswith('_Liq')]
# Training-schema (unsuffixed opx / liq_ prefix) fallback if the _Opx / _Liq
# columns do not exist on _p3_arcpl (they usually do not -- Part 3 uses the
# training schema). Build suffixed frames from training-schema columns so
# Thermobar sees the oxide names it expects.
if not _pvm_opx_cols:
    _pvm_opx_map = {
        'SiO2': 'SiO2_Opx', 'TiO2': 'TiO2_Opx', 'Al2O3': 'Al2O3_Opx',
        'Cr2O3': 'Cr2O3_Opx', 'FeO_total': 'FeOt_Opx', 'MnO': 'MnO_Opx',
        'MgO': 'MgO_Opx', 'CaO': 'CaO_Opx', 'Na2O': 'Na2O_Opx',
    }
    _pvm_opx_in = _pvm_pd.DataFrame({
        tgt: _pvm_pd.to_numeric(_pvm_df[src], errors='coerce').fillna(0.0)
        for src, tgt in _pvm_opx_map.items() if src in _pvm_df.columns
    })
else:
    _pvm_opx_in = _pvm_df[_pvm_opx_cols].apply(
        _pvm_pd.to_numeric, errors='coerce').fillna(0.0)

if not _pvm_liq_cols:
    _pvm_liq_map = {
        'liq_SiO2': 'SiO2_Liq', 'liq_TiO2': 'TiO2_Liq', 'liq_Al2O3': 'Al2O3_Liq',
        'liq_FeO': 'FeOt_Liq', 'liq_MnO': 'MnO_Liq', 'liq_MgO': 'MgO_Liq',
        'liq_CaO': 'CaO_Liq', 'liq_Na2O': 'Na2O_Liq', 'liq_K2O': 'K2O_Liq',
        'liq_Cr2O3': 'Cr2O3_Liq', 'P2O5_Liq': 'P2O5_Liq', 'H2O_Liq': 'H2O_Liq',
    }
    _pvm_liq_in = _pvm_pd.DataFrame({
        tgt: _pvm_pd.to_numeric(_pvm_df[src], errors='coerce').fillna(0.0)
        for src, tgt in _pvm_liq_map.items() if src in _pvm_df.columns
    })
else:
    _pvm_liq_in = _pvm_df[_pvm_liq_cols].apply(
        _pvm_pd.to_numeric, errors='coerce').fillna(0.0)

# v8-fix: Fe3Fet_Liq REQUIRED by eq28a/29a -- use 0.0 (reduced) per spec.
_pvm_liq_in['Fe3Fet_Liq'] = 0.0

try:
    _pvm_out = _pvm_tb.calculate_opx_liq_press_temp(
        opx_comps=_pvm_opx_in, liq_comps=_pvm_liq_in,
        equationP='P_Put2008_eq29a', equationT='T_Put2008_eq28a',
    )
    _pvm_put_T = _pvm_np.asarray(_pvm_out['T_K_calc']) - 273.15
    _pvm_put_P = _pvm_np.asarray(_pvm_out['P_kbar_calc'])
except Exception as _pvm_e:
    print(f'Putirka eq28a/29a run failed: {_pvm_e}')
    _pvm_put_T = _pvm_np.full(_pvm_n_full, _pvm_np.nan)
    _pvm_put_P = _pvm_np.full(_pvm_n_full, _pvm_np.nan)

# ---- Fair subset: rows where Putirka returns finite T AND P -------------
_pvm_ok = _pvm_np.isfinite(_pvm_put_T) & _pvm_np.isfinite(_pvm_put_P)
_pvm_n_fair = int(_pvm_ok.sum())
_pvm_fail_rate = 100.0 * (1.0 - _pvm_n_fair / _pvm_n_full)
print(f'Putirka convergence: fair n={_pvm_n_fair}/{_pvm_n_full} '
      f'({100.0 * _pvm_n_fair / _pvm_n_full:.1f}% coverage, '
      f'failure rate {_pvm_fail_rate:.1f}%)')


def _pvm_rmse(a, b):
    a = _pvm_np.asarray(a, dtype=float); b = _pvm_np.asarray(b, dtype=float)
    m = _pvm_np.isfinite(a) & _pvm_np.isfinite(b)
    return float(_pvm_np.sqrt(_pvm_np.mean((a[m] - b[m]) ** 2))) if m.any() else _pvm_np.nan


# ---- Figure: 3 rows x 2 cols (T left, P right) --------------------------
_pvm_fig, _pvm_ax = _pvm_plt.subplots(3, 2, figsize=(11, 14),
                                      constrained_layout=False)
_pvm_COL_ML = '#0072B2'   # blue
_pvm_COL_PU = '#D55E00'   # orange

# T axis range: pool observed + all predictions
_pvm_T_pool = _pvm_np.concatenate([_pvm_yT, _pvm_ml_T, _pvm_put_T[_pvm_ok]])
_pvm_T_pool = _pvm_T_pool[_pvm_np.isfinite(_pvm_T_pool)]
_pvm_T_lo, _pvm_T_hi = float(_pvm_T_pool.min()) - 30, float(_pvm_T_pool.max()) + 30

_pvm_P_pool = _pvm_np.concatenate([_pvm_yP, _pvm_ml_P, _pvm_put_P[_pvm_ok]])
_pvm_P_pool = _pvm_P_pool[_pvm_np.isfinite(_pvm_P_pool)]
_pvm_P_lo, _pvm_P_hi = float(_pvm_P_pool.min()) - 1, float(_pvm_P_pool.max()) + 1


def _pvm_panel(ax, x, y, color, title, unit, lo, hi, extra_text=None):
    m = _pvm_np.isfinite(x) & _pvm_np.isfinite(y)
    ax.scatter(x[m], y[m], s=16, alpha=0.6, color=color, edgecolor='none')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'Observed ({unit})')
    ax.set_ylabel(f'Predicted ({unit})')
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.3)
    if extra_text:
        ax.text(0.03, 0.97, extra_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='0.6', alpha=0.85))


# Row 1: ML on ALL samples (n=197, 100% coverage) -------------------------
_pvm_rmse_T_all = _pvm_rmse(_pvm_ml_T, _pvm_yT)
_pvm_rmse_P_all = _pvm_rmse(_pvm_ml_P, _pvm_yP)
_pvm_panel(_pvm_ax[0, 0], _pvm_yT, _pvm_ml_T, _pvm_COL_ML,
           f'ML forest T - all ArcPL (n={_pvm_n_full}, 100% coverage)\n'
           f'RMSE = {_pvm_rmse_T_all:.1f} C',
           'C', _pvm_T_lo, _pvm_T_hi)
_pvm_panel(_pvm_ax[0, 1], _pvm_yP, _pvm_ml_P, _pvm_COL_ML,
           f'ML forest P - all ArcPL (n={_pvm_n_full}, 100% coverage)\n'
           f'RMSE = {_pvm_rmse_P_all:.2f} kbar',
           'kbar', _pvm_P_lo, _pvm_P_hi)

# Row 2: ML on fair subset (where Putirka converges) ----------------------
_pvm_rmse_T_fair = _pvm_rmse(_pvm_ml_T[_pvm_ok], _pvm_yT[_pvm_ok])
_pvm_rmse_P_fair = _pvm_rmse(_pvm_ml_P[_pvm_ok], _pvm_yP[_pvm_ok])
_pvm_panel(_pvm_ax[1, 0], _pvm_yT[_pvm_ok], _pvm_ml_T[_pvm_ok], _pvm_COL_ML,
           f'ML forest T - fair subset (n={_pvm_n_fair})\n'
           f'RMSE = {_pvm_rmse_T_fair:.1f} C',
           'C', _pvm_T_lo, _pvm_T_hi)
_pvm_panel(_pvm_ax[1, 1], _pvm_yP[_pvm_ok], _pvm_ml_P[_pvm_ok], _pvm_COL_ML,
           f'ML forest P - fair subset (n={_pvm_n_fair})\n'
           f'RMSE = {_pvm_rmse_P_fair:.2f} kbar',
           'kbar', _pvm_P_lo, _pvm_P_hi)

# Row 3: Putirka 28a/29a on fair subset -----------------------------------
_pvm_rmse_T_put = _pvm_rmse(_pvm_put_T[_pvm_ok], _pvm_yT[_pvm_ok])
_pvm_rmse_P_put = _pvm_rmse(_pvm_put_P[_pvm_ok], _pvm_yP[_pvm_ok])
_pvm_fail_text = f'Putirka failure rate on full set: {_pvm_fail_rate:.1f}%'
_pvm_panel(_pvm_ax[2, 0], _pvm_yT[_pvm_ok], _pvm_put_T[_pvm_ok], _pvm_COL_PU,
           f'Putirka 2008 eq28a T - fair subset (n={_pvm_n_fair})\n'
           f'RMSE = {_pvm_rmse_T_put:.1f} C',
           'C', _pvm_T_lo, _pvm_T_hi, extra_text=_pvm_fail_text)
_pvm_panel(_pvm_ax[2, 1], _pvm_yP[_pvm_ok], _pvm_put_P[_pvm_ok], _pvm_COL_PU,
           f'Putirka 2008 eq29a P - fair subset (n={_pvm_n_fair})\n'
           f'RMSE = {_pvm_rmse_P_put:.2f} kbar',
           'kbar', _pvm_P_lo, _pvm_P_hi, extra_text=_pvm_fail_text)

_pvm_fig.suptitle(
    'ArcPL opx-liq: ML forest vs Putirka 2008 (restored nb04b headline)',
    fontsize=13, fontweight='bold', y=0.995)
_pvm_fig.tight_layout(rect=[0, 0.06, 1, 0.975])

_pvm_caption = (
    f'Top: ML predictions on all n={_pvm_n_full} ArcPL opx-bearing experiments '
    f'(100% coverage). Middle: ML predictions on the n={_pvm_n_fair} subset '
    f'where Putirka 2008 eq 28a/29a converges. Bottom: Putirka predictions '
    f'on the same fair subset. Putirka fails to predict on '
    f'{_pvm_fail_rate:.1f}% of the full dataset due to equilibrium filter '
    f'rejections.'
)
_pvm_plt.figtext(0.5, 0.02, _pvm_caption, ha='center', va='bottom',
                 fontsize=9, wrap=True)

for _ext in ('png', 'pdf'):
    _pvm_fig.savefig(FIGURES / f'fig_nb04_putirka_vs_ml_arcpl.{_ext}',
                     dpi=300 if _ext == 'png' else None, bbox_inches='tight')
_pvm_plt.show()
_pvm_plt.close(_pvm_fig)

print(f'Saved figures/fig_nb04_putirka_vs_ml_arcpl.{{png,pdf}}')
print(f'  Row 1 (ML all):       T RMSE={_pvm_rmse_T_all:.1f} C  P RMSE={_pvm_rmse_P_all:.2f} kbar')
print(f'  Row 2 (ML fair):      T RMSE={_pvm_rmse_T_fair:.1f} C  P RMSE={_pvm_rmse_P_fair:.2f} kbar')
print(f'  Row 3 (Putirka fair): T RMSE={_pvm_rmse_T_put:.1f} C  P RMSE={_pvm_rmse_P_put:.2f} kbar')
assert _pvm_rmse_T_fair < _pvm_rmse_T_put, (
    f'ML middle-row T RMSE ({_pvm_rmse_T_fair:.1f}) not lower than '
    f'Putirka bottom-row ({_pvm_rmse_T_put:.1f}) -- check inputs')
'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # Already patched?
    for i, c in enumerate(nb.cells):
        if c.cell_type == 'code' and SENTINEL in c.source and '_pvm_' in c.source:
            print(f'putirka-vs-ml cell already present at index {i}; no-op')
            return 0

    new_cell = nbformat.v4.new_code_cell(CELL_SRC)
    new_cell['id'] = uuid.uuid4().hex[:8]
    # Insert immediately after cell 26 (Part 3 body)
    nb.cells.insert(27, new_cell)
    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'Inserted putirka-vs-ml cell at index 27; total cells now {len(nb.cells)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
