"""v8 nb04 fixes: F1 (Fe3Fet_Liq + opx-only API), F2 (Part 3 figure),
F3 (same F1 fix in Part 5), F4 (independent T/P sort in cell 23).

Applies surgical replacements to cells 21, 23, 26, 30. Idempotent via
sentinel comments. Does NOT execute the notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"

# ---------------------------------------------------------------------
# F1/F3 shared replacement: fix the Putirka opx block.
# - Add _liq_in['Fe3Fet_Liq'] = 0.0 after the H2O_Liq assignment.
# - Replace the broken calculate_opx_only_press_temp / _temp calls with
#   the encyclopedia's working pattern (calculate_opx_only_press for P,
#   calculate_opx_liq_temp with T_Put2008_eq28b_opx_sat for T).
# Idempotent via sentinel 'v8-fix: Putirka opx corrected API'.
# ---------------------------------------------------------------------

V8_SENTINEL = 'v8-fix: Putirka opx corrected API'

# Replacement 1: add Fe3Fet_Liq right after H2O_Liq handling. The same
# pattern shows up in cells 21, 23, and 30.
FE3_OLD_TEMPLATE = """if 'H2O_Liq' in {scope}.columns:
{indent}_liq_in['H2O_Liq'] = pd.to_numeric({scope}['H2O_Liq'], errors='coerce').fillna(0.0)"""

FE3_NEW_TEMPLATE = """if 'H2O_Liq' in {scope}.columns:
{indent}_liq_in['H2O_Liq'] = pd.to_numeric({scope}['H2O_Liq'], errors='coerce').fillna(0.0)
{indent}# v8-fix: Putirka opx-liq eq28a/29a require Fe3Fet_Liq
{indent}_liq_in['Fe3Fet_Liq'] = 0.0"""


# ---------------------------------------------------------------------
# CELL 21 specifics
# ---------------------------------------------------------------------
# Cell 21 has one extra bug vs. cells 23/30: the Cr2O3 mask uses the
# unsuffixed key 'Cr2O3' even though _opx_in columns are suffixed '_Opx'.
CELL21_MASK_OLD = (
    "        _mask_cr = (_opx_in.get('Cr2O3', pd.Series(np.zeros(len(_opx_in))))\n"
    "                    .fillna(0.0) > 1e-4).values"
)
CELL21_MASK_NEW = (
    "        # v8-fix: _opx_in columns are '_Opx'-suffixed; use Cr2O3_Opx.\n"
    "        _mask_cr = (_opx_in.get('Cr2O3_Opx', pd.Series(np.zeros(len(_opx_in))))\n"
    "                    .fillna(0.0) > 1e-4).values"
)

# Cell 21's broken opx-only iterative + ceiling block.
CELL21_OPX_OLD = """            _iter_only = _pt_opx.calculate_opx_only_press_temp(
                opx_comps=_opx_cr,
                equationT='T_Put2008_eq28b_opx_sat',
                equationP='P_Put2008_eq29c')
            _T_o = np.full(len(_opx_in), np.nan)
            _P_o = np.full(len(_opx_in), np.nan)
            _T_o[_mask_cr] = (_iter_only['T_K_calc'].values - 273.15
                              if 'T_K_calc' in _iter_only.columns
                              else _iter_only.iloc[:, 0].values - 273.15)
            _P_o[_mask_cr] = (_iter_only['P_kbar_calc'].values
                              if 'P_kbar_calc' in _iter_only.columns
                              else _iter_only.iloc[:, 1].values)
            preds['Putirka 2008 opx-only'] = (_T_o, _P_o)

            # (d) opx-only with true T input -> ceiling
            try:
                _Pceil_o = _pt_opx.calculate_opx_only_press(
                    opx_comps=_opx_cr,
                    equationP='P_Put2008_eq29c',
                    T=(y_T + 273.15)[_mask_cr])
                _Tceil_o = _pt_opx.calculate_opx_only_temp(
                    opx_comps=_opx_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    P=y_P[_mask_cr])"""

CELL21_OPX_NEW = """            # v8-fix: Thermobar has no calculate_opx_only_press_temp or _temp.
            # Two-step iteration: eq28a for initial T, then eq29c uses that T,
            # then eq28b_opx_sat (via calculate_opx_liq_temp) uses the new P.
            _liq_cr = _liq_in[_mask_cr].reset_index(drop=True)
            _T_init = _pt_opx.calculate_opx_liq_temp(
                opx_comps=_opx_cr, liq_comps=_liq_cr,
                equationT='T_Put2008_eq28a', P=10.0)
            _T_init_K = (_T_init['T_K_calc'].values
                         if hasattr(_T_init, 'columns') and 'T_K_calc' in _T_init.columns
                         else (_T_init.values if hasattr(_T_init, 'values')
                               else np.asarray(_T_init)))
            if np.nanmedian(_T_init_K) < 400:
                _T_init_K = _T_init_K + 273.15
            _P_step = _pt_opx.calculate_opx_only_press(
                opx_comps=_opx_cr,
                equationP='P_Put2008_eq29c',
                T=_T_init_K)
            _P_arr = (_P_step['P_kbar_calc'].values
                      if hasattr(_P_step, 'columns') and 'P_kbar_calc' in _P_step.columns
                      else (_P_step.values if hasattr(_P_step, 'values')
                            else np.asarray(_P_step)))
            _T_step = _pt_opx.calculate_opx_liq_temp(
                opx_comps=_opx_cr, liq_comps=_liq_cr,
                equationT='T_Put2008_eq28b_opx_sat', P=_P_arr)
            _T_K = (_T_step['T_K_calc'].values
                    if hasattr(_T_step, 'columns') and 'T_K_calc' in _T_step.columns
                    else (_T_step.values if hasattr(_T_step, 'values')
                          else np.asarray(_T_step)))
            _T_arr = _T_K - 273.15 if np.nanmedian(_T_K) > 400 else _T_K
            _T_o = np.full(len(_opx_in), np.nan)
            _P_o = np.full(len(_opx_in), np.nan)
            _T_o[_mask_cr] = _T_arr
            _P_o[_mask_cr] = _P_arr
            preds['Putirka 2008 opx-only'] = (_T_o, _P_o)

            # (d) opx-only ceiling: observed T -> 29c for P; observed P ->
            # 28b_opx_sat for T.
            try:
                _Pceil_o = _pt_opx.calculate_opx_only_press(
                    opx_comps=_opx_cr,
                    equationP='P_Put2008_eq29c',
                    T=(y_T + 273.15)[_mask_cr])
                _Tceil_o = _pt_opx.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28b_opx_sat', P=y_P[_mask_cr])"""

CELL21_CEIL_OLD = """                _Pc[_mask_cr] = (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                                 else np.asarray(_Pceil_o))
                _Tc[_mask_cr] = (_Tceil_o.values - 273.15
                                 if hasattr(_Tceil_o, 'values')
                                 else np.asarray(_Tceil_o) - 273.15)"""
CELL21_CEIL_NEW = """                _Pc[_mask_cr] = (_Pceil_o['P_kbar_calc'].values
                                 if hasattr(_Pceil_o, 'columns') and 'P_kbar_calc' in _Pceil_o.columns
                                 else (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                                       else np.asarray(_Pceil_o)))
                _Tck = (_Tceil_o['T_K_calc'].values
                        if hasattr(_Tceil_o, 'columns') and 'T_K_calc' in _Tceil_o.columns
                        else (_Tceil_o.values if hasattr(_Tceil_o, 'values')
                              else np.asarray(_Tceil_o)))
                _Tc[_mask_cr] = _Tck - 273.15 if np.nanmedian(_Tck) > 400 else _Tck"""


# ---------------------------------------------------------------------
# CELL 21 assertion at end
# ---------------------------------------------------------------------
CELL21_ASSERT_TAIL = """

# v8-fix: assert Putirka opx methods made it in.
assert len(preds) >= 9, f'Expected >=9 methods after v8 opx fixes, got {len(preds)}'
_expected_opx = {'Putirka 2008 opx-liq', 'Putirka 2008 opx-only'}
_missing_opx = _expected_opx - set(preds)
print(f'v8-check: preds has {len(preds)} methods; '
      f'Putirka opx variants present: {sorted(_expected_opx & set(preds))}')
if _missing_opx:
    print(f'v8-check WARNING: missing {_missing_opx}')
"""


# ---------------------------------------------------------------------
# CELL 23 broken opx-only block
# ---------------------------------------------------------------------
CELL23_OPX_OLD = """            try:
                _iter_only = _pt_opx.calculate_opx_only_press_temp(
                    opx_comps=_opx_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    equationP='P_Put2008_eq29c')
                _To = np.full(len(scope_df), np.nan)
                _Po = np.full(len(scope_df), np.nan)
                _To[_mask_cr] = (_iter_only['T_K_calc'].values - 273.15
                                 if 'T_K_calc' in _iter_only.columns
                                 else _iter_only.iloc[:, 0].values - 273.15)
                _Po[_mask_cr] = (_iter_only['P_kbar_calc'].values
                                 if 'P_kbar_calc' in _iter_only.columns
                                 else _iter_only.iloc[:, 1].values)
                p['Putirka 2008 opx-only'] = (_To, _Po)"""

CELL23_OPX_NEW = """            try:
                # v8-fix: use encyclopedia pattern (no calculate_opx_only_press_temp).
                _liq_cr = _liq_in[_mask_cr].reset_index(drop=True)
                _T_init = _pt_opx.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28a', P=10.0)
                _T_init_K = (_T_init['T_K_calc'].values
                             if hasattr(_T_init, 'columns') and 'T_K_calc' in _T_init.columns
                             else (_T_init.values if hasattr(_T_init, 'values')
                                   else np.asarray(_T_init)))
                if np.nanmedian(_T_init_K) < 400:
                    _T_init_K = _T_init_K + 273.15
                _P_step = _pt_opx.calculate_opx_only_press(
                    opx_comps=_opx_cr, equationP='P_Put2008_eq29c', T=_T_init_K)
                _P_arr = (_P_step['P_kbar_calc'].values
                          if hasattr(_P_step, 'columns') and 'P_kbar_calc' in _P_step.columns
                          else (_P_step.values if hasattr(_P_step, 'values')
                                else np.asarray(_P_step)))
                _T_step = _pt_opx.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28b_opx_sat', P=_P_arr)
                _T_K = (_T_step['T_K_calc'].values
                        if hasattr(_T_step, 'columns') and 'T_K_calc' in _T_step.columns
                        else (_T_step.values if hasattr(_T_step, 'values')
                              else np.asarray(_T_step)))
                _T_arr = _T_K - 273.15 if np.nanmedian(_T_K) > 400 else _T_K
                _To = np.full(len(scope_df), np.nan)
                _Po = np.full(len(scope_df), np.nan)
                _To[_mask_cr] = _T_arr
                _Po[_mask_cr] = _P_arr
                p['Putirka 2008 opx-only'] = (_To, _Po)"""


# ---------------------------------------------------------------------
# CELL 23 opx-only ceiling block
# ---------------------------------------------------------------------
CELL23_CEIL_SEARCH_TAG = "calculate_opx_only_temp"
# We'll find and replace any remaining calculate_opx_only_temp calls in
# cells 21/23/30 via a generic per-cell regex post-pass.


# ---------------------------------------------------------------------
# CELL 26 Part 3 figure (F2)
# ---------------------------------------------------------------------
PART3_FIG_SENTINEL = "# v8-fix: Part 3 scatter figure"

PART3_FIG_CODE = """

# v8-fix: Part 3 scatter figure
import matplotlib.pyplot as _p3_plt
from src.plot_style import apply_style as _p3_apply_style

_p3_apply_style()
_p3_col_forest  = '#0072B2'
_p3_col_boosted = '#D55E00'

_p3_yT = _p3_arcpl['T_C'].values
_p3_yP = _p3_arcpl['P_kbar'].values

_p3_fig, _p3_axes = _p3_plt.subplots(2, 4, figsize=(16, 8))
for _col, (_fam, _color) in enumerate([('forest', _p3_col_forest),
                                        ('boosted', _p3_col_boosted)]):
    if _fam not in _p3_pred_store:
        continue
    _pd_f = _p3_pred_store[_fam].reset_index(drop=True)
    _tp = _pd_f['T_pred'].values
    _pp = _pd_f['P_pred'].values
    _tr = _tp - _p3_yT
    _pr = _pp - _p3_yP
    _rT = float(np.sqrt(np.nanmean(_tr ** 2)))
    _rP = float(np.sqrt(np.nanmean(_pr ** 2)))
    _base = _col * 2
    # [0, base]: pred vs obs T
    _a = _p3_axes[0, _base]
    _a.scatter(_p3_yT, _tp, s=12, alpha=0.6, color=_color, edgecolor='none')
    _lo, _hi = min(_p3_yT.min(), _tp.min()), max(_p3_yT.max(), _tp.max())
    _a.plot([_lo, _hi], [_lo, _hi], 'k--', lw=0.8, alpha=0.5)
    _a.set_xlabel('Observed T (C)'); _a.set_ylabel('Predicted T (C)')
    _a.set_title(f'{_fam.capitalize()} T (RMSE={_rT:.1f} C)')
    # [0, base+1]: pred vs obs P
    _a = _p3_axes[0, _base + 1]
    _a.scatter(_p3_yP, _pp, s=12, alpha=0.6, color=_color, edgecolor='none')
    _lo, _hi = min(_p3_yP.min(), _pp.min()), max(_p3_yP.max(), _pp.max())
    _a.plot([_lo, _hi], [_lo, _hi], 'k--', lw=0.8, alpha=0.5)
    _a.set_xlabel('Observed P (kbar)'); _a.set_ylabel('Predicted P (kbar)')
    _a.set_title(f'{_fam.capitalize()} P (RMSE={_rP:.2f} kbar)')
    # [1, base]: T residual hist
    _a = _p3_axes[1, _base]
    _a.hist(_tr[np.isfinite(_tr)], bins=30, color=_color, edgecolor='black',
            linewidth=0.3, alpha=0.8)
    _a.axvline(0, color='k', lw=0.8, alpha=0.5)
    _a.set_xlabel('T residual (pred - obs, C)'); _a.set_ylabel('count')
    _a.set_title(f'{_fam.capitalize()} T residuals')
    # [1, base+1]: P residual hist
    _a = _p3_axes[1, _base + 1]
    _a.hist(_pr[np.isfinite(_pr)], bins=30, color=_color, edgecolor='black',
            linewidth=0.3, alpha=0.8)
    _a.axvline(0, color='k', lw=0.8, alpha=0.5)
    _a.set_xlabel('P residual (pred - obs, kbar)'); _a.set_ylabel('count')
    _a.set_title(f'{_fam.capitalize()} P residuals')

_p3_fig.suptitle(f'ArcPL opx-liq external validation (n={len(_p3_arcpl)})',
                 fontsize=12, fontweight='bold', y=1.00)
_p3_plt.tight_layout()
for _ext in ('png', 'pdf'):
    _p3_fig.savefig(FIGURES / f'fig_nb04_arcpl_opx_liq_scatter.{_ext}',
                    dpi=300 if _ext == 'png' else None, bbox_inches='tight')
_p3_plt.show()
_p3_plt.close(_p3_fig)
print('Saved figures/fig_nb04_arcpl_opx_liq_scatter.{png,pdf}')
"""


# ---------------------------------------------------------------------
# F4: independent T/P sort in cell 23.
# ---------------------------------------------------------------------
# The cell 23 figure builds panels A (T RMSE), B (P RMSE), C (coverage)
# via a single sort key. We need each RMSE panel sorted by its own metric.
# We'll add a post-processing step inside _plot_three_way_scope that
# rebuilds the P panel ordering and re-labels its y-axis.


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _apply_fe3(source: str, scope_var: str) -> tuple[str, int]:
    """Insert Fe3Fet_Liq = 0.0 after the H2O_Liq assignment in `source`.
    Returns (new_source, num_replacements). Idempotent via sentinel."""
    if 'Putirka opx-liq eq28a/29a require Fe3Fet_Liq' in source:
        return source, 0
    # Indent can be 4 or 8 spaces depending on cell. Try both.
    for indent in ('        ', '            '):
        old = FE3_OLD_TEMPLATE.format(scope=scope_var, indent=indent)
        new = FE3_NEW_TEMPLATE.format(scope=scope_var, indent=indent)
        if old in source:
            return source.replace(old, new, 1), 1
    return source, 0


def fix_cell_21(src: str) -> str:
    if V8_SENTINEL in src:
        print('cell 21: already v8-patched')
        return src
    n_changed = 0
    src, n = _apply_fe3(src, 'm')
    n_changed += n
    if CELL21_MASK_OLD in src:
        src = src.replace(CELL21_MASK_OLD, CELL21_MASK_NEW, 1)
        n_changed += 1
    else:
        print('  cell 21: MASK_OLD not found')
    if CELL21_OPX_OLD in src:
        src = src.replace(CELL21_OPX_OLD, CELL21_OPX_NEW, 1)
        n_changed += 1
    else:
        print('  cell 21: OPX_OLD not found')
    if CELL21_CEIL_OLD in src:
        src = src.replace(CELL21_CEIL_OLD, CELL21_CEIL_NEW, 1)
        n_changed += 1
    else:
        print('  cell 21: CEIL_OLD not found')
    if '# v8-fix: assert Putirka opx methods made it in.' not in src:
        src = src.rstrip() + '\n' + CELL21_ASSERT_TAIL
        n_changed += 1
    # Tag with sentinel
    if n_changed > 0:
        src = f'# {V8_SENTINEL}\n' + src
    print(f'cell 21: {n_changed} change(s)')
    return src


def fix_cell_23(src: str) -> str:
    if V8_SENTINEL in src:
        print('cell 23: already v8-patched')
        return src
    n_changed = 0
    src, n = _apply_fe3(src, 'scope_df')
    n_changed += n
    if CELL23_OPX_OLD in src:
        src = src.replace(CELL23_OPX_OLD, CELL23_OPX_NEW, 1)
        n_changed += 1
    else:
        print('  cell 23: OPX_OLD not found')
    # Replace the ceiling opx-only _temp call (same bug)
    old_ceil = """                _Tc_o = _pt_opx.calculate_opx_only_temp(
                    opx_comps=_opx_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    P=y_P_s[_mask_cr])"""
    new_ceil = """                _Tc_o = _pt_opx.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    P=y_P_s[_mask_cr])"""
    if old_ceil in src:
        src = src.replace(old_ceil, new_ceil, 1)
        n_changed += 1
    # Also replace Pceil_o unpacking + Tc_o unpacking to handle DataFrame returns
    old_unpack = """                _Pc_o = (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                         else np.asarray(_Pceil_o))
                _Tc_o_arr = (_Tc_o.values - 273.15 if hasattr(_Tc_o, 'values')
                             else np.asarray(_Tc_o) - 273.15)"""
    new_unpack = """                _Pc_o = (_Pceil_o['P_kbar_calc'].values
                         if hasattr(_Pceil_o, 'columns') and 'P_kbar_calc' in _Pceil_o.columns
                         else (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                               else np.asarray(_Pceil_o)))
                _Tc_K = (_Tc_o['T_K_calc'].values
                         if hasattr(_Tc_o, 'columns') and 'T_K_calc' in _Tc_o.columns
                         else (_Tc_o.values if hasattr(_Tc_o, 'values')
                               else np.asarray(_Tc_o)))
                _Tc_o_arr = _Tc_K - 273.15 if np.nanmedian(_Tc_K) > 400 else _Tc_K"""
    if old_unpack in src:
        src = src.replace(old_unpack, new_unpack, 1)
        n_changed += 1
    if n_changed > 0:
        src = f'# {V8_SENTINEL}\n' + src
    print(f'cell 23: {n_changed} change(s)')
    return src


def fix_cell_30(src: str) -> str:
    if V8_SENTINEL in src:
        print('cell 30: already v8-patched')
        return src
    n_changed = 0
    src, n = _apply_fe3(src, '_p5_merged')
    n_changed += n
    # Replace _io = calculate_opx_only_press_temp block
    old_block = """                _io = _p5_pt.calculate_opx_only_press_temp(
                    opx_comps=_opx_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    equationP='P_Put2008_eq29c')
                _To = np.full(len(_p5_merged), np.nan)
                _Po = np.full(len(_p5_merged), np.nan)
                _To[_mask_cr] = (_io['T_K_calc'].values - 273.15
                                 if 'T_K_calc' in _io.columns
                                 else _io.iloc[:, 0].values - 273.15)
                _Po[_mask_cr] = (_io['P_kbar_calc'].values
                                 if 'P_kbar_calc' in _io.columns
                                 else _io.iloc[:, 1].values)
                _p5_preds_all['Putirka 2008 opx-only'] = (_To, _Po)"""
    new_block = """                # v8-fix: no calculate_opx_only_press_temp in Thermobar.
                _liq_cr = _liq_in[_mask_cr].reset_index(drop=True)
                _T_init = _p5_pt.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28a', P=10.0)
                _T_init_K = (_T_init['T_K_calc'].values
                             if hasattr(_T_init, 'columns') and 'T_K_calc' in _T_init.columns
                             else (_T_init.values if hasattr(_T_init, 'values')
                                   else np.asarray(_T_init)))
                if np.nanmedian(_T_init_K) < 400:
                    _T_init_K = _T_init_K + 273.15
                _P_step = _p5_pt.calculate_opx_only_press(
                    opx_comps=_opx_cr, equationP='P_Put2008_eq29c', T=_T_init_K)
                _P_arr = (_P_step['P_kbar_calc'].values
                          if hasattr(_P_step, 'columns') and 'P_kbar_calc' in _P_step.columns
                          else (_P_step.values if hasattr(_P_step, 'values')
                                else np.asarray(_P_step)))
                _T_step = _p5_pt.calculate_opx_liq_temp(
                    opx_comps=_opx_cr, liq_comps=_liq_cr,
                    equationT='T_Put2008_eq28b_opx_sat', P=_P_arr)
                _T_K = (_T_step['T_K_calc'].values
                        if hasattr(_T_step, 'columns') and 'T_K_calc' in _T_step.columns
                        else (_T_step.values if hasattr(_T_step, 'values')
                              else np.asarray(_T_step)))
                _T_arr = _T_K - 273.15 if np.nanmedian(_T_K) > 400 else _T_K
                _To = np.full(len(_p5_merged), np.nan)
                _Po = np.full(len(_p5_merged), np.nan)
                _To[_mask_cr] = _T_arr
                _Po[_mask_cr] = _P_arr
                _p5_preds_all['Putirka 2008 opx-only'] = (_To, _Po)"""
    if old_block in src:
        src = src.replace(old_block, new_block, 1)
        n_changed += 1
    else:
        print('  cell 30: opx-only block not found')
    # Replace Tc_o opx_only_temp call
    old_tc = """                    _Tc_o = _p5_pt.calculate_opx_only_temp(
                        opx_comps=_opx_cr, equationT='T_Put2008_eq28b_opx_sat',
                        P=_yP_all[_mask_cr])"""
    new_tc = """                    _Tc_o = _p5_pt.calculate_opx_liq_temp(
                        opx_comps=_opx_cr, liq_comps=_liq_cr,
                        equationT='T_Put2008_eq28b_opx_sat',
                        P=_yP_all[_mask_cr])"""
    if old_tc in src:
        src = src.replace(old_tc, new_tc, 1)
        n_changed += 1
    # Fix unpacking
    old_unpack = """                    _Po2[_mask_cr] = (_Pc_o.values if hasattr(_Pc_o, 'values')
                                      else np.asarray(_Pc_o))
                    _To2[_mask_cr] = (_Tc_o.values - 273.15 if hasattr(_Tc_o, 'values')
                                      else np.asarray(_Tc_o) - 273.15)"""
    new_unpack = """                    _Po2[_mask_cr] = (_Pc_o['P_kbar_calc'].values
                                      if hasattr(_Pc_o, 'columns') and 'P_kbar_calc' in _Pc_o.columns
                                      else (_Pc_o.values if hasattr(_Pc_o, 'values')
                                            else np.asarray(_Pc_o)))
                    _Tck2 = (_Tc_o['T_K_calc'].values
                             if hasattr(_Tc_o, 'columns') and 'T_K_calc' in _Tc_o.columns
                             else (_Tc_o.values if hasattr(_Tc_o, 'values')
                                   else np.asarray(_Tc_o)))
                    _To2[_mask_cr] = _Tck2 - 273.15 if np.nanmedian(_Tck2) > 400 else _Tck2"""
    if old_unpack in src:
        src = src.replace(old_unpack, new_unpack, 1)
        n_changed += 1
    if n_changed > 0:
        src = f'# {V8_SENTINEL}\n' + src
    print(f'cell 30: {n_changed} change(s)')
    return src


def fix_cell_26(src: str) -> str:
    """F2: add Part 3 scatter figure at end of cell 26."""
    if PART3_FIG_SENTINEL in src:
        print('cell 26: Part 3 figure already present')
        return src
    print('cell 26: appending Part 3 figure block')
    return src.rstrip() + '\n' + PART3_FIG_CODE


def fix_cell_23_sort(src: str) -> str:
    """F4: make T and P panels sort independently. The current figure
    sorts `shown` by T_RMSE and reuses that order for both T and P
    panels (sharey=ax_T). We switch to per-panel sorts and give ax_P
    its own y-axis + tick labels."""
    if 'v8-fix: independent T/P sort' in src:
        print('cell 23 sort: already v8-patched')
        return src
    # 1. Build two sorted frames: shown_T (T_RMSE) and shown_P (P_RMSE).
    old_sort = "shown = shown.sort_values('T_RMSE', na_position='last').reset_index(drop=True)"
    new_sort = (
        "# v8-fix: independent T/P sort — each RMSE panel uses its own ordering\n"
        "    shown_T = shown.sort_values('T_RMSE', na_position='last').reset_index(drop=True)\n"
        "    shown_P = shown.sort_values('P_RMSE', na_position='last').reset_index(drop=True)\n"
        "    shown = shown_T  # coverage panel + footnotes stay T-sorted for reference"
    )
    if old_sort not in src:
        print('cell 23 sort: sort anchor not found')
        return src
    src = src.replace(old_sort, new_sort, 1)
    # 2. Stop sharing y-axis between ax_T and ax_P.
    old_ax_P = "ax_P = fig.add_subplot(gs[0, 1], sharey=ax_T)"
    new_ax_P = "ax_P = fig.add_subplot(gs[0, 1])  # v8-fix: no sharey (P sort differs)"
    src = src.replace(old_ax_P, new_ax_P, 1)
    # 3. Replace the T/P loop with two explicit renders using shown_T / shown_P.
    old_loop = """    for ax, col, col_lo, col_hi, xlab, fmt in [
        (ax_T, 'T_RMSE', 'T_RMSE_CI_lo', 'T_RMSE_CI_hi', 'T RMSE (C)', '{:.0f}'),
        (ax_P, 'P_RMSE', 'P_RMSE_CI_lo', 'P_RMSE_CI_hi', 'P RMSE (kbar)', '{:.2f}'),
    ]:
        vals = shown[col].values.astype(float)
        lo = shown[col_lo].values.astype(float)
        hi = shown[col_hi].values.astype(float)
        xerr_lo = np.where(np.isfinite(vals) & np.isfinite(lo), vals - lo, 0.0)
        xerr_hi = np.where(np.isfinite(vals) & np.isfinite(hi), hi - vals, 0.0)
        xerr = np.vstack([xerr_lo, xerr_hi])
        colors = [COLORS.get(n, '#777') for n in shown['Method']]
        bars = ax.barh(y_pos, np.nan_to_num(vals, nan=0.0), xerr=xerr,
                       color=colors, edgecolor='black',
                       error_kw={'elinewidth': 1.1, 'capsize': 2.5})
        for b, n in zip(bars, shown['Method']):
            if _is_putirka(n):
                b.set_hatch('///')
            if not np.isfinite(shown.loc[b.get_y() // 1, col]) if False else False:
                pass
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v * 1.01 + 0.001, i, fmt.format(v),
                        va='center', fontsize=8)
            else:
                ax.text(0.01, i, 'n/a', va='center', fontsize=8,
                        color='gray', style='italic')
        if np.isfinite(vals).any():
            best = float(np.nanmin(vals))
            ax.axvline(best, color='gray', linestyle=':', lw=1, alpha=0.6)
        ax.set_xlabel(xlab)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=0)"""
    new_loop = """    for ax, df, col, col_lo, col_hi, xlab, fmt in [
        (ax_T, shown_T, 'T_RMSE', 'T_RMSE_CI_lo', 'T_RMSE_CI_hi', 'T RMSE (C)', '{:.0f}'),
        (ax_P, shown_P, 'P_RMSE', 'P_RMSE_CI_lo', 'P_RMSE_CI_hi', 'P RMSE (kbar)', '{:.2f}'),
    ]:
        vals = df[col].values.astype(float)
        lo = df[col_lo].values.astype(float)
        hi = df[col_hi].values.astype(float)
        xerr_lo = np.where(np.isfinite(vals) & np.isfinite(lo), vals - lo, 0.0)
        xerr_hi = np.where(np.isfinite(vals) & np.isfinite(hi), hi - vals, 0.0)
        xerr = np.vstack([xerr_lo, xerr_hi])
        colors = [COLORS.get(n, '#777') for n in df['Method']]
        bars = ax.barh(y_pos, np.nan_to_num(vals, nan=0.0), xerr=xerr,
                       color=colors, edgecolor='black',
                       error_kw={'elinewidth': 1.1, 'capsize': 2.5})
        for b, n in zip(bars, df['Method']):
            if _is_putirka(n):
                b.set_hatch('///')
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v * 1.01 + 0.001, i, fmt.format(v),
                        va='center', fontsize=8)
            else:
                ax.text(0.01, i, 'n/a', va='center', fontsize=8,
                        color='gray', style='italic')
        if np.isfinite(vals).any():
            best = float(np.nanmin(vals))
            ax.axvline(best, color='gray', linestyle=':', lw=1, alpha=0.6)
        ax.set_xlabel(xlab)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_yticks(y_pos)
        _sup_labels = []
        for _name in df['Method']:
            _sup = footnotes.get(_name)
            _sup_labels.append(f'{_name}$^{{{_sup}}}$' if _sup else _name)
        ax.set_yticklabels(_sup_labels, fontsize=9)
        ax.invert_yaxis()"""
    if old_loop not in src:
        print('cell 23 sort: loop anchor not found')
        return src
    src = src.replace(old_loop, new_loop, 1)
    # 4. Now the ax_T.set_yticks + invert_yaxis + set_yticklabels block
    # that came AFTER the old loop is redundant (moved into loop). Remove
    # it and the setp visible=False (since P panel has its own labels now).
    old_post = """    ax_T.set_yticks(y_pos)
    labels = []
    for name in shown['Method']:
        sup = footnotes.get(name)
        labels.append(f'{name}$^{{{sup}}}$' if sup else name)
    ax_T.set_yticklabels(labels, fontsize=9)
    ax_T.invert_yaxis()
    plt.setp(ax_P.get_yticklabels(), visible=False)
    plt.setp(ax_C.get_yticklabels(), visible=False)"""
    new_post = """    # v8-fix: y-ticks now set inside the per-panel loop; C panel still hides labels
    plt.setp(ax_C.get_yticklabels(), visible=False)"""
    if old_post in src:
        src = src.replace(old_post, new_post, 1)
    # 5. Update the caption.
    old_cap = ('Hatched bars = '
               "classical (Putirka 2008); solid bars = ML methods. Methods "
               "sorted by T RMSE ascending.")
    new_cap = ('Hatched bars = '
               "classical (Putirka 2008); solid bars = ML methods. Panel A "
               "sorted by T RMSE; panel B sorted independently by P RMSE; "
               "panel C uses T order as a reference.")
    if old_cap in src:
        src = src.replace(old_cap, new_cap, 1)
    print('cell 23 sort: applied independent T/P sort + per-panel yticks')
    return src


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # F1
    nb.cells[21].source = fix_cell_21(nb.cells[21].source)
    # F1 applied to the three-way figure helper
    nb.cells[23].source = fix_cell_23(nb.cells[23].source)
    # F4 sort fix on the same cell
    nb.cells[23].source = fix_cell_23_sort(nb.cells[23].source)
    # F2: Part 3 figure
    nb.cells[26].source = fix_cell_26(nb.cells[26].source)
    # F3: Part 5 same fix
    nb.cells[30].source = fix_cell_30(nb.cells[30].source)

    # Clear outputs on the touched cells so papermill re-executes fresh
    for i in [21, 23, 26, 30]:
        nb.cells[i].outputs = []
        nb.cells[i].execution_count = None

    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'nb04 written: {len(nb.cells)} cells.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
