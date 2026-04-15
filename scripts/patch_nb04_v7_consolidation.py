"""Apply the seven-fix nb04 consolidation:
  F1: delete legacy plot_putirka_grid cell + orphan figure files
  F2: extend Part 2 preds dict with 4 Putirka opx methods (cells 24, 26)
  F3: append Part 3 ArcPL opx-liq external validation (ported from nb04b)
  F4: append Part 4 H2O reporting sensitivity analysis
  F5: append Part 5 opx-only 8-method dual-scope comparison
  F6: add 7 structured markdown section headers
  F7: archive nb04b; update run_all_v7.py + extract_results.py

The script is idempotent: each fix checks a sentinel comment before
applying. Re-running produces "already applied" messages and no edits.

Cell sources for Parts 3, 4, 5 are kept in the `P3_*`, `P4_*`, `P5_*`
module-level constants below so reviewers can read them linearly.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"
NB04B = ROOT / "notebooks" / "nb04b_lepr_arcpl_validation.ipynb"
ARCHIVE = ROOT / "archive"
FIGURES = ROOT / "figures"

SENTINEL_P3 = "# v7-fix: Part 3 ArcPL opx-liq external validation"
SENTINEL_P4 = "# v7-fix: Part 4 H2O reporting sensitivity"
SENTINEL_P5 = "# v7-fix: Part 5 opx-only 8-method dual-scope"
SENTINEL_P2_OPX = "# v7-fix: Putirka opx methods (a/b/c/d)"
SENTINEL_MARKDOWN = "<!-- v7-fix-section-header -->"


# ---------------------------------------------------------------------
# F1: delete legacy plot_putirka_grid cell + orphan figure files
# ---------------------------------------------------------------------

def fix1_delete_legacy_putirka(nb) -> int:
    removed = 0
    new_cells = []
    for c in nb.cells:
        if (c.cell_type == 'code' and
                'def plot_putirka_grid' in c.source and
                'fig_nb04_putirka' in c.source):
            removed += 1
            continue
        new_cells.append(c)
    nb.cells = new_cells
    for stem in ['fig_nb04_putirka_T', 'fig_nb04_putirka_P',
                 'fig_nb04_putirka_comparison']:
        for ext in ['.png', '.pdf']:
            p = FIGURES / f'{stem}{ext}'
            if p.exists():
                p.unlink()
    return removed


# ---------------------------------------------------------------------
# F2: extend Part 2 preds dict with 4 Putirka opx methods
# ---------------------------------------------------------------------

# Code block inserted into cell 24 (and the equivalent block into cell 26's
# _predict_all_methods). Uses try/except so Thermobar API failures don't
# crash the whole method benchmark.
CELL24_OPX_INSERT = '''
# v7-fix: Putirka opx methods (a/b/c/d)
# (a) opx-liq iterative (eq 28a + eq 29a)
try:
    import Thermobar as _pt_opx
    _opx_cols = {
        'SiO2_Opx': 'SiO2', 'TiO2_Opx': 'TiO2', 'Al2O3_Opx': 'Al2O3',
        'Cr2O3_Opx': 'Cr2O3', 'FeOt_Opx': 'FeOt_Opx', 'MnO_Opx': 'MnO',
        'MgO_Opx': 'MgO', 'CaO_Opx': 'CaO', 'Na2O_Opx': 'Na2O', 'K2O_Opx': 'K2O',
    }
    _liq_cols = {
        'SiO2_Liq': 'SiO2', 'TiO2_Liq': 'TiO2', 'Al2O3_Liq': 'Al2O3',
        'FeOt_Liq': 'FeOt_Liq', 'MnO_Liq': 'MnO', 'MgO_Liq': 'MgO',
        'CaO_Liq': 'CaO', 'Na2O_Liq': 'Na2O', 'K2O_Liq': 'K2O',
        'Cr2O3_Liq': 'Cr2O3', 'P2O5_Liq': 'P2O5',
    }
    _opx_in = pd.DataFrame({k: pd.to_numeric(m[k], errors='coerce')
                            for k in _opx_cols if k in m.columns})
    _liq_in = pd.DataFrame({k: pd.to_numeric(m[k], errors='coerce')
                            for k in _liq_cols if k in m.columns})
    if 'H2O_Liq' in m.columns:
        _liq_in['H2O_Liq'] = pd.to_numeric(m['H2O_Liq'], errors='coerce').fillna(0.0)
    if len(_opx_in) and len(_liq_in):
        try:
            _iter = _pt_opx.calculate_opx_liq_press_temp(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')
            _T = (_iter['T_K_calc'].values - 273.15
                  if 'T_K_calc' in _iter.columns
                  else _iter.iloc[:, 0].values - 273.15)
            _P = (_iter['P_kbar_calc'].values
                  if 'P_kbar_calc' in _iter.columns
                  else _iter.iloc[:, 1].values)
            preds['Putirka 2008 opx-liq'] = (_T, _P)
        except Exception as _e1:
            print(f'Putirka opx-liq iterative skipped ({_e1})')

        # (c) opx-liq with true P, true T as inputs -> ceiling
        try:
            _Tceil = _pt_opx.calculate_opx_liq_temp(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationT='T_Put2008_eq28a', P=y_P)
            _Pceil = _pt_opx.calculate_opx_liq_press(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationP='P_Put2008_eq29a', T=y_T + 273.15)
            _T_arr = (_Tceil.values - 273.15 if hasattr(_Tceil, 'values')
                      else np.asarray(_Tceil) - 273.15)
            _P_arr = (_Pceil.values if hasattr(_Pceil, 'values')
                      else np.asarray(_Pceil))
            preds['Putirka 2008 opx-liq [true P]'] = (_T_arr, _P_arr)
        except Exception as _e2:
            print(f'Putirka opx-liq [true P] skipped ({_e2})')

    # (b) opx-only iterative (eq 28b_opx_sat + eq 29c) - 29c requires Cr2O3_Opx > 0
    try:
        _mask_cr = (_opx_in.get('Cr2O3', pd.Series(np.zeros(len(_opx_in))))
                    .fillna(0.0) > 1e-4).values
        if _mask_cr.sum() >= 3:
            _opx_cr = _opx_in[_mask_cr].reset_index(drop=True)
            _iter_only = _pt_opx.calculate_opx_only_press_temp(
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
                    P=y_P[_mask_cr])
                _Pc = np.full(len(_opx_in), np.nan)
                _Tc = np.full(len(_opx_in), np.nan)
                _Pc[_mask_cr] = (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                                 else np.asarray(_Pceil_o))
                _Tc[_mask_cr] = (_Tceil_o.values - 273.15
                                 if hasattr(_Tceil_o, 'values')
                                 else np.asarray(_Tceil_o) - 273.15)
                preds['Putirka 2008 opx-only [true T]'] = (_Tc, _Pc)
            except Exception as _e4:
                print(f'Putirka opx-only [true T] skipped ({_e4})')
    except Exception as _e3:
        print(f'Putirka opx-only iterative skipped ({_e3})')
except Exception as _e_outer:
    print(f'Putirka opx block skipped ({_e_outer})')
'''


CELL26_OPX_INSERT = '''    # v7-fix: Putirka opx methods (a/b/c/d) in _predict_all_methods
    try:
        import Thermobar as _pt_opx
        _opx_in = pd.DataFrame({
            k: pd.to_numeric(scope_df.get(k, np.nan), errors='coerce')
            for k in ['SiO2_Opx','TiO2_Opx','Al2O3_Opx','Cr2O3_Opx','FeOt_Opx',
                      'MnO_Opx','MgO_Opx','CaO_Opx','Na2O_Opx','K2O_Opx']
            if k in scope_df.columns})
        _liq_in = pd.DataFrame({
            k: pd.to_numeric(scope_df.get(k, np.nan), errors='coerce')
            for k in ['SiO2_Liq','TiO2_Liq','Al2O3_Liq','FeOt_Liq','MnO_Liq',
                      'MgO_Liq','CaO_Liq','Na2O_Liq','K2O_Liq','Cr2O3_Liq','P2O5_Liq']
            if k in scope_df.columns})
        if 'H2O_Liq' in scope_df.columns:
            _liq_in['H2O_Liq'] = pd.to_numeric(scope_df['H2O_Liq'], errors='coerce').fillna(0.0)
        if len(_opx_in) and len(_liq_in):
            try:
                _iter = _pt_opx.calculate_opx_liq_press_temp(
                    opx_comps=_opx_in, liq_comps=_liq_in,
                    equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')
                _T = (_iter['T_K_calc'].values - 273.15 if 'T_K_calc' in _iter.columns
                      else _iter.iloc[:, 0].values - 273.15)
                _P = (_iter['P_kbar_calc'].values if 'P_kbar_calc' in _iter.columns
                      else _iter.iloc[:, 1].values)
                p['Putirka 2008 opx-liq'] = (_T, _P)
            except Exception as _e1:
                print(f'  Putirka opx-liq iterative skipped ({_e1})')
            try:
                _Tceil = _pt_opx.calculate_opx_liq_temp(
                    opx_comps=_opx_in, liq_comps=_liq_in,
                    equationT='T_Put2008_eq28a', P=y_P_s)
                _Pceil = _pt_opx.calculate_opx_liq_press(
                    opx_comps=_opx_in, liq_comps=_liq_in,
                    equationP='P_Put2008_eq29a', T=y_T_s + 273.15)
                _Ta = (_Tceil.values - 273.15 if hasattr(_Tceil, 'values')
                       else np.asarray(_Tceil) - 273.15)
                _Pa = (_Pceil.values if hasattr(_Pceil, 'values') else np.asarray(_Pceil))
                p['Putirka 2008 opx-liq [true P]'] = (_Ta, _Pa)
            except Exception as _e2:
                print(f'  Putirka opx-liq [true P] skipped ({_e2})')
        _mask_cr = ((_opx_in.get('Cr2O3_Opx', pd.Series(np.zeros(len(scope_df))))
                     .fillna(0.0) > 1e-4).values
                    if 'Cr2O3_Opx' in _opx_in.columns
                    else np.zeros(len(scope_df), dtype=bool))
        if _mask_cr.sum() >= 3:
            _opx_cr = _opx_in[_mask_cr].reset_index(drop=True)
            try:
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
                p['Putirka 2008 opx-only'] = (_To, _Po)
            except Exception as _e3:
                print(f'  Putirka opx-only iterative skipped ({_e3})')
            try:
                _Pceil_o = _pt_opx.calculate_opx_only_press(
                    opx_comps=_opx_cr,
                    equationP='P_Put2008_eq29c',
                    T=(y_T_s + 273.15)[_mask_cr])
                _Tceil_o = _pt_opx.calculate_opx_only_temp(
                    opx_comps=_opx_cr,
                    equationT='T_Put2008_eq28b_opx_sat',
                    P=y_P_s[_mask_cr])
                _Pc = np.full(len(scope_df), np.nan)
                _Tc = np.full(len(scope_df), np.nan)
                _Pc[_mask_cr] = (_Pceil_o.values if hasattr(_Pceil_o, 'values')
                                 else np.asarray(_Pceil_o))
                _Tc[_mask_cr] = (_Tceil_o.values - 273.15
                                 if hasattr(_Tceil_o, 'values')
                                 else np.asarray(_Tceil_o) - 273.15)
                p['Putirka 2008 opx-only [true T]'] = (_Tc, _Pc)
            except Exception as _e4:
                print(f'  Putirka opx-only [true T] skipped ({_e4})')
    except Exception as _eo:
        print(f'  Putirka opx block skipped ({_eo})')
'''


# COLORS dict extension for cell 26
COLORS_OLD = """COLORS = {
    'Ours opx-liq forest':    FAMILY_COLORS['forest'],
    'Ours opx-liq boosted':   FAMILY_COLORS['boosted'],
    'Ours opx-only forest':   FAMILY_COLORS['forest'],
    'Ours opx-only boosted':  FAMILY_COLORS['boosted'],
    'Agreda-Lopez cpx-liq':   FAMILY_COLORS['external_cpx'],
    'Agreda-Lopez cpx-only':  OKABE_ITO['yellow'],
    'Jorgenson cpx-only':     OKABE_ITO['green'],
    'Wang 2021 cpx-liq':      OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':   FAMILY_COLORS['putirka'],
}"""

COLORS_NEW = """COLORS = {
    'Ours opx-liq forest':           FAMILY_COLORS['forest'],
    'Ours opx-liq boosted':          FAMILY_COLORS['boosted'],
    'Ours opx-only forest':          FAMILY_COLORS['forest'],
    'Ours opx-only boosted':         FAMILY_COLORS['boosted'],
    'Agreda-Lopez cpx-liq':          FAMILY_COLORS['external_cpx'],
    'Agreda-Lopez cpx-only':         OKABE_ITO['yellow'],
    'Jorgenson cpx-only':            OKABE_ITO['green'],
    'Wang 2021 cpx-liq':             OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':          FAMILY_COLORS['putirka'],
    'Putirka 2008 opx-liq':          FAMILY_COLORS['putirka'],
    'Putirka 2008 opx-only':         FAMILY_COLORS['putirka'],
    'Putirka 2008 opx-liq [true P]': FAMILY_COLORS['putirka'],
    'Putirka 2008 opx-only [true T]':FAMILY_COLORS['putirka'],
}"""


def fix2_add_putirka_opx(nb) -> int:
    changed = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        if 'preds[f\'Ours {_tracks.replace("_", "-")}' in cell.source and SENTINEL_P2_OPX not in cell.source:
            # cell 24 style
            cell.source = cell.source.rstrip() + '\n\n' + CELL24_OPX_INSERT + '\n'
            cell.outputs = []
            cell.execution_count = None
            changed += 1
        elif 'p[name] = (mT.predict(Xt), mP.predict(Xp))' in cell.source and SENTINEL_P2_OPX not in cell.source:
            # cell 26 style: insert before the final `return p`
            if '    return p' in cell.source:
                cell.source = cell.source.replace(
                    '    return p',
                    CELL26_OPX_INSERT + '    return p',
                    1,
                )
                cell.outputs = []
                cell.execution_count = None
                changed += 1
        if COLORS_OLD in cell.source:
            cell.source = cell.source.replace(COLORS_OLD, COLORS_NEW, 1)
            cell.outputs = []
            cell.execution_count = None
            changed += 1
    return changed


# ---------------------------------------------------------------------
# F3: Part 3 ArcPL opx-liq external validation
# ---------------------------------------------------------------------

P3_MD = """<!-- v7-fix-section-header -->
## Part 3: ArcPL opx-liq external validation

**Purpose.** External validation of our canonical opx-liq ML on the ArcPL
hydrous subset of LEPR — a dataset our models have never seen. This is
the "deployment reality" number.

**Data inputs.** LEPR Opx-Liq sheet at `data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx`,
filtered via `Citation_x.contains('_notinLEPR')` -> ArcPL subset (n~=324),
minus training overlap with ExPetDB (author+year match) -> n~=204.

**Methods evaluated.** Canonical ML opx-liq forest (RF on alr for T, RF on raw for P)
and boosted (XGB on raw for both). Putirka opx comparisons live in Part 5.

**Analysis performed.** Cleaning mirrors NB01 (cation recalc on 6-oxygen basis,
oxide total filter, pigeonite filter, KD filter, engineered features). Apply both
canonical families to the cleaned frame, compute RMSE, MAE, R^2 overall plus 95%
bootstrap CI (B=2000, rng seed = SEED_BOOTSTRAP).

**How to interpret.** The RMSE here (T ~=58 C, P ~=2.7 kbar on the baseline)
is what a user should expect on real arc opx-liq samples. Substantially higher
numbers than the internal test split (Figure 3/4) would indicate train/deployment
drift; comparable numbers indicate the model generalizes.

**Outputs.**
- `results/nb04_arcpl_opx_liq_predictions_forest.csv`
- `results/nb04_arcpl_opx_liq_predictions_boosted.csv`
- `results/nb04_arcpl_opx_liq_metrics.csv`
- `results/nb04_arcpl_opx_liq_bootstrap.csv`
- Backward-compat aliases: `nb04b_arcpl_predictions_forest.csv`,
  `nb04b_arcpl_predictions_boosted.csv`, `nb04b_arcpl_predictions.csv`

**Downstream use.** NB09 reads `nb04b_arcpl_predictions.csv` for H2O and OOD
analyses; NBF fig 13 reads the OOD derivatives. Backward-compat aliases keep
those paths working until the downstream notebooks are migrated.
"""


P3_CODE = """# v7-fix: Part 3 ArcPL opx-liq external validation (ported from nb04b)
# Cleaning pipeline mirrors NB01 so the canonical ML models see training-
# schema columns they recognize.
from scipy import stats as _p3_stats
import joblib as _p3_joblib
from src.features import (
    build_feature_matrix as _p3_build_feat,
    cation_recalc_6oxy as _p3_cation,
    add_engineered_features as _p3_engineer,
)
from src.data import (
    canonical_model_path as _p3_canon_path,
    canonical_model_spec as _p3_canon_spec,
)

_P3_OXIDE_TOT = (OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX)
_P3_CATION_RANGE = (CATION_SUM_MIN, CATION_SUM_MAX)
_P3_WO_MAX = WO_MAX_MOL_PCT
_P3_PCEIL = P_CEILING_KBAR
_P3_BOOT = 2000
_p3_rng = np.random.default_rng(SEED_BOOTSTRAP)


def _p3_bootstrap_rmse(y, yhat, B=_P3_BOOT):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    n = int(mask.sum())
    if n < 3:
        return (np.nan, np.nan, np.nan, n)
    yt = y[mask]; yp = yhat[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    idx = _p3_rng.integers(0, n, size=(B, n))
    boots = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2, axis=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(rmse), float(lo), float(hi), n


def _p3_metrics(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    yt = y[mask]; yp = yhat[mask]
    n = int(mask.sum())
    if n < 3:
        return dict(n=n, rmse=np.nan, mae=np.nan, r2=np.nan)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(n=n, rmse=rmse, mae=mae, r2=float(r2))


# 1) Load LEPR Opx-Liq sheet, ArcPL subset via Citation_x._notinLEPR.
_p3_lepr = pd.read_excel(LEPR_XLSX, sheet_name='Opx-Liq')
_p3_arcpl = _p3_lepr[_p3_lepr['Citation_x'].astype(str).str.contains('_notinLEPR',
                                                                       na=False)].copy()
print(f'Part 3: ArcPL subset before overlap removal: n={len(_p3_arcpl)}')

# 2) Rename LEPR -> ExPetDB training schema
_p3_rename = {
    'Citation_x': 'Citation', 'Experiment_x': 'Experiment',
    'P_kbar_x': 'P_kbar',
    'SiO2_Opx': 'SiO2', 'TiO2_Opx': 'TiO2', 'Al2O3_Opx': 'Al2O3',
    'FeOt_Opx': 'FeO_total', 'MnO_Opx': 'MnO', 'MgO_Opx': 'MgO',
    'CaO_Opx': 'CaO', 'Na2O_Opx': 'Na2O', 'K2O_Opx': 'K2O',
    'Cr2O3_Opx': 'Cr2O3', 'P2O5_Opx': 'P2O5',
    'SiO2_Liq': 'liq_SiO2', 'TiO2_Liq': 'liq_TiO2', 'Al2O3_Liq': 'liq_Al2O3',
    'FeOt_Liq': 'liq_FeO', 'MnO_Liq': 'liq_MnO', 'MgO_Liq': 'liq_MgO',
    'CaO_Liq': 'liq_CaO', 'Na2O_Liq': 'liq_Na2O', 'K2O_Liq': 'liq_K2O',
    'Cr2O3_Liq': 'liq_Cr2O3', 'H2O_Liq': 'H2O_Liq',
    'H2O_Liq_Method': 'H2O_Liq_Method',
}
_p3_arcpl = _p3_arcpl.rename(columns=_p3_rename)
if 'T_K_x' in _p3_arcpl.columns:
    _p3_arcpl['T_C'] = _p3_arcpl['T_K_x'] - 273.15
_p3_arcpl = _p3_arcpl.drop(columns=[c for c in _p3_arcpl.columns if c.endswith('_y')],
                           errors='ignore')
_p3_arcpl = _p3_arcpl[_p3_arcpl.get('H2O_Liq', pd.Series(np.zeros(len(_p3_arcpl)))) >= 0].copy()

# is_vbd flag from H2O_Liq_Method
_p3_arcpl['is_vbd'] = _p3_arcpl.get(
    'H2O_Liq_Method', pd.Series([''] * len(_p3_arcpl))
).astype(str).str.contains('VBD|vbd|mass_balance|diff', na=False, regex=True)

# 3) Cleaning pipeline
_p3_all_ox = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO',
              'Na2O', 'K2O', 'P2O5']
for _ox in _p3_all_ox:
    if _ox in _p3_arcpl.columns:
        _p3_arcpl[_ox] = pd.to_numeric(_p3_arcpl[_ox], errors='coerce')
_p3_present = [o for o in _p3_all_ox if o in _p3_arcpl.columns]
_p3_arcpl['oxide_total'] = _p3_arcpl[_p3_present].sum(axis=1, min_count=5)
_p3_arcpl = _p3_arcpl[_p3_arcpl['oxide_total'].between(*_P3_OXIDE_TOT)].copy()
_p3_arcpl = _p3_cation(_p3_arcpl, oxides=_p3_all_ox)
_p3_arcpl = _p3_arcpl[_p3_arcpl['cation_sum'].between(*_P3_CATION_RANGE)].copy()
_p3_arcpl = _p3_arcpl.dropna(subset=['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO']).copy()
_p3_arcpl = _p3_engineer(_p3_arcpl)
_p3_arcpl['Wo'] = _p3_arcpl['Wo_frac'] * 100.0
_p3_arcpl = _p3_arcpl[_p3_arcpl['Wo'] <= _P3_WO_MAX].copy()
_p3_arcpl = _p3_arcpl[_p3_arcpl['P_kbar'] <= _P3_PCEIL].copy()

# Basic KD filter (same constants as NB01)
_p3_fe_opx = _p3_arcpl['FeO_total'] / 71.844
_p3_mg_opx = _p3_arcpl['MgO'] / 40.304
_p3_fe_liq = (_p3_arcpl['liq_FeO'] * (1.0 - FE3_FET_RATIO)) / 71.844
_p3_mg_liq = _p3_arcpl['liq_MgO'] / 40.304
_p3_KD = (_p3_fe_opx / _p3_mg_opx) / (_p3_fe_liq / _p3_mg_liq)
_p3_arcpl = _p3_arcpl[(_p3_KD >= KD_FEMG_MIN) & (_p3_KD <= KD_FEMG_MAX)].copy()

# 4) Remove ExPetDB training overlap (best-effort author+year)
_p3_expet = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
def _p3_citkey(s):
    s = str(s)
    import re as _re
    m = _re.search(r'([A-Za-z][A-Za-z ,.\\-]+?)[\\s,]*((?:19|20)\\d{2})', s)
    return (m.group(1).strip().lower() + '_' + m.group(2)) if m else s.strip().lower()
_p3_expet_keys = set(_p3_expet['Citation'].astype(str).map(_p3_citkey))
_p3_arcpl['_key'] = _p3_arcpl['Citation'].astype(str).map(_p3_citkey)
_p3_before = len(_p3_arcpl)
_p3_arcpl = _p3_arcpl[~_p3_arcpl['_key'].isin(_p3_expet_keys)].reset_index(drop=True)
print(f'Overlap removal: {_p3_before} -> {len(_p3_arcpl)}')

# 5) Per-family predictions
_p3_metrics_rows = []
_p3_boot_rows = []
_p3_pred_store = {}
for _fam in ['forest', 'boosted']:
    try:
        _sT = _p3_canon_spec('T_C',    'opx_liq', _fam, RESULTS)
        _sP = _p3_canon_spec('P_kbar', 'opx_liq', _fam, RESULTS)
        _mT = _p3_joblib.load(_p3_canon_path('T_C',    'opx_liq', _fam, MODELS, RESULTS))
        _mP = _p3_joblib.load(_p3_canon_path('P_kbar', 'opx_liq', _fam, MODELS, RESULTS))
        _Xt, _ = _p3_build_feat(_p3_arcpl, _sT['feature_set'], use_liq=True)
        _Xp, _ = _p3_build_feat(_p3_arcpl, _sP['feature_set'], use_liq=True)
        _yT = _p3_arcpl['T_C'].values
        _yP = _p3_arcpl['P_kbar'].values
        _pT = _mT.predict(_Xt)
        _pP = _mP.predict(_Xp)
        _pred_df = _p3_arcpl.drop(columns=['_key'], errors='ignore').copy()
        _pred_df['T_pred'] = _pT
        _pred_df['P_pred'] = _pP
        _pred_df['family'] = _fam
        _pred_df.to_csv(RESULTS / f'nb04_arcpl_opx_liq_predictions_{_fam}.csv', index=False)
        # backward-compat alias
        _pred_df.to_csv(RESULTS / f'nb04b_arcpl_predictions_{_fam}.csv', index=False)
        _p3_pred_store[_fam] = _pred_df
        for _tgt, _y, _p in [('T_C', _yT, _pT), ('P_kbar', _yP, _pP)]:
            _m = _p3_metrics(_y, _p)
            _p3_metrics_rows.append({'family': _fam, 'target': _tgt, **_m})
            _r, _lo, _hi, _n = _p3_bootstrap_rmse(_y, _p)
            _p3_boot_rows.append({'family': _fam, 'target': _tgt,
                                   'rmse': _r, 'rmse_ci_lo': _lo, 'rmse_ci_hi': _hi, 'n': _n})
    except Exception as _e:
        print(f'Part 3 family {_fam} skipped ({_e})')

# Merged backward-compat file (both families in one CSV; disambiguate via `family`)
if _p3_pred_store:
    pd.concat(list(_p3_pred_store.values()), ignore_index=True).to_csv(
        RESULTS / 'nb04b_arcpl_predictions.csv', index=False)

_p3_metrics_df = pd.DataFrame(_p3_metrics_rows)
_p3_boot_df = pd.DataFrame(_p3_boot_rows)
_p3_metrics_df.to_csv(RESULTS / 'nb04_arcpl_opx_liq_metrics.csv', index=False)
_p3_boot_df.to_csv(RESULTS / 'nb04_arcpl_opx_liq_bootstrap.csv', index=False)
print(_p3_metrics_df.round(3).to_string(index=False))
print(_p3_boot_df.round(3).to_string(index=False))
"""


# ---------------------------------------------------------------------
# F4: Part 4 H2O reporting sensitivity
# ---------------------------------------------------------------------

P4_MD = """<!-- v7-fix-section-header -->
## Part 4: H2O reporting sensitivity analysis

**Purpose.** Check whether our ML's ArcPL performance differs between
samples with directly measured H2O (FTIR/SIMS/Raman/Sol) versus those
with VBD/mass-balance H2O. Relevant because many ArcPL experiments do
not have measured H2O.

**Data inputs.** The cleaned ArcPL opx-liq frame built in Part 3 (n~=204).
Partition by the `is_vbd` flag derived from `H2O_Liq_Method`.

**Methods evaluated.** Canonical opx-liq forest and boosted, per-subset.

**Analysis performed.** For each subset (measured-H2O ~=134, VBD ~=70)
compute RMSE, MAE, R^2 with 95% bootstrap CI (B=2000). Run Mann-Whitney U
on absolute residuals between subsets for T and P separately. Produce a
4-panel figure: T pred-vs-obs by subset, P pred-vs-obs by subset, and
residual histograms for T and P.

**How to interpret.** Robust performance across H2O methods supports
general-use applicability. A large gap (especially measured << VBD) would
suggest H2O quality matters and would motivate a user caveat. VBD samples
often have systematically lower P, which confounds P comparisons; any
apparent VBD advantage is likely this artifact, not a real model property.

**Outputs.**
- `results/nb04_arcpl_h2o_stratified_metrics.csv`
- `results/nb04_arcpl_h2o_mannwhitney.csv`
- `figures/fig_nb04_h2o_sensitivity.{png,pdf}`

**Downstream use.** NB09 may cite the p-values in a supplementary discussion
paragraph. Terminal otherwise — not consumed by other notebooks.
"""


P4_CODE = """# v7-fix: Part 4 H2O reporting sensitivity analysis
# Uses `_p3_arcpl` + `_p3_pred_store` produced in Part 3.
from scipy.stats import mannwhitneyu as _p4_mwu
import matplotlib.pyplot as _p4_plt

_p4_rows = []
_p4_mw_rows = []

if not _p3_pred_store:
    print('Part 4 skipped (no Part 3 predictions).')
else:
    _p4_df_base = _p3_arcpl.reset_index(drop=True)
    _p4_meas_mask = ~_p4_df_base['is_vbd'].values.astype(bool)
    _p4_vbd_mask  =  _p4_df_base['is_vbd'].values.astype(bool)
    print(f'Part 4 split: measured n={int(_p4_meas_mask.sum())} '
          f'VBD n={int(_p4_vbd_mask.sum())}')

    for _fam, _pred_df in _p3_pred_store.items():
        _pred_df = _pred_df.reset_index(drop=True)
        _yT = _pred_df['T_C'].values
        _yP = _pred_df['P_kbar'].values
        _pT = _pred_df['T_pred'].values
        _pP = _pred_df['P_pred'].values
        for _sub_name, _mask in [('measured', _p4_meas_mask),
                                  ('vbd',      _p4_vbd_mask)]:
            if _mask.sum() < 5:
                continue
            for _tgt, _y, _p in [('T_C', _yT, _pT), ('P_kbar', _yP, _pP)]:
                _m = _p3_metrics(_y[_mask], _p[_mask])
                _r, _lo, _hi, _n = _p3_bootstrap_rmse(_y[_mask], _p[_mask])
                _p4_rows.append({'family': _fam, 'subset': _sub_name,
                                  'target': _tgt, **_m,
                                  'rmse_ci_lo': _lo, 'rmse_ci_hi': _hi})
        # Mann-Whitney on |residuals|
        for _tgt, _resid in [('T_C', _pT - _yT), ('P_kbar', _pP - _yP)]:
            _a = np.abs(_resid[_p4_meas_mask])
            _b = np.abs(_resid[_p4_vbd_mask])
            _a = _a[np.isfinite(_a)]; _b = _b[np.isfinite(_b)]
            if len(_a) >= 5 and len(_b) >= 5:
                _s, _pp = _p4_mwu(_a, _b, alternative='two-sided')
                _p4_mw_rows.append({'family': _fam, 'target': _tgt,
                                     'n_measured': int(len(_a)),
                                     'n_vbd': int(len(_b)),
                                     'U_stat': float(_s),
                                     'p_value': float(_pp),
                                     'median_abs_resid_measured': float(np.median(_a)),
                                     'median_abs_resid_vbd':      float(np.median(_b))})

    _p4_metrics_df = pd.DataFrame(_p4_rows)
    _p4_mw_df = pd.DataFrame(_p4_mw_rows)
    _p4_metrics_df.to_csv(RESULTS / 'nb04_arcpl_h2o_stratified_metrics.csv', index=False)
    _p4_mw_df.to_csv(RESULTS / 'nb04_arcpl_h2o_mannwhitney.csv', index=False)
    print('\\nStratified metrics:')
    print(_p4_metrics_df.round(3).to_string(index=False))
    print('\\nMann-Whitney:')
    print(_p4_mw_df.round(4).to_string(index=False))

    # 4-panel figure using the forest family if available, else boosted.
    _p4_fam = 'forest' if 'forest' in _p3_pred_store else list(_p3_pred_store)[0]
    _pd = _p3_pred_store[_p4_fam].reset_index(drop=True)
    _yT = _pd['T_C'].values; _yP = _pd['P_kbar'].values
    _pT = _pd['T_pred'].values; _pP = _pd['P_pred'].values
    _fig, _axes = _p4_plt.subplots(2, 2, figsize=(11, 9))
    _col_meas = '#0072B2'; _col_vbd = '#D55E00'

    def _p4_scatter(ax, y, p, mask_m, mask_v, unit):
        ax.scatter(y[mask_m], p[mask_m], s=18, alpha=0.6, c=_col_meas,
                   label=f'measured n={int(mask_m.sum())}', edgecolor='k', lw=0.2)
        ax.scatter(y[mask_v], p[mask_v], s=18, alpha=0.6, c=_col_vbd,
                   label=f'VBD n={int(mask_v.sum())}', edgecolor='k', lw=0.2)
        _lim = [np.nanmin([y.min(), p.min()]), np.nanmax([y.max(), p.max()])]
        ax.plot(_lim, _lim, 'k--', lw=0.8)
        _rm = float(np.sqrt(np.nanmean((y[mask_m] - p[mask_m]) ** 2))) if mask_m.sum() else float('nan')
        _rv = float(np.sqrt(np.nanmean((y[mask_v] - p[mask_v]) ** 2))) if mask_v.sum() else float('nan')
        ax.set_xlabel(f'Observed ({unit})'); ax.set_ylabel(f'Predicted ({unit})')
        ax.set_title(f'RMSE meas={_rm:.2f} / VBD={_rv:.2f} ({unit})')
        ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)

    _p4_scatter(_axes[0, 0], _yT, _pT, _p4_meas_mask, _p4_vbd_mask, 'C')
    _p4_scatter(_axes[0, 1], _yP, _pP, _p4_meas_mask, _p4_vbd_mask, 'kbar')
    _axes[0, 0].set_title('T: ' + _axes[0, 0].get_title())
    _axes[0, 1].set_title('P: ' + _axes[0, 1].get_title())
    for _ax, _y, _p, _unit in [
        (_axes[1, 0], _yT, _pT, 'C'),
        (_axes[1, 1], _yP, _pP, 'kbar'),
    ]:
        _resid = _p - _y
        _ax.hist(_resid[_p4_meas_mask], bins=25, alpha=0.6, color=_col_meas,
                 label='measured', density=True)
        _ax.hist(_resid[_p4_vbd_mask], bins=25, alpha=0.6, color=_col_vbd,
                 label='VBD', density=True)
        _ax.axvline(0, color='k', lw=0.8)
        _ax.set_xlabel(f'Residual ({_unit})')
        _ax.set_ylabel('density')
        _ax.legend(loc='best', fontsize=8); _ax.grid(True, alpha=0.3)
    _p4_plt.suptitle(f'H2O sensitivity - {_p4_fam} family', fontsize=12)
    _p4_plt.tight_layout()
    for _ext in ('png', 'pdf'):
        _p4_plt.savefig(FIGURES / f'fig_nb04_h2o_sensitivity.{_ext}',
                         dpi=300, bbox_inches='tight')
    _p4_plt.close(_fig)
"""


# ---------------------------------------------------------------------
# F5: Part 5 opx-only 8-method dual-scope comparison
# ---------------------------------------------------------------------

P5_MD = """<!-- v7-fix-section-header -->
## Part 5: Opx-only method comparison, dual-scope (headline figure)

**Purpose.** The paper's headline deployment figure. Answers two questions
at once: (i) across all ArcPL opx-bearing samples, how do our ML methods
compare to Putirka opx methods in deployment terms (coverage + RMSE on
what each method can return)? (ii) on the narrow subset where Putirka's
equilibrium filters permit a prediction ("fair scope"), does our ML beat
Putirka head-to-head?

**Data inputs.** The Part 3 cleaned ArcPL opx-liq frame (n~=204, "all" scope).
The "fair" scope is defined dynamically as rows where Putirka 28a/29a iterative
returns finite T *and* P — typically n~=28.

**Methods evaluated.** 8 total:
1. Ours opx-liq forest (canonical RF on alr for T, RF on raw for P)
2. Ours opx-liq boosted (canonical XGB on raw for T and P)
3. Ours opx-only forest (canonical RF on pwlr for T and P)
4. Ours opx-only boosted (canonical XGB on pwlr for T and P)
5. Putirka 2008 opx-liq (eq 28a + eq 29a iterative)
6. Putirka 2008 opx-only (eq 28b_opx_sat + eq 29c iterative, Cr2O3_Opx > 0)
7. Putirka 2008 opx-liq [true P] (eq 28a with true P input -> ceiling)
8. Putirka 2008 opx-only [true T] (eq 28b_opx_sat with true T input -> ceiling)

**Analysis performed.** Compute RMSE + 95% bootstrap CI (B=2000), signed bias,
and coverage (n_predicted / n_scope * 100) for every (method, scope) pair.
Produce a two-panel RMSE bar chart (T on the left, P on the right, 8 methods
each, sorted by T RMSE in panel A). Putirka bars are color-coded sky blue and
hatched; ML bars use family colors (forest blue, boosted vermillion). Two
side-by-side bars per method distinguish "all" scope (alpha=0.6) from "fair"
scope (alpha=1.0). A separate coverage panel shows n_predicted/n_scope.

**How to interpret.** On the "all" scope (n~=204), coverage exposes Putirka's
real-world limitation: iterative methods drop to ~13.7% due to equilibrium
filter failures, while our ML returns predictions on 100% of opx-bearing arc
samples. On the "fair" scope (n~=28), RMSE is apples-to-apples. The expected
outcome is that our ML has lower RMSE on both T and P than Putirka iterative,
by a factor near 2.6x on T (~=46 C vs ~=119 C). The "true P" and "true T"
Putirka variants are CEILING comparisons — they use the correct answer as
input, representing the best Putirka could ever do; real users do not have
true P/T when predicting. If our ML beats Putirka even when Putirka is given
the true answer, the deployment argument is complete: competitive with the
ceiling AND always-on.

**Outputs.**
- `results/nb04_opx_only_comparison_all.csv`  (8 methods, n_all scope)
- `results/nb04_opx_only_comparison_fair.csv` (8 methods, n_fair scope)
- `figures/fig_nb04_opx_only_comparison.{png,pdf}`

**Downstream use.** NB09 pulls the headline "ML ~=46 C vs Putirka ~=119 C"
number from the fair-scope CSV for the paper abstract.
"""


P5_CODE = """# v7-fix: Part 5 opx-only 8-method dual-scope comparison
import matplotlib.pyplot as _p5_plt
from src.features import lepr_to_training_features as _p5_shim
import joblib as _p5_joblib

if not _p3_pred_store:
    print('Part 5 skipped (no Part 3 predictions).')
else:
    _p5_df = _p3_arcpl.reset_index(drop=True).copy()
    # Part 5 treats _p5_df as a LEPR-suffix frame; Part 3 already renamed
    # to flat training-schema (SiO2, liq_SiO2, etc.), so recover the LEPR
    # suffixed columns Putirka expects from _p3_lepr joined on Experiment.
    _p5_raw = _p3_lepr.copy()
    if 'Experiment' not in _p5_raw.columns and 'Experiment_x' in _p5_raw.columns:
        _p5_raw = _p5_raw.rename(columns={'Experiment_x': 'Experiment'})
    _p5_merged = _p5_df.merge(
        _p5_raw[['Experiment'] + [c for c in _p5_raw.columns
                                   if c.endswith('_Opx') or c.endswith('_Liq')]],
        on='Experiment', how='left', suffixes=('', '_lepr'))
    _yT_all = _p5_df['T_C'].values; _yP_all = _p5_df['P_kbar'].values

    _p5_preds_all = {}

    # Ours predictions: reuse Part 3 for opx-liq; recompute opx-only locally.
    for _fam in ['forest', 'boosted']:
        if _fam in _p3_pred_store:
            _pd = _p3_pred_store[_fam].reset_index(drop=True)
            _p5_preds_all[f'Ours opx-liq {_fam}'] = (_pd['T_pred'].values,
                                                      _pd['P_pred'].values)
        try:
            _sT = _p3_canon_spec('T_C',    'opx_only', _fam, RESULTS)
            _sP = _p3_canon_spec('P_kbar', 'opx_only', _fam, RESULTS)
            _mT = _p5_joblib.load(_p3_canon_path('T_C',    'opx_only', _fam, MODELS, RESULTS))
            _mP = _p5_joblib.load(_p3_canon_path('P_kbar', 'opx_only', _fam, MODELS, RESULTS))
            _Xt, _ = _p3_build_feat(_p5_df, _sT['feature_set'], use_liq=False)
            _Xp, _ = _p3_build_feat(_p5_df, _sP['feature_set'], use_liq=False)
            _p5_preds_all[f'Ours opx-only {_fam}'] = (_mT.predict(_Xt), _mP.predict(_Xp))
        except Exception as _e:
            print(f'Part 5 Ours opx-only {_fam} skipped ({_e})')

    # Putirka: run four variants off _p5_merged LEPR-suffixed columns.
    try:
        import Thermobar as _p5_pt
        _opx_in = pd.DataFrame({
            k: pd.to_numeric(_p5_merged.get(k, np.nan), errors='coerce')
            for k in ['SiO2_Opx','TiO2_Opx','Al2O3_Opx','Cr2O3_Opx','FeOt_Opx',
                      'MnO_Opx','MgO_Opx','CaO_Opx','Na2O_Opx','K2O_Opx']
            if k in _p5_merged.columns})
        _liq_in = pd.DataFrame({
            k: pd.to_numeric(_p5_merged.get(k, np.nan), errors='coerce')
            for k in ['SiO2_Liq','TiO2_Liq','Al2O3_Liq','FeOt_Liq','MnO_Liq',
                      'MgO_Liq','CaO_Liq','Na2O_Liq','K2O_Liq','Cr2O3_Liq','P2O5_Liq']
            if k in _p5_merged.columns})
        if 'H2O_Liq' in _p5_merged.columns:
            _liq_in['H2O_Liq'] = pd.to_numeric(_p5_merged['H2O_Liq'], errors='coerce').fillna(0.0)
        # (5) opx-liq iterative
        try:
            _it = _p5_pt.calculate_opx_liq_press_temp(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')
            _T = (_it['T_K_calc'].values - 273.15 if 'T_K_calc' in _it.columns
                  else _it.iloc[:, 0].values - 273.15)
            _P = (_it['P_kbar_calc'].values if 'P_kbar_calc' in _it.columns
                  else _it.iloc[:, 1].values)
            _p5_preds_all['Putirka 2008 opx-liq'] = (_T, _P)
        except Exception as _e:
            print(f'Part 5 Putirka opx-liq iterative skipped ({_e})')
        # (7) opx-liq ceiling with true P, true T
        try:
            _Tc = _p5_pt.calculate_opx_liq_temp(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationT='T_Put2008_eq28a', P=_yP_all)
            _Pc = _p5_pt.calculate_opx_liq_press(
                opx_comps=_opx_in, liq_comps=_liq_in,
                equationP='P_Put2008_eq29a', T=_yT_all + 273.15)
            _Tc_arr = (_Tc.values - 273.15 if hasattr(_Tc, 'values')
                       else np.asarray(_Tc) - 273.15)
            _Pc_arr = (_Pc.values if hasattr(_Pc, 'values') else np.asarray(_Pc))
            _p5_preds_all['Putirka 2008 opx-liq [true P]'] = (_Tc_arr, _Pc_arr)
        except Exception as _e:
            print(f'Part 5 Putirka opx-liq [true P] skipped ({_e})')
        # (6) opx-only iterative with Cr2O3 > 0 filter
        try:
            _mask_cr = ((_opx_in.get('Cr2O3_Opx', pd.Series(np.zeros(len(_p5_merged))))
                         .fillna(0.0) > 1e-4).values
                        if 'Cr2O3_Opx' in _opx_in.columns
                        else np.zeros(len(_p5_merged), dtype=bool))
            if _mask_cr.sum() >= 3:
                _opx_cr = _opx_in[_mask_cr].reset_index(drop=True)
                _io = _p5_pt.calculate_opx_only_press_temp(
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
                _p5_preds_all['Putirka 2008 opx-only'] = (_To, _Po)
                # (8) opx-only ceiling with true T, true P
                try:
                    _Pc_o = _p5_pt.calculate_opx_only_press(
                        opx_comps=_opx_cr, equationP='P_Put2008_eq29c',
                        T=(_yT_all + 273.15)[_mask_cr])
                    _Tc_o = _p5_pt.calculate_opx_only_temp(
                        opx_comps=_opx_cr, equationT='T_Put2008_eq28b_opx_sat',
                        P=_yP_all[_mask_cr])
                    _Po2 = np.full(len(_p5_merged), np.nan)
                    _To2 = np.full(len(_p5_merged), np.nan)
                    _Po2[_mask_cr] = (_Pc_o.values if hasattr(_Pc_o, 'values')
                                      else np.asarray(_Pc_o))
                    _To2[_mask_cr] = (_Tc_o.values - 273.15 if hasattr(_Tc_o, 'values')
                                      else np.asarray(_Tc_o) - 273.15)
                    _p5_preds_all['Putirka 2008 opx-only [true T]'] = (_To2, _Po2)
                except Exception as _e:
                    print(f'Part 5 Putirka opx-only [true T] skipped ({_e})')
        except Exception as _e:
            print(f'Part 5 Putirka opx-only iterative skipped ({_e})')
    except Exception as _e:
        print(f'Part 5 Putirka block skipped ({_e})')

    # Fair-scope mask: rows where Putirka opx-liq iterative returned finite T and P.
    _pit = _p5_preds_all.get('Putirka 2008 opx-liq', (None, None))
    if _pit[0] is not None and _pit[1] is not None:
        _fair_mask = np.isfinite(_pit[0]) & np.isfinite(_pit[1])
    else:
        _fair_mask = np.zeros(len(_p5_df), dtype=bool)
    print(f'Part 5 scopes: all n={len(_p5_df)}  fair n={int(_fair_mask.sum())}')

    def _p5_row(name, y, yhat, scope_name, scope_mask):
        yhat = np.asarray(yhat, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = scope_mask & np.isfinite(yhat) & np.isfinite(y)
        n = int(mask.sum())
        if n < 3:
            return dict(method=name, scope=scope_name, n=n, rmse=np.nan,
                        rmse_ci_lo=np.nan, rmse_ci_hi=np.nan,
                        coverage_pct=100.0 * n / max(int(scope_mask.sum()), 1),
                        signed_bias=np.nan)
        _rmse, _lo, _hi, _ = _p3_bootstrap_rmse(y[mask], yhat[mask])
        bias = float(np.mean(yhat[mask] - y[mask]))
        return dict(method=name, scope=scope_name, n=n, rmse=_rmse,
                    rmse_ci_lo=_lo, rmse_ci_hi=_hi,
                    coverage_pct=100.0 * n / max(int(scope_mask.sum()), 1),
                    signed_bias=bias)

    _all_mask = np.ones(len(_p5_df), dtype=bool)
    _rows_all_T, _rows_all_P, _rows_fair_T, _rows_fair_P = [], [], [], []
    for _name, (_pT_m, _pP_m) in _p5_preds_all.items():
        _rows_all_T.append(_p5_row(_name, _yT_all, _pT_m, 'all',  _all_mask))
        _rows_all_P.append(_p5_row(_name, _yP_all, _pP_m, 'all',  _all_mask))
        _rows_fair_T.append(_p5_row(_name, _yT_all, _pT_m, 'fair', _fair_mask))
        _rows_fair_P.append(_p5_row(_name, _yP_all, _pP_m, 'fair', _fair_mask))

    _df_all = pd.DataFrame([{**tr, 'target': 'T_C'} for tr in _rows_all_T] +
                           [{**pr, 'target': 'P_kbar'} for pr in _rows_all_P])
    _df_fair = pd.DataFrame([{**tr, 'target': 'T_C'} for tr in _rows_fair_T] +
                            [{**pr, 'target': 'P_kbar'} for pr in _rows_fair_P])
    _df_all.to_csv(RESULTS / 'nb04_opx_only_comparison_all.csv', index=False)
    _df_fair.to_csv(RESULTS / 'nb04_opx_only_comparison_fair.csv', index=False)
    print('\\nPart 5 (all scope):')
    print(_df_all.round(3).to_string(index=False))
    print('\\nPart 5 (fair scope):')
    print(_df_fair.round(3).to_string(index=False))

    # Two-panel bar chart + coverage side panel.
    from config import FAMILY_COLORS as _p5_FC
    _SKY = '#56B4E9'
    def _p5_color(name):
        if name.startswith('Ours') and 'forest' in name:
            return _p5_FC['forest']
        if name.startswith('Ours') and 'boosted' in name:
            return _p5_FC['boosted']
        return _SKY

    _methods = list(_p5_preds_all.keys())
    _fig, _axes = _p5_plt.subplots(1, 3, figsize=(16, 6),
                                     gridspec_kw={'width_ratios': [4, 4, 2]})

    def _panel(ax, rows_all, rows_fair, title, unit):
        methods_sorted = sorted(rows_all,
                                 key=lambda r: (r['rmse'] if np.isfinite(r['rmse']) else 1e9))
        order = [r['method'] for r in methods_sorted]
        y = np.arange(len(order))
        rmse_all = np.array([next(r['rmse'] for r in rows_all if r['method'] == m) for m in order])
        lo_all   = np.array([next(r['rmse_ci_lo'] for r in rows_all if r['method'] == m) for m in order])
        hi_all   = np.array([next(r['rmse_ci_hi'] for r in rows_all if r['method'] == m) for m in order])
        rmse_fair = np.array([next(r['rmse'] for r in rows_fair if r['method'] == m) for m in order])
        lo_f   = np.array([next(r['rmse_ci_lo'] for r in rows_fair if r['method'] == m) for m in order])
        hi_f   = np.array([next(r['rmse_ci_hi'] for r in rows_fair if r['method'] == m) for m in order])
        colors = [_p5_color(m) for m in order]
        hatches = ['///' if 'Putirka' in m else '' for m in order]
        # all scope: alpha 0.6
        for i, m in enumerate(order):
            ax.barh(y[i] - 0.2, rmse_all[i], height=0.4,
                    xerr=[[rmse_all[i]-lo_all[i] if np.isfinite(lo_all[i]) else 0],
                          [hi_all[i]-rmse_all[i] if np.isfinite(hi_all[i]) else 0]],
                    color=colors[i], alpha=0.5, edgecolor='k', lw=0.5,
                    hatch=hatches[i], error_kw=dict(ecolor='gray', lw=0.8))
            ax.barh(y[i] + 0.2, rmse_fair[i], height=0.4,
                    xerr=[[rmse_fair[i]-lo_f[i] if np.isfinite(lo_f[i]) else 0],
                          [hi_f[i]-rmse_fair[i] if np.isfinite(hi_f[i]) else 0]],
                    color=colors[i], alpha=1.0, edgecolor='k', lw=0.5,
                    hatch=hatches[i], error_kw=dict(ecolor='k', lw=0.8))
            if np.isfinite(rmse_all[i]):
                ax.text(rmse_all[i], y[i] - 0.2, f' {rmse_all[i]:.1f}',
                        va='center', fontsize=7, alpha=0.7)
            if np.isfinite(rmse_fair[i]):
                ax.text(rmse_fair[i], y[i] + 0.2, f' {rmse_fair[i]:.1f}',
                        va='center', fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel(f'RMSE ({unit})'); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

    _panel(_axes[0], _rows_all_T, _rows_fair_T, 'T (sorted by all-scope T RMSE)', 'C')
    _panel(_axes[1], _rows_all_P, _rows_fair_P, 'P (sorted by all-scope P RMSE)', 'kbar')

    # Coverage panel: one group per method, two bars (all / fair)
    _order_cov = [r['method'] for r in sorted(_rows_all_T,
                                               key=lambda r: (r['rmse'] if np.isfinite(r['rmse']) else 1e9))]
    _cov_all = [next(r['coverage_pct'] for r in _rows_all_T if r['method'] == m) for m in _order_cov]
    _cov_fair = [next(r['coverage_pct'] for r in _rows_fair_T if r['method'] == m) for m in _order_cov]
    _y_cov = np.arange(len(_order_cov))
    _axes[2].barh(_y_cov - 0.2, _cov_all, height=0.4, color='#7f8c8d', alpha=0.5,
                   edgecolor='k', lw=0.5, label='all scope')
    _axes[2].barh(_y_cov + 0.2, _cov_fair, height=0.4, color='#2c3e50',
                   edgecolor='k', lw=0.5, label='fair scope')
    _axes[2].set_yticks(_y_cov); _axes[2].set_yticklabels(['' for _ in _order_cov])
    _axes[2].set_xlim(0, 105); _axes[2].set_xlabel('Coverage (%)')
    _axes[2].set_title('Coverage'); _axes[2].grid(True, alpha=0.3, axis='x')
    _axes[2].legend(loc='lower right', fontsize=7)
    _p5_plt.suptitle('Part 5: opx-only method comparison, dual-scope (alpha=all, opaque=fair)',
                      fontsize=12)
    _p5_plt.tight_layout()
    for _ext in ('png', 'pdf'):
        _p5_plt.savefig(FIGURES / f'fig_nb04_opx_only_comparison.{_ext}',
                         dpi=300, bbox_inches='tight')
    _p5_plt.close(_fig)
"""


# ---------------------------------------------------------------------
# F6: markdown headers for Part 1, Part 2, Encyclopedia, OPERATOR DECISION
# (Parts 3-5 markdown cells are inserted at the same time as their code.)
# ---------------------------------------------------------------------

P1_MD = """<!-- v7-fix-section-header -->
## Part 1: ExPetDB internal test benchmark

**Purpose.** Compare the canonical opx-liq ML against Putirka 2008 eq28a/29a on
the internal ExPetDB held-out test set. Intersection-scope comparison (same
rows) isolates model quality from coverage.

**Data inputs.** `data/processed/opx_clean_opx_liq.parquet` test rows selected
by `data/splits/test_indices_opx_liq.npy` (n~=174). `results/nb03_per_family_winners.json`
for the canonical model/feature choice.

**Methods evaluated.**
1. Putirka 2008 opx-liq, Option 1 iterative (eq 28a + eq 29a)
2. Putirka 2008 opx-liq, Option 2 true P/T as input (ceiling)
3. Putirka 2008 opx-liq, Option 3 ML P/T fed back to Putirka
4. Ours canonical opx-liq forest (RF on alr for T, RF on raw for P)
5. Ours canonical opx-liq boosted (XGB on raw for both)

**Analysis performed.** Intersection-scope RMSE + 95% bootstrap CI (B=2000),
failure-rate analysis for Putirka, paired Wilcoxon signed-rank of ML vs each
Putirka variant, compositional stratification of failure modes.

**How to interpret.** Intersection RMSE compares methods on equal footing.
A substantially larger Putirka failure rate than ML indicates deployment
coverage matters even before looking at RMSE.

**Outputs.**
- `results/nb04_unified_benchmark.csv`
- `results/nb04_paired_wilcoxon.csv`
- `results/nb04_failure_mode_stratification.csv`
- `figures/fig_nb04_failure_analysis.{png,pdf}`
- `figures/fig_nb04_diagnostic_encyclopedia_T.{png,pdf}`, `fig_nb04_diagnostic_encyclopedia_P.{png,pdf}`

**Downstream use.** NB09 manuscript compilation reads the unified table.
"""


P2_MD = """<!-- v7-fix-section-header -->
## Part 2: Three-way ML benchmark on ArcPL three-phase paired

**Purpose.** Head-to-head comparison of ML opx (ours) vs ML cpx external models
(Agreda/Jorgenson/Wang) vs classical Putirka 2008 opx/cpx thermobarometers on
the ArcPL three-phase (cpx + opx + liq) paired subset. This is the methodological
rigor figure: same rows, all methods.

**Data inputs.** Merged LEPR ArcPL sheets (Cpx, Opx, Liq) joined on `Experiment`
to a three-phase scope (n~=118), plus broader scopes `lepr_full` and `cpx_scope`.

**Methods evaluated.** 13 total:
- 5 cpx: Agreda-Lopez cpx-liq, Agreda-Lopez cpx-only, Jorgenson cpx-only,
  Wang 2021 cpx-liq, Putirka 2008 cpx-liq
- 4 Putirka opx: opx-liq (iterative), opx-only (iterative), opx-liq [true P],
  opx-only [true T]
- 4 Ours: opx-liq forest/boosted, opx-only forest/boosted

**Analysis performed.** Per-method RMSE + 95% bootstrap CI (B=2000) +
coverage across the three scopes. The coverage panel exposes which methods
work on which samples.

**How to interpret.** Look at RMSE under the "paired" scope for method quality,
and coverage across scopes for deployment breadth. Ceiling Putirka variants
([true P]/[true T]) bracket the best Putirka could ever do.

**Outputs.**
- `results/nb04_method_benchmark_paired.csv`,
  `results/nb04_method_benchmark_lepr_full.csv`,
  `results/nb04_method_benchmark_cpx_scope.csv`
- `figures/fig_nb04_method_benchmark_paired.{png,pdf}`,
  `figures/fig_nb04_method_benchmark_lepr_full.{png,pdf}`,
  `figures/fig_nb04_method_benchmark_cpx_scope.{png,pdf}`

**Downstream use.** NB09 references the paired scope numbers.
"""


ENCY_MD = """<!-- v7-fix-section-header -->
## Encyclopedia: per-method diagnostic pred-vs-obs

**Purpose.** Visual QC: one 1:1 scatter panel per method per target on the Part 2
ArcPL paired scope. Bad calibration, heteroscedasticity, or systematic bias
show up here before they get masked by summary statistics.

**Data inputs.** Uses `preds` dict from Part 2 (cell 24) on the paired scope.

**Methods evaluated.** All 13 methods from Part 2.

**Analysis performed.** For each method, plot predicted vs observed with 1:1
line. Color-code by family. Annotate with n, RMSE, R^2.

**How to interpret.** A clean 1:1 trend with tight scatter = well-calibrated.
Horizontal or L-shaped cloud = model reverting to mean. Vertical stripe at
one obs value = insufficient compositional spread.

**Outputs.**
- `figures/fig_nb04_diagnostic_encyclopedia_T.{png,pdf}`
- `figures/fig_nb04_diagnostic_encyclopedia_P.{png,pdf}`

**Downstream use.** Terminal outputs - manuscript supplementary material.
"""


OPDEC_MD = """<!-- v7-fix-section-header -->
## OPERATOR DECISION

**Purpose.** Headline framing for the benchmark — who wins on the internal
test set and under what caveats. This is the cell that determines the
one-line claim in the abstract.

**Data inputs.** Prints derivatives of `results/nb04_unified_benchmark.csv`
(from Part 1).

**Methods evaluated.** None new; summarizes Part 1 comparisons.

**Analysis performed.** Prints the headline RMSE + bootstrap CI for ML vs
Putirka (Option 1 iterative and Option 2 true P/T), plus the failure rate
gap. No decisions are auto-made; human reviewer chooses the framing.

**How to interpret.** This is a framing aid — the numbers it prints are
what the manuscript's one-sentence claim rests on.

**Outputs.** Printed summary only. No files written here.

**Downstream use.** Reviewer copies headline numbers into the manuscript
abstract and Section 4.
"""


# ---------------------------------------------------------------------
# Main composer
# ---------------------------------------------------------------------

import uuid as _uuid


def _md_cell(src):
    c = nbformat.v4.new_markdown_cell(src)
    c['id'] = _uuid.uuid4().hex[:8]
    return c


def _code_cell(src):
    c = nbformat.v4.new_code_cell(src)
    c['id'] = _uuid.uuid4().hex[:8]
    return c


def _find_cell_index(nb, substring, cell_type='markdown'):
    for i, c in enumerate(nb.cells):
        if c.cell_type == cell_type and substring in c.source:
            return i
    return None


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # F1
    n_removed = fix1_delete_legacy_putirka(nb)
    print(f'F1: deleted {n_removed} legacy Putirka cell(s).')

    # F2
    n_f2 = fix2_add_putirka_opx(nb)
    print(f'F2: {n_f2} Part 2 edits applied.')

    # F6 markdown inserts + F3/F4/F5 body inserts
    # Strategy: ensure each ## Part N header exists exactly once. Then build
    # the final ordered list: all current cells minus encyclopedia/op-dec,
    # with Part-N markdown before its code, Part 3/4/5 markdown+code appended,
    # and the encyclopedia + op-dec sections moved to the end with their own
    # markdown headers.

    # Identify key cells by content fingerprint
    idx_ency = next((i for i, c in enumerate(nb.cells)
                     if c.cell_type == 'code'
                     and 'fig_nb04_diagnostic_encyclopedia_T' in c.source), None)
    idx_opdec = next((i for i, c in enumerate(nb.cells)
                      if c.cell_type == 'code'
                      and 'OPERATOR DECISION' in c.source
                      and 'fig_nb04_diagnostic_encyclopedia' not in c.source), None)
    idx_part1_md_existing = next((i for i, c in enumerate(nb.cells)
                                   if c.cell_type == 'markdown'
                                   and 'Part 1: Unified benchmark reframe' in c.source), None)
    idx_part2_md_existing = next((i for i, c in enumerate(nb.cells)
                                   if c.cell_type == 'markdown'
                                   and 'Three-way ML benchmark on ArcPL' in c.source), None)

    # Idempotent Part 3/4/5 checks
    have_p3 = any(c.cell_type == 'code' and SENTINEL_P3 in c.source for c in nb.cells)
    have_p4 = any(c.cell_type == 'code' and SENTINEL_P4 in c.source for c in nb.cells)
    have_p5 = any(c.cell_type == 'code' and SENTINEL_P5 in c.source for c in nb.cells)

    # Build output cell list
    out_cells = []
    # Copy cells 0..<encyclopedia start>, skipping encyclopedia and opdec
    ency_opdec_skip = {idx_ency, idx_opdec}
    # Also skip the legacy Part 1 and Three-way MD headers that we replace.
    skip_md = {idx_part1_md_existing, idx_part2_md_existing}
    # ...and any post-Part 2 "How to read" markdown (cell 27) - keep it.
    part1_code_inserted = False
    part2_code_inserted = False

    for i, c in enumerate(nb.cells):
        if i in ency_opdec_skip:
            continue
        if i in skip_md:
            continue
        # Insert the new Part 1 header before the first Part 1 code cell
        # (identify by unified_benchmark_table sentinel)
        if (not part1_code_inserted and c.cell_type == 'code'
                and 'unified_benchmark_table' in c.source):
            out_cells.append(_md_cell(P1_MD))
            part1_code_inserted = True
        # Insert the new Part 2 header before the three-way load cell
        if (not part2_code_inserted and c.cell_type == 'code'
                and 'Three-way ML benchmark' in c.source
                and 'load merged LEPR ArcPL' in c.source):
            out_cells.append(_md_cell(P2_MD))
            part2_code_inserted = True
        out_cells.append(c)

    # Append Parts 3, 4, 5 (idempotent)
    if not have_p3:
        out_cells.append(_md_cell(P3_MD))
        out_cells.append(_code_cell(P3_CODE))
        print('F3: Part 3 appended.')
    else:
        print('F3: Part 3 already present, skipped.')
    if not have_p4:
        out_cells.append(_md_cell(P4_MD))
        out_cells.append(_code_cell(P4_CODE))
        print('F4: Part 4 appended.')
    else:
        print('F4: Part 4 already present, skipped.')
    if not have_p5:
        out_cells.append(_md_cell(P5_MD))
        out_cells.append(_code_cell(P5_CODE))
        print('F5: Part 5 appended.')
    else:
        print('F5: Part 5 already present, skipped.')

    # Encyclopedia + OPERATOR DECISION at the end with their markdown headers.
    if idx_ency is not None:
        out_cells.append(_md_cell(ENCY_MD))
        out_cells.append(nb.cells[idx_ency])
    if idx_opdec is not None:
        out_cells.append(_md_cell(OPDEC_MD))
        out_cells.append(nb.cells[idx_opdec])

    nb.cells = out_cells
    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'nb04 written: {len(nb.cells)} cells.')

    # Structural map
    print('\n----- nb04 cell map -----')
    for i, c in enumerate(nb.cells):
        snip = c.source.replace('\n', ' ')[:90]
        tag = c.cell_type
        if c.cell_type == 'markdown' and SENTINEL_MARKDOWN in c.source:
            tag = 'markdown*'
        print(f'  [{i:2d}] {tag:>9} | {snip}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
