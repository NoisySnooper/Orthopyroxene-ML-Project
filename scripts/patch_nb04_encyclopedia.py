"""Replace nb04 cell 18 with the diagnostic encyclopedia cell.

The current cell 18 is the 4-panel pred-vs-obs + residual grid that saves
figures/fig_nb04_putirka_comparison.{png,pdf}. Per operator spec the cell
is superseded by a unified diagnostic encyclopedia: two 4x4 grids (T and
P), one 1:1 scatter panel per method, three info panels (legend/summary/
scope). Family-colored, Putirka marker = square, standardized axes.

Side effects:
    * rewrites notebooks/nb04_putirka_benchmark.ipynb cell 18 in place
    * nbformat.validate after write
    * deletes figures/fig_nb04_putirka_comparison.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"
STALE_FIGS = [
    ROOT / "figures" / "fig_nb04_putirka_comparison.png",
    ROOT / "figures" / "fig_nb04_putirka_comparison.pdf",
]

ANCHOR_HEADER = "# Part 1d: new 4-panel figure replacing the old 2x3 grid."
ANCHOR_OUTNAME = "fig_nb04_putirka_comparison"

NEW_CELL = r'''# Diagnostic encyclopedia: 4x4 pred-vs-obs grids for T and P.
# Every benchmark method gets its own 1:1 scatter panel on the ArcPL
# three-phase paired scope (cpx + opx + liq). Family-colored, Putirka =
# square markers. 13 method panels + legend + summary + scope info.
# Self-contained: loads LEPR data and arcpl prediction manifest locally.
import joblib
import Thermobar as tb
from matplotlib.lines import Line2D

from src.plot_style import apply_style, OKABE_ITO
from src.data import canonical_model_path, canonical_model_spec
from src.features import build_feature_matrix, lepr_to_training_features
from src.external_models import (
    predict_agreda_from_df, predict_jorgenson, predict_wang,
    predict_putirka_cpx_liq, AGREDA_CPX_COLS,
)
from config import FAMILY_COLORS, LEPR_XLSX

apply_style()

# ---- Load LEPR cpx / liq / opx sheets (self-contained) ----
_xls = pd.ExcelFile(LEPR_XLSX)
_cpx_enc = pd.read_excel(_xls, sheet_name='Cpx')
_liq_enc = pd.read_excel(_xls, sheet_name='Liq')
_opx_enc = pd.read_excel(_xls, sheet_name='Opx')
def _numerify_enc(_df, _suffix):
    for _c in _df.columns:
        if _c.endswith(_suffix):
            _df[_c] = pd.to_numeric(_df[_c], errors='coerce').fillna(0.0)
    return _df
_cpx_enc = _numerify_enc(_cpx_enc.drop_duplicates('Experiment'), '_Cpx')
_liq_enc = _numerify_enc(_liq_enc.drop_duplicates('Experiment'), '_Liq')
_opx_enc = _numerify_enc(_opx_enc.drop_duplicates('Experiment'), '_Opx')
_arcpl_ids = pd.read_csv(RESULTS / 'nb04b_arcpl_predictions.csv')
keep_exp_enc = set(_arcpl_ids['Experiment'].astype(str))

# ---- Build ArcPL three-phase paired scope (cpx + opx + liq) ----
_scope = _cpx_enc.merge(_liq_enc, on='Experiment', how='inner',
                        suffixes=('', '_liq_dup'))
_scope = _scope.merge(_opx_enc, on='Experiment', how='inner',
                      suffixes=('', '_opx_dup'))
for _tgt in ['T_K', 'P_kbar']:
    for _src in ['', '_liq_dup', '_opx_dup']:
        _col = f'{_tgt}{_src}'
        if _col in _scope.columns:
            if _tgt not in _scope.columns:
                _scope[_tgt] = _scope[_col]
            else:
                _scope[_tgt] = _scope[_tgt].combine_first(_scope[_col])
_scope['T_C'] = pd.to_numeric(_scope['T_K'], errors='coerce') - 273.15
_scope['P_kbar'] = pd.to_numeric(_scope['P_kbar'], errors='coerce')
if 'Fe3Fet_Liq' not in _scope.columns:
    _scope['Fe3Fet_Liq'] = 0.0
_scope = _scope[_scope['Experiment'].astype(str).isin(keep_exp_enc)]
_has_cpx = _scope[AGREDA_CPX_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) > 80
_scope = _scope[_has_cpx & np.isfinite(_scope['T_C']) & np.isfinite(_scope['P_kbar'])].reset_index(drop=True)
n_paired = len(_scope)
y_T_enc = _scope['T_C'].values
y_P_enc = _scope['P_kbar'].values

# ---- Training-schema copy for Ours inference ----
# LEPR uses SiO2_Opx / SiO2_Liq; the training schema uses SiO2 / liq_SiO2.
# Apply shim on a .copy() so _scope keeps its _Cpx/_Liq/_Opx columns for
# Thermobar + external ML methods.
_scope_train = lepr_to_training_features(_scope.copy())
for _fs in ['raw', 'pwlr', 'alr']:
    _Xt, _ = build_feature_matrix(_scope_train, _fs, use_liq=True)
    _Xo, _ = build_feature_matrix(_scope_train, _fs, use_liq=False)
    print(f'  feature check {_fs}: use_liq=True {_Xt.shape} '
          f'({int(np.isnan(_Xt).sum())} NaN) | use_liq=False {_Xo.shape} '
          f'({int(np.isnan(_Xo).sum())} NaN)')

FAMILY_MAP = {
    'Ours opx-liq forest':    'forest',
    'Ours opx-liq boosted':   'boosted',
    'Ours opx-only forest':   'forest',
    'Ours opx-only boosted':  'boosted',
    'Agreda-Lopez cpx-liq':   'external_cpx',
    'Agreda-Lopez cpx-only':  'external_cpx',
    'Jorgenson cpx-liq':      'external_cpx',
    'Jorgenson cpx-only':     'external_cpx',
    'Wang 2021 cpx-liq':      'external_cpx',
    'Wang 2021 cpx-only':     'external_cpx',
    'Putirka 2008 opx-liq':   'putirka',
    'Putirka 2008 opx-only':  'putirka',
    'Putirka 2008 cpx-liq':   'putirka',
    'Putirka 2008 cpx-only':  'putirka',
}

# ---- Compute every method on _scope ----
enc_preds = {}  # name -> (T_pred, P_pred)

# Ours opx-liq forest / boosted (training schema via shim)
for _fam in ['forest', 'boosted']:
    try:
        _sT = canonical_model_spec('T_C', 'opx_liq', _fam, RESULTS)
        _sP = canonical_model_spec('P_kbar', 'opx_liq', _fam, RESULTS)
        _mT = joblib.load(canonical_model_path('T_C', 'opx_liq', _fam, MODELS, RESULTS))
        _mP = joblib.load(canonical_model_path('P_kbar', 'opx_liq', _fam, MODELS, RESULTS))
        _Xt, _ = build_feature_matrix(_scope_train, _sT['feature_set'], use_liq=True)
        _Xp, _ = build_feature_matrix(_scope_train, _sP['feature_set'], use_liq=True)
        enc_preds[f'Ours opx-liq {_fam}'] = (_mT.predict(_Xt), _mP.predict(_Xp))
    except Exception as e:
        print(f'  Ours opx-liq {_fam} skipped: {e}')

# Ours opx-only forest / boosted (training schema via shim)
for _fam in ['forest', 'boosted']:
    try:
        _sT = canonical_model_spec('T_C', 'opx_only', _fam, RESULTS)
        _sP = canonical_model_spec('P_kbar', 'opx_only', _fam, RESULTS)
        _mT = joblib.load(canonical_model_path('T_C', 'opx_only', _fam, MODELS, RESULTS))
        _mP = joblib.load(canonical_model_path('P_kbar', 'opx_only', _fam, MODELS, RESULTS))
        _Xt, _ = build_feature_matrix(_scope_train, _sT['feature_set'], use_liq=False)
        _Xp, _ = build_feature_matrix(_scope_train, _sP['feature_set'], use_liq=False)
        enc_preds[f'Ours opx-only {_fam}'] = (_mT.predict(_Xt), _mP.predict(_Xp))
    except Exception as e:
        print(f'  Ours opx-only {_fam} skipped: {e}')

# Agreda-Lopez cpx-only / cpx-liq
for _kind, _label in [('cpx_only', 'cpx-only'), ('cpx_liq', 'cpx-liq')]:
    try:
        _aT = predict_agreda_from_df(_scope, MODELS / 'external', _kind, 'T')['median']
        _aP = predict_agreda_from_df(_scope, MODELS / 'external', _kind, 'P')['median']
        enc_preds[f'Agreda-Lopez {_label}'] = (_aT, _aP)
    except Exception as e:
        print(f'  Agreda {_kind} skipped: {e}')

# Jorgenson cpx-only / cpx-liq (Thermobar needs Sample_ID_Liq + NiO/CoO/CO2_Liq present)
_scope_jorg = _scope.copy()
for _c in ['Sample_ID_Liq', 'NiO_Liq', 'CoO_Liq', 'CO2_Liq']:
    if _c not in _scope_jorg.columns:
        _scope_jorg[_c] = 0.0 if _c != 'Sample_ID_Liq' else _scope_jorg['Experiment'].astype(str).values
for _phase, _label in [('cpx_only', 'cpx-only'), ('cpx_liq', 'cpx-liq')]:
    try:
        _jT = predict_jorgenson(_scope_jorg, 'T', phase=_phase, P_kbar=y_P_enc)
        _jP = predict_jorgenson(_scope_jorg, 'P', phase=_phase, T_K=y_T_enc + 273.15)
        enc_preds[f'Jorgenson {_label}'] = (_jT, _jP)
    except Exception as e:
        print(f'  Jorgenson {_phase} skipped: {e}')

# Wang 2021 cpx-liq
try:
    _wT = predict_wang(_scope, 'T', P_kbar=y_P_enc)
    _wP = predict_wang(_scope, 'P', T_K=y_T_enc + 273.15)
    enc_preds['Wang 2021 cpx-liq'] = (_wT, _wP)
except Exception as e:
    print(f'  Wang skipped: {e}')

# Putirka 2008 cpx-liq
try:
    _pT = predict_putirka_cpx_liq(_scope, 'T', P_kbar=y_P_enc)
    _pP = predict_putirka_cpx_liq(_scope, 'P', T_K=y_T_enc + 273.15)
    enc_preds['Putirka 2008 cpx-liq'] = (_pT, _pP)
except Exception as e:
    print(f'  Putirka cpx-liq skipped: {e}')

# Putirka 2008 cpx-only (eq32b P + eq32d T, iterative — eq32c is not a valid Thermobar id)
try:
    _cpx_df = _scope[[c for c in _scope.columns if c.endswith('_Cpx')]].apply(
        pd.to_numeric, errors='coerce').fillna(0.0)
    _out_cpx = tb.calculate_cpx_only_press_temp(
        cpx_comps=_cpx_df,
        equationP='P_Put2008_eq32b',
        equationT='T_Put2008_eq32d',
    )
    enc_preds['Putirka 2008 cpx-only'] = (
        np.asarray(_out_cpx['T_K_calc']) - 273.15,
        np.asarray(_out_cpx['P_kbar_calc']),
    )
except Exception as e:
    print(f'  Putirka cpx-only skipped: {e}')

# Wang 2021 cpx-only (direct Thermobar call; predict_wang is cpx-liq only)
try:
    _cpx_df_w = _scope[[c for c in _scope.columns if c.endswith('_Cpx')]].apply(
        pd.to_numeric, errors='coerce').fillna(0.0)
    _out_wco = tb.calculate_cpx_only_press_temp(
        cpx_comps=_cpx_df_w,
        equationP='P_Wang2021_eq1',
        equationT='T_Wang2021_eq2',
    )
    enc_preds['Wang 2021 cpx-only'] = (
        np.asarray(_out_wco['T_K_calc']) - 273.15,
        np.asarray(_out_wco['P_kbar_calc']),
    )
except Exception as e:
    print(f'  Wang 2021 cpx-only skipped: {e}')

# Putirka 2008 opx-liq (eq28a T, eq29a P, iterative)
try:
    _opx_cols = [c for c in _scope.columns if c.endswith('_Opx')]
    _liq_cols = [c for c in _scope.columns if c.endswith('_Liq')]
    _opx_df = _scope[_opx_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    _liq_df = _scope[_liq_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    _out_ol = tb.calculate_opx_liq_press_temp(
        opx_comps=_opx_df, liq_comps=_liq_df,
        equationP='P_Put2008_eq29a', equationT='T_Put2008_eq28a',
    )
    enc_preds['Putirka 2008 opx-liq'] = (
        np.asarray(_out_ol['T_K_calc']) - 273.15,
        np.asarray(_out_ol['P_kbar_calc']),
    )
except Exception as e:
    print(f'  Putirka opx-liq skipped: {e}')

# Putirka 2008 opx-only (eq28b_opx_sat T with P_obs, eq29c P with T_obs)
try:
    _opx_cols = [c for c in _scope.columns if c.endswith('_Opx')]
    _liq_cols = [c for c in _scope.columns if c.endswith('_Liq')]
    _opx_df = _scope[_opx_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    _liq_df = _scope[_liq_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    _out_T = tb.calculate_opx_liq_temp(
        equationT='T_Put2008_eq28b_opx_sat',
        opx_comps=_opx_df, liq_comps=_liq_df,
        P=y_P_enc,
    )
    _out_P = tb.calculate_opx_only_press(
        opx_comps=_opx_df, equationP='P_Put2008_eq29c',
        T=y_T_enc + 273.15,
    )
    # Thermobar returns DataFrame or Series depending on eq_tests
    def _as_T_C(obj):
        if isinstance(obj, pd.DataFrame):
            col = 'T_K_calc' if 'T_K_calc' in obj.columns else obj.columns[0]
            return np.asarray(obj[col]) - 273.15
        arr = np.asarray(obj)
        return arr - 273.15 if np.nanmedian(arr) > 400 else arr
    def _as_P_kbar(obj):
        if isinstance(obj, pd.DataFrame):
            col = 'P_kbar_calc' if 'P_kbar_calc' in obj.columns else obj.columns[0]
            return np.asarray(obj[col])
        return np.asarray(obj)
    enc_preds['Putirka 2008 opx-only'] = (_as_T_C(_out_T), _as_P_kbar(_out_P))
except Exception as e:
    print(f'  Putirka opx-only skipped: {e}')

print(f'Methods computed ({len(enc_preds)}):')
for _k in enc_preds:
    print(f'  - {_k}')

# ---- Render 4x4 grids ----
LAYOUT = [
    ('Ours opx-liq forest',  'Ours opx-liq boosted',  'Putirka 2008 opx-liq',  'Agreda-Lopez cpx-liq'),
    ('Ours opx-only forest', 'Ours opx-only boosted', 'Putirka 2008 opx-only', 'Jorgenson cpx-liq'),
    ('Putirka 2008 cpx-liq', 'Putirka 2008 cpx-only', 'Wang 2021 cpx-liq',     'Agreda-Lopez cpx-only'),
    ('Jorgenson cpx-only',   'Wang 2021 cpx-only',    '__LEGEND__',            '__SUMMARY__'),
]
T_LIMS = (700, 1700)
P_LIMS = (-2, 35)


def _enc_metrics(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    n = int(mask.sum())
    if n < 3:
        return {'n': n, 'rmse': np.nan, 'r2': np.nan, 'cov': 100 * n / len(y)}
    yt = y[mask]; yp = yhat[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_r = float(np.sum((yt - yp) ** 2))
    ss_t = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    return {'n': n, 'rmse': rmse, 'r2': r2, 'cov': 100 * n / len(y)}


def _is_put(name):
    return name.startswith('Putirka')


def _render_grid(obs, pred_idx, lims, unit, fmt_rmse, out_stem, suptitle):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    off_notes = []
    low_cov_notes = []
    summary = []
    for r in range(4):
        for c in range(4):
            name = LAYOUT[r][c]
            ax = axes[r, c]
            if name.startswith('__'):
                continue
            if name not in enc_preds:
                ax.text(0.5, 0.5, f'{name}\n(unavailable)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray', style='italic')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            yhat = np.asarray(enc_preds[name][pred_idx], dtype=float)
            m = _enc_metrics(obs, yhat)
            fam = FAMILY_MAP.get(name, 'putirka' if _is_put(name) else 'external_cpx')
            color = FAMILY_COLORS.get(fam, OKABE_ITO['yellow'])
            marker = 's' if _is_put(name) else 'o'
            mask = np.isfinite(obs) & np.isfinite(yhat)
            ax.scatter(obs[mask], yhat[mask], s=22, alpha=0.55,
                       c=color, edgecolor='k', linewidths=0.3, marker=marker)
            off = int(((yhat[mask] < lims[0]) | (yhat[mask] > lims[1])).sum())
            if off > 0:
                off_notes.append(f'{name}: {off}/{int(mask.sum())} off-axis')
            ax.plot(lims, lims, '--', color='gray', lw=0.9, alpha=0.8)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f'Observed ({unit})', fontsize=9)
            ax.set_ylabel(f'Predicted ({unit})', fontsize=9)
            title = (f'{name}\n'
                     f'RMSE {fmt_rmse.format(m["rmse"])} {unit}  '
                     f'R$^2$={m["r2"]:+.2f}  n={m["n"]}')
            if m['cov'] < 95:
                title += f'\n{m["cov"]:.0f}% coverage'
                low_cov_notes.append(f'{name}: {m["cov"]:.0f}% coverage')
            ax.set_title(title, fontsize=8.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            summary.append({**m, 'name': name})
    # Legend panel (3,2)
    ax = axes[3, 2]
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['forest'],
               markeredgecolor='k', markersize=11, label='Forest (RF/ERT) - Ours'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['boosted'],
               markeredgecolor='k', markersize=11, label='Boosted (GB/XGB) - Ours'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS['external_cpx'],
               markeredgecolor='k', markersize=11, label='External ML (cpx)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=FAMILY_COLORS['putirka'],
               markeredgecolor='k', markersize=11, label='Putirka 2008 (classical)'),
        Line2D([0], [0], linestyle='--', color='gray', label='1:1 line'),
    ]
    ax.legend(handles=handles, loc='center', fontsize=10, frameon=False)
    ax.set_title('Legend', fontsize=11, fontweight='bold')
    # Summary panel (3,3) - top 6 by RMSE + scope info merged in
    ax = axes[3, 3]
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    txt = (f'Scope\n{"-" * 28}\n'
           f'ArcPL three-phase paired\n(cpx + opx + liq)\n'
           f'n_paired = {n_paired}\n'
           f'T range: {T_LIMS[0]}-{T_LIMS[1]} C\n'
           f'P range: {P_LIMS[0]}-{P_LIMS[1]} kbar\n\n')
    if summary:
        sdf = pd.DataFrame(summary).sort_values('rmse', na_position='last')
        txt += 'Top 6 by RMSE\n' + '-' * 28 + '\n'
        for _, row in sdf.head(6).iterrows():
            name_short = row['name'].replace('Putirka 2008 ', 'Put. ').replace('Agreda-Lopez ', 'Agreda ').replace('Jorgenson ', 'Jorg. ').replace('Wang 2021 ', 'Wang ')
            txt += f"{name_short[:20]:<20}{fmt_rmse.format(row['rmse']):>8}\n"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace')
    ax.set_title('Scope + top methods', fontsize=11, fontweight='bold')

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ('png', 'pdf'):
        fp = FIGURES / f'{out_stem}.{ext}'
        fig.savefig(fp, dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'\nSaved {out_stem}.')
    print(f'  Off-axis: {off_notes or ["none"]}')
    print(f'  <95% coverage: {low_cov_notes or ["none"]}')


_render_grid(y_T_enc, 0, T_LIMS, 'C', '{:.0f}',
             'fig_nb04_diagnostic_encyclopedia_T',
             f'Diagnostic encyclopedia: Temperature pred vs obs '
             f'(ArcPL three-phase paired, n={n_paired})')
_render_grid(y_P_enc, 1, P_LIMS, 'kbar', '{:.2f}',
             'fig_nb04_diagnostic_encyclopedia_P',
             f'Diagnostic encyclopedia: Pressure pred vs obs '
             f'(ArcPL three-phase paired, n={n_paired})')
'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    if len(nb.cells) <= 18:
        print(f'nb04 has only {len(nb.cells)} cells')
        return 1
    cell = nb.cells[18]
    if cell.cell_type != 'code':
        print(f'cell 18 is {cell.cell_type}, expected code')
        return 1

    src = cell.source
    already_exact = (src == NEW_CELL)
    is_encyc = 'fig_nb04_diagnostic_encyclopedia' in src
    is_orig = ANCHOR_HEADER in src and ANCHOR_OUTNAME in src

    if already_exact:
        print('nb04 cell 18: already on target NEW_CELL exactly.')
    elif is_orig or is_encyc:
        cell.source = NEW_CELL
        cell.outputs = []
        cell.execution_count = None
        nbformat.validate(nb)
        nbformat.write(nb, str(NB))
        reason = 'reverting stale encyclopedia cell' if is_encyc else 'replacing original cell 18'
        print(f'nb04 cell 18: rewritten ({reason}).')
    else:
        print('nb04 cell 18: anchor mismatch - refusing to touch.')
        print(f'  first 120 chars: {src[:120]!r}')
        return 1

    # Delete stale comparison figures (also post-execution garbage).
    for fp in STALE_FIGS:
        if fp.exists():
            fp.unlink()
            print(f'  deleted stale figure: {fp.name}')

    # Also purge any cached embedded pngs in executed notebook copy
    exec_nb = ROOT / 'notebooks' / 'executed' / 'nb04_putirka_benchmark_executed.ipynb'
    if exec_nb.exists():
        enb = nbformat.read(str(exec_nb), as_version=4)
        if len(enb.cells) > 18:
            enb.cells[18].source = NEW_CELL
            enb.cells[18].outputs = []
            enb.cells[18].execution_count = None
            nbformat.validate(enb)
            nbformat.write(enb, str(exec_nb))
            print('  executed-notebook copy cell 18 also updated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
