"""Apply lepr_to_training_schema shim to nb04 cells 24 and 26.

Cell 24: swap the CSV-cache Ours opx-liq lookup for direct model prediction
via `build_feature_matrix(lepr_to_training_schema(m.copy()), ...)`, and add
the missing Ours opx-only forest/boosted branch. Also print the final
method-vs-metric table to stdout for operator verification.

Cell 26: inside `_predict_all_methods(scope_df)`, apply the shim to a copy
before the `build_feature_matrix` call, add the Ours opx-only loop, and
print the benchmark table for each scope.

Idempotent: re-running on an already-patched notebook is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"
EXEC_NB = ROOT / "notebooks" / "executed" / "nb04_putirka_benchmark_executed.ipynb"

# ---------- CELL 24 ----------
CELL24_OLD_OURS = """# 6. Ours opx-liq (v7 two-family: forest + boosted)
for _family in ['forest', 'boosted']:
    try:
        ours_df = pd.read_csv(
            RESULTS / f'nb04b_arcpl_predictions_{_family}.csv'
        )
        if 'Experiment' in ours_df.columns:
            _o = m.merge(ours_df[['Experiment', 'T_pred', 'P_pred']],
                         on='Experiment', how='left')
            preds[f'Ours opx-liq {_family}'] = (
                _o['T_pred'].values, _o['P_pred'].values
            )
        else:
            preds[f'Ours opx-liq {_family}'] = (
                ours_df['T_pred'].values[:len(m)],
                ours_df['P_pred'].values[:len(m)],
            )
    except Exception as e:
        print(f'Ours opx-liq {_family} skipped ({e})')"""

CELL24_NEW_OURS = """# 6. Ours opx-liq + opx-only (direct inference via training-schema shim).
# LEPR uses _Opx/_Liq column suffixes; training schema uses unsuffixed opx
# oxides + liq_* prefix. Apply lepr_to_training_schema on a .copy() so m
# keeps its original columns for Thermobar and external cpx models above.
import joblib as _jl_ours
from src.data import canonical_model_path, canonical_model_spec
from src.features import lepr_to_training_features
m_for_ours = lepr_to_training_features(m.copy())
print(f'\\nOurs shim: m_for_ours shape={m_for_ours.shape}')
for _ox in ['SiO2', 'FeO_total', 'liq_SiO2', 'liq_FeO']:
    print(f'  {_ox} present: {_ox in m_for_ours.columns}')
for _tracks in ['opx_liq', 'opx_only']:
    _use_liq = (_tracks == 'opx_liq')
    for _family in ['forest', 'boosted']:
        try:
            _sT = canonical_model_spec('T_C', _tracks, _family, RESULTS)
            _sP = canonical_model_spec('P_kbar', _tracks, _family, RESULTS)
            _mT = _jl_ours.load(canonical_model_path('T_C', _tracks, _family, MODELS, RESULTS))
            _mP = _jl_ours.load(canonical_model_path('P_kbar', _tracks, _family, MODELS, RESULTS))
            _Xt, _ = build_feature_matrix(m_for_ours, _sT['feature_set'], use_liq=_use_liq)
            _Xp, _ = build_feature_matrix(m_for_ours, _sP['feature_set'], use_liq=_use_liq)
            preds[f'Ours {_tracks.replace("_", "-")} {_family}'] = (
                _mT.predict(_Xt), _mP.predict(_Xp),
            )
        except Exception as e:
            print(f'Ours {_tracks} {_family} skipped ({e})')"""


# ---------- CELL 26 ----------
# Inside _predict_all_methods: swap the Ours opx-liq loop for a shim+both-tracks loop.
CELL26_OLD_OURS = """    for fam in ['forest', 'boosted']:
        try:
            sT = canonical_model_spec('T_C', 'opx_liq', fam, RESULTS)
            sP = canonical_model_spec('P_kbar', 'opx_liq', fam, RESULTS)
            mT = joblib.load(canonical_model_path('T_C', 'opx_liq', fam, MODELS, RESULTS))
            mP = joblib.load(canonical_model_path('P_kbar', 'opx_liq', fam, MODELS, RESULTS))
            Xt, _ = build_feature_matrix(scope_df, sT['feature_set'], use_liq=True)
            Xp, _ = build_feature_matrix(scope_df, sP['feature_set'], use_liq=True)
            p[f'Ours opx-liq {fam}'] = (mT.predict(Xt), mP.predict(Xp))
        except Exception as e:
            print(f'  Ours opx-liq {fam} partial/skipped ({e})')
            p[f'Ours opx-liq {fam}'] = (np.full(len(scope_df), np.nan),
                                         np.full(len(scope_df), np.nan))
    return p"""

CELL26_NEW_OURS = """    # Training-schema shim: LEPR _Opx/_Liq suffixes -> ExPetDB flat columns
    # + cation recalc + engineered features (Mg_num, En/Fs/Wo, Al_IV/VI, MgTs).
    from src.features import lepr_to_training_features
    scope_train = lepr_to_training_features(scope_df.copy())
    for tracks in ['opx_liq', 'opx_only']:
        use_liq = (tracks == 'opx_liq')
        for fam in ['forest', 'boosted']:
            name = f'Ours {tracks.replace("_", "-")} {fam}'
            try:
                sT = canonical_model_spec('T_C', tracks, fam, RESULTS)
                sP = canonical_model_spec('P_kbar', tracks, fam, RESULTS)
                mT = joblib.load(canonical_model_path('T_C', tracks, fam, MODELS, RESULTS))
                mP = joblib.load(canonical_model_path('P_kbar', tracks, fam, MODELS, RESULTS))
                Xt, _ = build_feature_matrix(scope_train, sT['feature_set'], use_liq=use_liq)
                Xp, _ = build_feature_matrix(scope_train, sP['feature_set'], use_liq=use_liq)
                p[name] = (mT.predict(Xt), mP.predict(Xp))
            except Exception as e:
                print(f'  {name} partial/skipped ({e})')
                p[name] = (np.full(len(scope_df), np.nan),
                           np.full(len(scope_df), np.nan))
    return p"""


# Add a COLORS entry for the new Ours opx-only rows in cell 26.
CELL26_OLD_COLORS = """COLORS = {
    'Ours opx-liq forest':    FAMILY_COLORS['forest'],
    'Ours opx-liq boosted':   FAMILY_COLORS['boosted'],
    'Agreda-Lopez cpx-liq':   FAMILY_COLORS['external_cpx'],
    'Agreda-Lopez cpx-only':  OKABE_ITO['yellow'],
    'Jorgenson cpx-only':     OKABE_ITO['green'],
    'Wang 2021 cpx-liq':      OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':   FAMILY_COLORS['putirka'],
}"""

CELL26_NEW_COLORS = """COLORS = {
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


# Also print the per-scope benchmark table at the end of cell 26 for operator verification.
CELL26_OLD_LOOP = """for df_scope, label, fn, slug in SCOPES:
    if len(df_scope) < 10:
        print(f'\\n[{slug}] SKIPPED: n={len(df_scope)} too small')
        continue
    print(f'\\n[{slug}] running {len(df_scope)} rows ...')
    preds_s = _predict_all_methods(df_scope)
    ms = _metrics_df(preds_s, df_scope['T_C'].values,
                     df_scope['P_kbar'].values, len(df_scope))
    ms.round(3).to_csv(RESULTS / f'nb04_method_benchmark_{slug}.csv', index=False)
    _render_three_panel(ms, len(df_scope), label, fn)"""

CELL26_NEW_LOOP = """for df_scope, label, fn, slug in SCOPES:
    if len(df_scope) < 10:
        print(f'\\n[{slug}] SKIPPED: n={len(df_scope)} too small')
        continue
    print(f'\\n[{slug}] running {len(df_scope)} rows ...')
    preds_s = _predict_all_methods(df_scope)
    ms = _metrics_df(preds_s, df_scope['T_C'].values,
                     df_scope['P_kbar'].values, len(df_scope))
    ms.round(3).to_csv(RESULTS / f'nb04_method_benchmark_{slug}.csv', index=False)
    _render_three_panel(ms, len(df_scope), label, fn)
    _table_cols = ['Method', 'T_n', 'T_RMSE', 'T_R2', 'T_coverage_pct',
                   'P_n', 'P_RMSE', 'P_R2', 'P_coverage_pct']
    print(f'\\n[{slug}] benchmark table:')
    print(ms[_table_cols].round(3).to_string(index=False))"""


def _patch(cell, old, new, label):
    if new in cell.source:
        print(f'  {label}: already patched.')
        return False
    if old not in cell.source:
        print(f'  {label}: anchor not found.')
        return None
    cell.source = cell.source.replace(old, new, 1)
    print(f'  {label}: patched.')
    return True


def _apply(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    print(f'{nb_path.name} ({len(nb.cells)} cells):')
    changed = False
    if len(nb.cells) > 24:
        r = _patch(nb.cells[24], CELL24_OLD_OURS, CELL24_NEW_OURS, 'cell 24 Ours block')
        if r:
            nb.cells[24].outputs = []
            nb.cells[24].execution_count = None
            changed = True
        elif r is None:
            return False
    if len(nb.cells) > 26:
        r1 = _patch(nb.cells[26], CELL26_OLD_COLORS, CELL26_NEW_COLORS, 'cell 26 COLORS')
        r2 = _patch(nb.cells[26], CELL26_OLD_OURS, CELL26_NEW_OURS, 'cell 26 Ours loop')
        r3 = _patch(nb.cells[26], CELL26_OLD_LOOP, CELL26_NEW_LOOP, 'cell 26 scope loop')
        if r1 or r2 or r3:
            nb.cells[26].outputs = []
            nb.cells[26].execution_count = None
            changed = True
        if r2 is None:
            return False
    if changed:
        nbformat.validate(nb)
        nbformat.write(nb, str(nb_path))
        print(f'  wrote {nb_path.name}.')
    else:
        print(f'  {nb_path.name}: no changes.')
    return True


def main() -> int:
    ok = _apply(NB)
    if not ok:
        return 1
    if EXEC_NB.exists():
        _apply(EXEC_NB)
    return 0


if __name__ == '__main__':
    sys.exit(main())
