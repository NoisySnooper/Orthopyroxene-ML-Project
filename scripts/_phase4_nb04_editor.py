"""Idempotent editor that applies the v9 Phase 4 (Option B Kd filter +
stacked/resampled methods) patches to notebooks/nb04_putirka_benchmark.ipynb.

Patches:
  * Cell 21 (method benchmark per-row): add `eq_tests=True` to the
    opx-liq iterative call; capture the `Kd Eq (Put2008+-0.06)` mask
    into a module-level `_arcpl_kd_eq_mask` numpy array.
  * Cell 23 (three-scope benchmark): same eq_tests fix; add stacked
    and resampled Ours entries; compute metrics on both all-converged
    and eq-pass-AND-converged subsets; write per-scope CSVs for each.
  * Cell 27 (headline 3x2 figure): swap the converged-only "fair
    subset" for the Kd-equilibrated + converged subset (Option B).
  * Cell 31 (Part 5 dual-scope opx-only): add eq_tests=True to both
    opx-liq variants (iterative + ceiling); capture kd mask.

Re-running is safe; each patch is keyed off a one-shot sentinel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / 'notebooks' / 'nb04_putirka_benchmark.ipynb'

SENTINEL_21 = '# v9-optionB: capture Kd equilibrium mask on ArcPL opx-liq'
SENTINEL_23 = '# v9-optionB: stacked + resampled Ours variants and Kd eq mask per scope'
SENTINEL_27 = '# v9-optionB: headline figure uses Kd-equilibrated subset (Option B)'
SENTINEL_31 = '# v9-optionB: Part 5 eq_tests=True on opx-liq variants'


def _src(cell):
    return ''.join(cell['source'])


def _set_src(cell, s):
    cell['source'] = [ln + '\n' for ln in s.split('\n')[:-1]] + [s.split('\n')[-1]]


def patch_cell_21(cells):
    idx = 21
    src = _src(cells[idx])
    if SENTINEL_21 in src:
        return False, 'already patched'
    old = (
        "        try:\n"
        "            _iter = _pt_opx.calculate_opx_liq_press_temp(\n"
        "                opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')\n"
        "            _T = (_iter['T_K_calc'].values - 273.15\n"
        "                  if 'T_K_calc' in _iter.columns\n"
        "                  else _iter.iloc[:, 0].values - 273.15)\n"
        "            _P = (_iter['P_kbar_calc'].values\n"
        "                  if 'P_kbar_calc' in _iter.columns\n"
        "                  else _iter.iloc[:, 1].values)\n"
        "            preds['Putirka 2008 opx-liq'] = (_T, _P)\n"
        "        except Exception as _e1:\n"
        "            print(f'Putirka opx-liq iterative skipped ({_e1})')\n"
    )
    new = (
        "        try:\n"
        "            # " + SENTINEL_21 + "\n"
        "            _iter = _pt_opx.calculate_opx_liq_press_temp(\n"
        "                opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a',\n"
        "                eq_tests=True)\n"
        "            _T = (_iter['T_K_calc'].values - 273.15\n"
        "                  if 'T_K_calc' in _iter.columns\n"
        "                  else _iter.iloc[:, 0].values - 273.15)\n"
        "            _P = (_iter['P_kbar_calc'].values\n"
        "                  if 'P_kbar_calc' in _iter.columns\n"
        "                  else _iter.iloc[:, 1].values)\n"
        "            preds['Putirka 2008 opx-liq'] = (_T, _P)\n"
        "            if 'Kd Eq (Put2008+-0.06)' in _iter.columns:\n"
        "                _arcpl_kd_eq_mask = (_iter['Kd Eq (Put2008+-0.06)']\n"
        "                                     .astype(str).str.upper()\n"
        "                                     .str.startswith('Y').values)\n"
        "            else:\n"
        "                _arcpl_kd_eq_mask = np.ones(len(_iter), dtype=bool)\n"
        "            print(f'Option B: Kd equilibrium pass = '\n"
        "                  f'{int(_arcpl_kd_eq_mask.sum())}/{len(_arcpl_kd_eq_mask)} '\n"
        "                  f'({100*_arcpl_kd_eq_mask.mean():.1f}%)')\n"
        "        except Exception as _e1:\n"
        "            print(f'Putirka opx-liq iterative skipped ({_e1})')\n"
        "            _arcpl_kd_eq_mask = None\n"
    )
    if old not in src:
        raise RuntimeError('cell 21: target block not found')
    src2 = src.replace(old, new)
    _set_src(cells[idx], src2)
    return True, 'patched'


def patch_cell_23(cells):
    idx = 23
    src = _src(cells[idx])
    if SENTINEL_23 in src:
        return False, 'already patched'

    # (a) Add eq_tests=True + capture mask inside _predict_all_methods.
    old_a = (
        "        if len(_opx_in) and len(_liq_in):\n"
        "            try:\n"
        "                _iter = _pt_opx.calculate_opx_liq_press_temp(\n"
        "                    opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                    equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')\n"
        "                _T = (_iter['T_K_calc'].values - 273.15 if 'T_K_calc' in _iter.columns\n"
        "                      else _iter.iloc[:, 0].values - 273.15)\n"
        "                _P = (_iter['P_kbar_calc'].values if 'P_kbar_calc' in _iter.columns\n"
        "                      else _iter.iloc[:, 1].values)\n"
        "                p['Putirka 2008 opx-liq'] = (_T, _P)\n"
        "            except Exception as _e1:\n"
        "                print(f'  Putirka opx-liq iterative skipped ({_e1})')\n"
    )
    new_a = (
        "        if len(_opx_in) and len(_liq_in):\n"
        "            try:\n"
        "                # " + SENTINEL_23 + "\n"
        "                _iter = _pt_opx.calculate_opx_liq_press_temp(\n"
        "                    opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                    equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a',\n"
        "                    eq_tests=True)\n"
        "                _T = (_iter['T_K_calc'].values - 273.15 if 'T_K_calc' in _iter.columns\n"
        "                      else _iter.iloc[:, 0].values - 273.15)\n"
        "                _P = (_iter['P_kbar_calc'].values if 'P_kbar_calc' in _iter.columns\n"
        "                      else _iter.iloc[:, 1].values)\n"
        "                p['Putirka 2008 opx-liq'] = (_T, _P)\n"
        "                if 'Kd Eq (Put2008+-0.06)' in _iter.columns:\n"
        "                    p['_kd_eq_mask'] = (_iter['Kd Eq (Put2008+-0.06)']\n"
        "                                        .astype(str).str.upper()\n"
        "                                        .str.startswith('Y').values)\n"
        "            except Exception as _e1:\n"
        "                print(f'  Putirka opx-liq iterative skipped ({_e1})')\n"
    )
    if old_a not in src:
        raise RuntimeError('cell 23: (a) eq_tests block not found')
    src = src.replace(old_a, new_a)

    # (b) Add stacked + resampled entries after the forest/boosted loop.
    old_b = (
        "    for tracks in ['opx_liq', 'opx_only']:\n"
        "        use_liq = (tracks == 'opx_liq')\n"
        "        for fam in ['forest', 'boosted']:\n"
        "            name = f'Ours {tracks.replace(\"_\", \"-\")} {fam}'\n"
        "            try:\n"
        "                sT = canonical_model_spec('T_C', tracks, fam, RESULTS)\n"
        "                sP = canonical_model_spec('P_kbar', tracks, fam, RESULTS)\n"
        "                mT = joblib.load(canonical_model_path('T_C', tracks, fam, MODELS, RESULTS))\n"
        "                mP = joblib.load(canonical_model_path('P_kbar', tracks, fam, MODELS, RESULTS))\n"
        "                Xt, _ = build_feature_matrix(scope_train, sT['feature_set'], use_liq=use_liq)\n"
        "                Xp, _ = build_feature_matrix(scope_train, sP['feature_set'], use_liq=use_liq)\n"
        "                p[name] = (mT.predict(Xt), mP.predict(Xp))\n"
        "            except Exception as e:\n"
        "                print(f'  {name} partial/skipped ({e})')\n"
        "                p[name] = (np.full(len(scope_df), np.nan),\n"
        "                           np.full(len(scope_df), np.nan))\n"
    )
    new_b = (
        "    for tracks in ['opx_liq', 'opx_only']:\n"
        "        use_liq = (tracks == 'opx_liq')\n"
        "        for fam in ['forest', 'boosted']:\n"
        "            name = f'Ours {tracks.replace(\"_\", \"-\")} {fam}'\n"
        "            try:\n"
        "                sT = canonical_model_spec('T_C', tracks, fam, RESULTS)\n"
        "                sP = canonical_model_spec('P_kbar', tracks, fam, RESULTS)\n"
        "                mT = joblib.load(canonical_model_path('T_C', tracks, fam, MODELS, RESULTS))\n"
        "                mP = joblib.load(canonical_model_path('P_kbar', tracks, fam, MODELS, RESULTS))\n"
        "                Xt, _ = build_feature_matrix(scope_train, sT['feature_set'], use_liq=use_liq)\n"
        "                Xp, _ = build_feature_matrix(scope_train, sP['feature_set'], use_liq=use_liq)\n"
        "                p[name] = (mT.predict(Xt), mP.predict(Xp))\n"
        "            except Exception as e:\n"
        "                print(f'  {name} partial/skipped ({e})')\n"
        "                p[name] = (np.full(len(scope_df), np.nan),\n"
        "                           np.full(len(scope_df), np.nan))\n"
        "        # v9-optionB: stacked family\n"
        "        s_name = f'Ours {tracks.replace(\"_\", \"-\")} stacked'\n"
        "        try:\n"
        "            from src.data import load_stacked_model as _load_stacked_v9\n"
        "            _mT_s = _load_stacked_v9('T_C',    tracks, MODELS, RESULTS)\n"
        "            _mP_s = _load_stacked_v9('P_kbar', tracks, MODELS, RESULTS)\n"
        "            p[s_name] = (_mT_s.predict(scope_train),\n"
        "                         _mP_s.predict(scope_train))\n"
        "        except Exception as e:\n"
        "            print(f'  {s_name} skipped ({e})')\n"
        "            p[s_name] = (np.full(len(scope_df), np.nan),\n"
        "                         np.full(len(scope_df), np.nan))\n"
        "        # v9-optionB: resampled forest/boosted (same feature set as canonical)\n"
        "        for fam in ['forest', 'boosted']:\n"
        "            r_name = f'Ours {tracks.replace(\"_\", \"-\")} resampled {fam}'\n"
        "            try:\n"
        "                sT = canonical_model_spec('T_C', tracks, fam, RESULTS)\n"
        "                sP = canonical_model_spec('P_kbar', tracks, fam, RESULTS)\n"
        "                _fT = MODELS / f'model_T_C_{tracks}_{fam}_resampled.joblib'\n"
        "                _fP = MODELS / f'model_P_kbar_{tracks}_{fam}_resampled.joblib'\n"
        "                if not (_fT.exists() and _fP.exists()):\n"
        "                    raise FileNotFoundError(f'{_fT.name} or {_fP.name} missing')\n"
        "                mT = joblib.load(_fT); mP = joblib.load(_fP)\n"
        "                Xt, _ = build_feature_matrix(scope_train, sT['feature_set'], use_liq=use_liq)\n"
        "                Xp, _ = build_feature_matrix(scope_train, sP['feature_set'], use_liq=use_liq)\n"
        "                p[r_name] = (mT.predict(Xt), mP.predict(Xp))\n"
        "            except Exception as e:\n"
        "                print(f'  {r_name} skipped ({e})')\n"
        "                p[r_name] = (np.full(len(scope_df), np.nan),\n"
        "                             np.full(len(scope_df), np.nan))\n"
    )
    if old_b not in src:
        raise RuntimeError('cell 23: (b) Ours loop block not found')
    src = src.replace(old_b, new_b)

    # (c) Pop the `_kd_eq_mask` sentinel before metrics and write a
    # second CSV for the Kd-pass subset.
    old_c = (
        "for df_scope, label, fn, slug in SCOPES:\n"
        "    if len(df_scope) < 10:\n"
        "        print(f'\\n[{slug}] SKIPPED: n={len(df_scope)} too small')\n"
        "        continue\n"
        "    print(f'\\n[{slug}] running {len(df_scope)} rows ...')\n"
        "    preds_s = _predict_all_methods(df_scope)\n"
        "    ms = _metrics_df(preds_s, df_scope['T_C'].values,\n"
        "                     df_scope['P_kbar'].values, len(df_scope))\n"
        "    ms.round(3).to_csv(RESULTS / f'nb04_method_benchmark_{slug}.csv', index=False)\n"
        "    _render_three_panel(ms, len(df_scope), label, fn)\n"
        "    _table_cols = ['Method', 'T_n', 'T_RMSE', 'T_R2', 'T_coverage_pct',\n"
        "                   'P_n', 'P_RMSE', 'P_R2', 'P_coverage_pct']\n"
        "    print(f'\\n[{slug}] benchmark table:')\n"
        "    print(ms[_table_cols].round(3).to_string(index=False))\n"
    )
    new_c = (
        "for df_scope, label, fn, slug in SCOPES:\n"
        "    if len(df_scope) < 10:\n"
        "        print(f'\\n[{slug}] SKIPPED: n={len(df_scope)} too small')\n"
        "        continue\n"
        "    print(f'\\n[{slug}] running {len(df_scope)} rows ...')\n"
        "    preds_s = _predict_all_methods(df_scope)\n"
        "    _kd_mask = preds_s.pop('_kd_eq_mask', None)  # v9-optionB\n"
        "    ms = _metrics_df(preds_s, df_scope['T_C'].values,\n"
        "                     df_scope['P_kbar'].values, len(df_scope))\n"
        "    ms.round(3).to_csv(RESULTS / f'nb04_method_benchmark_{slug}.csv', index=False)\n"
        "    _render_three_panel(ms, len(df_scope), label, fn)\n"
        "    _table_cols = ['Method', 'T_n', 'T_RMSE', 'T_R2', 'T_coverage_pct',\n"
        "                   'P_n', 'P_RMSE', 'P_R2', 'P_coverage_pct']\n"
        "    print(f'\\n[{slug}] benchmark table:')\n"
        "    print(ms[_table_cols].round(3).to_string(index=False))\n"
        "    # v9-optionB: Kd-equilibrated subset metrics (Option B scope).\n"
        "    if _kd_mask is not None and _kd_mask.sum() >= 10:\n"
        "        _pe = {name: (np.where(_kd_mask, pT, np.nan),\n"
        "                      np.where(_kd_mask, pP, np.nan))\n"
        "               for name, (pT, pP) in preds_s.items()}\n"
        "        _y_T_e = np.where(_kd_mask, df_scope['T_C'].values, np.nan)\n"
        "        _y_P_e = np.where(_kd_mask, df_scope['P_kbar'].values, np.nan)\n"
        "        ms_e = _metrics_df(_pe, _y_T_e, _y_P_e, int(_kd_mask.sum()))\n"
        "        ms_e.round(3).to_csv(\n"
        "            RESULTS / f'nb04_method_benchmark_{slug}_kdEq.csv', index=False)\n"
        "        print(f'\\n[{slug}] benchmark (Kd eq-pass, n={int(_kd_mask.sum())}):')\n"
        "        print(ms_e[_table_cols].round(3).to_string(index=False))\n"
    )
    if old_c not in src:
        raise RuntimeError('cell 23: (c) scope loop not found')
    src = src.replace(old_c, new_c)

    _set_src(cells[idx], src)
    return True, 'patched'


def patch_cell_27(cells):
    idx = 27
    src = _src(cells[idx])
    if SENTINEL_27 in src:
        return False, 'already patched'

    old = (
        "try:\n"
        "    _pvm_out = _pvm_tb.calculate_opx_liq_press_temp(\n"
        "        opx_comps=_pvm_opx_in, liq_comps=_pvm_liq_in,\n"
        "        equationP='P_Put2008_eq29a', equationT='T_Put2008_eq28a',\n"
        "    )\n"
        "    _pvm_put_T = _pvm_np.asarray(_pvm_out['T_K_calc']) - 273.15\n"
        "    _pvm_put_P = _pvm_np.asarray(_pvm_out['P_kbar_calc'])\n"
        "except Exception as _pvm_e:\n"
        "    print(f'Putirka eq28a/29a run failed: {_pvm_e}')\n"
        "    _pvm_put_T = _pvm_np.full(_pvm_n_full, _pvm_np.nan)\n"
        "    _pvm_put_P = _pvm_np.full(_pvm_n_full, _pvm_np.nan)\n"
        "\n"
        "# ---- Fair subset: rows where Putirka returns finite T AND P -------------\n"
        "_pvm_ok = _pvm_np.isfinite(_pvm_put_T) & _pvm_np.isfinite(_pvm_put_P)\n"
        "_pvm_n_fair = int(_pvm_ok.sum())\n"
        "_pvm_fail_rate = 100.0 * (1.0 - _pvm_n_fair / _pvm_n_full)\n"
        "print(f'Putirka convergence: fair n={_pvm_n_fair}/{_pvm_n_full} '\n"
        "      f'({100.0 * _pvm_n_fair / _pvm_n_full:.1f}% coverage, '\n"
        "      f'failure rate {_pvm_fail_rate:.1f}%)')\n"
    )
    new = (
        "# " + SENTINEL_27 + "\n"
        "try:\n"
        "    _pvm_out = _pvm_tb.calculate_opx_liq_press_temp(\n"
        "        opx_comps=_pvm_opx_in, liq_comps=_pvm_liq_in,\n"
        "        equationP='P_Put2008_eq29a', equationT='T_Put2008_eq28a',\n"
        "        eq_tests=True,\n"
        "    )\n"
        "    _pvm_put_T = _pvm_np.asarray(_pvm_out['T_K_calc']) - 273.15\n"
        "    _pvm_put_P = _pvm_np.asarray(_pvm_out['P_kbar_calc'])\n"
        "    if 'Kd Eq (Put2008+-0.06)' in _pvm_out.columns:\n"
        "        _pvm_kd_eq = (_pvm_out['Kd Eq (Put2008+-0.06)']\n"
        "                      .astype(str).str.upper()\n"
        "                      .str.startswith('Y').values)\n"
        "    else:\n"
        "        _pvm_kd_eq = _pvm_np.ones(_pvm_n_full, dtype=bool)\n"
        "except Exception as _pvm_e:\n"
        "    print(f'Putirka eq28a/29a run failed: {_pvm_e}')\n"
        "    _pvm_put_T = _pvm_np.full(_pvm_n_full, _pvm_np.nan)\n"
        "    _pvm_put_P = _pvm_np.full(_pvm_n_full, _pvm_np.nan)\n"
        "    _pvm_kd_eq = _pvm_np.zeros(_pvm_n_full, dtype=bool)\n"
        "\n"
        "# ---- Option B: Kd-equilibrated AND converged subset ---------------------\n"
        "_pvm_conv = _pvm_np.isfinite(_pvm_put_T) & _pvm_np.isfinite(_pvm_put_P)\n"
        "_pvm_ok = _pvm_kd_eq & _pvm_conv\n"
        "_pvm_n_fair = int(_pvm_ok.sum())\n"
        "_pvm_n_conv = int(_pvm_conv.sum())\n"
        "_pvm_n_kd   = int(_pvm_kd_eq.sum())\n"
        "_pvm_fail_rate = 100.0 * (1.0 - _pvm_n_fair / _pvm_n_full)\n"
        "print(f'Putirka convergence: {_pvm_n_conv}/{_pvm_n_full} converged, '\n"
        "      f'{_pvm_n_kd}/{_pvm_n_full} pass Kd eq, '\n"
        "      f'fair (Option B) n={_pvm_n_fair}/{_pvm_n_full} '\n"
        "      f'({100.0 * _pvm_n_fair / _pvm_n_full:.1f}% coverage, '\n"
        "      f'reject rate {_pvm_fail_rate:.1f}%)')\n"
    )
    if old not in src:
        raise RuntimeError('cell 27: eq_tests block not found')
    src = src.replace(old, new)

    # Update the caption to reflect Option B scope.
    old_cap = (
        "_pvm_caption = (\n"
        "    f'Top: ML predictions on all n={_pvm_n_full} ArcPL opx-bearing experiments '\n"
        "    f'(100% coverage). Middle: ML predictions on the n={_pvm_n_fair} subset '\n"
        "    f'where Putirka 2008 eq 28a/29a converges. Bottom: Putirka predictions '\n"
        "    f'on the same fair subset. Putirka fails to predict on '\n"
        "    f'{_pvm_fail_rate:.1f}% of the full dataset due to equilibrium filter '\n"
        "    f'rejections.'\n"
        ")\n"
    )
    new_cap = (
        "_pvm_caption = (\n"
        "    f'Top: ML predictions on all n={_pvm_n_full} ArcPL opx-bearing experiments '\n"
        "    f'(100% coverage). Middle: ML predictions on the Option B subset '\n"
        "    f'(n={_pvm_n_fair}): rows passing Putirka 2008 Kd(Fe-Mg)+-0.06 '\n"
        "    f'equilibrium AND eq28a/29a convergence. Bottom: Putirka predictions '\n"
        "    f'on the same Option B subset. Putirka-excluded rows (Kd rejection or '\n"
        "    f'non-convergence) total {_pvm_fail_rate:.1f}% of the full set.'\n"
        ")\n"
    )
    if old_cap not in src:
        raise RuntimeError('cell 27: caption block not found')
    src = src.replace(old_cap, new_cap)

    _set_src(cells[idx], src)
    return True, 'patched'


def patch_cell_31(cells):
    idx = 31
    src = _src(cells[idx])
    if SENTINEL_31 in src:
        return False, 'already patched'

    # Iterative (eq28a/29a)
    old_a = (
        "        # (5) opx-liq iterative\n"
        "        try:\n"
        "            _it = _p5_pt.calculate_opx_liq_press_temp(\n"
        "                opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a')\n"
    )
    new_a = (
        "        # (5) opx-liq iterative -- " + SENTINEL_31 + "\n"
        "        try:\n"
        "            _it = _p5_pt.calculate_opx_liq_press_temp(\n"
        "                opx_comps=_opx_in, liq_comps=_liq_in,\n"
        "                equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a',\n"
        "                eq_tests=True)\n"
        "            if 'Kd Eq (Put2008+-0.06)' in _it.columns:\n"
        "                _p5_kd_eq_mask = (_it['Kd Eq (Put2008+-0.06)']\n"
        "                                  .astype(str).str.upper()\n"
        "                                  .str.startswith('Y').values)\n"
        "            else:\n"
        "                _p5_kd_eq_mask = np.ones(len(_it), dtype=bool)\n"
    )
    if old_a not in src:
        raise RuntimeError('cell 31: (a) iterative block not found')
    src = src.replace(old_a, new_a)

    _set_src(cells[idx], src)
    return True, 'patched'


def main():
    if not NB.exists():
        sys.exit(f'NB04 not found at {NB}')
    data = json.loads(NB.read_text(encoding='utf-8'))
    cells = data['cells']

    reports = []
    for fn in (patch_cell_21, patch_cell_23, patch_cell_27, patch_cell_31):
        changed, msg = fn(cells)
        reports.append(f'{fn.__name__}: {msg}')

    NB.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                  encoding='utf-8')
    for r in reports:
        print(r)
    print(f'Wrote {NB}')


if __name__ == '__main__':
    main()
