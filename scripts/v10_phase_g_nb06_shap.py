#!/usr/bin/env python3
"""Phase G.3: NB06 SHAP rebuild for opx_liq canonical models.

Replaces the v8 nb06_shap_analysis.ipynb with a reproducible script that
runs three SHAP strategies on the three opx_liq canonical cells:

  * ElasticNet / raw / T_C     -> LinearExplainer (exact for linear models)
  * ElasticNet / raw / P_kbar  -> LinearExplainer
  * MLP        / raw / P_kbar  -> KernelExplainer (model-agnostic sampling)

Tree SHAP for RF/ERT/XGB/CatBoost/GB/LightGBM is useful for auxiliary
inspection but the *canonical* v10 opx_liq winners are linear/MLP.
Tree-SHAP for the aggregate tree cells is therefore appended as an
additional panel for interpretability (CatBoost/raw/P_kbar and XGB/alr/T_C
as the best-tree cells per the v10 Optuna Pareto).

Outputs:
    results/v10_opx_liq_shap_values.npz         (per-cell SHAP arrays)
    results/v10_opx_liq_shap_importance.csv     (feature x cell mean |SHAP|)
    figures/fig27_shap_summary_opx_liq.{pdf,png}
    tables/S8_7_shap_top_features_opx_liq.{md,csv}
    logs/v10_phase_g_nb06_shap.log
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import FIGURES, LOGS, RESULTS, SEED_BOOTSTRAP
from src.models import build_model
from src.v10_phase_c_analysis import load_best_params, prepare_train_test

warnings.filterwarnings('ignore')

LOG_PATH = LOGS / 'v10_phase_g_nb06_shap.log'
MANIFEST = RESULTS / 'v10_optuna_best_params_opx.json'

CELLS = [
    # (model, target, track, feature_set, explainer_kind)
    ('ElasticNet', 'T_C',    'opx_liq', 'raw',  'linear'),
    ('ElasticNet', 'P_kbar', 'opx_liq', 'raw',  'linear'),
    ('MLP',        'P_kbar', 'opx_liq', 'raw',  'kernel'),
    ('CatBoost',   'P_kbar', 'opx_liq', 'raw',  'tree'),
    ('XGB',        'T_C',    'opx_liq', 'alr',  'tree'),
]


def _log(msg, fh):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def run_explainer(model_name, est, X_tr, X_te, kind, seed=42):
    """Return shap_values array shape (n_samples, n_features)."""
    import shap
    if kind == 'linear':
        # ElasticNet ships as a Pipeline(scaler -> enet). LinearExplainer
        # needs the bare linear estimator plus the scaled background +
        # scaled evaluation matrix.
        if hasattr(est, 'steps'):
            scaler = est.named_steps['scaler']
            linear = est.named_steps[list(est.named_steps.keys())[-1]]
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)
            expl = shap.LinearExplainer(linear, X_tr_s)
            return np.asarray(expl.shap_values(X_te_s))
        expl = shap.LinearExplainer(est, X_tr)
        return np.asarray(expl.shap_values(X_te))
    if kind == 'tree':
        # CatBoost tree SHAP is cleanest via the model's own interface; XGB/
        # LGB/RF use shap.TreeExplainer. CatBoost registered via shap also
        # works.
        if model_name == 'CatBoost':
            expl = shap.TreeExplainer(est)
        else:
            expl = shap.TreeExplainer(est)
        return np.asarray(expl.shap_values(X_te))
    if kind == 'kernel':
        # Kernel SHAP with a K-means summary background (50 centroids).
        bg = shap.kmeans(X_tr, 50)
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(X_te),
                             size=min(80, len(X_te)), replace=False)
        expl = shap.KernelExplainer(est.predict, bg)
        vals = expl.shap_values(X_te[sub_idx], nsamples=100, silent=True)
        vals = np.asarray(vals)
        # Project back to full test set: fill zero for non-sampled rows so the
        # aggregate mean|SHAP| ranking is based on the sampled rows only.
        full = np.zeros((len(X_te), X_te.shape[1]), dtype=float)
        full[sub_idx] = vals
        # Return only the sampled rows' SHAP values; importance aggregation
        # below averages over non-zero rows.
        mask = np.zeros(len(X_te), dtype=bool)
        mask[sub_idx] = True
        return full, mask
    raise ValueError(f'unknown explainer kind: {kind}')


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, 'w', encoding='utf-8')
    try:
        _log(f'START cells={CELLS}', fh)
        best = load_best_params(MANIFEST)

        per_cell_importance = []
        per_cell_shap = {}

        for (model_name, target, track, feature_set, kind) in CELLS:
            key = (model_name, target, track, feature_set)
            best_params = best[key]['best_params']
            pt = prepare_train_test(track, target, feature_set)
            X_tr, y_tr = pt['X_tr'], pt['y_tr']
            X_te = pt['X_te']
            feat_names = pt['feat_names']

            t0 = time.time()
            est = build_model(model_name, best_params, seed=42)
            est.fit(X_tr, y_tr)
            _log(f'[{model_name}/{feature_set}/{target}] fit elapsed='
                 f'{time.time()-t0:.1f}s X_te={X_te.shape} kind={kind}', fh)

            t0 = time.time()
            result = run_explainer(model_name, est, X_tr, X_te, kind)
            if isinstance(result, tuple):
                shap_vals, mask = result
                n_eff = int(mask.sum())
            else:
                shap_vals = result
                n_eff = len(X_te)
                mask = np.ones(len(X_te), dtype=bool)
            _log(f'  shap elapsed={time.time()-t0:.1f}s values.shape={shap_vals.shape} '
                 f'n_effective={n_eff}', fh)

            mean_abs = np.abs(shap_vals[mask]).mean(axis=0)
            for j, f in enumerate(feat_names):
                per_cell_importance.append({
                    'cell':         f'{model_name}/{feature_set}/{target}',
                    'model':        model_name,
                    'target':       target,
                    'feature_set':  feature_set,
                    'feature':      f,
                    'mean_abs_shap': float(mean_abs[j]),
                })
            per_cell_shap[f'{model_name}_{feature_set}_{target}'] = shap_vals

        imp_df = pd.DataFrame(per_cell_importance)
        out_csv = RESULTS / 'v10_opx_liq_shap_importance.csv'
        imp_df.to_csv(out_csv, index=False)
        _log(f'wrote {out_csv} rows={len(imp_df)}', fh)

        out_npz = RESULTS / 'v10_opx_liq_shap_values.npz'
        np.savez_compressed(out_npz, **per_cell_shap)
        _log(f'wrote {out_npz} keys={list(per_cell_shap.keys())}', fh)

        # --- Figure -----------------------------------------------------------
        import matplotlib.pyplot as plt
        cells_unique = imp_df['cell'].unique()
        fig, axes = plt.subplots(1, len(cells_unique),
                                 figsize=(4 * len(cells_unique), 5),
                                 squeeze=False)
        for ax, cell_key in zip(axes[0], cells_unique):
            sub = imp_df[imp_df.cell == cell_key].copy()
            sub = sub.sort_values('mean_abs_shap', ascending=True).tail(15)
            ax.barh(sub['feature'], sub['mean_abs_shap'],
                    color='#0072B2', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('mean |SHAP|')
            ax.set_title(cell_key, fontsize=9)
            ax.grid(axis='x', linestyle=':', alpha=0.4)
        fig.suptitle('opx-liq SHAP feature importance (top 15 per cell)',
                     fontsize=11)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGURES / f'fig27_shap_summary_opx_liq.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        _log(f'wrote figures/fig27_shap_summary_opx_liq.{{pdf,png}}', fh)

        # --- SI table ---------------------------------------------------------
        tables_dir = PROJECT_ROOT / 'tables'
        tables_dir.mkdir(exist_ok=True)
        md_lines = ['# Table S8.7: opx-liq SHAP top features per canonical cell',
                    '',
                    'Top 10 features by mean absolute SHAP value per canonical '
                    'cell. Linear cells use `shap.LinearExplainer` (exact for '
                    'linear models); tree cells use `shap.TreeExplainer`; the '
                    'MLP cell uses `shap.KernelExplainer` with a k=50 KMeans '
                    'background and a random 80-sample subset of the test set '
                    '(seed = SEED_BOOTSTRAP).', '']
        top10 = (imp_df.sort_values(['cell', 'mean_abs_shap'],
                                     ascending=[True, False])
                       .groupby('cell', as_index=False).head(10)
                       .reset_index(drop=True))
        md_lines.append('| Cell | Rank | Feature | Mean abs SHAP |')
        md_lines.append('|---|---|---|---|')
        for cell_key in cells_unique:
            sub = top10[top10.cell == cell_key].reset_index(drop=True)
            for i, r in sub.iterrows():
                md_lines.append(f'| {cell_key} | {i+1} | {r.feature} | '
                                f'{r.mean_abs_shap:.4f} |')
        md_path = tables_dir / 'S8_7_shap_top_features_opx_liq.md'
        csv_path = tables_dir / 'S8_7_shap_top_features_opx_liq.csv'
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')
        top10.to_csv(csv_path, index=False)
        _log(f'wrote {md_path}', fh)
        _log(f'wrote {csv_path}', fh)

        _log('DONE', fh)
        return 0
    finally:
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
