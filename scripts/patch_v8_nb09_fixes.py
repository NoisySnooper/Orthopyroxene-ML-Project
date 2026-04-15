"""v8 nb09 fixes: F6 (strategy name + feature_set filter),
F7 (canonical figure roster + captions), F8 (OOD percentile threshold),
F10 (pipeline health check cell). F9 (structured markdown) is deferred
to keep this patch surgical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb09_manuscript_compilation.ipynb"

V8_SENTINEL = 'v8-fix'


# ---------------------------------------------------------------------
# F6: Cell 17 — per-target feature_set filter (opx_liq winners).
# ---------------------------------------------------------------------
CELL17_OLD = """pooled_rf = pooled[(pooled['model_name'] == 'RF') &
                    (pooled['feature_set'] == WIN_FEAT)].copy()"""

CELL17_NEW = """# v8-fix: nb05 pool contains rows from multiple feature_sets; keep the
# full RF slice so downstream lookup sees whichever feature_set nb05
# actually wrote. (Historic bug: filtering to a fixed WIN_FEAT dropped
# every row when the canonical winner differed from the nb05 training
# feature set.)
pooled_rf = pooled[pooled['model_name'] == 'RF'].copy()
_pool_feats = sorted(pooled_rf['feature_set'].unique().tolist())
_pool_strats = sorted(pooled_rf['strategy'].unique().tolist())
print(f'v8-check: pooled_rf has {len(pooled_rf)} rows; '
      f'feature_sets={_pool_feats}; strategies={_pool_strats}')"""


# ---------------------------------------------------------------------
# F6: Cell 22 — rename Gridded-PT -> TargetBinKFold in the manifest.
# ---------------------------------------------------------------------
CELL22_STRAT_OLD1 = "_pooled_rmse('Gridded-PT',    'T_C'),"
CELL22_STRAT_NEW1 = "_pooled_rmse('TargetBinKFold', 'T_C'),  # v8-fix: renamed from Gridded-PT"
CELL22_STRAT_OLD2 = "_pooled_rmse('Gridded-PT',    'P_kbar'),"
CELL22_STRAT_NEW2 = "_pooled_rmse('TargetBinKFold', 'P_kbar'),  # v8-fix: renamed from Gridded-PT"


# ---------------------------------------------------------------------
# F7: Cell 22 — replace the legacy figure roster with v7/v8 canonical set.
# ---------------------------------------------------------------------
CANONICAL_FIGURES_SRC = '''CANONICAL_FIGURES = [
    {'num': 1,  'stem': 'fig_nb02_clusters',
     'caption': 'PCA decomposition of opx composition space, colored by k-means cluster.'},
    {'num': 2,  'stem': 'fig_nb03c_multiseed_rmse',
     'caption': 'Multi-seed test RMSE (mean +/- std over 20 seeds) per model x feature set.'},
    {'num': 3,  'stem': 'fig_nb04_method_benchmark_paired',
     'caption': 'Method benchmark on ArcPL three-phase paired scope. T RMSE (A), P RMSE (B, sorted independently), coverage (C).'},
    {'num': 4,  'stem': 'fig_nb04_diagnostic_encyclopedia_T',
     'caption': 'Per-method predicted-vs-observed T scatter, ArcPL three-phase paired scope.'},
    {'num': 5,  'stem': 'fig_nb04_diagnostic_encyclopedia_P',
     'caption': 'Per-method predicted-vs-observed P scatter, same scope.'},
    {'num': 6,  'stem': 'fig_nb04_arcpl_opx_liq_scatter',
     'caption': 'ArcPL external validation (n=204): forest and boosted families pred vs obs scatter + residuals.'},
    {'num': 7,  'stem': 'fig_nb04_h2o_sensitivity',
     'caption': 'Performance stratified by H2O reporting method (measured vs VBD).'},
    {'num': 8,  'stem': 'fig_nb04_opx_only_comparison',
     'caption': 'Eight-method opx-only comparison: 4 ML (ours, forest/boosted x opx-liq/opx-only) + 4 Putirka variants.'},
    {'num': 9,  'stem': 'fig_nb05_generalization',
     'caption': 'Generalization under three CV strategies: LOSO, Cluster-KFold, TargetBinKFold. Random-split reference lines dashed.'},
    {'num': 10, 'stem': 'fig_nb06_shap_P_beeswarm',
     'caption': 'SHAP attributions for P. Rows = features; dots = per-experiment contributions; color = feature value.'},
    {'num': 11, 'stem': 'fig_nb06_shap_T_beeswarm',
     'caption': 'Same as Fig 10 for T.'},
    {'num': 12, 'stem': 'fig_nb07_conformal_coverage',
     'caption': 'Conformal interval coverage at nominal 0.68 / 0.80 / 0.90 / 0.95 levels.'},
    {'num': 13, 'stem': 'fig_nb08_twopx_1to1',
     'caption': 'Cross-mineral validation: ML opx-only vs Jorgenson cpx-only on LEPR paired-pyroxene experiments.'},
]
'''

CELL22_ROSTER_OLD = """expected_figures = [
    'fig01_pt_distribution.png',
    'fig02_pca_biplot.png',
    'fig03_pred_vs_obs_P.png',
    'fig04_pred_vs_obs_T.png',
    'fig05_model_comparison.png',
    'fig06_loso_pooled.png',
    'fig07_shap_P_beeswarm.png',
    'fig08_shap_T_beeswarm.png',
    'fig09_shap_P_dependence.png',
    'fig09_shap_T_dependence.png',
    'fig10_putirka_vs_ml.png',
    'fig11_bias_correction_residuals.png',
    'fig12_natural_samples_geotherm.png',
    'fig13_ood_score_vs_residual.png',
    'fig14_mc_vs_iqr_uncertainty.png',
]"""

CELL22_ROSTER_NEW = f"""# v8-fix: canonical v7 figure roster (replaces legacy fig01-fig14 names)
{CANONICAL_FIGURES_SRC}
expected_figures = [f"{{spec['stem']}}.png" for spec in CANONICAL_FIGURES]"""


# ---------------------------------------------------------------------
# F8: Cell 10 — replace contamination='auto' with percentile threshold
# so the OOD subset is a meaningful minority.
# ---------------------------------------------------------------------
CELL10_ISO_OLD = """iso = IsolationForest(n_estimators=300, contamination='auto',
                      random_state=SEED_MODEL, n_jobs=-1)
iso.fit(X_train)

# IsolationForest in T feature space (= X_arcpl_T). The canonical RF T and P
# models get their own target-specific matrices for prediction.
arcpl_score = iso.score_samples(X_arcpl_T)   # higher = more in-distribution
arcpl_label = iso.predict(X_arcpl_T)         # +1 inlier, -1 outlier"""

CELL10_ISO_NEW = """# v8-fix: contamination='auto' often labels 0 or ~50% outliers on this
# composition space. Score the training set to pick a data-driven cutoff
# (bottom 10% of TRAINING scores) so the ArcPL OOD bucket is a sensible
# minority (~5-25%) rather than 0% or 100%.
iso = IsolationForest(n_estimators=300, contamination='auto',
                      random_state=SEED_MODEL, n_jobs=-1)
iso.fit(X_train)
_train_score = iso.score_samples(X_train)
_ood_threshold = float(np.percentile(_train_score, 10))
arcpl_score = iso.score_samples(X_arcpl_T)
arcpl_label = np.where(arcpl_score < _ood_threshold, -1, 1)
print(f'v8-check OOD: training 10th percentile cutoff={_ood_threshold:.3f} | '
      f'ArcPL score min={arcpl_score.min():.3f} max={arcpl_score.max():.3f} | '
      f'OOD n={int((arcpl_label == -1).sum())}/{len(arcpl_label)} '
      f'({100*(arcpl_label == -1).mean():.1f}%)')"""


# ---------------------------------------------------------------------
# F10: pipeline_health.txt cell — appended at end of nb09.
# ---------------------------------------------------------------------
F10_HEALTH_CODE = '''# v8-fix: pipeline health check. Writes logs/pipeline_health.txt with
# PASS/FAIL per upstream notebook output. Intended to surface silent
# breakage before the manuscript is compiled.
from datetime import datetime as _dt
import json as _json

_health = []

def _check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    line = f'[{status}] {name}'
    if detail:
        line += f' -- {detail}'
    _health.append(line)
    return ok

# NB01
_check('NB01 cleaned core',
       (DATA_PROC / 'opx_clean_core.parquet').exists())
# NB02
_check('NB02 clusters fig', (FIGURES / 'fig_nb02_clusters.png').exists())
# NB03
_winners_path = RESULTS / 'nb03_per_family_winners.json'
_check('NB03 winners json', _winners_path.exists())
if _winners_path.exists():
    _w = _json.loads(_winners_path.read_text())
    _n = sum(len(_w.get(f, {})) for f in ['forest_family', 'boosted_family'])
    _check('NB03 winner count == 8', _n == 8, f'found {_n}')
# NB04 core outputs
for _f in ['nb04_method_benchmark_arcpl_paired.csv',
           'nb04_arcpl_opx_liq_metrics.csv',
           'nb04_arcpl_h2o_stratified_metrics.csv',
           'nb04_opx_only_comparison_all.csv']:
    _check(f'NB04 {_f}', (RESULTS / _f).exists())
for _f in ['fig_nb04_method_benchmark_paired.png',
           'fig_nb04_diagnostic_encyclopedia_T.png',
           'fig_nb04_diagnostic_encyclopedia_P.png',
           'fig_nb04_arcpl_opx_liq_scatter.png',
           'fig_nb04_h2o_sensitivity.png',
           'fig_nb04_opx_only_comparison.png']:
    _check(f'NB04 {_f}', (FIGURES / _f).exists())
# NB04 specific: Putirka opx in benchmark
_bench_path = RESULTS / 'nb04_method_benchmark_arcpl_paired.csv'
if _bench_path.exists():
    _b = pd.read_csv(_bench_path)
    _has_put_opx = any('Putirka 2008 opx' in m for m in _b['Method'])
    _check('NB04 benchmark includes Putirka opx methods', _has_put_opx,
           f'methods: {_b["Method"].tolist()}')
# NB05
_check('NB05 LOSO pooled', (RESULTS / 'nb05_loso_pooled.csv').exists())
_check('NB05 generalization fig',
       (FIGURES / 'fig_nb05_generalization.png').exists())
# NB06
_check('NB06 SHAP P', (FIGURES / 'fig_nb06_shap_P_beeswarm.png').exists())
_check('NB06 SHAP T', (FIGURES / 'fig_nb06_shap_T_beeswarm.png').exists())
# NB07
_check('NB07 bias correction null',
       (RESULTS / 'nb07_bias_correction_null_result.csv').exists() or
       (RESULTS / 'nb07_ab_test_main.csv').exists())
# NB08
_check('NB08 predictions',
       (RESULTS / 'nb08_natural_predictions.csv').exists() or
       (RESULTS / 'nb08_disagreement_metrics.csv').exists())
# NB09 manifest sanity
_mani_path = RESULTS / 'manuscript_key_results.csv'
if _mani_path.exists():
    _mani = pd.read_csv(_mani_path)
    _nan_cols = [c for c in _mani.columns if pd.isna(_mani[c].iloc[0])]
    _check('NB09 manifest has no NaN fields', len(_nan_cols) == 0,
           f'NaN fields: {_nan_cols}' if _nan_cols else 'all populated')
# NB09 OOD separation
_ood_path = RESULTS / 'nb10_ood_isoforest.csv'
if _ood_path.exists():
    _ood = pd.read_csv(_ood_path)
    _distinct = False
    for _t in ['T_C', 'P_kbar']:
        _sub = _ood[_ood['target'] == _t]
        if len(_sub) >= 2 and 'rmse' in _sub.columns:
            _vals = _sub['rmse'].round(3).unique()
            if len(_vals) > 1:
                _distinct = True
                break
    _check('NB09 OOD subsets have distinct RMSE', _distinct,
           'at least one target shows different all/in/out metrics')

_n_pass = sum(1 for x in _health if x.startswith('[PASS]'))
_n_fail = sum(1 for x in _health if x.startswith('[FAIL]'))
_summary = f'SUMMARY: {_n_pass} PASS, {_n_fail} FAIL of {len(_health)} checks'
print(chr(10).join(_health))
print(_summary)

_health_path = LOGS / 'pipeline_health.txt'
_health_path.parent.mkdir(parents=True, exist_ok=True)
_health_path.write_text(
    f'Pipeline health report -- {_dt.now().isoformat()}\\n'
    + '=' * 60 + '\\n'
    + chr(10).join(_health) + '\\n'
    + _summary + '\\n'
)
print(f'Wrote {_health_path}')
if _n_fail > 0:
    print(f'WARNING: {_n_fail} pipeline checks failed. See pipeline_health.txt.')
'''


# ---------------------------------------------------------------------
def _patch_cell(nb, idx, old, new, label):
    src = nb.cells[idx].source
    if new in src:
        print(f'{label}: already patched')
        return False
    if old not in src:
        print(f'{label}: anchor not found')
        return False
    nb.cells[idx].source = src.replace(old, new, 1)
    nb.cells[idx].outputs = []
    nb.cells[idx].execution_count = None
    print(f'{label}: patched')
    return True


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # F6 cell 17
    _patch_cell(nb, 17, CELL17_OLD, CELL17_NEW, 'F6 cell 17 filter')
    # F6 cell 22 strategy rename
    _patch_cell(nb, 22, CELL22_STRAT_OLD1, CELL22_STRAT_NEW1,
                'F6 cell 22 strategy T')
    _patch_cell(nb, 22, CELL22_STRAT_OLD2, CELL22_STRAT_NEW2,
                'F6 cell 22 strategy P')
    # F7 cell 22 figure roster
    _patch_cell(nb, 22, CELL22_ROSTER_OLD, CELL22_ROSTER_NEW,
                'F7 cell 22 figure roster')
    # F8 cell 10 IsolationForest threshold
    _patch_cell(nb, 10, CELL10_ISO_OLD, CELL10_ISO_NEW,
                'F8 cell 10 IsoForest threshold')

    # F10: append pipeline health cell at end if not present
    _health_sentinel = 'v8-fix: pipeline health check'
    already = any(_health_sentinel in c.source for c in nb.cells
                  if c.cell_type == 'code')
    if not already:
        import uuid as _uuid
        new_cell = nbformat.v4.new_code_cell(F10_HEALTH_CODE)
        new_cell['id'] = _uuid.uuid4().hex[:8]
        nb.cells.append(new_cell)
        print('F10 health cell: appended')
    else:
        print('F10 health cell: already present')

    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'nb09 written: {len(nb.cells)} cells.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
