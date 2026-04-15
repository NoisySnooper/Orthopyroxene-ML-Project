"""Final v6 verification: notebook validity, artifact inventory, markdown ratio.
"""
from __future__ import annotations
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent.parent

LIVE = [
    'nb01_data_cleaning', 'nb02_eda_pca', 'nb03_baseline_models',
    'nb04_putirka_benchmark', 'nb04b_lepr_arcpl_validation',
    'nb05_loso_validation', 'nb06_shap_analysis', 'nb07_bias_correction',
    'nb08_natural_twopx', 'nb09_manuscript_compilation', 'nbF_figures',
]

total_md = total_code = 0
all_ok = True
print('=== Notebook validity + md/code ratio ===')
for name in LIVE:
    try:
        nb = nbformat.read(str(ROOT / 'notebooks' / f'{name}.ipynb'), as_version=4)
        nbformat.validate(nb)
        md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
        code = sum(1 for c in nb.cells if c.cell_type == 'code')
        total_md += md; total_code += code
        ratio = md / max(code, 1)
        print(f'  {name:<32s} md={md:>2} code={code:>2} ratio={ratio:.2f}')
    except Exception as e:
        all_ok = False
        print(f'  {name:<32s} INVALID: {e}')

print(f'\nTotals: md={total_md} code={total_code} ratio={total_md/max(total_code,1):.2f}')
print(f'All notebooks valid: {all_ok}')

req_results = [
    'nb03_winning_configurations.json', 'nb03_multi_seed_results.csv',
    'nb04b_arcpl_predictions.csv', 'nb04b_arcpl_metrics.csv',
    'nb07_conformal_qhat.json',
    'nb08_natural_predictions.csv', 'nb08_cross_mineral_agreement.csv',
    'nb11_model_family_ceiling.csv',
    'nb10_two_pyroxene_benchmark.csv', 'nb10_h2o_engineered_test_rmse.csv',
    'nb10_iqr_uncertainty.csv', 'nb10_analytical_uncertainty.csv',
    'nb10_ood_paradox_methods.csv',
]
missing_r = [f for f in req_results if not (ROOT / 'results' / f).exists()]
print(f'\nMissing results CSVs: {missing_r if missing_r else "NONE"}')

req_models = [
    'model_RF_T_C_opx_liq.joblib', 'model_RF_P_kbar_opx_liq.joblib',
    'model_RF_T_C_opx_liq_H2O.joblib', 'model_RF_P_kbar_opx_liq_H2O.joblib',
    'model_IsolationForest_opx_liq.joblib',
]
missing_m = [m for m in req_models if not (ROOT / 'models' / m).exists()]
print(f'Missing models: {missing_m if missing_m else "NONE"}')

req_figs = [
    'fig_nb08_twopx_1to1.png', 'fig_nb11_model_family_ceiling.png',
    'fig_nb02_clusters.png',
]
missing_f = [f for f in req_figs if not (ROOT / 'figures' / f).exists()]
print(f'Missing figures: {missing_f if missing_f else "NONE"}')

print('\n=== Summary ===')
print(f'Notebooks in notebooks/: {len(list((ROOT/"notebooks").glob("*.ipynb")))}')
print(f'Archived v5/v4 notebooks: {len(list((ROOT/"archive").glob("*.ipynb")))}')
print(f'Result CSVs: {len(list((ROOT/"results").glob("*.csv")))}')
print(f'Figures: {len(list((ROOT/"figures").glob("*.png")))}')
print(f'Models: {len(list((ROOT/"models").glob("*.joblib")))}')
