"""Insert a Purpose/Inputs/Outputs/Canonical-decisions markdown block after
the title cell of each target notebook. Idempotent: if a cell tagged
`v5-docs-template` already exists in the notebook, we skip.

Target notebooks: nb02, nb03, nb04, nb04b, nb06, nb07, nb08, nb10.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / 'notebooks'

DOCS = {
    'nb02_eda_pca': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Exploratory data analysis of the cleaned opx-only and "
        "opx-liq tables: univariate distributions, oxide correlation heatmap, "
        "principal components of the training feature space, and k-means "
        "chemical clusters used to sanity-check provenance.",
        "**Inputs:** `data/processed/opx_clean_core.csv`, "
        "`data/processed/opx_clean_opx_liq.parquet`.",
        "**Outputs:** `figures/fig_eda_distributions.png`, "
        "`figures/fig_eda_correlation.png`, `figures/fig02_pca_biplot.png`, "
        "`results/nb02_pca_loadings.csv` (if produced).",
        "**Canonical decisions:** This notebook does not set any canonical "
        "model choice; it is descriptive only. Downstream notebooks must not "
        "read cluster labels from here without first re-validating.",
    ),
    'nb03_baseline_models': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Multi-seed baseline-model evaluation. Tests three "
        "feature engineering methods (raw, alr, pwlr) across four tree "
        "ensembles (RF, ERT, XGB, GB) for opx-only and opx-liq tracks under "
        "a 10-fold StratifiedGroupKFold CV, repeated across 5 split seeds.",
        "**Inputs:** `data/processed/opx_clean_opx_liq.parquet`, "
        "`data/processed/opx_clean_core.csv`, split-index arrays.",
        "**Outputs:** `results/nb03_multi_seed_results.csv`, "
        "`results/nb03_multi_seed_summary.csv`, "
        "`results/nb03_winning_configurations.json` "
        "(read by every downstream notebook via `load_winning_config`), "
        "`models/model_<family>_<target>_<track>_<feat>.joblib`.",
        "**Canonical decisions:** `nb03_winning_configurations.json` sets "
        "the **global** feature set (`WIN_FEAT`) that all downstream work "
        "consumes. Do not hard-code feature names elsewhere - route through "
        "`WIN_FEAT` and `canonical_model_filename`.",
    ),
    'nb04_putirka_benchmark': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Benchmark the canonical opx-liq ML model against the "
        "Putirka (2008) thermobarometer using `Thermobar`. Reports a unified "
        "table with three scopes (full / Putirka-valid subset / "
        "intersection), paired Wilcoxon tests on |residuals|, Delta-RMSE "
        "bootstrap CIs, and a failure-mode Mann-Whitney on Putirka-fail vs "
        "Putirka-valid.",
        "**Inputs:** `data/processed/opx_clean_opx_liq.parquet`, "
        "`results/nb03_winning_configurations.json`, canonical RF models.",
        "**Outputs:** `results/nb04_putirka_comparison.csv` (+ `_fair`, "
        "`_practical`, `_unified`), `results/nb04_putirka_vs_ml.csv` "
        "(legacy schema for NBF fig10), "
        "`results/nb04_putirka_vs_ml_wilcoxon.csv`, "
        "`figures/fig_nb04_putirka_comparison.{png,pdf}`.",
        "**Canonical decisions:** Uses Putirka Option 2 (passing true "
        "observed P and T as inputs) as the canonical Putirka scope for the "
        "legacy comparison CSV. The OPERATOR DECISION block (A/B/C) controls "
        "which scope becomes the manuscript headline.",
    ),
    'nb04b_lepr_arcpl_validation': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** External validation of the canonical opx-liq ML model "
        "on the ArcPL hydrous subset of LEPR (measured vs VBD-imputed "
        "experiments). Reports RMSE with bootstrap CIs for all / measured / "
        "VBD scopes plus Mann-Whitney on |residuals| between measured and "
        "VBD experiments.",
        "**Inputs:** "
        "`data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx`, "
        "canonical opx-liq RF models, `WIN_FEAT`.",
        "**Outputs:** `results/nb04b_arcpl_metrics.csv`, "
        "`results/nb04b_arcpl_predictions.csv` (consumed by NB10), "
        "`figures/fig_nb04b_arcpl_4panel.{png,pdf}`.",
        "**Canonical decisions:** The ArcPL predictions CSV is the canonical "
        "natural-hydrous validation surface consumed by NB10's H2O and OOD "
        "analyses. OPERATOR DECISION chooses the reported headline scope.",
    ),
    'nb06_shap_analysis': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** SHAP analysis on the canonical opx-liq RF models. "
        "Beeswarm plots, dependence plots for top features, and a "
        "feature-importance ranking used to verify that the model learned "
        "geochemically plausible signals.",
        "**Inputs:** canonical opx-liq RF models, `WIN_FEAT`, "
        "`data/processed/opx_clean_opx_liq.parquet`.",
        "**Outputs:** `results/nb06_shap_feature_importance.csv`, "
        "`figures/fig07_shap_*`, `figures/fig08_shap_*`, "
        "`figures/fig09_shap_*`.",
        "**Canonical decisions:** Does not set canonical choices. Results "
        "are descriptive. If a SHAP-based feature pruning is adopted, it "
        "must be re-routed through NB03's feature-set winner selection.",
    ),
    'nb07_bias_correction': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Bias correction and uncertainty quantification. "
        "Four-track A/B test of piecewise linear residual corrections (plus "
        "paired-bootstrap null-result check), QRF coverage leakage A/B, and "
        "multi-alpha split-conformal calibration (primary: 90% nominal).",
        "**Inputs:** canonical opx-liq RF models, "
        "`nb03_multi_seed_results.csv` (for RF hyperparameters), split "
        "indices.",
        "**Outputs:** `results/nb07_ab_test_report.csv`, "
        "`results/nb07_piecewise_params.json`, "
        "`results/nb07_qrf_ab_coverage.csv`, "
        "`results/nb07_test_predictions.csv`, "
        "`results/nb07_conformal_multi_alpha.csv`, "
        "`results/nb07_conformal_qhat.json`, "
        "`results/nb07_bias_correction_null_result.csv`, "
        "`models/model_QRF_*_opx_liq.joblib`.",
        "**Canonical decisions:** Split-conformal is the **primary** "
        "uncertainty estimator for NB08 / NB10 / manuscript. The legacy "
        "single-alpha JSON format is preserved (populated from alpha=0.10) "
        "for back-compat. Piecewise bias correction is reported as "
        "supplementary because the Delta-RMSE 95% CI contains zero.",
    ),
    'nb08_natural_samples': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Apply the canonical opx-liq ML model to natural "
        "samples (Lin et al. 2023 NE China peridotites), assign confidence "
        "tiers from max(T_CV, P_CV), overlay H&C 2011 geotherm, and flag "
        "IsolationForest OOD samples as `unreliable`.",
        "**Inputs:** `data/natural/lin_2023_ne_china_peridotites.csv`, "
        "canonical opx-liq RF / QRF models, `WIN_FEAT`, "
        "`results/nb07_conformal_qhat.json`.",
        "**Outputs:** `results/nb08_natural_predictions_all.csv`, "
        "`results/nb08_natural_predictions_filtered.csv` (legacy CV<0.3), "
        "`figures/fig12_natural_samples_geotherm.png`.",
        "**Canonical decisions:** Confidence tiers use thresholds "
        "`CV_THRESHOLDS = {'high': 0.20, 'moderate': 0.35, 'low': 0.50}`. "
        "Samples above low are labeled `unreliable`; IsoForest outliers are "
        "also labeled `unreliable`. OPERATOR DECISION chooses A (opx-only "
        "tiered) / B (opx-liq) / C (hybrid).",
    ),
    'nb10_extended_analyses': (
        "## Purpose / Inputs / Outputs / Canonical decisions",
        "**Purpose:** Downstream analyses on top of the canonical RF / QRF: "
        "two-pyroxene benchmark (moved to NB10b), H2O dependence + "
        "engineered-feature retrain, IQR per-sample uncertainty, analytical "
        "noise propagation, IsolationForest OOD filter, and OOD-method "
        "paradox diagnosis.",
        "**Inputs:** `data/processed/opx_clean_opx_liq.parquet`, canonical "
        "RF + QRF opx-liq models, `results/nb04b_arcpl_predictions.csv`, "
        "`results/nb03_winning_configurations.json`.",
        "**Outputs:** `results/nb10_two_pyroxene_benchmark.csv`, "
        "`results/nb10_h2o_dependence.csv`, "
        "`results/nb10_h2o_engineered_arcpl.csv`, "
        "`results/nb10_h2o_engineered_test_rmse.csv`, "
        "`results/nb10_iqr_uncertainty.csv`, "
        "`results/nb10_analytical_uncertainty.csv`, "
        "`results/nb10_mc_per_sample.csv`, "
        "`results/nb10_ood_isoforest.csv`, "
        "`results/nb10_arcpl_ood_scores.csv`, "
        "`results/nb10_ood_paradox_methods.csv`, "
        "`results/nb10_ood_scores_all_methods.csv`, "
        "`models/model_IsolationForest_opx_liq.joblib`, "
        "`models/model_RF_*_opx_liq_H2O.joblib`.",
        "**Canonical decisions:** H2O-engineered retrain is a *sensitivity "
        "check*, not a canonical replacement - canonical RF remains the one "
        "built in NB03. OOD-method OPERATOR DECISION (A/B/C) picks the "
        "score used for NB08 downstream filtering.",
    ),
}


def render(title, *lines):
    return title + "\n\n" + "\n\n".join(lines) + "\n"


def ensure_template(nb_path: Path, doc: tuple) -> bool:
    nb = nbformat.read(nb_path, as_version=4)
    # idempotency: skip if any cell already tagged as the template
    for c in nb.cells:
        if c.get('metadata', {}).get('id') == 'v5-docs-template':
            return False

    src = render(*doc)
    template_cell = nbformat.v4.new_markdown_cell(src, metadata={'id': 'v5-docs-template'})
    # insert after the first (title) cell
    nb.cells.insert(1, template_cell)
    nbformat.write(nb, nb_path)
    return True


def main() -> None:
    for stem, doc in DOCS.items():
        p = NB_DIR / f'{stem}.ipynb'
        if not p.exists():
            print(f'missing {p}; skipping')
            continue
        changed = ensure_template(p, doc)
        print(f'{stem}: {"added template" if changed else "template already present"}')


if __name__ == '__main__':
    main()
