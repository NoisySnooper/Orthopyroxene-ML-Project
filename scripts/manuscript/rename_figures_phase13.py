"""Phase 13 figure rename: copy from figures/legacy/ to figures/opx_only/.

Reads scripts/manuscript/figure_inventory.txt for the legacy_name -> new_name
map, copies PNG and PDF, writes a cleaned .txt caption sidecar.

Captions are sourced from a CAPTIONS dict below (manually curated to drop
v10 / Amendment / cpx wording per the Phase 13 spec).
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LEGACY = PROJECT_ROOT / 'figures' / 'legacy'
DEST = PROJECT_ROOT / 'figures' / 'opx_only'
DEST.mkdir(parents=True, exist_ok=True)

# (legacy_stem, new_stem) — only entries that should be copied.
COPY_MAP = [
    ('Core_01_fig_dataset_map',                       'main_fig_1'),
    ('Core_03_fig_methods_flowchart',                 'main_fig_2'),
    ('Core_04_fig_nb04_cross_pipeline_heatmap',       'main_fig_3'),
    ('Core_05_fig30_bias_correction_per_regime_rmse', 'main_fig_4'),
    ('Core_06_fig31_bias_correction_residuals',       'main_fig_5'),
    ('Core_07_fig34_bias_correction_scorecard_delta', 'main_fig_6'),
    ('Core_08_fig45_opx_headline',                    'main_fig_7'),
    ('Core_09a_fig_opx_regime_families',              'main_fig_8a'),
    ('Core_09b_fig_opx_overall_families',             'main_fig_8b'),
    ('Core_10_fig_best_vs_putirka',                   'main_fig_9'),
    ('Core_12_fig_shap_winners',                      'main_fig_10'),
    ('Core_15_fig_feature_concordance',               'main_fig_11'),
    ('Core_16_fig_classical_equivalence',             'main_fig_12'),
    ('Core_17_fig_partial_dependence',                'main_fig_13'),
    ('Core_18_fig_surrogate_trees',                   'main_fig_14'),
    ('fig24_per_regime_rmse_opx_liq',                 'supp_fig_1'),
    ('fig25_per_regime_residual_violins_opx_liq',     'supp_fig_2'),
    ('fig26_generalization_opx_liq',                  'supp_fig_3'),
    ('fig27_shap_summary_opx_liq',                    'supp_fig_4'),
    ('fig28_bias_correction_opx_liq',                 'supp_fig_5'),
    ('fig30_bias_correction_per_regime_rmse',         'supp_fig_6'),
    ('fig31_bias_correction_residuals',               'supp_fig_7'),
    ('fig32_bias_correction_form_comparison',         'supp_fig_8'),
    ('fig33_bias_correction_per_seed_stability',      'supp_fig_9'),
    ('fig34_bias_correction_scorecard_delta',         'supp_fig_10'),
    ('fig_aug01_ship_verdict_comparison',             'supp_fig_11'),
    ('fig_aug02_aggregate_rmse_delta',                'supp_fig_12'),
    ('fig_aug03_residual_structure_per_regime',       'supp_fig_13'),
    ('fig_aug04_form_b_breakpoint_stability',         'supp_fig_14'),
]

CAPTIONS = {
    'main_fig_1': (
        'Figure 1. ExPetDB experimental coverage for the two opx pipelines, '
        'colored by pre-registered pressure regime. Panel a: opx + liquid '
        '(n = 600, 93 citations). Panel b: opx only (n = 1035, 123 citations). '
        'Train and test splits are citation-grouped at the 80/20 ratio with '
        'StratifiedGroupKFold. Coverage is densest in the shallow_crustal '
        'and deep_crustal_MASH regimes; deeper_mantle is the smallest '
        'partition and is reported without directional claims when its test '
        'count falls below 20.'
    ),
    'main_fig_2': (
        'Figure 2. Methods flowchart. Three phases: (1) data preprocessing '
        'with citation-grouped 80/20 split, KD = 0.23-0.35 Fe-Mg equilibrium '
        'filter for opx-liq, and three feature-engineering schemes evaluated '
        'in parallel; (2) nine-family model training with Optuna inner CV '
        'and a TabPFN v2 default-hyperparameter peer; (3) evaluation pipeline '
        'with 10-fold OOF residuals feeding bias-correction fitting and a '
        'held-out test partition for the canonical scorecard.'
    ),
    'main_fig_3': (
        'Figure 3. Cross-family RMSE heatmap on the held-out test partition '
        'across the two opx pipelines and two targets. Rows are model '
        'families; columns are (track, target) cells. Cell text is mean test '
        'RMSE across 20 seeds. Per-cell winners are framed in bold: '
        'ElasticNet for opx-liq T, MLP for opx-liq P, LightGBM for opx-only T, '
        'Random Forest for opx-only P. Four families win across the four '
        'cells, indicating no single architecture dominates at this sample '
        'size.'
    ),
    'main_fig_4': (
        'Figure 4. Bias-correction effect on per-regime test RMSE. For each '
        'of the four (track, target) opx cells, paired bars compare '
        'pre-correction (left) and post-correction (right) RMSE in the four '
        'pre-registered pressure regimes. Form A (regime-piecewise OLS) is '
        'the shipped form for opx-liq P, opx-only T, and opx-only P; Form B '
        '(quantile-thresholded sigmoid, after Agreda-Lopez et al. 2024) is '
        'shown for completeness on opx-liq T where it accepts marginally. '
        'The pre-registered tolerance band is overlaid: a regime is flagged '
        'as a degradation veto if post RMSE exceeds pre RMSE by more than '
        'max(absolute_floor, 10%% of pre RMSE) and that regime has n >= 20.'
    ),
    'main_fig_5': (
        'Figure 5. Per-regime residual structure before and after bias '
        'correction, for each of the four opx cells. Top row: pre-correction '
        'residual violin plots stratified by pressure regime. Bottom row: '
        'post-correction residuals after the regime-piecewise correction '
        'is applied. Reduction in mean absolute residual is largest where '
        'the pre-correction model exhibits a regime-systematic offset; the '
        'opx-only P cell shows the cleanest improvement across all four '
        'regimes.'
    ),
    'main_fig_6': (
        'Figure 6. Per-regime scorecard delta between our post-bias-'
        'correction RMSE and the best-available external Putirka equation '
        'for the same (track, target, regime) cell. Green cells are wins '
        'for our method; red cells are wins for Putirka. Cell text is the '
        'absolute RMSE delta in the target unit (degrees C for T, kbar for '
        'P) followed by the family that produced the post-correction '
        'prediction. The opx-only P column is uniformly green: the '
        'Random Forest plus pairwise log-ratio features wins every '
        'pre-registered pressure regime against Putirka equation 29c.'
    ),
    'main_fig_7': (
        'Figure 7. Headline opx-only P scorecard for the four candidate '
        'shipping configurations: tuned pre-correction (gray), tuned '
        'post-correction Form A (blue), TabPFN pre-correction (orange), '
        'TabPFN post-correction Form A (green). Per-regime test RMSE in kbar '
        'with the pre-registered honesty bar (n >= 20) marked. Form A on '
        'the tuned RF/pwlr winner reduces aggregate test RMSE from 10.35 '
        'to 6.05 kbar (-41.9%%) and beats Putirka equation 29c by 54.7%%.'
    ),
    'main_fig_8a': (
        'Figure 8a. Per-regime test RMSE by model family, opx pipelines '
        'only. Each panel is a (track, target, regime) cell; bar height is '
        'mean test RMSE across 20 seeds with one-sigma whiskers. Family '
        'colors are consistent across panels (Okabe-Ito palette). The '
        'shipped per-cell winner is highlighted in each panel.'
    ),
    'main_fig_8b': (
        'Figure 8b. Aggregate test RMSE by family across both opx pipelines '
        'and both targets, ordered left to right. The four cells share no '
        'single dominant family; this is the empirical justification for '
        'evaluating nine families and selecting per cell rather than '
        'pre-committing to one architecture.'
    ),
    'main_fig_9': (
        'Figure 9. Per-cell best ML method versus the corresponding '
        'best-available Putirka equation, on the held-out test partition. '
        'Bars are ordered: ML pre-correction, ML post-correction (shipped '
        'form), and Putirka. Labels report the absolute aggregate test '
        'RMSE in the target unit. The four opx cells all show the ML '
        'post-correction below Putirka.'
    ),
    'main_fig_10': (
        'Figure 10. SHAP feature importance for the best-explainable tuned '
        'family in each of the four opx (track, target) cells. Each panel '
        'shows the top 10 features ranked by mean absolute SHAP on the '
        'held-out test partition at canonical seed 42. Bar length is mean '
        '|SHAP| in the target native unit. Tree families use TreeExplainer '
        '(exact); ElasticNet uses LinearExplainer; MLP uses permutation '
        'importance as a SHAP surrogate.'
    ),
    'main_fig_11': (
        'Figure 11. Feature concordance heatmap. For each of the four opx '
        '(track, target) cells, pairwise Spearman rho across the eight '
        'tuned-family permutation-importance ranks. High median off-diagonal '
        'rho means families agree on what features matter; low rho means '
        'families disagree, which weakens any single-family interpretive '
        'claim about feature importance.'
    ),
    'main_fig_12': (
        'Figure 12. Classical-equivalence regression. For each of the four '
        'opx (track, target) cells, columns: (1) ML predicted versus '
        'classical-fit predicted in the target unit, with the 1:1 line and '
        'the per-cell tolerance band; (2) the classical-fit residuals; '
        '(3) the ML-minus-classical-fit disequilibrium map showing where '
        'the ML diverges from a closed-form classical functional form. '
        'High R^2 in column 1 means the ML prediction is approximately '
        'a classical formula plus a smooth residual; low R^2 means the '
        'ML is doing something a classical equation cannot capture.'
    ),
    'main_fig_13': (
        'Figure 13. Partial-dependence functions across the eight tuned '
        'families for the top three permutation-important features in each '
        'opx (track, target) cell. Family curves are colored consistently '
        'with the Okabe-Ito palette. Convergence across families on the same '
        'monotone shape supports the corresponding feature direction; '
        'divergence flags a feature whose effect is family-specific and '
        'should not be cited as a robust geological inference.'
    ),
    'main_fig_14': (
        'Figure 14. Surrogate decision trees fit to each per-cell winning '
        'model. A depth-3 tree is trained to predict the winning model '
        'output as a function of input features on the OOF training '
        'predictions; reported R^2 quantifies how well a small tree '
        'approximates the deeper learner. A high surrogate R^2 means the '
        'underlying model is dominated by a few axis-aligned splits; '
        'a low surrogate R^2 means the model uses richer structure '
        '(interactions, smoothness) that a tree cannot recover.'
    ),
    'supp_fig_1':  ('Supp. Figure 1. Per-regime test RMSE for opx-liq, by '
                    'pre-registered pressure regime, with Wilson 95%% '
                    'confidence intervals.'),
    'supp_fig_2':  ('Supp. Figure 2. Per-regime residual violin plots for '
                    'opx-liq winners (T and P).'),
    'supp_fig_3':  ('Supp. Figure 3. Generalization curves for opx-liq T '
                    'and P winners across 20 seeds.'),
    'supp_fig_4':  ('Supp. Figure 4. SHAP summary beeswarm plots for the '
                    'opx-liq winners (ElasticNet for T, MLP for P).'),
    'supp_fig_5':  ('Supp. Figure 5. Bias-correction diagnostic plots for '
                    'the opx-liq cells: fitted Form A piecewise correction '
                    'and post-correction residual scatter.'),
    'supp_fig_6':  ('Supp. Figure 6. Bias-correction per-regime RMSE table '
                    'across the four opx cells. Pre and post bars per regime, '
                    'with the pre-registered tolerance band marked.'),
    'supp_fig_7':  ('Supp. Figure 7. Bias-correction residual structure per '
                    'regime, four opx cells.'),
    'supp_fig_8':  ('Supp. Figure 8. Form A versus Form B side-by-side on '
                    'the four opx cells. Form B (Agreda-Lopez quantile-'
                    'thresholded sigmoid) accepts only marginally on opx-liq '
                    'T and not at all on the other three opx cells.'),
    'supp_fig_9':  ('Supp. Figure 9. Per-seed bias-correction stability '
                    'across the 20 seeds, four opx cells. Stripplot of '
                    'post-correction aggregate test RMSE per seed.'),
    'supp_fig_10': ('Supp. Figure 10. Per-cell scorecard delta against '
                    'Putirka, four opx cells, with per-regime breakdown.'),
    'supp_fig_11': ('Supp. Figure 11. Augmentation ablation: ship-rule '
                    'verdict comparison between N_AUG = 1 (canonical) and '
                    'N_AUG = 15 (Gaussian composition-noise augmentation) '
                    'across the four opx cells.'),
    'supp_fig_12': ('Supp. Figure 12. Augmentation ablation: aggregate test '
                    'RMSE delta from N_AUG = 15 vs N_AUG = 1, four opx cells. '
                    'Augmentation degrades RMSE on every opx cell.'),
    'supp_fig_13': ('Supp. Figure 13. Augmentation ablation: residual '
                    'structure per regime under N_AUG = 15. No regime '
                    'shows the structured pre-correction bias that Form B '
                    'is designed to correct.'),
    'supp_fig_14': ('Supp. Figure 14. Augmentation ablation: Form B '
                    'breakpoint stability across the 20 seeds and the four '
                    'opx cells. Breakpoints jump unstably across seeds, '
                    'consistent with Form B not finding a robust target '
                    'for the opx residuals.'),
}


def main():
    n_copied = 0
    n_missing = 0
    for legacy_stem, new_stem in COPY_MAP:
        for ext in ('png', 'pdf'):
            src = LEGACY / f'{legacy_stem}.{ext}'
            dst = DEST / f'{new_stem}.{ext}'
            if src.exists():
                shutil.copy2(src, dst)
                n_copied += 1
            else:
                print(f'MISSING: {src}')
                n_missing += 1
        cap_path = DEST / f'{new_stem}.txt'
        cap_text = CAPTIONS.get(new_stem, '')
        if not cap_text:
            print(f'NO CAPTION FOR {new_stem}')
        cap_path.write_text(cap_text + '\n', encoding='utf-8')
    print(f'\nCopied {n_copied} files, {n_missing} missing.')
    print(f'Captions written: {len(COPY_MAP)} sidecars.')


if __name__ == '__main__':
    main()
