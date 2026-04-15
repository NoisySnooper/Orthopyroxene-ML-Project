"""Central path configuration for the opx ML thermobarometer pipeline.

Every notebook imports from this module. No hardcoded paths allowed
anywhere else in the codebase.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_EXTERNAL = ROOT / 'data' / 'raw' / 'external'
DATA_PROC = ROOT / 'data' / 'processed'
DATA_SPLITS = ROOT / 'data' / 'splits'
DATA_NATURAL = ROOT / 'data' / 'natural'
MODELS = ROOT / 'models'
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'
LOGS = ROOT / 'logs'

for d in [DATA_RAW, DATA_EXTERNAL, DATA_PROC, DATA_SPLITS, DATA_NATURAL,
          MODELS, FIGURES, RESULTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# Input data files
EXPETDB = DATA_RAW / 'ExPetDB_download_ExPetDB-2025-07-21.xlsx'
LEPR_XLSX = DATA_EXTERNAL / 'LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx'
LIN2023_NATURAL = DATA_NATURAL / 'natural_opx_cleaned.csv'

# Fixed scientific parameters (do not change without user approval)
FE3_FET_RATIO = 0.15            # Fixed Fe3+/FeT assumption
KD_FEMG_MIN = 0.23              # Putirka 2008 equilibrium window
KD_FEMG_MAX = 0.35
WO_MAX_MOL_PCT = 5.0            # Pigeonite filter
P_CEILING_KBAR = 100.0          # Opx stability ceiling
CATION_SUM_MIN = 3.95           # 6-oxygen basis
CATION_SUM_MAX = 4.05
OXIDE_TOTAL_MIN = 95.0
OXIDE_TOTAL_MAX = 102.0

# Random seeds (centralized for reproducibility)
SEED_SPLIT = 42
SEED_MODEL = 42
SEED_NOISE_AUG = 42
SEED_KMEANS = 42
SEED_BOOTSTRAP = 42  # dedicated seed for bootstrap CIs (replaces hardcoded default_rng(42))

# Multi-seed training protocol (v7: 20 seeds)
N_SPLIT_REPS = 20
SPLIT_SEEDS = list(range(42, 42 + N_SPLIT_REPS))

# Feature sets (used in Phase 3 CoDA benchmark)
OPX_RAW_OXIDES = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO']
OPX_FULL_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O']
LIQ_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']

# Feature-engineering + augmentation defaults (NB03 frozen after sensitivity test)
FEATURE_METHODS = ('raw', 'alr', 'pwlr')
N_AUG_DEFAULT = 1                              # augmentation disabled (see NB03 appendix)
EPMA_NOISE_REL_MAJOR = 0.03                    # Agreda-Lopez 2024 analytical noise
EPMA_NOISE_REL_MINOR = 0.08

# Quantile regression forest settings (NB07 bias-corrected models, NB10 analytical noise)
QRF_QUANTILES = (0.05, 0.5, 0.95)

# Conformal prediction miscoverage rate. Target coverage = 1 - CONFORMAL_ALPHA.
CONFORMAL_ALPHA = 0.10

# Hasterok & Chapman 2011 reference surface heat flows for geotherm overlays (mW/m^2)
GEOTHERM_Q_S_CRATONIC = 40.0
GEOTHERM_Q_S_AVERAGE  = 60.0
GEOTHERM_Q_S_HOT      = 80.0

# Canonical output file names for downstream notebooks. File extensions carry
# the convention, so consumers never need to know the parent directory.
OPX_CORE_CLUSTERED_FILE = 'opx_clean_core_with_clusters.parquet'
WINNING_CONFIG_FILE = 'nb03_winning_configurations.json'        # deprecated (pre-v7)
PER_FAMILY_WINNERS_FILE = 'nb03_per_family_winners.json'        # v7 single source of truth
MULTI_SEED_RESULTS_FILE = 'nb03_multi_seed_results.csv'
MULTI_SEED_SUMMARY_FILE = 'nb03_multi_seed_summary.csv'
CANONICAL_PREDICTIONS_FILE = 'nb03_canonical_test_predictions.npz'

# v7 two-family model design; v9 adds the stacked family (single member per target/track)
MODEL_FAMILIES = {
    'forest':  {'candidates': ('RF', 'ERT'),  'tiebreaker_preferred': 'RF'},
    'boosted': {'candidates': ('GB', 'XGB'), 'tiebreaker_preferred': 'XGB'},
    'stacked': {'candidates': ('STACKED',),   'tiebreaker_preferred': 'STACKED'},
}
TIEBREAKER_RULE = (
    "when means within 1 std, prefer RF over ERT (forest), "
    "prefer XGB over GB (boosted)"
)
TIEBREAKER_RATIONALE = (
    "more commonly cited in the ML petrology literature per user preference"
)

# Thermobar contract (v7 Part E.1)
THERMOBAR_PINNED_VERSION = '1.0.70'
THERMOBAR_T_RETURNS_KELVIN = True

# Figure color convention (v7 Part I.1)
FAMILY_COLORS = {
    'forest': '#0072B2',        # Okabe-Ito blue
    'boosted': '#D55E00',       # Okabe-Ito vermillion
    'external_cpx': '#E69F00',  # orange (Agreda-Lopez, Jorgenson, Wang)
    'putirka': '#56B4E9',       # sky blue (classical reference)
    'onetoone': '#000000',      # black dashed 1:1 line
}

# Canonical manuscript figure roster (v7 Part C.3). nb09 and nbF read this.
CANONICAL_FIGURES = [
    {'num': 1,  'stem': 'fig_nb02_pca_variance',
     'caption': 'PCA cumulative variance'},
    {'num': 2,  'stem': 'fig_nb02_clusters',
     'caption': 'k-means clusters in PC1/PC2 space'},
    {'num': 3,  'stem': 'fig_nb03c_multiseed_rmse',
     'caption': '20-seed RMSE by model x feature set'},
    {'num': 4,  'stem': 'fig_nb04_putirka_comparison',
     'caption': 'Putirka 2008 vs ML on ExPetDB test'},
    {'num': 5,  'stem': 'fig_nb04_three_way',
     'caption': 'Three-way ML benchmark on ArcPL'},
    {'num': 6,  'stem': 'fig_nb05_generalization',
     'caption': 'LOSO / cluster / TargetBinKFold generalization'},
    {'num': 7,  'stem': 'fig_nb06_shap_T',
     'caption': 'SHAP feature importance for T'},
    {'num': 8,  'stem': 'fig_nb06_shap_P',
     'caption': 'SHAP feature importance for P'},
    {'num': 9,  'stem': 'fig_nb07_conformal_coverage',
     'caption': 'Conformal interval calibration'},
    {'num': 10, 'stem': 'fig_nb07_bias_correction',
     'caption': 'Bias correction null result'},
    {'num': 11, 'stem': 'fig_nb08_twopx_1to1',
     'caption': 'Cross-mineral validation on paired pyroxenes'},
    {'num': 12, 'stem': 'fig_nb08_disagreement_map',
     'caption': 'Opx-cpx disagreement diagnostic'},
    {'num': 13, 'stem': 'fig_nb09_ood_residuals',
     'caption': 'OOD score vs residual'},
    {'num': 14, 'stem': 'fig_nb09_h2o_residuals',
     'caption': 'H2O dependence of residuals'},
    {'num': 15, 'stem': 'fig_nb11_model_family_ceiling',
     'caption': 'Model-family ceiling (supplementary)'},
    # v9 Phase 2 additions (NB03 Optuna + resampling + stacking)
    {'num': 16, 'stem': 'fig_nb03_resampling_pt_distribution',
     'caption': 'Tempered resampling: P-T distribution before vs after'},
    {'num': 17, 'stem': 'fig_nb03_resampling_impact_on_metrics',
     'caption': 'Tempered resampling: T/P RMSE by P bin'},
    {'num': 18, 'stem': 'fig_nb03_optuna_search_progress',
     'caption': 'Optuna search convergence (best-so-far per study)'},
    {'num': 19, 'stem': 'fig_nb03_optuna_hyperparameter_importance',
     'caption': 'Optuna hyperparameter importance per base model'},
    {'num': 20, 'stem': 'fig_nb03_stacking_weights',
     'caption': 'Ridge stacking weights per (target, track)'},
    {'num': 21, 'stem': 'fig_nb03_stacking_oof_correlation',
     'caption': 'OOF correlation heatmap across base learners'},
    {'num': 22, 'stem': 'fig_nb03_stacking_vs_base_comparison',
     'caption': 'Stacking vs best base model with bootstrap CIs'},
    {'num': 23, 'stem': 'fig_nb03_three_family_comparison',
     'caption': 'Forest vs boosted vs stacked family overview'},
]

# ---------------------------------------------------------------------------
# v9 Phase 2: Optuna / resampling / stacking constants
# ---------------------------------------------------------------------------

OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_PER_TRIAL = 300
OPTUNA_N_JOBS_INNER = 4
OPTUNA_SEED = 42
OPTUNA_STUDIES_DIR = RESULTS / 'optuna_studies'
OPTUNA_BEST_PARAMS_FILE = 'nb03_optuna_best_params.json'
OPTUNA_BEST_PARAMS_PARTIAL_FILE = 'nb03_optuna_best_params_partial.json'
OPTUNA_MODELS = ('RF', 'ERT', 'XGB', 'GB')
OPTUNA_TARGETS = ('T_C', 'P_kbar')
OPTUNA_TRACKS = ('opx_only', 'opx_liq')
OPTUNA_FEATURE_SETS = ('raw', 'alr', 'pwlr')

RESAMPLING_N_P_BINS = 5
RESAMPLING_N_T_BINS = 5
RESAMPLING_SEED = 42
RESAMPLING_DIAGNOSTICS_FILE = 'nb03_resampling_diagnostics.csv'
RESAMPLING_SUMMARY_FILE = 'nb03_resampling_summary.json'

STACKING_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
STACKING_CV_FOLDS = 5
STACKING_SEED = 42
STACKING_BASE_ORDER = ('RF', 'ERT', 'XGB', 'GB')
STACKING_OOF_MATRIX_TEMPLATE = 'nb03_stacking_oof_matrix_{target}_{track}.npz'
STACKING_MANIFEST_TEMPLATE = 'nb03_stacked_members_{target}_{track}.json'
STACKING_META_TEMPLATE = 'meta_ridge_{target}_{track}_stacked.joblib'
STACKING_DIAGNOSTICS_FILE = 'nb03_stacking_diagnostics.json'
