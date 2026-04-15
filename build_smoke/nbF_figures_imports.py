import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from config import (
    ROOT, DATA_RAW, DATA_EXTERNAL, DATA_PROC, DATA_SPLITS, DATA_NATURAL,
    MODELS, FIGURES, RESULTS, LOGS,
    EXPETDB, LEPR_XLSX, LIN2023_NATURAL,
    FE3_FET_RATIO, KD_FEMG_MIN, KD_FEMG_MAX, WO_MAX_MOL_PCT,
    P_CEILING_KBAR, CATION_SUM_MIN, CATION_SUM_MAX,
    OXIDE_TOTAL_MIN, OXIDE_TOTAL_MAX,
    SEED_SPLIT, SEED_MODEL, SEED_NOISE_AUG, SEED_KMEANS,
    OPX_RAW_OXIDES, OPX_FULL_OXIDES, LIQ_OXIDES,
)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Canonical features, models, and plotting from src/ (one source of truth).
from src.features import (
    build_feature_matrix,
    make_raw_features,
    make_alr_features,
    make_pwlr_features,
    augment_dataframe,
)
from src.models import predict_median, predict_iqr
from src.geotherm import hasterok_chapman_geotherm
from src.plot_style import (
    apply_style, panel_label, stats_box, one_to_one, regression_line,
    save_figure, fmt_value,
    load_winning_config, canonical_model_filename,
    TOL_BRIGHT, MODEL_COLORS, PUTIRKA_COLOR, ML_COLOR,
)

apply_style()
