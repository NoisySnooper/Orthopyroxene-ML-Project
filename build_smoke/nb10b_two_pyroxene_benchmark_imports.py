import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from config import (
    ROOT, DATA_RAW, DATA_EXTERNAL, DATA_PROC, DATA_SPLITS, DATA_NATURAL,
    MODELS, FIGURES, RESULTS, LOGS,
    LEPR_XLSX,
    FE3_FET_RATIO, SEED_MODEL,
)
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats

from src.features import build_feature_matrix
from src.plot_style import (
    apply_style, PUTIRKA_COLOR, ML_COLOR, TOL_BRIGHT,
    load_winning_config, canonical_model_filename,
)
apply_style()

import Thermobar as pt
print(f'Thermobar version: {pt.__version__ if hasattr(pt, "__version__") else "unknown"}')
