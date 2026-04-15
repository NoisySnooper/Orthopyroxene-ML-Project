import os
os.chdir(r"C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks")
import sys
import json
import ast
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
from config import (
    ROOT, DATA_PROC, DATA_SPLITS, MODELS, FIGURES, RESULTS,
    SEED_SPLIT, SEED_MODEL,
)
from src.features import build_feature_matrix
from src.plot_style import load_winning_config, apply_style
from src.models import predict_median

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

apply_style()  # Okabe-Ito colorblind-safe palette, 300 dpi PNG+PDF defaults
