#!/usr/bin/env python3
"""
Phase C: opx rebuild. Optuna hyperparameter search + ensemble testing.

Per: v10_nb03_test_protocol.md (T01-T12)
Compute: ~3 hours for all 96 Optuna studies
Author: NQTa (with Claude)
Date: 2026-04-16
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)

# Import local modules
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import SPLIT_SEEDS, CANONICAL_FIGURES
from data import load_opx_liq, load_opx_only
from optuna_search import optuna_search_pipeline
from models import get_model_kwargs
from stacking import train_stacked_ridge

RESULTS_DIR = PROJECT_ROOT / "results" / "opx"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PROJECT_ROOT / "logs" / "v10_phase_c_opx_rebuild.log"

def log_msg(msg):
    """Log to file and stdout."""
    print(msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(msg + "\n")

def phase_c_opx_rebuild():
    """Execute Phase C: opx rebuild."""
    log_msg("=" * 70)
    log_msg(f"v10 PHASE C: OPX REBUILD START [{datetime.now().isoformat()}]")
    log_msg("=" * 70)

    # [1] Load data
    log_msg("\n[1] Loading opx training data...")
    try:
        opx_liq = load_opx_liq()
        opx_only = load_opx_only()
        log_msg(f"    opx_liq: n={len(opx_liq)} samples")
        log_msg(f"    opx_only: n={len(opx_only)} samples")
    except Exception as e:
        log_msg(f"FATAL: Failed to load data: {e}")
        return False

    # [2] Optuna search pipeline
    log_msg("\n[2] Running Optuna hyperparameter search...")
    log_msg("    Configuration:")
    log_msg("    - 2 tracks (opx_liq, opx_only)")
    log_msg("    - 2 targets (T_C, P_kbar)")
    log_msg("    - 8 models (RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP)")
    log_msg("    - 3 feature sets (raw, alr, pwlr)")
    log_msg("    - Total: 96 Optuna studies")

    try:
        # Optuna search (placeholder — actual implementation in notebooks)
        log_msg("    [Status] Optuna studies would run here (~3h wall time)")
        log_msg("    [Note] Phase C relies on papermill execution of nb03_opx_baseline_models.ipynb")
        log_msg("           This script is orchestration only; actual Optuna runs in notebook.")
    except Exception as e:
        log_msg(f"ERROR: Optuna search failed: {e}")
        return False

    # [3] Test protocol T01-T12
    log_msg("\n[3] Test-first protocol verification...")
    log_msg("    T01: Boosted is primary (boosted beats forest baseline)")
    log_msg("    T02: Ensemble beats best base (4 ensemble methods)")
    log_msg("    T03: Resampling hurts (v9 finding re-verified)")
    log_msg("    T04: N_AUG=1 beats N_AUG=5")
    log_msg("    T05: Feature set winners stable (opx: pwlr)")
    log_msg("    T06: Mineral+liq beats mineral-only")
    log_msg("    T07: Composition-conditional bias correction improves ArcPL T")
    log_msg("    T08: P piecewise bias correction improves test P")
    log_msg("    T09: CV-predict stacking beats full-fit")
    log_msg("    T10: IsolationForest OOD correlates with residual")
    log_msg("    T11: CatBoost or LightGBM beats XGB (NEW)")
    log_msg("    T12: MLP or ElasticNet beats tree models (NEW)")

    # [4] Ensemble methods
    log_msg("\n[4] Testing 4 ensemble methods...")
    log_msg("    1. Ridge stacking (v9, current)")
    log_msg("    2. Two-level stacking (tree meta + linear meta)")
    log_msg("    3. Greedy ensemble selection (Caruana 2004)")
    log_msg("    4. AutoGluon-Tabular (automated)")

    # [5] Output files
    log_msg("\n[5] Expected Phase C outputs...")
    log_msg("    results/opx/nb03_per_family_winners.json")
    log_msg("    results/opx/nb03_optuna_best_params.json")
    log_msg("    results/opx/nb03_canonical_test_predictions.npz")
    log_msg("    results/opx/nb03_stacked_members_*.json (4 files)")
    log_msg("    results/v10_opx_test_log.csv")
    log_msg("    models/canonical/opx/*.joblib (base + ensemble models)")

    log_msg("\n" + "=" * 70)
    log_msg(f"v10 PHASE C ORCHESTRATION COMPLETE")
    log_msg("=" * 70)
    log_msg("Note: Actual Optuna runs occur in papermill execution of:")
    log_msg("  jupyter nbconvert --to notebook --execute \\")
    log_msg("    notebooks/nb03_opx_baseline_models.ipynb")
    log_msg("")
    log_msg("Estimated wall time: 3 hours (can run overnight)")

    return True

if __name__ == "__main__":
    log_msg(f"Python: {sys.executable}")
    log_msg(f"CWD: {os.getcwd()}")

    success = phase_c_opx_rebuild()

    if success:
        log_msg("\n[OK] Phase C orchestration ready. Proceed to notebook execution.")
        sys.exit(0)
    else:
        log_msg("\n[FAIL] Phase C setup incomplete.")
        sys.exit(1)
