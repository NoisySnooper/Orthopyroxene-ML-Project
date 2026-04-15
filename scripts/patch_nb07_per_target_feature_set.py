"""Patch nb07 to use per-target feature matrices.

Under v7 per-family canonical specs, the forest T_C opx_liq winner is 'alr'
(23 features) while the forest P_kbar opx_liq winner is 'raw' (25 features).
nb07 currently builds a single X_train/X_test matrix from T_C's feature set
and feeds it to both T and P pipelines, which fails with a shape-mismatch.

This patch:
  * rewrites cell 4 to build WIN_FEAT_T / WIN_FEAT_P and four matrices
    X_train_T, X_test_T, X_train_P, X_test_P (keeping WIN_FEAT = WIN_FEAT_T
    and X_train, X_test as aliases for any downstream code referring to
    "the" winning feature set);
  * rewrites cells 5, 9, 11 so every T usage references *_T matrices and
    every P usage references *_P matrices.

Idempotent: re-running is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb07_bias_correction.ipynb"

CELL4_NEW = '''# Phase 7R setup: per-target canonical feature matrices + RF models.
# Forest T_C and P_kbar winners can use different feature sets; build X
# separately for each so model.predict() shape checks pass.
from src.data import canonical_model_spec
_spec_T = canonical_model_spec('T_C',    'opx_liq', 'forest', RESULTS)
_spec_P = canonical_model_spec('P_kbar', 'opx_liq', 'forest', RESULTS)
WIN_FEAT_T = _spec_T['feature_set']
WIN_FEAT_P = _spec_P['feature_set']
WIN_FEAT   = WIN_FEAT_T  # legacy alias: "the" feature set = T winner
print(f'v7 forest opx_liq feature sets: T={WIN_FEAT_T}  P={WIN_FEAT_P}')

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')

df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()

def parse_params(s):
    import ast, json
    if isinstance(s, dict): return s
    try: return ast.literal_eval(s)
    except:
        try: return json.loads(s)
        except: return {}

# Reconstruct RF hyperparameters per target from the canonical winning
# feature set. Seed-42 rows are the reference.
multi = pd.read_csv(RESULTS / 'nb03_multi_seed_results.csv')
def _rf_params_for(target, feat):
    sub = multi[(multi.track == 'opx_liq') &
                (multi.model_name == 'RF') &
                (multi.feature_set == feat) &
                (multi.split_seed == 42) &
                (multi.target == target)]
    return parse_params(sub.iloc[0]['best_params'])
params_T = _rf_params_for('T_C',    WIN_FEAT_T)
params_P = _rf_params_for('P_kbar', WIN_FEAT_P)

# Canonical RF models (one per target, each on its own feature set).
model_T = joblib.load(MODELS / canonical_model_filename('T_C',    'opx_liq', 'forest', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))

# Per-target feature matrices.
X_train_T, feat_names_T = build_feature_matrix(df_train, WIN_FEAT_T, use_liq=True)
X_test_T,  _            = build_feature_matrix(df_test,  WIN_FEAT_T, use_liq=True)
X_train_P, feat_names_P = build_feature_matrix(df_train, WIN_FEAT_P, use_liq=True)
X_test_P,  _            = build_feature_matrix(df_test,  WIN_FEAT_P, use_liq=True)

# Legacy aliases (unused for T/P-specific ops but kept for any downstream
# generic references that are actually target-agnostic).
X_train, feat_names = X_train_T, feat_names_T
X_test              = X_test_T

y_train_T = df_train['T_C'].values
y_train_P = df_train['P_kbar'].values
y_test_T  = df_test['T_C'].values
y_test_P  = df_test['P_kbar'].values

groups_train = df_train['Citation'].astype(str).values
print(f'train n={len(df_train)} test n={len(df_test)} '
      f'features: T={len(feat_names_T)} P={len(feat_names_P)}')
'''


# Cell 5 direct replacements (T uses *_T, P uses *_P).
CELL5_REPLS = [
    ('oof_T = oof_rf(X_train, y_train_T, groups_train, params_T, SEED_MODEL)',
     'oof_T = oof_rf(X_train_T, y_train_T, groups_train, params_T, SEED_MODEL)'),
    ('oof_P = oof_rf(X_train, y_train_P, groups_train, params_P, SEED_MODEL)',
     'oof_P = oof_rf(X_train_P, y_train_P, groups_train, params_P, SEED_MODEL)'),
    ('train_pred_T_raw = predict_median(model_T, X_train)',
     'train_pred_T_raw = predict_median(model_T, X_train_T)'),
    ('train_pred_P_raw = predict_median(model_P, X_train)',
     'train_pred_P_raw = predict_median(model_P, X_train_P)'),
    ('test_pred_T_raw = predict_median(model_T, X_test)',
     'test_pred_T_raw = predict_median(model_T, X_test_T)'),
    ('test_pred_P_raw = predict_median(model_P, X_test)',
     'test_pred_P_raw = predict_median(model_P, X_test_P)'),
]

# Cell 9 replacements.
CELL9_REPLS = [
    ('qrf_T_full.fit(X_train, y_train_T)',
     'qrf_T_full.fit(X_train_T, y_train_T)'),
    ('qrf_P_full.fit(X_train, y_train_P)',
     'qrf_P_full.fit(X_train_P, y_train_P)'),
    ('q_T_tr = qrf_T_full.predict(X_train, quantiles=[0.16, 0.5, 0.84])',
     'q_T_tr = qrf_T_full.predict(X_train_T, quantiles=[0.16, 0.5, 0.84])'),
    ('q_P_tr = qrf_P_full.predict(X_train, quantiles=[0.16, 0.5, 0.84])',
     'q_P_tr = qrf_P_full.predict(X_train_P, quantiles=[0.16, 0.5, 0.84])'),
    ('oof_T_lo, oof_T_md, oof_T_hi = oof_qrf(X_train, y_train_T, groups_train, params_T, SEED_MODEL)',
     'oof_T_lo, oof_T_md, oof_T_hi = oof_qrf(X_train_T, y_train_T, groups_train, params_T, SEED_MODEL)'),
    ('oof_P_lo, oof_P_md, oof_P_hi = oof_qrf(X_train, y_train_P, groups_train, params_P, SEED_MODEL)',
     'oof_P_lo, oof_P_md, oof_P_hi = oof_qrf(X_train_P, y_train_P, groups_train, params_P, SEED_MODEL)'),
    ('q_T_te = qrf_T_full.predict(X_test, quantiles=[0.16, 0.5, 0.84])',
     'q_T_te = qrf_T_full.predict(X_test_T, quantiles=[0.16, 0.5, 0.84])'),
    ('q_P_te = qrf_P_full.predict(X_test, quantiles=[0.16, 0.5, 0.84])',
     'q_P_te = qrf_P_full.predict(X_test_P, quantiles=[0.16, 0.5, 0.84])'),
]

# Cell 11: train_test_split needs separate splits per target.
CELL11_OLD = """X_fit, X_cal, y_fit_P, y_cal_P = train_test_split(
    X_train, y_train_P, test_size=0.10, random_state=SEED_SPLIT
)
_, _, y_fit_T, y_cal_T = train_test_split(
    X_train, y_train_T, test_size=0.10, random_state=SEED_SPLIT
)

qrf_P_cal = clone(qrf_P_full).fit(X_fit, y_fit_P)
qrf_T_cal = clone(qrf_T_full).fit(X_fit, y_fit_T)

pred_P_cal = predict_median(qrf_P_cal, X_cal)
pred_T_cal = predict_median(qrf_T_cal, X_cal)
resid_P = y_cal_P - pred_P_cal
resid_T = y_cal_T - pred_T_cal

pred_P_test = predict_median(qrf_P_full, X_test)
pred_T_test = predict_median(qrf_T_full, X_test)"""

CELL11_NEW = """X_fit_P, X_cal_P, y_fit_P, y_cal_P = train_test_split(
    X_train_P, y_train_P, test_size=0.10, random_state=SEED_SPLIT
)
X_fit_T, X_cal_T, y_fit_T, y_cal_T = train_test_split(
    X_train_T, y_train_T, test_size=0.10, random_state=SEED_SPLIT
)

qrf_P_cal = clone(qrf_P_full).fit(X_fit_P, y_fit_P)
qrf_T_cal = clone(qrf_T_full).fit(X_fit_T, y_fit_T)

pred_P_cal = predict_median(qrf_P_cal, X_cal_P)
pred_T_cal = predict_median(qrf_T_cal, X_cal_T)
resid_P = y_cal_P - pred_P_cal
resid_T = y_cal_T - pred_T_cal

pred_P_test = predict_median(qrf_P_full, X_test_P)
pred_T_test = predict_median(qrf_T_full, X_test_T)"""


def _apply_cell(cell, old_new_list, label):
    changed = 0
    for old, new in old_new_list:
        if new in cell.source:
            continue
        if old not in cell.source:
            print(f'  {label}: anchor not found -> {old[:60]}')
            return -1
        cell.source = cell.source.replace(old, new, 1)
        changed += 1
    if changed:
        cell.outputs = []
        cell.execution_count = None
        print(f'  {label}: patched {changed} anchor(s).')
    else:
        print(f'  {label}: already patched.')
    return changed


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)

    # Cell 4 full rewrite.
    cell4 = nb.cells[4]
    if cell4.source.strip() == CELL4_NEW.strip():
        print('cell 4: already on target.')
    else:
        cell4.source = CELL4_NEW
        cell4.outputs = []
        cell4.execution_count = None
        print('cell 4: rewritten.')

    # Cell 5 targeted substitutions.
    if _apply_cell(nb.cells[5], CELL5_REPLS, 'cell 5') < 0:
        return 1

    # Cell 9 targeted substitutions.
    if _apply_cell(nb.cells[9], CELL9_REPLS, 'cell 9') < 0:
        return 1

    # Cell 11 block rewrite.
    cell11 = nb.cells[11]
    if CELL11_NEW in cell11.source:
        print('cell 11: already patched.')
    elif CELL11_OLD not in cell11.source:
        print('cell 11: anchor not found.')
        return 1
    else:
        cell11.source = cell11.source.replace(CELL11_OLD, CELL11_NEW, 1)
        cell11.outputs = []
        cell11.execution_count = None
        print('cell 11: patched.')

    nbformat.validate(nb)
    nbformat.write(nb, str(NB))
    print(f'nb07 updated at {NB}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
