"""Patch nb09_manuscript_compilation for per-target canonical feature sets.

Like nb07/nbF, nb09 builds one `feat_fn` from WIN_FEAT and pushes the same
X through both T and P models. Forest opx_liq winners: T=alr (23),
P=raw (25) — the P model trips ValueError on the alr matrix.

This patch:
 * expands cell 3 to build X_{train,test,arcpl}_{T,P} + feat_fn_T/P;
   keeps X_train/X_test/X_arcpl as T aliases so the non-P code paths
   (IsolationForest fit, PCA+Mahalanobis, QRF standardizer) keep working.
 * cell 5: ArcPL H2O residual correlation -> X_arcpl_T for T, X_arcpl_P for P.
 * cell 7: H2O engineered features -> retrain RF_T on X_train_T+H2O,
   RF_P on X_train_P+H2O; test predictions use the matching X_test_*.
 * cell 8: predict_iqr -> X_test_T for T, X_test_P for P.
 * cell 9: MC analytical noise -> per-target feature matrices each MC draw.
 * cell 10: IsolationForest keeps using X_train (=X_train_T); T pred uses
   X_arcpl_T; P pred uses X_arcpl_P.
 * cell 12: QRF scoring -> X_arcpl_T and X_arcpl_P.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb09_manuscript_compilation.ipynb"


# ---------- CELL 3 ----------
CELL3_OLD = """feat_fn = lambda df, use_liq: build_feature_matrix(df, WIN_FEAT, use_liq=use_liq)
model_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))
qrf_T_path = MODELS / 'model_QRF_T_C_opx_liq.joblib'
qrf_P_path = MODELS / 'model_QRF_P_kbar_opx_liq.joblib'
qrf_T = joblib.load(qrf_T_path) if qrf_T_path.exists() else None
qrf_P = joblib.load(qrf_P_path) if qrf_P_path.exists() else None

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()
X_train, feat_names = feat_fn(df_train, use_liq=True)
X_test,  _          = feat_fn(df_test,  use_liq=True)

arcpl_path = RESULTS / 'nb04b_arcpl_predictions.csv'
if not arcpl_path.exists():
    raise FileNotFoundError(f'Missing {arcpl_path}. Run nb04b first.')
df_arcpl = pd.read_csv(arcpl_path)
print(f'Ext setup OK: train n={len(df_train)} test n={len(df_test)} ArcPL n={len(df_arcpl)}')"""

CELL3_NEW = """# v7 per-family canonical spec: T and P forests may use different feature
# sets (T_C opx_liq = alr / P_kbar opx_liq = raw). Build one feat_fn per
# target and wire X_*_T vs X_*_P everywhere so both model.predict() calls
# get the matrix their trained shape expects.
_spec_T_cell3 = canonical_model_spec('T_C',    'opx_liq', 'forest', RESULTS)
_spec_P_cell3 = canonical_model_spec('P_kbar', 'opx_liq', 'forest', RESULTS)
WIN_FEAT_T = _spec_T_cell3['feature_set']
WIN_FEAT_P = _spec_P_cell3['feature_set']
feat_fn_T = lambda df, use_liq: build_feature_matrix(df, WIN_FEAT_T, use_liq=use_liq)
feat_fn_P = lambda df, use_liq: build_feature_matrix(df, WIN_FEAT_P, use_liq=use_liq)
feat_fn = feat_fn_T  # back-compat: legacy calls default to T winner

model_T = joblib.load(MODELS / canonical_model_filename('T_C', 'opx_liq', 'forest', RESULTS))
model_P = joblib.load(MODELS / canonical_model_filename('P_kbar', 'opx_liq', 'forest', RESULTS))
qrf_T_path = MODELS / 'model_QRF_T_C_opx_liq.joblib'
qrf_P_path = MODELS / 'model_QRF_P_kbar_opx_liq.joblib'
qrf_T = joblib.load(qrf_T_path) if qrf_T_path.exists() else None
qrf_P = joblib.load(qrf_P_path) if qrf_P_path.exists() else None

df_liq = pd.read_parquet(DATA_PROC / 'opx_clean_opx_liq.parquet')
idx_tr = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
idx_te = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_train = df_liq.loc[idx_tr].copy()
df_test  = df_liq.loc[idx_te].copy()
X_train_T, feat_names_T = feat_fn_T(df_train, use_liq=True)
X_train_P, feat_names_P = feat_fn_P(df_train, use_liq=True)
X_test_T,  _            = feat_fn_T(df_test,  use_liq=True)
X_test_P,  _            = feat_fn_P(df_test,  use_liq=True)
# Legacy aliases for target-agnostic code paths (IsoForest fit, PCA).
X_train, feat_names = X_train_T, feat_names_T
X_test              = X_test_T

arcpl_path = RESULTS / 'nb04b_arcpl_predictions.csv'
if not arcpl_path.exists():
    raise FileNotFoundError(f'Missing {arcpl_path}. Run nb04b first.')
df_arcpl = pd.read_csv(arcpl_path)
# ArcPL comes from nb04b with the training-schema columns already applied,
# so build_feature_matrix can be called directly.
X_arcpl_T, _ = feat_fn_T(df_arcpl, use_liq=True)
X_arcpl_P, _ = feat_fn_P(df_arcpl, use_liq=True)
X_arcpl      = X_arcpl_T  # legacy alias
print(f'Ext setup OK: train n={len(df_train)} test n={len(df_test)} ArcPL n={len(df_arcpl)}')
print(f'  feature sets: T={WIN_FEAT_T} ({X_train_T.shape[1]})  P={WIN_FEAT_P} ({X_train_P.shape[1]})')"""


# ---------- CELL 5 ----------
CELL5_OLD = """    X_arcpl, _ = feat_fn(df_arcpl, use_liq=True)
    t_pred = predict_median(model_T, X_arcpl)
    p_pred = predict_median(model_P, X_arcpl)"""

CELL5_NEW = """    # Per-target feature matrices (T=alr/P=raw for opx_liq forest).
    t_pred = predict_median(model_T, X_arcpl_T)
    p_pred = predict_median(model_P, X_arcpl_P)"""


# ---------- CELL 7 ----------
CELL7_OLD = """X_train_h2o, feat_h2o = _add_h2o_engineered(df_train, X_train, feat_names)
X_test_h2o,  _        = _add_h2o_engineered(df_test,  X_test,  feat_names)
print(f'Augmented feature count: {X_train.shape[1]} -> {X_train_h2o.shape[1]}')"""

CELL7_NEW = """X_train_h2o_T, feat_h2o_T = _add_h2o_engineered(df_train, X_train_T, feat_names_T)
X_test_h2o_T,  _          = _add_h2o_engineered(df_test,  X_test_T,  feat_names_T)
X_train_h2o_P, feat_h2o_P = _add_h2o_engineered(df_train, X_train_P, feat_names_P)
X_test_h2o_P,  _          = _add_h2o_engineered(df_test,  X_test_P,  feat_names_P)
# Legacy aliases (T is the canonical primary for feat-count reporting).
X_train_h2o, feat_h2o = X_train_h2o_T, feat_h2o_T
X_test_h2o            = X_test_h2o_T
print(f'Augmented feature count: T {X_train_T.shape[1]} -> {X_train_h2o_T.shape[1]} | '
      f'P {X_train_P.shape[1]} -> {X_train_h2o_P.shape[1]}')"""

CELL7_FIT_OLD = """rf_T_h2o.fit(X_train_h2o, df_train['T_C'].values)
rf_P_h2o.fit(X_train_h2o, df_train['P_kbar'].values)"""

CELL7_FIT_NEW = """rf_T_h2o.fit(X_train_h2o_T, df_train['T_C'].values)
rf_P_h2o.fit(X_train_h2o_P, df_train['P_kbar'].values)"""

CELL7_ARCPL_OLD = """    X_arcpl_h2o, _ = _add_h2o_engineered(df_arcpl, feat_fn(df_arcpl, use_liq=True)[0], feat_names)
    t_pred_h2o = predict_median(rf_T_h2o, X_arcpl_h2o)
    p_pred_h2o = predict_median(rf_P_h2o, X_arcpl_h2o)"""

CELL7_ARCPL_NEW = """    X_arcpl_h2o_T, _ = _add_h2o_engineered(df_arcpl, X_arcpl_T, feat_names_T)
    X_arcpl_h2o_P, _ = _add_h2o_engineered(df_arcpl, X_arcpl_P, feat_names_P)
    t_pred_h2o = predict_median(rf_T_h2o, X_arcpl_h2o_T)
    p_pred_h2o = predict_median(rf_P_h2o, X_arcpl_h2o_P)"""

CELL7_TEST_OLD = """    ('T_C',    df_test['T_C'].values, predict_median(model_T, X_test),
     predict_median(rf_T_h2o, X_test_h2o)),
    ('P_kbar', df_test['P_kbar'].values, predict_median(model_P, X_test),
     predict_median(rf_P_h2o, X_test_h2o)),"""

CELL7_TEST_NEW = """    ('T_C',    df_test['T_C'].values, predict_median(model_T, X_test_T),
     predict_median(rf_T_h2o, X_test_h2o_T)),
    ('P_kbar', df_test['P_kbar'].values, predict_median(model_P, X_test_P),
     predict_median(rf_P_h2o, X_test_h2o_P)),"""


# ---------- CELL 8 ----------
CELL8_OLD = """_, q16_T, q84_T = predict_iqr(model_T, X_test)
_, q16_P, q84_P = predict_iqr(model_P, X_test)
test_pred_T = predict_median(model_T, X_test)
test_pred_P = predict_median(model_P, X_test)"""

CELL8_NEW = """_, q16_T, q84_T = predict_iqr(model_T, X_test_T)
_, q16_P, q84_P = predict_iqr(model_P, X_test_P)
test_pred_T = predict_median(model_T, X_test_T)
test_pred_P = predict_median(model_P, X_test_P)"""


# ---------- CELL 9 ----------
CELL9_OLD = """    X_mc, _ = feat_fn(df_perturb, use_liq=True)
    mc_T[:, k] = predict_median(model_T, X_mc)
    mc_P[:, k] = predict_median(model_P, X_mc)"""

CELL9_NEW = """    X_mc_T, _ = feat_fn_T(df_perturb, use_liq=True)
    X_mc_P, _ = feat_fn_P(df_perturb, use_liq=True)
    mc_T[:, k] = predict_median(model_T, X_mc_T)
    mc_P[:, k] = predict_median(model_P, X_mc_P)"""


# ---------- CELL 10 ----------
# IsolationForest is fit on X_train (=X_train_T) and scores ArcPL in the same
# T feature space - that's a design choice, not a correctness bug. Only the
# T/P predict_median calls need the per-target matrices.
CELL10_OLD = """X_arcpl, _ = feat_fn(df_arcpl, use_liq=True)
arcpl_score = iso.score_samples(X_arcpl)   # higher = more in-distribution
arcpl_label = iso.predict(X_arcpl)         # +1 inlier, -1 outlier

pred_T_arcpl = predict_median(model_T, X_arcpl)
pred_P_arcpl = predict_median(model_P, X_arcpl)"""

CELL10_NEW = """# IsolationForest in T feature space (= X_arcpl_T). The canonical RF T and P
# models get their own target-specific matrices for prediction.
arcpl_score = iso.score_samples(X_arcpl_T)   # higher = more in-distribution
arcpl_label = iso.predict(X_arcpl_T)         # +1 inlier, -1 outlier

pred_T_arcpl = predict_median(model_T, X_arcpl_T)
pred_P_arcpl = predict_median(model_P, X_arcpl_P)"""


# ---------- CELL 12 ----------
# qrf_T / qrf_P were fit in nb07 on X_train_T / X_train_P respectively.
CELL12_OLD = """    q_T_ar = qrf_T.predict(X_arcpl, quantiles=[0.16, 0.5, 0.84])
    q_P_ar = qrf_P.predict(X_arcpl, quantiles=[0.16, 0.5, 0.84])"""

CELL12_NEW = """    q_T_ar = qrf_T.predict(X_arcpl_T, quantiles=[0.16, 0.5, 0.84])
    q_P_ar = qrf_P.predict(X_arcpl_P, quantiles=[0.16, 0.5, 0.84])"""


PATCHES = [
    (3, [(CELL3_OLD, CELL3_NEW)]),
    (5, [(CELL5_OLD, CELL5_NEW)]),
    (7, [(CELL7_OLD, CELL7_NEW),
         (CELL7_FIT_OLD, CELL7_FIT_NEW),
         (CELL7_ARCPL_OLD, CELL7_ARCPL_NEW),
         (CELL7_TEST_OLD, CELL7_TEST_NEW)]),
    (8, [(CELL8_OLD, CELL8_NEW)]),
    (9, [(CELL9_OLD, CELL9_NEW)]),
    (10, [(CELL10_OLD, CELL10_NEW)]),
    (12, [(CELL12_OLD, CELL12_NEW)]),
]


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    total_changes = 0
    for idx, repls in PATCHES:
        cell = nb.cells[idx]
        changed = 0
        for old, new in repls:
            if new in cell.source:
                continue
            if old not in cell.source:
                print(f'cell {idx}: anchor not found -> {old[:60]!r}')
                return 1
            cell.source = cell.source.replace(old, new, 1)
            changed += 1
        if changed:
            cell.outputs = []
            cell.execution_count = None
            print(f'cell {idx}: patched {changed} anchor(s).')
            total_changes += changed
        else:
            print(f'cell {idx}: already patched.')
    if total_changes:
        # cell 3 needs canonical_model_spec import; check and add at the top.
        src3 = nb.cells[3].source
        if 'canonical_model_spec' in src3 and 'from src.data import canonical_model_spec' not in src3:
            nb.cells[3].source = ('from src.data import canonical_model_spec\n' +
                                  nb.cells[3].source)
            print('cell 3: added canonical_model_spec import.')
        nbformat.validate(nb)
        nbformat.write(nb, str(NB))
        print(f'nb09 updated: {total_changes} total change(s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
