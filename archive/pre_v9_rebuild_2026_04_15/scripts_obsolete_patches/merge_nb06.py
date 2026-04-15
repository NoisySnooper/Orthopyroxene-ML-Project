"""Append robustness checks from deprecated nb06b into nb06.

Inserts six appendix sections after the existing SHAP analysis:
  A. Baseline retrain (to record original RMSE)
  B. Ablation test: drop liq_SiO2 + liq_MgO features
  C. Proxy check scatter plots (liq_SiO2 vs P, liq_MgO vs T)
  D. Feature correlation heatmap among candidate dominators
  E. Y-randomization (permutation) test
  F. Dummy regressor baseline
  G. Perfect-signal injection + unconstrained variant + linear sanity check

Each test is rewired to use the canonical feature matrix (X_train built via
build_feature_matrix on the winning feature set) rather than the raw numeric
column slice from the old nb06b. Figures are written to FIGURES, not cwd.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / 'notebooks' / 'nb06_shap_analysis.ipynb'


APPENDIX_INTRO = """## Robustness checks (merged from deprecated nb06b)

Six diagnostic tests rerun on the canonical opx_liq pressure model (using the
winning feature set from Phase 3R). Each test is a targeted probe of whether
the model is learning real physicochemical signal versus exploiting a
statistical proxy of the laboratory experimental design.
"""

SETUP_BASELINE = """# Load the canonical train + test feature matrices for robustness testing.
train_idx = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
df_train = df_liq.loc[train_idx].copy()

X_train, _ = build_feature_matrix(df_train, WIN_FEAT, use_liq=True)
y_train_P = df_train['P_kbar'].values
y_train_T = df_train['T_C'].values
y_test_P = df_test['P_kbar'].values
y_test_T = df_test['T_C'].values

# X_train/X_test are numpy arrays; wrap in DataFrames so we can drop columns by
# feature name in the ablation test below.
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

# Refit the canonical models once and record original RMSE.
model_P_fit = clone(model_P); model_P_fit.fit(X_train_df.values, y_train_P)
model_T_fit = clone(model_T); model_T_fit.fit(X_train_df.values, y_train_T)
original_rmse_P = float(np.sqrt(mean_squared_error(
    y_test_P, predict_median(model_P_fit, X_test_df.values))))
original_rmse_T = float(np.sqrt(mean_squared_error(
    y_test_T, predict_median(model_T_fit, X_test_df.values))))
print(f'Baseline RMSE P = {original_rmse_P:.3f} kbar | T = {original_rmse_T:.2f} C')
"""

ABLATION_MD = """### Robustness A. Ablation test

Drop every feature whose name contains `liq_SiO2` or `liq_MgO` (these dominate
SHAP importance) and refit the pressure and temperature models to see how
much performance degrades when the candidate proxies are removed.
"""

ABLATION_CODE = """suspect_substr = ['liq_SiO2', 'liq_MgO']
suspect_cols = [c for c in feature_names
                if any(s in c for s in suspect_substr)]
print(f'Dropping {len(suspect_cols)} features: {suspect_cols}')

X_tr_ab = X_train_df.drop(columns=suspect_cols)
X_te_ab = X_test_df.drop(columns=suspect_cols)

model_ab_P = clone(model_P); model_ab_P.fit(X_tr_ab.values, y_train_P)
model_ab_T = clone(model_T); model_ab_T.fit(X_tr_ab.values, y_train_T)
rmse_ab_P = float(np.sqrt(mean_squared_error(
    y_test_P, predict_median(model_ab_P, X_te_ab.values))))
rmse_ab_T = float(np.sqrt(mean_squared_error(
    y_test_T, predict_median(model_ab_T, X_te_ab.values))))

print(f'P RMSE: {original_rmse_P:.3f} -> {rmse_ab_P:.3f} kbar '
      f'(delta = {rmse_ab_P - original_rmse_P:+.3f})')
print(f'T RMSE: {original_rmse_T:.2f} -> {rmse_ab_T:.2f} C    '
      f'(delta = {rmse_ab_T - original_rmse_T:+.2f})')
"""

PROXY_MD = """### Robustness B. Experimental proxy check

If liquid SiO2 and MgO are simply acting as proxies for the experimental P
and T set points (because different labs run different ranges), the raw
scatterplots should show strong monotonic trends.
"""

PROXY_CODE = """fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df_liq['P_kbar'], df_liq['liq_SiO2'],
                alpha=0.5, c='#2c7bb6', edgecolor='k', linewidth=0.5)
axes[0].set_xlabel('Pressure (kbar)')
axes[0].set_ylabel('Liquid SiO2 (wt%)')
axes[0].set_title('liq_SiO2 vs. Pressure')

axes[1].scatter(df_liq['T_C'], df_liq['liq_MgO'],
                alpha=0.5, c='#d7191c', edgecolor='k', linewidth=0.5)
axes[1].set_xlabel('Temperature (C)')
axes[1].set_ylabel('Liquid MgO (wt%)')
axes[1].set_title('liq_MgO vs. Temperature')

plt.tight_layout()
plt.savefig(FIGURES / 'fig_nb06_proxy_check.png', dpi=200, bbox_inches='tight')
plt.show()
"""

CORR_MD = """### Robustness C. Feature correlation heatmap

Spearman-style visual check among the leading feature candidates and the
targets. Strong cross-correlation between `liq_*` oxides and P or T would
support the proxy concern.
"""

CORR_CODE = """import seaborn as sns

candidate_cols = [c for c in
                  ['Al_VI', 'CaO', 'liq_SiO2', 'liq_MgO', 'P_kbar', 'T_C']
                  if c in df_liq.columns]
if len(candidate_cols) < 4:
    print(f'Skipping correlation check: only {candidate_cols} present')
else:
    corr = df_liq[candidate_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1,
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation: candidate dominators vs. targets')
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_nb06_correlation_check.png',
                dpi=200, bbox_inches='tight')
    plt.show()
"""

YRAND_MD = """### Robustness D. Y-randomization (permutation) test

Shuffle the training labels to destroy the physical relationship, refit with
identical hyperparameters, and predict the real test set. A healthy model
collapses to `R2 <= 0` here; leakage would keep `R2 > 0`.
"""

YRAND_CODE = """rng = np.random.default_rng(SEED_SPLIT)
y_train_P_shuffled = rng.permutation(y_train_P)

model_sanity = clone(model_P)
model_sanity.fit(X_train_df.values, y_train_P_shuffled)
pred_shuffled = predict_median(model_sanity, X_test_df.values)
rmse_shuffled = float(np.sqrt(mean_squared_error(y_test_P, pred_shuffled)))
r2_shuffled = float(r2_score(y_test_P, pred_shuffled))

print(f'Baseline P RMSE:   {original_rmse_P:.3f} kbar')
print(f'Shuffled P RMSE:   {rmse_shuffled:.3f} kbar')
print(f'Shuffled R^2:      {r2_shuffled:+.3f}   (expected <= 0)')
"""

DUMMY_MD = """### Robustness E. Dummy regressor baseline

Constant-mean predictor. Any nontrivial model must beat this.
"""

DUMMY_CODE = """from sklearn.dummy import DummyRegressor

dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train_df.values, y_train_P)
rmse_dummy = float(np.sqrt(mean_squared_error(
    y_test_P, dummy_model.predict(X_test_df.values))))
print(f'RF P RMSE:    {original_rmse_P:.3f} kbar')
print(f'Dummy P RMSE: {rmse_dummy:.3f} kbar')
"""

PERFECT_MD = """### Robustness F. Perfect-signal injection

Inject a fake feature that equals `y_P + small_noise`. Under the canonical
constrained hyperparameters, a single perfect feature may not dominate
because `max_features` controls the split pool. The unconstrained variant
forces the model to consider the fake feature every split, and a linear
sanity check confirms the fake feature is present in the data.
"""

PERFECT_CODE = """# Constrained: injects fake feature, uses canonical hyperparameters.
X_tr_fake = X_train_df.copy()
X_te_fake = X_test_df.copy()
X_tr_fake['FAKE_PERFECT_SIGNAL'] = (y_train_P
                                    + rng.normal(0, 0.1, len(y_train_P)))
X_te_fake['FAKE_PERFECT_SIGNAL'] = (y_test_P
                                    + rng.normal(0, 0.1, len(y_test_P)))

model_fake = clone(model_P)
model_fake.fit(X_tr_fake.values, y_train_P)
rmse_fake = float(np.sqrt(mean_squared_error(
    y_test_P, predict_median(model_fake, X_te_fake.values))))
print(f'Constrained perfect-signal P RMSE: {rmse_fake:.3f} kbar '
      f'(baseline {original_rmse_P:.3f})')

# Unconstrained: max_features=None forces the fake feature to be seen every split.
from sklearn.ensemble import RandomForestRegressor
model_fake_unc = RandomForestRegressor(
    n_estimators=200, max_features=None,
    random_state=SEED_MODEL, n_jobs=-1,
)
model_fake_unc.fit(X_tr_fake.values, y_train_P)
rmse_fake_unc = float(np.sqrt(mean_squared_error(
    y_test_P, predict_median(model_fake_unc, X_te_fake.values))))
print(f'Unconstrained perfect-signal P RMSE: {rmse_fake_unc:.3f} kbar '
      f'(expected ~0.1)')

# Linear sanity check: a plain OLS with the fake feature included should nail it.
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_tr_fake.fillna(0).values, y_train_P)
rmse_linear = float(np.sqrt(mean_squared_error(
    y_test_P, linear_model.predict(X_te_fake.fillna(0).values))))
print(f'Linear perfect-signal P RMSE: {rmse_linear:.3f} kbar')
"""


def md(source):
    return nbformat.v4.new_markdown_cell(source=source)


def code(source):
    return nbformat.v4.new_code_cell(source=source)


def main():
    nb = nbformat.read(str(NB), as_version=4)

    # Skip if appendix already present (idempotent).
    if any('Robustness checks (merged from deprecated nb06b)' in c.source
           for c in nb.cells if c.cell_type == 'markdown'):
        print('Appendix already present; skipping.')
        return

    appendix = [
        md(APPENDIX_INTRO),
        code(SETUP_BASELINE),
        md(ABLATION_MD), code(ABLATION_CODE),
        md(PROXY_MD), code(PROXY_CODE),
        md(CORR_MD), code(CORR_CODE),
        md(YRAND_MD), code(YRAND_CODE),
        md(DUMMY_MD), code(DUMMY_CODE),
        md(PERFECT_MD), code(PERFECT_CODE),
    ]
    nb.cells.extend(appendix)
    nbformat.write(nb, str(NB))
    print(f'Appended {len(appendix)} cells; nb06 now has {len(nb.cells)} cells')


if __name__ == '__main__':
    main()
