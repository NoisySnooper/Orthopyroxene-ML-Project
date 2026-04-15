"""One-shot NB03 editor for Phase 2: replaces Phase 3.3b MD + code with
Optuna variants and appends Phase 3.9 (resampling), Phase 3.10 (stacking),
Phase 3.11 (Optuna diagnostics), and Phase 3.12 (three-family comparison).

Run once. Safe to re-run: it replaces cells by ID for the 3.3b pair and
looks for existing v9-tagged cell IDs before inserting new ones."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / 'notebooks' / 'nb03_baseline_models.ipynb'


def _md(source: str, cell_id: str) -> dict:
    return {
        'cell_type': 'markdown',
        'id': cell_id,
        'metadata': {},
        'source': source.splitlines(keepends=True),
    }


def _code(source: str, cell_id: str) -> dict:
    return {
        'cell_type': 'code',
        'id': cell_id,
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# New cell sources
# ---------------------------------------------------------------------------

MD_3_3B = """## Phase 3.3b: Optuna TPE hyperparameter search (pre-training)

**What this replaces.** v7 and v8 used `HalvingRandomSearchCV`. Halving allocates more CV budget to survivors of each successive round, which is defensible in principle but has a known failure mode: promising hyperparameter regions can be eliminated prematurely when their initial trials happen to underperform on small early-round budgets. For regression with noisy CV scores this produces search paths sensitive to the ordering of the first few trials.

**What this uses now.** Optuna's Tree-structured Parzen Estimator (`TPESampler(multivariate=True)`) models the joint posterior of hyperparameter values and validation performance, then draws new trials from regions of high expected improvement. `MedianPruner(n_startup_trials=10)` kills trials below the running median after a 10-trial warm-up. See `docs/optuna_strategy.md` for full rationale.

**Sweep layout.** 4 models x 2 targets x 2 tracks x 3 feature_sets = **48 studies**. 50 TPE trials per study, 3-fold `GroupKFold` by Citation, `n_jobs=4` per inner CV, neg-RMSE scoring. Tune-once-then-freeze: one TPE study produces frozen params reused across all 20 outer split seeds in Phase 3.4.

**Artifacts produced.**
- `results/optuna_studies/study_{model}_{target}_{track}_{feature_set}.joblib` (48 files).
- `results/nb03_optuna_best_params.json` - consumed by Phase 3.4 as `frozen_params_store`.
- `results/nb03_optuna_best_params_partial.json` - checkpoint every 10 studies.
- `results/nb03_hyperparameter_search.csv` - per-study summary (best RMSE, n_trials, n_pruned, elapsed).

**Sanity checks in the cell output.**
1. Plateau: fractional improvement between trial 25 and trial 50. Flags non-converged configs.
2. Trial failure rate: >20% signals misconfigured search space.
3. Pruner kill rate: ~0% means the pruner did nothing; ~100% means the sampler struggled.
"""

CODE_3_3B = '''# Phase 3.3b: Optuna TPE hyperparameter search (replaces HalvingRandomSearchCV).
# One study per (model, target, track, feature_set). Outer seed-42 train/test
# split by Citation groups. Frozen best params go to results/nb03_optuna_best_params.json
# and drive Phase 3.4 unchanged.
from config import (
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_PER_TRIAL, OPTUNA_N_JOBS_INNER,
    OPTUNA_SEED, OPTUNA_STUDIES_DIR, OPTUNA_BEST_PARAMS_FILE,
    OPTUNA_BEST_PARAMS_PARTIAL_FILE, OPTUNA_MODELS, OPTUNA_TARGETS,
    OPTUNA_TRACKS, OPTUNA_FEATURE_SETS,
)
from src.optuna_search import optuna_search
from sklearn.model_selection import GroupShuffleSplit

OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_LOCAL = TEMP_DIR / OPTUNA_BEST_PARAMS_PARTIAL_FILE
FROZEN_LOCAL  = TEMP_DIR / OPTUNA_BEST_PARAMS_FILE
for p in [PARTIAL_LOCAL, FROZEN_LOCAL]:
    if p.exists():
        p.unlink()

LOG_PATH_SEARCH = LOGS / 'nb03_search.log'
logger_search = logging.getLogger('nb03_search')
logger_search.setLevel(logging.INFO)
logger_search.handlers.clear()
_fh = logging.FileHandler(LOG_PATH_SEARCH, mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger_search.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logger_search.addHandler(_sh)
logger_search.info('Phase 3.3b: Optuna TPE | tune_seed=%d | n_trials=%d | n_jobs_inner=%d',
                   OPTUNA_SEED, OPTUNA_N_TRIALS, OPTUNA_N_JOBS_INNER)

frozen_params_store = {}
search_records = []

_totals = len(OPTUNA_TRACKS) * len(OPTUNA_FEATURE_SETS) * len(OPTUNA_MODELS) * len(OPTUNA_TARGETS)
pbar = tqdm(total=_totals, desc='Optuna TPE sweep (seed 42)', smoothing=0.05)
t0_search = time.time()
_study_counter = 0

for track_name, df_track, use_liq in [('opx_liq', df_liq, True),
                                       ('opx_only', df_opx, False)]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=OPTUNA_SEED)
    tr_pos, te_pos = next(gss.split(df_track,
                                    groups=df_track['Citation'].values))
    df_train = df_track.iloc[tr_pos].copy()
    df_test  = df_track.iloc[te_pos].copy()

    for feat_name in OPTUNA_FEATURE_SETS:
        X_tr, _ = build_feature_matrix(df_train, feat_name, use_liq)
        X_te, _ = build_feature_matrix(df_test,  feat_name, use_liq)
        groups_tr = df_train['Citation'].values

        for model_name in OPTUNA_MODELS:
            for target_name in OPTUNA_TARGETS:
                y_tr = df_train[target_name].values
                y_te = df_test[target_name].values
                pbar.set_postfix(trk=track_name, f=feat_name,
                                 m=model_name, t=target_name)
                t0 = time.time()
                study_path = OPTUNA_STUDIES_DIR / f'study_{model_name}_{target_name}_{track_name}_{feat_name}.joblib'
                study_name = f'{model_name}|{target_name}|{track_name}|{feat_name}'
                try:
                    out = optuna_search(
                        model_name, X_tr, y_tr, groups_tr,
                        n_trials=OPTUNA_N_TRIALS,
                        seed=OPTUNA_SEED,
                        timeout_per_trial=OPTUNA_TIMEOUT_PER_TRIAL,
                        n_jobs_inner=OPTUNA_N_JOBS_INNER,
                        study_save_path=study_path,
                        study_name=study_name,
                    )
                    frozen_key = (track_name, feat_name, model_name, target_name)
                    frozen_params_store[frozen_key] = out['best_params']

                    # Per-study diagnostics.
                    study = out['study']
                    completed = [t for t in study.trials
                                 if t.state.name == 'COMPLETE']
                    pruned   = [t for t in study.trials
                                 if t.state.name == 'PRUNED']
                    failed   = [t for t in study.trials
                                 if t.state.name == 'FAIL']
                    if len(completed) >= 50:
                        best25 = min(t.value for t in completed[:25])
                        best50 = min(t.value for t in completed[:50])
                        plateau_frac = 0.0 if best25 == 0 else (best25 - best50) / abs(best25)
                    else:
                        plateau_frac = float('nan')
                    search_records.append({
                        'track': track_name, 'feature_set': feat_name,
                        'model_name': model_name, 'target': target_name,
                        'best_rmse_cv': out['best_score'],
                        'best_trial':   out['best_trial'],
                        'n_trials_completed': len(completed),
                        'n_trials_pruned':    len(pruned),
                        'n_trials_failed':    len(failed),
                        'pruner_kill_rate':   len(pruned) / max(len(study.trials), 1),
                        'failure_rate':       len(failed) / max(len(study.trials), 1),
                        'plateau_frac_25_to_50': plateau_frac,
                        'best_params':        str(out['best_params']),
                        'elapsed_s':          round(time.time() - t0, 1),
                    })
                    logger_search.info(
                        '[OPTUNA] %s %s %s %s rmse_cv=%.3f pruned=%d failed=%d (%.0fs)',
                        track_name, feat_name, model_name, target_name,
                        out['best_score'], len(pruned), len(failed),
                        time.time() - t0,
                    )
                except Exception as e:
                    logger_search.error(
                        'FAIL %s/%s/%s/%s: %s: %s',
                        track_name, feat_name, model_name, target_name,
                        type(e).__name__, e,
                    )
                finally:
                    _study_counter += 1
                    pbar.update(1)
                    if _study_counter % 10 == 0:
                        _partial = {'||'.join(k): v for k, v in frozen_params_store.items()}
                        with open(PARTIAL_LOCAL, 'w') as f:
                            json.dump(_partial, f, default=str)
                        with open(RESULTS / OPTUNA_BEST_PARAMS_PARTIAL_FILE, 'w') as f:
                            json.dump(_partial, f, default=str)

pbar.close()

_frozen_flat = {'||'.join(k): {kk: (vv.item() if isinstance(vv, np.generic) else vv)
                                for kk, vv in v.items()}
                for k, v in frozen_params_store.items()}
with open(FROZEN_LOCAL, 'w') as f:
    json.dump(_frozen_flat, f, default=str)
with open(RESULTS / OPTUNA_BEST_PARAMS_FILE, 'w') as f:
    json.dump(_frozen_flat, f, default=str)

search_df = pd.DataFrame(search_records)
search_df.to_csv(RESULTS / 'nb03_hyperparameter_search.csv', index=False)

elapsed_h = (time.time() - t0_search) / 3600
logger_search.info('Optuna sweep complete: %d combos in %.2fh',
                   len(frozen_params_store), elapsed_h)
print(f'\\nOptuna sweep complete: {len(frozen_params_store)} studies in {elapsed_h:.2f}h')
print(f'Frozen params persisted to {RESULTS / OPTUNA_BEST_PARAMS_FILE}')
print(f'Per-study summary: {RESULTS / "nb03_hyperparameter_search.csv"}')
print('\\nPer-study diagnostics (first 24 rows):')
print(search_df.sort_values(['track', 'target', 'best_rmse_cv'])
                 .round(3).head(24).to_string(index=False))

# Flag non-converged or unstable studies.
_flag_plateau = search_df['plateau_frac_25_to_50'].fillna(0) > 0.05
_flag_failure = search_df['failure_rate'] > 0.20
if _flag_plateau.any():
    print('\\nPLATEAU FLAG (>5% improvement after trial 25):')
    print(search_df.loc[_flag_plateau, ['track','feature_set','model_name','target','plateau_frac_25_to_50']].to_string(index=False))
if _flag_failure.any():
    print('\\nFAILURE FLAG (>20% trials failed):')
    print(search_df.loc[_flag_failure, ['track','feature_set','model_name','target','n_trials_failed','failure_rate']].to_string(index=False))
'''

MD_PHASE_3_9 = """## Phase 3.9: Tempered resampling training track

**Motivation.** The opx-liq ExPetDB training set is highly imbalanced in P-T space: roughly 40% of samples sit at 0-10 kbar, only 20% span 20-60 kbar, and T coverage is similarly skewed toward crustal conditions. Tree regressors trained on imbalanced data regress toward the training mean on the underrepresented tail; Agreda-Lopez et al. (2024) document this exact mechanism as the cause of the ArcPL T underprediction we see in v7/v8 models.

**Strategy.** Tempered resampling on a range-based 5x5 P-T grid. For each occupied cell with current count `c_i`, the target count is `t_i = round((c_i + u_i) / 2)` where `u_i = n_total / n_occupied_cells`. Cells above target are subsampled without replacement; cells below target are bootstrapped with replacement. Unoccupied cells stay empty - we do not invent experimental support that does not exist. See `docs/resampling_strategy.md` for the bin-edge justification (range, not quintile) and the `bootstrap=False` discipline for RF/ERT on resampled training sets.

**Scope of application.**
- Applied to outer training fold only, for the **opx_liq** track at the seed-42 canonical split.
- Produces a parallel set of canonical `_resampled` models alongside the baseline canonical set from Phase 3.4. Baseline models stay untouched.
- **Not** applied to CV folds (biases hyperparameter search), outer test set (biases in-domain evaluation), or ArcPL external validation (biases external reporting).

**RF/ERT bootstrap discipline.** `RandomForestRegressor` and `ExtraTreesRegressor` default to `bootstrap=True`, which resamples internally for each tree. Stacking that on top of an already-bootstrap-augmented training set creates a bootstrap-of-a-bootstrap that shrinks effective per-tree diversity and inflates ensemble variance. For the `_resampled` models we pass `bootstrap=False` so the outer tempered resampling is the sole source of sample-level randomization. XGBoost and HistGradientBoosting use per-tree row fractions (`subsample`), not replacement sampling, so no adjustment is needed for them.

**Artifacts.**
- `models/model_{MODEL}_{target}_opx_liq_{feat}_resampled.joblib` x up to 12 models (4 base x 3 feature_sets, minus any that fail).
- `results/nb03_resampling_diagnostics.csv` - per-cell action log (current, target, action taken).
- `results/nb03_resampling_summary.json` - n_in/n_out/n_occupied/grown/shrunk/held totals.
- Two figures: `fig_nb03_resampling_pt_distribution` (hexbin before/after + marginals) and `fig_nb03_resampling_impact_on_metrics` (T/P RMSE by P bin).

**Expected outcomes and interpretation.** Medium probability of 2-5 C improvement on T RMSE (ArcPL head-to-head), minor risk of worse RMSE on the most common P-T region. If RMSE gets worse after resampling we keep the non-resampled canonical set as primary and document the experiment as a negative-result ablation in the manuscript.
"""

CODE_3_9_A = '''# Phase 3.9 (part A): produce the resampled opx-liq training set and save diagnostics.
from config import (
    RESAMPLING_N_P_BINS, RESAMPLING_N_T_BINS, RESAMPLING_SEED,
    RESAMPLING_DIAGNOSTICS_FILE, RESAMPLING_SUMMARY_FILE,
)
from src.resampling import tempered_resample, compute_pt_grid_bins, assign_pt_cells
from src.features import build_feature_matrix

# Load canonical seed-42 split written in Phase 3.3b.
_tr_liq = np.load(DATA_SPLITS / 'train_indices_opx_liq.npy')
_te_liq = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
df_train_liq = df_liq.loc[_tr_liq].reset_index(drop=True)
df_test_liq  = df_liq.loc[_te_liq].reset_index(drop=True)

df_train_res, resampling_diag = tempered_resample(
    df_train_liq,
    target_col_p='P_kbar', target_col_t='T_C',
    n_p_bins=RESAMPLING_N_P_BINS, n_t_bins=RESAMPLING_N_T_BINS,
    seed=RESAMPLING_SEED,
)
print('Tempered resampling complete:')
print('  n_in :', resampling_diag['summary']['n_in'])
print('  n_out:', resampling_diag['summary']['n_out'])
print('  occupied cells:', resampling_diag['summary']['n_occupied_cells'])
print('  grown / shrunk / held:',
      resampling_diag['summary']['n_cells_grown'],
      resampling_diag['summary']['n_cells_shrunk'],
      resampling_diag['summary']['n_cells_held'])

# Persist diagnostics.
resampling_diag['actions'].to_csv(RESULTS / RESAMPLING_DIAGNOSTICS_FILE, index=False)
with open(RESULTS / RESAMPLING_SUMMARY_FILE, 'w') as f:
    _summ = dict(resampling_diag['summary'])
    _summ['p_edges'] = resampling_diag['bin_edges'][0].tolist()
    _summ['t_edges'] = resampling_diag['bin_edges'][1].tolist()
    json.dump(_summ, f, indent=2)

# Load frozen Optuna best params.
with open(RESULTS / 'nb03_optuna_best_params.json') as f:
    _frozen = json.load(f)
def _params_for(track, feat, model, target):
    return _frozen.get('||'.join([track, feat, model, target]), None)

# Train one resampled model per (model, feature_set, target) for opx_liq track.
resampled_records = []
for feat_name in FEATURE_METHODS:
    X_tr_res, _ = build_feature_matrix(df_train_res, feat_name, use_liq=True)
    X_te,    _  = build_feature_matrix(df_test_liq,  feat_name, use_liq=True)
    for model_name in ['RF', 'ERT', 'XGB', 'GB']:
        for target_name in ['T_C', 'P_kbar']:
            params = _params_for('opx_liq', feat_name, model_name, target_name)
            if params is None:
                print('SKIP missing frozen params:', feat_name, model_name, target_name)
                continue
            params = dict(params)
            # RF/ERT on resampled training: disable internal bootstrap.
            if model_name in ('RF', 'ERT'):
                params['bootstrap'] = False
            y_tr = df_train_res[target_name].values
            y_te = df_test_liq[target_name].values
            est = clone(BASE_MODELS[model_name]())
            est.set_params(**params)
            est.fit(X_tr_res, y_tr)
            pred_te = predict_median(est, X_te)
            rmse_te = float(np.sqrt(mean_squared_error(y_te, pred_te)))
            mae_te  = float(mean_absolute_error(y_te, pred_te))
            r2_te   = float(r2_score(y_te, pred_te))
            fname = f'model_{model_name}_{target_name}_opx_liq_{feat_name}_resampled.joblib'
            joblib.dump(est, MODELS / fname)
            resampled_records.append({
                'model_name': model_name, 'feature_set': feat_name,
                'target': target_name, 'track': 'opx_liq',
                'rmse_test': rmse_te, 'mae_test': mae_te, 'r2_test': r2_te,
                'filename': fname,
                'best_params': str(params),
            })

resampled_df = pd.DataFrame(resampled_records)
resampled_df.to_csv(RESULTS / 'nb03_resampled_results.csv', index=False)
print('\\nResampled canonical models saved:', len(resampled_df))
print(resampled_df[['model_name','feature_set','target','rmse_test']]
      .sort_values(['target','rmse_test']).round(3).to_string(index=False))
'''

CODE_3_9_FIG1 = '''# Figure: fig_nb03_resampling_pt_distribution
# 2x2 hexbin (before/after) x (P on y, T on x) plus marginal histograms
# on top and right of each hexbin. Two rows, two cols.
from matplotlib import gridspec

fig = plt.figure(figsize=(14, 10))
outer = gridspec.GridSpec(1, 2, wspace=0.35)

def _hex_panel(ax_hex, ax_top, ax_right, df_in, title):
    hb = ax_hex.hexbin(df_in['T_C'], df_in['P_kbar'], gridsize=28,
                       cmap='viridis', mincnt=1)
    ax_hex.set_xlabel('T (C)')
    ax_hex.set_ylabel('P (kbar)')
    ax_hex.set_title(title, fontsize=11)
    ax_top.hist(df_in['T_C'], bins=30, color='#444444', alpha=0.85)
    ax_top.set_xticks([]); ax_top.set_ylabel('n')
    ax_right.hist(df_in['P_kbar'], bins=30, orientation='horizontal',
                  color='#444444', alpha=0.85)
    ax_right.set_yticks([]); ax_right.set_xlabel('n')
    return hb

for col, (dfin, title) in enumerate([
    (df_train_liq,  f'Before resampling (n={len(df_train_liq)})'),
    (df_train_res,  f'After tempered resampling (n={len(df_train_res)})'),
]):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0, col],
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )
    ax_top   = fig.add_subplot(inner[0, 0])
    ax_hex   = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])
    hb = _hex_panel(ax_hex, ax_top, ax_right, dfin, title)
    plt.colorbar(hb, ax=ax_hex, shrink=0.7, label='count')

fig.suptitle('P-T distribution of opx_liq training data, tempered resampling',
             fontsize=13, y=0.995)

for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_resampling_pt_distribution.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()

# Companion bar chart: per-cell count bars.
fig2, ax = plt.subplots(figsize=(11, 5))
actions = resampling_diag['actions'].copy()
actions['cell'] = actions.apply(lambda r: f'P{r.p_cell},T{r.t_cell}', axis=1)
x = np.arange(len(actions))
width = 0.4
ax.bar(x - width/2, actions['current'], width=width, color='#888888', label='before')
ax.bar(x + width/2, actions['target'],  width=width, color='#0072B2', label='after')
ax.set_xticks(x); ax.set_xticklabels(actions['cell'], rotation=90, fontsize=8)
ax.set_ylabel('count'); ax.set_xlabel('P-T cell (5x5 range-based grid)')
ax.set_title('Per-cell current vs target counts (tempered target)')
ax.legend()
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig2.savefig(FIGURES / f'fig_nb03_resampling_pt_distribution_bars.{ext}',
                 bbox_inches='tight', dpi=300)
plt.show()
'''

CODE_3_9_FIG2 = '''# Figure: fig_nb03_resampling_impact_on_metrics
# For opx_liq track, per target: baseline vs resampled per-P-bin RMSE.
# Uses the best per-model baseline from the Phase 3.4 multi-seed table
# (seed 42) and the matching resampled model from results/nb03_resampled_results.csv.
from config import PER_FAMILY_WINNERS_FILE
from src.data import load_per_family_winners

seed42 = multi_seed_df[multi_seed_df['split_seed'] == 42]
_winners = load_per_family_winners(RESULTS)

P_BIN_EDGES = np.array([0, 5, 10, 15, 20, 30, 60], dtype=float)

def _per_bin_rmse(y_true, y_pred, edges=P_BIN_EDGES):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) if n else float('nan')
        rows.append({'lo': lo, 'hi': hi, 'n': n, 'rmse': rmse})
    return pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
P_test = df_test_liq['P_kbar'].values
for ax, target in zip(axes, ['T_C', 'P_kbar']):
    # Baseline: pick per-family forest winner (best RF/ERT).
    fam = _winners['forest_family'][f'opx_liq_{target}']
    model_name = fam['model_name']; feat_name = fam['feature_set']
    base_model = joblib.load(MODELS / fam['filename'])
    X_te_base, _ = build_feature_matrix(df_test_liq, feat_name, use_liq=True)
    base_pred = predict_median(base_model, X_te_base)
    y_true = df_test_liq[target].values
    base_bins = _per_bin_rmse(y_true, base_pred)
    base_bins['P_mid'] = (base_bins['lo'] + base_bins['hi']) / 2
    ax.plot(base_bins['P_mid'], base_bins['rmse'], '-o', color='#444444',
            label=f'baseline {model_name}/{feat_name}')

    # Resampled: same model/feat pair if present.
    match = resampled_df.query('model_name == @model_name and feature_set == @feat_name and target == @target')
    if len(match):
        row = match.iloc[0]
        res_model = joblib.load(MODELS / row['filename'])
        X_te_res, _ = build_feature_matrix(df_test_liq, feat_name, use_liq=True)
        res_pred = predict_median(res_model, X_te_res)
        res_bins = _per_bin_rmse(y_true, res_pred)
        res_bins['P_mid'] = (res_bins['lo'] + res_bins['hi']) / 2
        ax.plot(res_bins['P_mid'], res_bins['rmse'], '-s', color='#0072B2',
                label=f'resampled {model_name}/{feat_name}')
    ax.set_xlabel('P bin midpoint (kbar)')
    ax.set_ylabel(f'{target} test RMSE')
    ax.set_title(f'opx_liq {target}: per-P-bin RMSE')
    ax.legend()
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_resampling_impact_on_metrics.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

MD_PHASE_3_10 = """## Phase 3.10: Ridge stacking meta-model

**Motivation.** Phase 3.4 trains four base learners per (target, track, feature_set): RF, ERT, XGB, GB. When base models make partly uncorrelated errors, a weighted blend cancels noise and outperforms any single model. When they make near-identical predictions, stacking collapses to the best single weight - an honest, informative result. Either way we learn something about the model-family ceiling.

**Meta-model.** `sklearn.linear_model.RidgeCV` with 5-fold CV over `alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`. Input: out-of-fold predictions from the four base models; target: the same ground truth used in Phase 3.4. Ridge regularization prevents the unregularized-linear failure mode where correlated base predictors produce near-singular Gram matrices and wildly swinging coefficients. RF/GB meta-models were rejected as overparameterized for 4-column input and non-interpretable.

**Per-base-model feature_set.** Each of the four base models uses its own winning feature_set from Phase 3.5 canonical selection (e.g., RF-T-opx_liq winner might be alr, XGB-T-opx_liq winner might be raw). The stack therefore blends predictions from heterogeneous feature representations. This matches deployment behavior: in production each canonical model uses its own feature_set, so the meta-model sees the same input distribution at training and prediction time. See `docs/stacking_strategy.md`.

**OOF generation.** 5-fold `GroupKFold` by Citation on the seed-42 canonical training set. For each base model: for each fold, fit on train indices, predict on val indices, store predictions at val positions. Result: one length-N OOF vector per base model. Column-stack the four into an (N, 4) matrix; fit RidgeCV against ground truth.

**New canonical family.** `stacked_family` joins `forest_family` and `boosted_family`. Canonical count grows from 8 (4 winners x 2 targets) to 12 (8 base + 4 stacked). Base winners remain valid; stacking adds one more canonical model per (target, track), it does not replace the base winners.

**Artifacts per (target, track).**
- `models/meta_ridge_{target}_{track}_stacked.joblib` - fitted RidgeCV (coef_, alpha_, intercept_).
- `results/nb03_stacking_oof_matrix_{target}_{track}.npz` - (N, 4) OOF matrix + y_train for reproducibility.
- `results/nb03_stacked_members_{target}_{track}.json` - manifest read by `src.data.load_stacked_model`.
- Three figures: `fig_nb03_stacking_weights`, `fig_nb03_stacking_oof_correlation`, `fig_nb03_stacking_vs_base_comparison`.

**Sanity checks in the cell output.**
1. *Alpha at endpoint*: if `alpha_ == 100.0` or `0.001`, widen the search range.
2. *Single-model collapse*: if one weight `> 0.9` and others near zero, stacking picked one model. Honest finding, not a bug.
3. *Negative weights*: small negatives (|c| < 0.5) are fine; large negatives signal severe multicollinearity.
4. *Blank OOF column*: drop that base from the stack and note in the manifest.
"""

CODE_3_10 = '''# Phase 3.10: Ridge stacking meta-model per (target, track).
from config import (
    STACKING_ALPHAS, STACKING_CV_FOLDS, STACKING_SEED, STACKING_BASE_ORDER,
    STACKING_OOF_MATRIX_TEMPLATE, STACKING_MANIFEST_TEMPLATE,
    STACKING_META_TEMPLATE, STACKING_DIAGNOSTICS_FILE,
)
from src.stacking import (
    generate_oof_predictions, fit_ridge_meta_model,
    compute_oof_correlation_matrix,
)
from src.data import load_per_family_winners, load_splits
from sklearn.model_selection import GroupKFold

# Load winners so we know each base model's winning feature_set per (target, track).
_winners = load_per_family_winners(RESULTS)

# Map base model -> winning feature_set from either family's winner list.
def _find_feature_set(track, target, model_name):
    # Check forest + boosted family blocks; the model_name will be in whichever
    # family it belongs to, but we accept either.
    for fam in ('forest_family', 'boosted_family'):
        block = _winners.get(fam, {}).get(f'{track}_{target}')
        if block and block.get('model_name') == model_name:
            return block.get('feature_set')
    # Fall back: use the per-family winning feature_set of whichever family contains model_name.
    if model_name in ('RF', 'ERT'):
        return _winners['forest_family'][f'{track}_{target}']['feature_set']
    if model_name in ('XGB', 'GB'):
        return _winners['boosted_family'][f'{track}_{target}']['feature_set']
    raise KeyError(model_name)

# Load frozen Optuna best params for each study.
with open(RESULTS / 'nb03_optuna_best_params.json') as f:
    _frozen = json.load(f)
def _params_for(track, feat, model, target):
    return _frozen.get('||'.join([track, feat, model, target]), None)

STACKING_DIAGNOSTICS = {}

for track_name, df_track, use_liq in [('opx_liq', df_liq, True),
                                       ('opx_only', df_opx, False)]:
    tr_idx, te_idx = load_splits(track_name)
    df_train = df_track.loc[tr_idx].reset_index(drop=True)
    df_test  = df_track.loc[te_idx].reset_index(drop=True)
    groups_tr = df_train['Citation'].values
    cv = GroupKFold(n_splits=STACKING_CV_FOLDS)

    for target_name in ['T_C', 'P_kbar']:
        y_train = df_train[target_name].values
        y_test  = df_test[target_name].values

        # Build one OOF vector per base model using that base's winning feature_set.
        oof_dict = {}
        feat_per_base = {}
        for base in STACKING_BASE_ORDER:
            feat = _find_feature_set(track_name, target_name, base)
            feat_per_base[base] = feat
            X_train_b, _ = build_feature_matrix(df_train, feat, use_liq=use_liq)
            params = _params_for(track_name, feat, base, target_name)
            if params is None:
                raise RuntimeError(f'Missing frozen params for {track_name} {target_name} {base} {feat}')
            params = dict(params)
            def _ctor(seed, _p=params, _b=base):
                est = clone(BASE_MODELS[_b]())
                est.set_params(**_p)
                return est
            oof_dict[base] = generate_oof_predictions(
                _ctor, X_train_b, y_train, groups_tr, cv, seed=STACKING_SEED,
            )

        # OOF correlation matrix (diagnostic + figure input).
        oof_corr = compute_oof_correlation_matrix(oof_dict, STACKING_BASE_ORDER)

        # Stack OOFs and fit RidgeCV.
        oof_matrix = np.column_stack([oof_dict[b] for b in STACKING_BASE_ORDER])
        meta = fit_ridge_meta_model(oof_matrix, y_train, alphas=STACKING_ALPHAS,
                                    cv=STACKING_CV_FOLDS)

        # Persist artifacts.
        oof_path = RESULTS / STACKING_OOF_MATRIX_TEMPLATE.format(
            target=target_name, track=track_name)
        np.savez(oof_path, oof_matrix=oof_matrix, y_train=y_train,
                 base_order=np.array(STACKING_BASE_ORDER))

        meta_filename = STACKING_META_TEMPLATE.format(
            target=target_name, track=track_name)
        joblib.dump(meta, MODELS / meta_filename)

        # Manifest for load_stacked_model.
        members = {}
        for base in STACKING_BASE_ORDER:
            feat = feat_per_base[base]
            # Look up the on-disk filename from seed-42 multi-seed results.
            fam_key = 'forest_family' if base in ('RF', 'ERT') else 'boosted_family'
            # Baseline winner filename (per-family winner uses feature_set winning for that family).
            # For stacking members we use each base's own (model, target, track, feature_set)
            # canonical joblib, which follows the convention model_{MODEL}_{target}_{track}_{feat}.joblib.
            base_filename = f'model_{base}_{target_name}_{track_name}_{feat}.joblib'
            members[base] = {'filename': base_filename, 'feature_set': feat}
        manifest = {
            'target': target_name, 'track': track_name,
            'members': members,
            'meta_filename': meta_filename,
        }
        with open(RESULTS / STACKING_MANIFEST_TEMPLATE.format(
                target=target_name, track=track_name), 'w') as f:
            json.dump(manifest, f, indent=2)

        # Compute test-set stacked prediction via load_stacked_model.
        from src.data import load_stacked_model
        predictor = load_stacked_model(target_name, track_name,
                                       models_dir=MODELS, results_dir=RESULTS)
        stacked_pred = predictor.predict(df_test)
        stacked_rmse = float(np.sqrt(mean_squared_error(y_test, stacked_pred)))
        stacked_mae  = float(mean_absolute_error(y_test, stacked_pred))
        stacked_r2   = float(r2_score(y_test, stacked_pred))

        # Sanity flags.
        alpha_sel = float(meta.alpha_)
        coefs = dict(zip(STACKING_BASE_ORDER, meta.coef_.tolist()))
        max_abs = max(abs(c) for c in coefs.values())
        min_coef = min(coefs.values())
        flags = []
        if alpha_sel == max(STACKING_ALPHAS) or alpha_sel == min(STACKING_ALPHAS):
            flags.append('alpha_at_endpoint')
        if max_abs > 0.9 and sum(abs(c) > 0.1 for c in coefs.values()) == 1:
            flags.append('single_model_collapse')
        if min_coef < -0.5:
            flags.append('large_negative_weight')

        STACKING_DIAGNOSTICS[f'{target_name}|{track_name}'] = {
            'alpha_selected': alpha_sel,
            'intercept': float(meta.intercept_),
            'coefficients': coefs,
            'feature_set_per_base': feat_per_base,
            'oof_correlation': oof_corr.to_dict(),
            'test_rmse_stacked': stacked_rmse,
            'test_mae_stacked':  stacked_mae,
            'test_r2_stacked':   stacked_r2,
            'flags': flags,
        }

        print(f'[STACKING] {target_name:6s} / {track_name:8s} '
              f'alpha={alpha_sel:.3g} weights={coefs} '
              f'test_rmse={stacked_rmse:.3f} flags={flags}')

with open(RESULTS / STACKING_DIAGNOSTICS_FILE, 'w') as f:
    json.dump(STACKING_DIAGNOSTICS, f, indent=2)
print(f'\\nStacking diagnostics saved to {RESULTS / STACKING_DIAGNOSTICS_FILE}')
'''

CODE_3_10_FIG_WEIGHTS = '''# Figure: fig_nb03_stacking_weights
# 2x2 bar chart: RidgeCV coef_ per base model, one subplot per (target, track).
# Alpha value annotated on each subplot.
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
panels = [('T_C','opx_liq'), ('P_kbar','opx_liq'),
          ('T_C','opx_only'), ('P_kbar','opx_only')]
palette = ['#0072B2', '#56B4E9', '#D55E00', '#E69F00']
for ax, (t, trk) in zip(axes.flat, panels):
    key = f'{t}|{trk}'
    d = STACKING_DIAGNOSTICS[key]
    bases = list(d['coefficients'].keys())
    vals  = [d['coefficients'][b] for b in bases]
    bars  = ax.bar(bases, vals, color=palette)
    ax.axhline(0, color='k', linewidth=0.7)
    ax.set_title(f'{t} / {trk}\\nalpha={d["alpha_selected"]:.3g}, '
                 f'intercept={d["intercept"]:.2f}')
    ax.set_ylabel('Ridge coefficient')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f'{v:.2f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9)
fig.suptitle('Ridge stacking weights per (target, track)')
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_stacking_weights.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

CODE_3_10_FIG_OOF_CORR = '''# Figure: fig_nb03_stacking_oof_correlation
# 2x2 grid of 4x4 OOF correlation heatmaps.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
panels = [('T_C','opx_liq'), ('P_kbar','opx_liq'),
          ('T_C','opx_only'), ('P_kbar','opx_only')]
for ax, (t, trk) in zip(axes.flat, panels):
    key = f'{t}|{trk}'
    corr = pd.DataFrame(STACKING_DIAGNOSTICS[key]['oof_correlation'])
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    ax.set_title(f'{t} / {trk}')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(corr.iloc[i,j]) > 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.7, label='corr')
fig.suptitle('OOF prediction correlation across base learners')
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_stacking_oof_correlation.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

CODE_3_10_FIG_COMPARE = '''# Figure: fig_nb03_stacking_vs_base_comparison
# Bar chart: stacked test RMSE vs best base test RMSE per (target, track),
# with bootstrap 95% CIs on the stacked RMSE.
from numpy.random import default_rng

rng_b = default_rng(42)
def _bootstrap_rmse_ci(y_true, y_pred, n_boot=1000):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_b.integers(0, n, size=n)
        vals[i] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

compare_rows = []
panels = [('T_C','opx_liq'), ('P_kbar','opx_liq'),
          ('T_C','opx_only'), ('P_kbar','opx_only')]
seed42 = multi_seed_df[multi_seed_df['split_seed'] == 42]
for target, track in panels:
    # Best base at seed 42: lowest rmse_test among the 4 models x 3 feats.
    sub = seed42[(seed42['track']==track) & (seed42['target']==target)]
    best_row = sub.sort_values('rmse_test').iloc[0]
    # Stacked RMSE from diagnostics.
    d = STACKING_DIAGNOSTICS[f'{target}|{track}']
    # Bootstrap CI for stacked prediction on test set.
    from src.data import load_stacked_model, load_splits
    _tr, _te = load_splits(track)
    df_te = (df_liq if track=='opx_liq' else df_opx).loc[_te].reset_index(drop=True)
    y_true = df_te[target].values
    predictor = load_stacked_model(target, track, models_dir=MODELS, results_dir=RESULTS)
    stacked_pred = predictor.predict(df_te)
    lo, hi = _bootstrap_rmse_ci(y_true, stacked_pred)
    compare_rows.append({
        'target': target, 'track': track,
        'best_base_name':  best_row['model_name'],
        'best_base_feat':  best_row['feature_set'],
        'best_base_rmse':  best_row['rmse_test'],
        'stacked_rmse':    d['test_rmse_stacked'],
        'stacked_lo':      lo, 'stacked_hi': hi,
    })
compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(RESULTS / 'nb03_stacking_vs_base.csv', index=False)
print(compare_df.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(compare_df))
w = 0.35
ax.bar(x - w/2, compare_df['best_base_rmse'], width=w, color='#888888',
       label='best base model')
stacked_vals = compare_df['stacked_rmse'].values
yerr = np.vstack([stacked_vals - compare_df['stacked_lo'].values,
                  compare_df['stacked_hi'].values - stacked_vals])
ax.bar(x + w/2, stacked_vals, width=w, color='#0072B2',
       yerr=yerr, capsize=4, label='stacked (95% boot CI)')
labels = [f'{r.target}\\n{r.track}\\n({r.best_base_name}/{r.best_base_feat})'
          for r in compare_df.itertuples()]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Test RMSE')
ax.set_title('Stacked vs best single base model (seed-42 test split)')
ax.legend()
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_stacking_vs_base_comparison.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

MD_PHASE_3_11 = """## Phase 3.11: Optuna search diagnostics

**What this shows.** Two companion figures that justify our Optuna setup to readers who want to see that the TPE sweep was carefully specified rather than just "set n_trials=50 and hope":

1. `fig_nb03_optuna_search_progress` - a 4x2 grid of best-so-far convergence curves, one panel per (model, target), overlaying all 6 (track x feature_set) curves within each panel. A flat tail past trial 30-40 means the search converged within our budget; a still-decreasing tail flags configs that would have benefitted from more trials.

2. `fig_nb03_optuna_hyperparameter_importance` - horizontal bars per base model showing `optuna.importance.get_param_importances` averaged across the 6 studies of that model. This tells us which hyperparameters actually mattered and which were irrelevant within the searched ranges.

Reads the persisted `study_*.joblib` files from `results/optuna_studies/` (written in Phase 3.3b).
"""

CODE_3_11_PROGRESS = '''# Figure: fig_nb03_optuna_search_progress
# 4x2 grid: rows = models (RF,ERT,XGB,GB), cols = targets (T_C, P_kbar).
# Each panel overlays 6 convergence curves (2 tracks x 3 feature_sets).
import optuna as _optuna
from config import OPTUNA_STUDIES_DIR

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
model_order  = ['RF','ERT','XGB','GB']
target_order = ['T_C','P_kbar']
track_order  = ['opx_liq', 'opx_only']
feat_order   = ['raw','alr','pwlr']
palette = {
    ('opx_liq','raw'):   '#0072B2', ('opx_liq','alr'):   '#009E73',
    ('opx_liq','pwlr'):  '#CC79A7', ('opx_only','raw'):  '#D55E00',
    ('opx_only','alr'):  '#E69F00', ('opx_only','pwlr'): '#F0E442',
}
for i, m in enumerate(model_order):
    for j, t in enumerate(target_order):
        ax = axes[i, j]
        for trk in track_order:
            for feat in feat_order:
                path = OPTUNA_STUDIES_DIR / f'study_{m}_{t}_{trk}_{feat}.joblib'
                if not path.exists():
                    continue
                study = joblib.load(path)
                completed = [tr for tr in study.trials
                             if tr.state.name == 'COMPLETE']
                if not completed:
                    continue
                values = [tr.value for tr in completed]
                best_so_far = np.minimum.accumulate(values)
                ax.plot(range(1, len(best_so_far)+1), best_so_far,
                        color=palette[(trk, feat)], alpha=0.85,
                        label=f'{trk}|{feat}')
        ax.set_title(f'{m} / {t}')
        ax.set_ylabel('best CV RMSE')
        if i == 3:
            ax.set_xlabel('trial')
        if i == 0 and j == 1:
            ax.legend(fontsize=8, loc='upper right')
fig.suptitle('Optuna TPE convergence (best-so-far per study)', y=0.995)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_optuna_search_progress.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

CODE_3_11_IMPORTANCE = '''# Figure: fig_nb03_optuna_hyperparameter_importance
# Hyperparameter importance per base model, averaged across the 6 studies
# belonging to that model (2 tracks x 3 feature_sets x 1 target x 1 model
# ... actually 2 tracks x 3 feats x 2 targets = 12 studies per model).
# Horizontal bars, one subplot per model.
from collections import defaultdict
try:
    from optuna.importance import get_param_importances
except Exception:
    get_param_importances = None

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, m in zip(axes.flat, ['RF','ERT','XGB','GB']):
    agg = defaultdict(list)
    paths = sorted(OPTUNA_STUDIES_DIR.glob(f'study_{m}_*.joblib'))
    for p in paths:
        study = joblib.load(p)
        if len(study.trials) < 10 or get_param_importances is None:
            continue
        try:
            imp = get_param_importances(study)
        except Exception:
            continue
        for k, v in imp.items():
            agg[k].append(v)
    if not agg:
        ax.set_title(f'{m}: no importance available'); continue
    means = {k: float(np.mean(v)) for k, v in agg.items()}
    order = sorted(means, key=means.get)
    ax.barh(order, [means[k] for k in order], color='#0072B2')
    ax.set_xlabel('mean importance')
    ax.set_title(f'{m} (n_studies={len(paths)})')
fig.suptitle('Optuna hyperparameter importance per base model')
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_optuna_hyperparameter_importance.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()
'''

MD_PHASE_3_12 = """## Phase 3.12: Three-family comparison figure

Summary plot rolling up the v9 canonical families: **forest** (winner of RF vs ERT), **boosted** (winner of XGB vs GB), and **stacked** (Ridge meta of all four). One bar per family x target x track, seed-42 test RMSE with 95% bootstrap CIs. Intended as the NB03 headline figure for the manuscript Methods section.
"""

CODE_3_12 = '''# Figure: fig_nb03_three_family_comparison
# Seed-42 test RMSE per (target, track, family) with 95% bootstrap CIs.
from src.data import load_canonical_model, load_splits, load_stacked_model
from src.features import build_feature_matrix
from src.models import predict_median

rng_tf = np.random.default_rng(42)
def _boot_rmse(y_true, y_pred, n_boot=1000):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_tf.integers(0, n, size=n)
        vals[i] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

rows = []
panels = [('T_C','opx_liq'), ('P_kbar','opx_liq'),
          ('T_C','opx_only'), ('P_kbar','opx_only')]
for target, track in panels:
    tr_idx, te_idx = load_splits(track)
    df_track = df_liq if track=='opx_liq' else df_opx
    df_test  = df_track.loc[te_idx].reset_index(drop=True)
    y_true   = df_test[target].values
    use_liq  = (track=='opx_liq')
    # forest
    fam = _winners['forest_family'][f'{track}_{target}']
    model = joblib.load(MODELS / fam['filename'])
    X_te, _ = build_feature_matrix(df_test, fam['feature_set'], use_liq=use_liq)
    pred = predict_median(model, X_te)
    m_f, lo_f, hi_f = _boot_rmse(y_true, pred)
    rows.append({'target': target, 'track': track, 'family': 'forest',
                 'label': f'{fam["model_name"]}/{fam["feature_set"]}',
                 'mean': m_f, 'lo': lo_f, 'hi': hi_f})
    # boosted
    fam = _winners['boosted_family'][f'{track}_{target}']
    model = joblib.load(MODELS / fam['filename'])
    X_te, _ = build_feature_matrix(df_test, fam['feature_set'], use_liq=use_liq)
    pred = predict_median(model, X_te)
    m_b, lo_b, hi_b = _boot_rmse(y_true, pred)
    rows.append({'target': target, 'track': track, 'family': 'boosted',
                 'label': f'{fam["model_name"]}/{fam["feature_set"]}',
                 'mean': m_b, 'lo': lo_b, 'hi': hi_b})
    # stacked
    predictor = load_stacked_model(target, track, models_dir=MODELS, results_dir=RESULTS)
    pred = predictor.predict(df_test)
    m_s, lo_s, hi_s = _boot_rmse(y_true, pred)
    rows.append({'target': target, 'track': track, 'family': 'stacked',
                 'label': 'RidgeCV(4 bases)',
                 'mean': m_s, 'lo': lo_s, 'hi': hi_s})

tf_df = pd.DataFrame(rows)
tf_df.to_csv(RESULTS / 'nb03_three_family_comparison.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 6))
fam_order = ['forest', 'boosted', 'stacked']
fam_colors = {'forest':'#0072B2', 'boosted':'#D55E00', 'stacked':'#009E73'}
x = np.arange(len(panels))
w = 0.27
for i, fam in enumerate(fam_order):
    sub = tf_df[tf_df['family']==fam]
    means = sub['mean'].values
    yerr = np.vstack([means - sub['lo'].values, sub['hi'].values - means])
    ax.bar(x + (i-1)*w, means, width=w, color=fam_colors[fam],
           yerr=yerr, capsize=3, label=fam)
ax.set_xticks(x)
ax.set_xticklabels([f'{t}\\n{trk}' for t, trk in panels])
ax.set_ylabel('Test RMSE (seed 42, 95% boot CI)')
ax.set_title('Forest vs boosted vs stacked family (canonical comparison)')
ax.legend()
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGURES / f'fig_nb03_three_family_comparison.{ext}',
                bbox_inches='tight', dpi=300)
plt.show()

print(tf_df.round(3).to_string(index=False))
'''


# ---------------------------------------------------------------------------
# Apply edits
# ---------------------------------------------------------------------------

def main():
    with open(NB_PATH) as f:
        nb = json.load(f)
    cells = nb['cells']

    def _find(id_):
        for i, c in enumerate(cells):
            if c.get('id') == id_:
                return i
        return -1

    # 1) Replace Phase 3.3b MD and code.
    i_md  = _find('md-phase3r-3b')
    i_cd  = _find('code-phase3r-3b')
    if i_md < 0 or i_cd < 0:
        print('Could not locate Phase 3.3b cells'); sys.exit(1)
    cells[i_md] = _md(MD_3_3B,  'md-phase3r-3b')
    cells[i_cd] = _code(CODE_3_3B, 'code-phase3r-3b')

    # 2) Remove any existing v9 Phase 3.9..3.12 cells (idempotency).
    v9_ids = {
        'md-phase3r-9', 'code-phase3r-9a', 'code-phase3r-9-fig1', 'code-phase3r-9-fig2',
        'md-phase3r-10', 'code-phase3r-10', 'code-phase3r-10-fig-weights',
        'code-phase3r-10-fig-oofcorr', 'code-phase3r-10-fig-compare',
        'md-phase3r-11', 'code-phase3r-11-progress', 'code-phase3r-11-importance',
        'md-phase3r-12', 'code-phase3r-12',
    }
    cells[:] = [c for c in cells if c.get('id') not in v9_ids]

    # 3) Append new v9 phases at the end.
    tail = [
        _md(MD_PHASE_3_9, 'md-phase3r-9'),
        _code(CODE_3_9_A, 'code-phase3r-9a'),
        _code(CODE_3_9_FIG1, 'code-phase3r-9-fig1'),
        _code(CODE_3_9_FIG2, 'code-phase3r-9-fig2'),

        _md(MD_PHASE_3_10, 'md-phase3r-10'),
        _code(CODE_3_10, 'code-phase3r-10'),
        _code(CODE_3_10_FIG_WEIGHTS, 'code-phase3r-10-fig-weights'),
        _code(CODE_3_10_FIG_OOF_CORR, 'code-phase3r-10-fig-oofcorr'),
        _code(CODE_3_10_FIG_COMPARE, 'code-phase3r-10-fig-compare'),

        _md(MD_PHASE_3_11, 'md-phase3r-11'),
        _code(CODE_3_11_PROGRESS, 'code-phase3r-11-progress'),
        _code(CODE_3_11_IMPORTANCE, 'code-phase3r-11-importance'),

        _md(MD_PHASE_3_12, 'md-phase3r-12'),
        _code(CODE_3_12, 'code-phase3r-12'),
    ]
    cells.extend(tail)

    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'NB03 edited in place: {len(cells)} cells total')


if __name__ == '__main__':
    main()
