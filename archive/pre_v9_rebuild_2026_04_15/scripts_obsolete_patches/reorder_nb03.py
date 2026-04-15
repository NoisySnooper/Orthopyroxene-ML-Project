"""Reorder nb03 cells so hyperparameter search + n_aug sensitivity precede
the final training loop.

New narrative:
  Phase 3R.1-2  : imports, data load, feature helpers, model defs
  Phase 3R.3a   : N_AUG sensitivity (was appendix) -> motivates N_AUG=1
  Phase 3R.3b   : Hyperparameter search (seed 42, HalvingRandomSearchCV)
                  -> produces frozen_params_store
  Phase 3R.4    : Final training (frozen params, all 10 seeds, eval_frozen)
  Phase 3R.5-6  : heatmaps, winner selection, canonical save
  Phase 3R.7    : verification

The prior design interleaved search and training in one loop. This split makes
the workflow explicit: first decide (augmentation + hyperparameters), then
train.
"""
from __future__ import annotations

from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / 'nb03_baseline_models.ipynb'

nb = nbformat.read(str(NB_PATH), as_version=4)
by_id = {c.get('id'): c for c in nb.cells}


def md(src: str, cell_id: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source=src)
    cell.id = cell_id
    return cell


def code(src: str, cell_id: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(source=src)
    cell.id = cell_id
    cell.outputs = []
    cell.execution_count = None
    return cell


HYPERPARAM_SEARCH_SRC = '''# Phase 3R.3b: Hyperparameter search (HalvingRandomSearchCV on seed 42 only)
# Fits HalvingRandomSearchCV for every (track, feature_set, model, target)
# combination using the tune seed. Persists best_params to frozen_params_store
# and the refit best estimators to seed42_models. The final training pass below
# fits all 10 seeds (including seed 42) with these frozen parameters, so the
# hyperparameter decision is explicit and auditable before any multi-seed work.

# Fresh checkpoint state.
PARTIAL_LOCAL = TEMP_DIR / 'nb03c_partial.csv'
FROZEN_LOCAL = TEMP_DIR / 'nb03c_frozen_params.json'
for p in [PARTIAL_LOCAL, FROZEN_LOCAL]:
    if p.exists():
        p.unlink()

LOG_PATH_SEARCH = TEMP_DIR / 'nb03_search.log'
logger_search = logging.getLogger('nb03_search')
logger_search.setLevel(logging.INFO)
logger_search.handlers.clear()
_fh = logging.FileHandler(LOG_PATH_SEARCH, mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger_search.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logger_search.addHandler(_sh)
logger_search.info('Phase 3R.3b: HalvingRandomSearchCV | tune_seed=%d', TUNE_SEED)

frozen_params_store = {}
seed42_models = {}
search_records = []

total_combos = 2 * len(FEATURE_METHODS) * 4 * 2
pbar = tqdm(total=total_combos, desc='Hyperparam search (seed 42)', smoothing=0.05)
t0_search = time.time()

for track_name, df_track, use_liq in [('opx_liq', df_liq, True),
                                       ('opx_only', df_opx, False)]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=TUNE_SEED)
    tr_pos, te_pos = next(gss.split(df_track,
                                     groups=df_track['Citation'].values))
    df_train = df_track.iloc[tr_pos].copy()
    df_test  = df_track.iloc[te_pos].copy()

    for feat_name in FEATURE_METHODS:
        X_tr, _ = build_feature_matrix(df_train, feat_name, use_liq)
        X_te, _ = build_feature_matrix(df_test,  feat_name, use_liq)
        groups_tr = df_train['Citation'].values

        for model_name in ['RF', 'ERT', 'XGB', 'GB']:
            for target_name in ['T_C', 'P_kbar']:
                y_tr = df_train[target_name].values
                y_te = df_test[target_name].values
                pbar.set_postfix(trk=track_name, f=feat_name,
                                 m=model_name, t=target_name)
                t0 = time.time()
                try:
                    result = tune_with_halving(
                        X_tr, y_tr, X_te, y_te,
                        model_name, target_name, TUNE_SEED, groups_tr,
                    )
                    frozen_key = (track_name, feat_name, model_name,
                                  target_name)
                    frozen_params_store[frozen_key] = result['best_params']
                    seed42_models[(track_name, target_name,
                                   model_name, feat_name)] = result.pop('model')
                    search_records.append({
                        'track':               track_name,
                        'feature_set':         feat_name,
                        'model_name':          model_name,
                        'target':              target_name,
                        'rmse_test_seed42':    result['rmse_test'],
                        'mae_test_seed42':     result['mae_test'],
                        'r2_test_seed42':      result['r2_test'],
                        'overfit_ratio_seed42': result['overfit_ratio'],
                        'best_params':         str(result['best_params']),
                        'elapsed_s':           round(time.time() - t0, 1),
                    })
                    logger_search.info(
                        '[SEARCH] %s %s %s %s rmse=%.2f r2=%.3f (%.0fs)',
                        track_name, feat_name, model_name, target_name,
                        result['rmse_test'], result['r2_test'],
                        time.time() - t0,
                    )
                except Exception as e:
                    logger_search.error(
                        'FAIL %s/%s/%s/%s: %s: %s',
                        track_name, feat_name, model_name, target_name,
                        type(e).__name__, e,
                    )
                finally:
                    pbar.update(1)

pbar.close()

with open(FROZEN_LOCAL, 'w') as f:
    json.dump({'||'.join(k): {kk: (vv.item() if isinstance(vv, np.generic) else vv)
                              for kk, vv in v.items()}
               for k, v in frozen_params_store.items()}, f, default=str)

search_df = pd.DataFrame(search_records)
search_df.to_csv(RESULTS / 'nb03_hyperparameter_search.csv', index=False)

elapsed_h = (time.time() - t0_search) / 3600
logger_search.info('Search complete: %d combos in %.2fh',
                   len(frozen_params_store), elapsed_h)
print(f'\\nSearch complete: {len(frozen_params_store)} combos in {elapsed_h:.2f}h')
print(f'Frozen params persisted to {FROZEN_LOCAL.name}')
print('\\nSeed-42 search summary (top of table):')
print(search_df.sort_values(['track', 'target', 'rmse_test_seed42'])
                 .round(3).head(24).to_string(index=False))
'''


FINAL_TRAINING_SRC = '''# Phase 3R.4: Final training (frozen params, all 10 seeds)
# Uses frozen_params_store built by the hyperparameter search above. Each
# (track, feature, model, target) combo is fit 10 times with the same frozen
# params on different 80/20 train/test splits (seeds 42-51). No retuning.

LOG_PATH = TEMP_DIR / 'nb03c_training.log'
logger = logging.getLogger('nb03c')
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fh = logging.FileHandler(LOG_PATH, mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(_sh)
logger.info('=' * 70)
logger.info('Phase 3R.4: final training (frozen) | N_AUG=%d | features=%s',
            N_AUG, FEATURE_METHODS)

multi_seed_results = []

total = 2 * N_SPLIT_REPS * len(FEATURE_METHODS) * 4 * 2
pbar = tqdm(total=total, desc='Training (frozen)', smoothing=0.05)
t0_global = time.time()

for track_name, df_track, use_liq in [('opx_liq', df_liq, True),
                                       ('opx_only', df_opx, False)]:
    for seed in SPLIT_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_pos, te_pos = next(gss.split(df_track,
                                         groups=df_track['Citation'].values))
        df_train = df_track.iloc[tr_pos].copy()
        df_test  = df_track.iloc[te_pos].copy()
        assert set(df_train['Citation']).isdisjoint(set(df_test['Citation']))

        for feat_name in FEATURE_METHODS:
            # N_AUG=1 -> augment_dataframe is an identity copy (see 3R.3a).
            df_tr_use = df_train
            X_tr, _ = build_feature_matrix(df_tr_use, feat_name, use_liq)
            X_te, _ = build_feature_matrix(df_test,  feat_name, use_liq)

            y_T_tr = df_tr_use['T_C'].values
            y_P_tr = df_tr_use['P_kbar'].values
            y_T_te = df_test['T_C'].values
            y_P_te = df_test['P_kbar'].values

            for model_name in ['RF', 'ERT', 'XGB', 'GB']:
                for target_name, y_tr, y_te_arr in [('T_C', y_T_tr, y_T_te),
                                                     ('P_kbar', y_P_tr, y_P_te)]:
                    pbar.set_postfix(trk=track_name, s=seed, f=feat_name,
                                     m=model_name, t=target_name)
                    frozen_key = (track_name, feat_name, model_name,
                                  target_name)
                    if frozen_key not in frozen_params_store:
                        logger.error('Missing frozen params: %s', frozen_key)
                        pbar.update(1)
                        continue
                    t1 = time.time()
                    try:
                        result = eval_frozen(
                            X_tr, y_tr, X_te, y_te_arr,
                            model_name, target_name,
                            frozen_params_store[frozen_key],
                        )
                        result.pop('model')
                        result['split_seed']  = seed
                        result['track']       = track_name
                        result['feature_set'] = feat_name
                        multi_seed_results.append(result)
                        logger.info(
                            '[FROZEN] %s s=%d %s %s %s rmse=%.2f r2=%.3f (%.0fs)',
                            track_name, seed, feat_name, model_name, target_name,
                            result['rmse_test'], result['r2_test'],
                            time.time() - t1,
                        )
                    except Exception as e:
                        logger.error('FAIL %s: %s: %s',
                                     frozen_key, type(e).__name__, e)
                    finally:
                        pbar.update(1)

        pd.DataFrame(multi_seed_results).to_csv(PARTIAL_LOCAL, index=False)
        elapsed_h = (time.time() - t0_global) / 3600
        logger.info('CHECKPOINT %s seed=%d | %d rows | %.2fh',
                    track_name, seed, len(multi_seed_results), elapsed_h)

pbar.close()

multi_seed_df = pd.DataFrame(multi_seed_results)
n_nan = multi_seed_df['rmse_test'].isna().sum()
if n_nan > 0:
    logger.error('CRITICAL: %d NaN rmse_test rows detected', n_nan)
    print(f'WARNING: {n_nan} NaN rows detected. Check log file.')

multi_seed_df['best_params'] = multi_seed_df['best_params'].astype(str)
multi_seed_df.to_csv(RESULTS / 'nb03_multi_seed_results.csv', index=False)
if PARTIAL_LOCAL.exists():
    PARTIAL_LOCAL.unlink()

total_h = (time.time() - t0_global) / 3600
logger.info('COMPLETE: %d rows in %.2fh', len(multi_seed_df), total_h)
print(f'\\nFinal training complete: {len(multi_seed_df)} rows in {total_h:.2f}h')
print(f'NaN rows: {n_nan}')
'''


# --- Assemble new cell list ---

new_cells = [
    by_id['eb48b5eb'],                               # title MD
    by_id['e41790dc'],                               # imports
    by_id['f24a9b8a'],                               # load data
    by_id['c3f20977'],                               # save opx_liq
    by_id['50db43e6'],                               # feature helpers
    by_id['713a561c'],                               # model defs (FEATURE_METHODS, BASE_MODELS, PARAM_GRIDS, tune_with_halving, eval_frozen)
    md('## Phase 3R.3a: N_AUG sensitivity (pre-training exploration)\n\n'
       'Tests `n_aug in {1, 3, 5, 10, 15}` across all three representations, '
       'both targets, and both tracks with default RF and XGB hyperparameters, '
       'then runs a Wilcoxon signed-rank test (1 vs N) per representation. '
       'Augmentation either hurt or failed to significantly help in every cell, '
       'so the 3-method design uses N_AUG=1 (identity augmentation).\n\n'
       'Writes:\n'
       '- `results/nb03_n_aug_sensitivity.csv`\n'
       '- `figures/fig_nb03_n_aug_sensitivity.png`\n'
       '- `figures/fig_nb03_n_aug_overfit.png`',
       'md-phase3r-3a'),
    by_id['e7e9bbcc'],                               # N_AUG sensitivity experiment
    by_id['5e192a1f'],                               # N_AUG plots
    md('## Phase 3R.3b: Hyperparameter search (pre-training)\n\n'
       'HalvingRandomSearchCV on seed 42 for every '
       '(track, feature_set, model, target) combo. The resulting best_params '
       'are frozen and reused across all 10 split seeds in the final training '
       'pass below. Separating search from training makes the hyperparameter '
       'decision explicit and auditable in `results/nb03_hyperparameter_search.csv`.',
       'md-phase3r-3b'),
    code(HYPERPARAM_SEARCH_SRC.rstrip(), 'code-phase3r-3b'),
    md('## Phase 3R.4: Final training (frozen params, 10 seeds)\n\n'
       'Expected: 480 fits (2 tracks x 10 seeds x 3 features x 4 models x 2 '
       'targets). No retuning. Per-seed RMSE/MAE/R2 written to '
       '`results/nb03_multi_seed_results.csv`.',
       'md-phase3r-4'),
    code(FINAL_TRAINING_SRC.rstrip(), 'code-phase3r-4'),
    by_id['359a683c'],                               # section: heatmaps
    by_id['63de16e2'],                               # summary table
    by_id['e84828d1'],                               # seed-42 heatmap
    by_id['83cff4dc'],                               # 10-seed mean heatmap
    by_id['24dca445'],                               # boxplot
    by_id['34dd6d0d'],                               # wilcoxon
    by_id['ef908071'],                               # section: winner
    by_id['9b033565'],                               # winner selection
    by_id['addc7609'],                               # section: canonical save
    by_id['8c9ebb9f'],                               # canonical save
    by_id['b2fecb42'],                               # section: verification
    by_id['b0ec2f7e'],                               # verification
]

nb.cells = new_cells
nbformat.write(nb, str(NB_PATH))
print(f'Rewrote {NB_PATH} with {len(new_cells)} cells.')
