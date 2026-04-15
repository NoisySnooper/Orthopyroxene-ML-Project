"""Rebuild nb03 for v7: Parts B, D, G in one idempotent pass.

- Part G: N_SPLIT_REPS=20 (via config.py) instead of hardcoded 10.
- Part D: Write canonical seed-42 train/test splits BEFORE the 20-seed
  training loop so every downstream phase reads from disk.
- Part B: Per-family winner selection. Writes
  `results/nb03_per_family_winners.json` (single source of truth) and
  saves 8 canonical joblibs named `model_{target}_{track}_{family}.joblib`.

Run:
    .venv\\Scripts\\python.exe scripts/rebuild_nb03_v7.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NBPATH = ROOT / "notebooks" / "nb03_baseline_models.ipynb"

V7_MARKER_IMPORTS = "MODEL_FAMILIES, TIEBREAKER_RULE"
V7_MARKER_SPLIT_WRITE = "Phase 3.3b: canonical split write (v7 Part D)"
V7_MARKER_PER_FAMILY = "Phase 3.5: per-family winner selection (v7 Part B)"
V7_MARKER_3_6 = "Phase 3.6: canonical test predictions per (family, target, track)"
V7_MARKER_3_7 = "Phase 3.7: verification (v7 Part B/D/G)"


def _patch_imports(nb) -> bool:
    cell = nb.cells[2]
    if V7_MARKER_IMPORTS in cell.source:
        return False
    pat = re.compile(
        r"(from config import \([^)]*?)"
        r"(OPX_RAW_OXIDES, OPX_FULL_OXIDES, LIQ_OXIDES,\s*\))",
        re.DOTALL,
    )
    m = pat.search(cell.source)
    if not m:
        raise RuntimeError("import cell does not match expected shape")
    addition = (
        "    OPX_RAW_OXIDES, OPX_FULL_OXIDES, LIQ_OXIDES,\n"
        "    N_SPLIT_REPS, SPLIT_SEEDS, FEATURE_METHODS,\n"
        "    MODEL_FAMILIES, TIEBREAKER_RULE,\n"
        ")"
    )
    cell.source = pat.sub(r"\1" + addition, cell.source, count=1)
    return True


def _patch_cell_6_config(nb) -> bool:
    cell = nb.cells[6]
    # Remove hardcoded N_SPLIT_REPS / SPLIT_SEEDS / FEATURE_METHODS / N_AUG lines
    # so they are picked up from config.py via the import cell.
    if "SPLIT_SEEDS = list(range(42, 42 + N_SPLIT_REPS))" not in cell.source:
        return False  # already patched
    replacement = (
        "# Model definitions and global config. N_SPLIT_REPS, SPLIT_SEEDS,\n"
        "# FEATURE_METHODS come from config.py. v7 Part G raises N_SPLIT_REPS\n"
        "# from 10 to 20.\n"
        "TUNE_SEED = 42\n"
        "N_AUG = 1  # augmentation disabled per sensitivity test results\n"
    )
    cell.source = re.sub(
        r"# Model definitions and global config\n"
        r"N_SPLIT_REPS = \d+\n"
        r"SPLIT_SEEDS = list\(range\(42, 42 \+ N_SPLIT_REPS\)\)\n"
        r"TUNE_SEED = 42\n\n"
        r"# 3-method design[^\n]*\n# mathematically identical[^\n]*\n"
        r"FEATURE_METHODS = \['raw', 'alr', 'pwlr'\]\n"
        r"N_AUG = 1[^\n]*\n",
        replacement,
        cell.source,
        count=1,
    )
    return True


def _insert_canonical_split_write(nb) -> bool:
    # Insert at the top of the Phase 3.4 training cell (cell 13).
    cell = nb.cells[13]
    if V7_MARKER_SPLIT_WRITE in cell.source:
        return False
    preamble = (
        "# Phase 3.3b: canonical split write (v7 Part D).\n"
        "# Seed-42 train/test indices are persisted to disk BEFORE the 20-seed\n"
        "# training loop begins. Every downstream phase and notebook loads from\n"
        "# these NPY files rather than recomputing GroupShuffleSplit.\n"
        "_gss42_liq = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
        "_tr_l, _te_l = next(_gss42_liq.split(df_liq, groups=df_liq['Citation'].values))\n"
        "np.save(DATA_SPLITS / 'train_indices_opx_liq.npy', df_liq.index.values[_tr_l])\n"
        "np.save(DATA_SPLITS / 'test_indices_opx_liq.npy',  df_liq.index.values[_te_l])\n"
        "_gss42_opx = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
        "_tr_o, _te_o = next(_gss42_opx.split(df_opx, groups=df_opx['Citation'].values))\n"
        "np.save(DATA_SPLITS / 'train_indices_opx.npy', df_opx.index.values[_tr_o])\n"
        "np.save(DATA_SPLITS / 'test_indices_opx.npy',  df_opx.index.values[_te_o])\n"
        "print(f'Canonical seed-42 splits written: '\n"
        "      f'{len(_tr_l)}/{len(_te_l)} liq, {len(_tr_o)}/{len(_te_o)} opx-only')\n\n"
    )
    cell.source = preamble + cell.source
    return True


PER_FAMILY_CELL = '''# Phase 3.5: per-family winner selection (v7 Part B)
#
# Two families (forest = RF+ERT, boosted = GB+XGB). For each family and
# each (target, track), pick the (feature_set, model_name) with lowest
# 20-seed mean test RMSE. Tiebreaker: if another family candidate is
# within 1 std of the minimum, prefer the family's `tiebreaker_preferred`
# (RF for forest, XGB for boosted).

agg = multi_seed_df.groupby(
    ['track', 'target', 'feature_set', 'model_name']
).agg(
    rmse_test_mean=('rmse_test', 'mean'),
    rmse_test_std=('rmse_test', 'std'),
    r2_test_mean=('r2_test', 'mean'),
    r2_test_std=('r2_test', 'std'),
    overfit_ratio_mean=('overfit_ratio', 'mean'),
).reset_index()
agg.to_csv(RESULTS / 'nb03_multi_seed_summary.csv', index=False)


def _choose_family_winner(fam_subset, pref_model):
    """Pick min-RMSE row; if pref_model is within 1 std, prefer it."""
    if fam_subset.empty:
        return None
    min_row = fam_subset.loc[fam_subset['rmse_test_mean'].idxmin()]
    tol = float(min_row['rmse_test_std']) if pd.notna(min_row['rmse_test_std']) else 0.0
    band = fam_subset[
        fam_subset['rmse_test_mean'] <= min_row['rmse_test_mean'] + tol
    ]
    if pref_model in band['model_name'].values:
        pref_rows = band[band['model_name'] == pref_model]
        return pref_rows.loc[pref_rows['rmse_test_mean'].idxmin()]
    return min_row


per_family_winners = {
    'forest_family': {},
    'boosted_family': {},
    'tiebreaker_rule': TIEBREAKER_RULE,
    'selection_metadata': {
        'n_seeds': int(N_SPLIT_REPS),
        'split_seeds': list(SPLIT_SEEDS),
        'feature_methods': list(FEATURE_METHODS),
    },
}

canonical_files_written = []

for family_name, fam_cfg in MODEL_FAMILIES.items():
    fam_key = f'{family_name}_family'
    for track in ['opx_only', 'opx_liq']:
        for target in ['T_C', 'P_kbar']:
            subset = agg[
                (agg['track'] == track) &
                (agg['target'] == target) &
                (agg['model_name'].isin(fam_cfg['candidates']))
            ]
            chosen = _choose_family_winner(subset, fam_cfg['tiebreaker_preferred'])
            if chosen is None:
                print(f'WARNING: no candidates for {family_name}/{track}/{target}')
                continue
            model_name = str(chosen['model_name'])
            feat_set = str(chosen['feature_set'])
            filename = f'model_{target}_{track}_{family_name}.joblib'
            min_row = subset.loc[subset['rmse_test_mean'].idxmin()]
            winning_model = seed42_models.get(
                (track, target, model_name, feat_set)
            )
            if winning_model is None:
                print(f'WARNING: missing seed42 model for '
                      f'({track}, {target}, {model_name}, {feat_set})')
                continue
            joblib.dump(winning_model, MODELS / filename)
            canonical_files_written.append(filename)
            per_family_winners[fam_key][f'{track}_{target}'] = {
                'model_name': model_name,
                'feature_set': feat_set,
                'filename': filename,
                'rmse_test_mean': float(chosen['rmse_test_mean']),
                'rmse_test_std': float(chosen['rmse_test_std']),
                'r2_test_mean': float(chosen['r2_test_mean']),
                'r2_test_std': float(chosen['r2_test_std']),
                'tiebreaker_applied': bool(model_name != min_row['model_name']),
            }

with open(RESULTS / 'nb03_per_family_winners.json', 'w') as f:
    json.dump(per_family_winners, f, indent=2)

# Flat summary CSV for quick scanning / reporting.
flat_rows = []
for fam in ['forest_family', 'boosted_family']:
    for track in ['opx_only', 'opx_liq']:
        for target in ['T_C', 'P_kbar']:
            spec = per_family_winners[fam].get(f'{track}_{target}')
            if spec is None:
                continue
            flat_rows.append({
                'family': fam.replace('_family', ''),
                'track': track,
                'target': target,
                'model_name': spec['model_name'],
                'feature_set': spec['feature_set'],
                'rmse_test_mean': spec['rmse_test_mean'],
                'rmse_test_std': spec['rmse_test_std'],
                'r2_test_mean': spec['r2_test_mean'],
                'filename': spec['filename'],
                'tiebreaker_applied': spec['tiebreaker_applied'],
            })
pd.DataFrame(flat_rows).to_csv(
    RESULTS / 'nb03_per_family_winners_flat.csv', index=False
)

print(f'Per-family winners written: {RESULTS / "nb03_per_family_winners.json"}')
print(f'Canonical joblibs written: {len(canonical_files_written)}')
for fn in canonical_files_written:
    print(f'  {fn}')
'''

PHASE_3_6_CELL = '''# Phase 3.6: canonical test predictions per (family, target, track)
from src.data import canonical_model_path, load_per_family_winners

_winners = load_per_family_winners(RESULTS)
_test_idx_liq = np.load(DATA_SPLITS / 'test_indices_opx_liq.npy')
_test_idx_opx = np.load(DATA_SPLITS / 'test_indices_opx.npy')
_df_test_liq = df_liq.loc[_test_idx_liq].copy()
_df_test_opx = df_opx.loc[_test_idx_opx].copy()

for _family in ['forest', 'boosted']:
    _fam_block = _winners[f'{_family}_family']
    for _track, _df_test_track, _use_liq in [
        ('opx_liq', _df_test_liq, True),
        ('opx_only', _df_test_opx, False),
    ]:
        _records = {
            'idx': _df_test_track.index,
            'y_T_true': _df_test_track['T_C'].values,
            'y_P_true': _df_test_track['P_kbar'].values,
        }
        for _target in ['T_C', 'P_kbar']:
            _spec = _fam_block.get(f'{_track}_{_target}')
            if _spec is None:
                continue
            _X_test, _ = build_feature_matrix(
                _df_test_track, _spec['feature_set'], _use_liq
            )
            _mdl = joblib.load(
                canonical_model_path(_target, _track, _family, MODELS, RESULTS)
            )
            _records[f'{_target}_pred'] = predict_median(_mdl, _X_test)
        pd.DataFrame(_records).to_csv(
            RESULTS / f'nb03_canonical_test_predictions_{_track}_{_family}.csv',
            index=False,
        )
print('Per-family canonical test predictions saved.')
'''

PHASE_3_7_CELL = '''# Phase 3.7: verification (v7 Part B/D/G)
errors = []

required_files = [
    RESULTS / 'nb03_per_family_winners.json',
    RESULTS / 'nb03_per_family_winners_flat.csv',
    RESULTS / 'nb03_multi_seed_results.csv',
    RESULTS / 'nb03_multi_seed_summary.csv',
    DATA_SPLITS / 'train_indices_opx_liq.npy',
    DATA_SPLITS / 'test_indices_opx_liq.npy',
    DATA_SPLITS / 'train_indices_opx.npy',
    DATA_SPLITS / 'test_indices_opx.npy',
]
for _path in required_files:
    if not _path.exists():
        errors.append(f'MISSING: {_path}')

expected_rows = N_SPLIT_REPS * 2 * len(FEATURE_METHODS) * 4 * 2
actual_rows = len(multi_seed_df)
if actual_rows < expected_rows * 0.95:
    errors.append(
        f'multi_seed_df too sparse: got {actual_rows}, expected {expected_rows}'
    )

n_nan = multi_seed_df['rmse_test'].isna().sum()
if n_nan > 0:
    errors.append(f'multi_seed_df has {n_nan} NaN rmse_test rows')

with open(RESULTS / 'nb03_per_family_winners.json') as f:
    wjson = json.load(f)

for fam in ('forest_family', 'boosted_family'):
    fam_block = wjson.get(fam)
    if not fam_block:
        errors.append(f'per_family_winners missing {fam} block')
        continue
    for track in ('opx_only', 'opx_liq'):
        for target in ('T_C', 'P_kbar'):
            k = f'{track}_{target}'
            spec = fam_block.get(k)
            if spec is None:
                errors.append(f'missing winner: {fam}/{k}')
                continue
            fname = spec.get('filename')
            if not fname or not (MODELS / fname).exists():
                errors.append(f'missing canonical joblib: {MODELS / (fname or "")}')

if errors:
    print('=== VERIFICATION FAILED ===')
    for e in errors:
        print(f'  - {e}')
    raise AssertionError(f'{len(errors)} verification errors')

print('=== NB03 complete ===')
print(f'  {len(multi_seed_df)} multi-seed rows (expected {expected_rows})')
print(f'  two families (forest, boosted); 8 canonical joblibs written')
print(f'  selection source of truth: {RESULTS / "nb03_per_family_winners.json"}')
'''


def _replace_cell(nb, idx, new_source, marker):
    cell = nb.cells[idx]
    if marker in cell.source:
        return False
    cell.source = new_source
    return True


def main() -> int:
    if not NBPATH.exists():
        print(f"ERROR: {NBPATH} not found")
        return 1
    nb = nbformat.read(str(NBPATH), as_version=4)
    changed = {
        "imports": _patch_imports(nb),
        "cell6_config": _patch_cell_6_config(nb),
        "split_write": _insert_canonical_split_write(nb),
        "phase35": _replace_cell(nb, 21, PER_FAMILY_CELL, V7_MARKER_PER_FAMILY),
        "phase36": _replace_cell(nb, 23, PHASE_3_6_CELL, V7_MARKER_3_6),
        "phase37": _replace_cell(nb, 25, PHASE_3_7_CELL, V7_MARKER_3_7),
    }
    if any(changed.values()):
        nbformat.write(nb, str(NBPATH))
        print("nb03 v7 rebuild applied:")
        for k, v in changed.items():
            print(f"  {k}: {'changed' if v else 'no-op (already patched)'}")
    else:
        print("nb03 v7 rebuild: no changes (already applied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
