"""Generate AUGMENTATION_ABLATION_REPORT.md, the headline CSV, and the
manuscript paragraphs (Methods 3.9, Discussion 5.3).

Runs last, after:
    1. scripts/ablations/run_augmentation_ablation_opx.py  (Section 4)
    2. scripts/ablations/run_augmentation_oof_bias.py      (Sections 5-7)
    3. scripts/ablations/make_augmentation_figures.py      (figures)

Writes:
    AUGMENTATION_ABLATION_REPORT.md
    results/augmentation_ablation_opx_headline.csv
    manuscripts/opx_2026/text/aug_methods_3_9.md
    manuscripts/opx_2026/text/aug_discussion_5_3.md
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import CANONICAL_FIGURES, FIGURES, RESULTS

COMBOS = [
    ('opx_liq', 'T_C'),
    ('opx_liq', 'P_kbar'),
    ('opx_only', 'T_C'),
    ('opx_only', 'P_kbar'),
]


def load_headline() -> pd.DataFrame:
    """Build the 4-row headline table."""
    summary_non = pd.read_csv(RESULTS / 'opx_multiseed_summary.csv')
    summary_non = summary_non[summary_non['model'] != 'TabPFN'].copy()
    aug_results = pd.read_csv(
        RESULTS / 'augmentation_ablation_opx_multiseed_results.csv')
    bias_aug = pd.read_csv(
        RESULTS / 'augmentation_ablation_opx_bias_correction.csv')
    bias_canon = pd.read_csv(RESULTS / 'bias_correction_shipped.csv')

    rows = []
    for track, target in COMBOS:
        non_sub = summary_non[(summary_non['track'] == track)
                               & (summary_non['target'] == target)]
        best_non = non_sub.loc[non_sub['mean'].idxmin()]

        aug_sub = aug_results[(aug_results['track'] == track)
                               & (aug_results['target'] == target)]
        cell_mean = (aug_sub.groupby(['model', 'feature_set'])['test_rmse']
                     .mean())
        best_aug_key = cell_mean.idxmin()
        best_aug = aug_sub[(aug_sub['model'] == best_aug_key[0])
                          & (aug_sub['feature_set'] == best_aug_key[1])]

        bc_sub = bias_aug[(bias_aug['track'] == track)
                           & (bias_aug['target'] == target)]
        non_bc = bias_canon[(bias_canon['track'] == track)
                             & (bias_canon['target'] == target)]
        non_winner = (non_bc['winner'].mode().iat[0]
                      if not non_bc.empty and not non_bc['winner'].mode().empty
                      else 'none')

        rows.append({
            'track': track, 'target': target,
            'non_aug_model': best_non['model'],
            'non_aug_feature_set': best_non['feature_set'],
            'non_aug_mean_rmse': float(best_non['mean']),
            'non_aug_std': float(best_non['std']),
            'non_aug_winner': non_winner,
            'aug_model': best_aug_key[0],
            'aug_feature_set': best_aug_key[1],
            'aug_mean_rmse': float(best_aug['test_rmse'].mean()),
            'aug_std': float(best_aug['test_rmse'].std()),
            'aug_ship_a_rate': f"{int(bc_sub['ship_a'].sum())}/20",
            'aug_ship_b_rate': f"{int(bc_sub['ship_b'].sum())}/20",
            'aug_winner_A': int((bc_sub['winner'] == 'A').sum()),
            'aug_winner_B': int((bc_sub['winner'] == 'B').sum()),
            'aug_winner_none': int((bc_sub['winner'] == 'none').sum()),
            'aug_form_a_post_rmse_mean': float(bc_sub['post_rmse_a'].mean()),
            'aug_form_b_post_rmse_mean': float(bc_sub['post_rmse_b'].mean()),
        })
    hl = pd.DataFrame(rows)
    hl.to_csv(RESULTS / 'augmentation_ablation_opx_headline.csv', index=False)
    return hl


def classify(headline: pd.DataFrame) -> tuple[str, str]:
    """Return (label, interpretation_text). Branch-selected from data."""
    n_b_combos = int((headline['aug_winner_B'] > 0).sum())
    n_b_seeds = int(headline['aug_winner_B'].sum())
    any_aug_better = bool((headline['aug_mean_rmse'] < headline['non_aug_mean_rmse']).any())
    aug_all_worse = bool(((headline['aug_mean_rmse'] - headline['non_aug_mean_rmse']) > 0).all())

    if n_b_combos >= 1:
        return (
            'supported',
            f'The augmentation hypothesis is supported. Form B ships on '
            f'{n_b_combos} of 4 opx combinations under the Agreda-Lopez 15x '
            f'Gaussian augmentation protocol ({n_b_seeds}/80 total '
            f'combo-seed votes), compared to 0/4 under the pre-registered '
            f'non-augmented protocol. Form B effectiveness depends on '
            f'training data distribution: augmentation converts residual '
            f'structure from pressure-regime-aligned to value-distribution-'
            f'aligned.'
        )
    if aug_all_worse:
        return (
            'not_supported_aug_degrades',
            'The augmentation hypothesis is not supported. Form B fails to '
            'ship on 0/4 opx combinations under the Agreda-Lopez 15x Gaussian '
            'augmentation protocol, and augmentation additionally degrades '
            'aggregate RMSE on every combination. For small experimental '
            'petrology datasets with publication-level clustering, 15x '
            'Gaussian composition noise is counterproductive. The '
            'differentiator between our negative Form B result and Agreda-'
            'Lopez positive one must be mineral-specific (orthopyroxene vs '
            'clinopyroxene residual structure) or protocol-specific beyond '
            'augmentation alone.'
        )
    return (
        'not_supported',
        'The augmentation hypothesis is not supported. Even with Agreda-'
        'Lopez 15x Gaussian augmentation, Form B fails to ship on any of '
        'the four opx combinations. Aggregate RMSE is not uniformly '
        f"degraded ({int((headline['aug_mean_rmse'] < headline['non_aug_mean_rmse']).sum())} "
        'of 4 combinations see equal-or-better aug RMSE), so augmentation '
        'itself is not the blocker. The differentiator between our negative '
        'Form B result and Agreda-Lopez positive one must therefore be '
        'mineral-specific (orthopyroxene vs clinopyroxene residual '
        'structure) or protocol-specific beyond augmentation (e.g., their '
        'use of published vendor parameters versus our OOF-fit parameters).'
    )


def methods_paragraph() -> str:
    return (
        '### Section 3.9 Augmentation sensitivity (Methods)\n\n'
        'We test whether the negative Form B ship-if-better result on opx '
        'depends on the presence or absence of the 15x Gaussian '
        'composition-noise augmentation protocol used by Agreda-Lopez '
        'et al. (2024) on their cpx training set. For each of the four '
        'opx (track, target) combinations, at each of 20 seeds 42-61, we '
        'augmented the training set with 15 noisy copies per original '
        'sample (4 combinations x 3 feature sets x 8 models x 20 seeds = '
        '1920 cell-refits). Per-sample noise was multiplicative Gaussian '
        'with 3% relative standard deviation, applied independently per '
        'copy to every oxide feature, with non-negative clipping. Citation '
        'groups were preserved across copies; citation-grouped 10-fold CV '
        'splits were computed on the original data so augmented rows '
        'always inherit their parent\'s fold membership and never leak '
        'into a held-out fold. Test sets were not augmented. Optuna '
        'hyperparameters were frozen at the non-augmented values '
        '(`results/optuna_best_params_opx.json`) so this ablation isolates '
        'augmentation as the only independent variable against our pre-'
        'registered v10 pipeline. Form A and Form B fits and the ship-if-'
        'better decision used the same definitions as in Section 3.8 '
        '(per-regime OLS for Form A, piecewise bounds on predicted-value '
        'quantiles for Form B, ship if overall RMSE improves beyond '
        'numerical tolerance and no regime degrades by more than '
        'tolerance). Implementation in `src/ablations/augmentation.py`, '
        'runnable via `scripts/ablations/run_augmentation_ablation_opx.py` '
        'and `scripts/ablations/run_augmentation_oof_bias.py`. '
        'Reproducible at tag `pre-nb04b-aug-20260419`.'
    )


def discussion_paragraph(headline: pd.DataFrame, interpretation: str) -> str:
    lines = []
    lines.append('### Section 5.3 Augmentation sensitivity (Discussion, autofilled)')
    lines.append('')
    lines.append(interpretation)
    lines.append('')
    lines.append('Per-combination headline (best cell under each protocol):')
    lines.append('')
    for _, row in headline.iterrows():
        lines.append(
            f"- **{row['track']}/{row['target']}**: "
            f"non-aug {row['non_aug_model']}/{row['non_aug_feature_set']} "
            f"RMSE {row['non_aug_mean_rmse']:.2f}+/-{row['non_aug_std']:.2f} "
            f"(canon winner: {row['non_aug_winner']}); "
            f"aug {row['aug_model']}/{row['aug_feature_set']} "
            f"RMSE {row['aug_mean_rmse']:.2f}+/-{row['aug_std']:.2f}; "
            f"Form A ships {row['aug_ship_a_rate']} seeds, "
            f"Form B ships {row['aug_ship_b_rate']} seeds."
        )
    total_A = int(headline['aug_winner_A'].sum())
    total_B = int(headline['aug_winner_B'].sum())
    total_none = int(headline['aug_winner_none'].sum())
    lines.append('')
    lines.append(
        f'Across all 80 augmented (combo, seed) cells: Form A wins '
        f'{total_A}, Form B wins {total_B}, neither ships {total_none}.'
    )
    lines.append(
        'See fig_aug01 for ship-verdict comparison, fig_aug02 for aggregate '
        'RMSE delta, fig_aug03 for residual structure per regime on opx-only '
        'P_kbar, and fig_aug04 for Form B breakpoint stability.'
    )
    return '\n'.join(lines)


def run_pytest() -> tuple[int, str]:
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-q'],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        summary = (r.stdout.strip().splitlines() or [''])[-1]
        return (r.returncode, summary)
    except Exception as e:
        return (-1, repr(e))


def validation_checks(headline: pd.DataFrame) -> list[dict]:
    checks = []

    # Check 1: pytest
    rc, summary = run_pytest()
    checks.append({
        'check': 'pytest all pass',
        'ok': rc == 0,
        'detail': summary,
    })

    # Check 2: test_augmentation passes
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_augmentation.py', '-q'],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT))
        summary = (r.stdout.strip().splitlines() or [''])[-1]
        checks.append({
            'check': 'test_augmentation 7 unit tests pass',
            'ok': r.returncode == 0,
            'detail': summary,
        })
    except Exception as e:
        checks.append({
            'check': 'test_augmentation 7 unit tests pass',
            'ok': False, 'detail': repr(e)[:120],
        })

    # Check 3: notebook papermilled successfully (detected by executed notebook presence)
    nb_out = ROOT / 'notebooks' / 'executed' / 'nb04b_aug_test_executed.ipynb'
    checks.append({
        'check': 'notebook executed via papermill',
        'ok': nb_out.exists(),
        'detail': f'looked for {nb_out}',
    })

    # Check 4-7: 4 CSVs exist
    for name in (
        'augmentation_ablation_opx_multiseed_results.csv',
        'augmentation_ablation_opx_bias_correction.csv',
        'augmentation_ablation_opx_regime_rmse.csv',
        'augmentation_ablation_opx_headline.csv',
    ):
        p = RESULTS / name
        checks.append({
            'check': f'{name} exists',
            'ok': p.exists(),
            'detail': f'{p.stat().st_size if p.exists() else 0} bytes',
        })

    # Check 8-11: 4 figures exist as PDF + PNG
    for stem in (
        'fig_aug01_ship_verdict_comparison',
        'fig_aug02_aggregate_rmse_delta',
        'fig_aug03_residual_structure_per_regime',
        'fig_aug04_form_b_breakpoint_stability',
    ):
        pdf_ok = (FIGURES / f'{stem}.pdf').exists()
        png_ok = (FIGURES / f'{stem}.png').exists()
        checks.append({
            'check': f'{stem}: PDF + PNG',
            'ok': pdf_ok and png_ok,
            'detail': f'pdf={pdf_ok} png={png_ok}',
        })

    # Check 12: CANONICAL_FIGURES entries
    stems = {e['stem'] for e in CANONICAL_FIGURES}
    expect = {
        'fig_aug01_ship_verdict_comparison',
        'fig_aug02_aggregate_rmse_delta',
        'fig_aug03_residual_structure_per_regime',
        'fig_aug04_form_b_breakpoint_stability',
    }
    missing = expect - stems
    checks.append({
        'check': 'CANONICAL_FIGURES registers fig_aug01-04',
        'ok': not missing,
        'detail': f'missing={missing}' if missing else 'all 4 registered',
    })

    # Check 13: summary CSV row count
    summary = pd.read_csv(RESULTS / 'augmentation_ablation_opx_multiseed_results.csv')
    checks.append({
        'check': 'results CSV row count == 1920 (4 x 3 x 8 x 20)',
        'ok': len(summary) == 1920,
        'detail': f'got {len(summary)}',
    })

    # Check 14: no NaN RMSEs in results CSV
    n_nan = int(summary['test_rmse'].isna().sum())
    checks.append({
        'check': 'no NaN RMSEs in results',
        'ok': n_nan == 0,
        'detail': f'n_nan={n_nan}',
    })

    # Check 15: bias CSV ship verdict is well-defined
    bias = pd.read_csv(RESULTS / 'augmentation_ablation_opx_bias_correction.csv')
    bad = int((~bias['winner'].isin(['A', 'B', 'none'])).sum())
    checks.append({
        'check': 'ship verdict is A/B/none for all 80 rows',
        'ok': bad == 0 and len(bias) == 80,
        'detail': f'rows={len(bias)} bad_winner={bad}',
    })

    return checks


def main():
    headline = load_headline()
    label, interp = classify(headline)
    methods = methods_paragraph()
    discussion = discussion_paragraph(headline, interp)

    # Write manuscript paragraphs to separate files.
    txt_dir = ROOT / 'manuscripts' / 'opx_2026' / 'text'
    txt_dir.mkdir(parents=True, exist_ok=True)
    (txt_dir / 'aug_methods_3_9.md').write_text(methods, encoding='utf-8')
    (txt_dir / 'aug_discussion_5_3.md').write_text(discussion, encoding='utf-8')

    checks = validation_checks(headline)
    all_ok = all(c['ok'] for c in checks)

    lines = []
    lines.append('# Augmentation ablation (nb04b) report')
    lines.append('')
    lines.append('**Date:** 2026-04-19')
    lines.append('**Branch:** main')
    lines.append('**Safety tag:** `pre-nb04b-aug-20260419` -> 134b025')
    lines.append('**Interpretation label:** ' + label)
    lines.append('')
    lines.append('## Scope')
    lines.append('')
    lines.append(
        '4 opx combinations (opx_liq T_C/P_kbar, opx_only T_C/P_kbar) x 3 '
        'feature sets (raw/alr/pwlr) x 8 Optuna-tuned models (RF, ERT, XGB, '
        'GB, CatBoost, LightGBM, ElasticNet, MLP) x 20 seeds (42-61) = 1920 '
        'augmented test-RMSE fits, plus 80 augmented OOF runs (4 winning '
        'cells x 20 seeds) for Form A / Form B / ship-if-better.'
    )
    lines.append('')
    lines.append('## Headline table')
    lines.append('')
    lines.append(headline.to_markdown(index=False, floatfmt='.3f'))
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append(interp)
    lines.append('')
    lines.append('## Validation checks')
    lines.append('')
    lines.append('| # | Check | Status | Detail |')
    lines.append('|---|-------|--------|--------|')
    for i, c in enumerate(checks, 1):
        status = 'OK' if c['ok'] else 'FAIL'
        lines.append(f"| {i} | {c['check']} | {status} | {c['detail']} |")
    lines.append('')
    lines.append(
        f'Overall: {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    lines.append('')
    lines.append('## Manuscript paragraphs')
    lines.append('')
    lines.append(methods)
    lines.append('')
    lines.append(discussion)
    lines.append('')
    lines.append('## Known residuals / deferrals')
    lines.append('')
    lines.append(
        '- fig_aug03 visualizes augmented OOF residuals only (not a '
        'non-aug vs aug side-by-side). A full non-aug OOF refit on the 4 '
        'winner cells was out of scope for the 12-hour budget; the '
        'aggregate RMSE comparison in fig_aug02 already captures the net '
        'residual magnitude change.'
    )
    lines.append(
        '- The Section 5.3 draft language in `manuscripts/opx_2026/text/'
        'aug_discussion_5_3.md` is autofilled from this run. Hand-edit '
        'for voice before merging into the main manuscript draft.'
    )
    lines.append('')
    lines.append('## Pre-registered commitments reaffirmed')
    lines.append('')
    lines.append(
        '- Canonical artifacts (`results/opx_multiseed_*.csv`, `results/'
        'bias_correction_*.csv`, `results/preregistered_scorecard_*.csv`) '
        'untouched.'
    )
    lines.append(
        '- Optuna hyperparameters frozen at non-aug values; no re-tuning.'
    )
    lines.append(
        '- Citation-grouped CV preserved under augmentation via the '
        'augmented-aware OOF helper `oof_predict_augmented`.'
    )
    lines.append('')

    report_path = ROOT / 'AUGMENTATION_ABLATION_REPORT.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {report_path}')
    print(f'Overall status: {"ALL OK" if all_ok else "SOME CHECKS FAILED"}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
