# Claude Code Prompt: Activate Phase H — Natural Sample Worldwide Plan

## Context

The natural-sample comparison plan is fully specified at `docs/natural_worldwide_plan.md` (revised 2026-04-18 with Phase G collisions resolved). GEOROC is back online as of today, which unblocks the cpx pull (H.1b) and the twopx pair construction (H.1d). This prompt activates the full Phase H execution per that plan, end-to-end, autonomously.

Repository root: `C:\Users\NQTa\Documents\MLCourse\Final Project`
Target branch: `phase_h_natural_worldwide`
Canonical spec: `docs/natural_worldwide_plan.md`
Companion docs: `docs/master_plan.md`, `docs/figure_audit.md`

Read-write scope: `data/natural/`, `results/`, `src/`, `scripts/`, `figures/`, `notebooks/`, `tests/`. Do NOT edit `docs/preregistration/` or any pre-existing ExPetDB training data.

## Critical pre-registration boundaries

The natural-sample work collides with the Phase G pre-registration framework (`docs/preregistration/p_regime_preregistration.md`) in six ways. Resolutions are locked in `docs/natural_worldwide_plan.md` Section 0 and must be enforced throughout execution:

1. Per-regime RMSE claims on natural samples are circular (no ground truth P). Visualizations may stratify by predicted P regime; numeric claims may not.
2. The G.4 piecewise P bias correction (now the tolerance-band Form A correction) is NOT applied to natural samples by default. Raw predictions are the primary artifact. Corrected values appear ONLY as a supplementary panel on curated localities where literature P assigns the regime non-circularly.
3. Canonical cell roster for cpx and twopx must be locked at H.0b before inference. opx_liq is already canonical from Phase G.
4. The dual robustness check (bootstrap CI + 20-seed spread) only exists for opx_liq. cpx and twopx natural-sample claims are descriptive with bootstrap CI only, no 20-seed axis.
5. Per-locality n>=20 floor for any quantitative claim. Localities below 20 samples report pooled-by-tectonic-setting RMSE only.
6. ArcPL opx (n=197, paired opx+liq with literature P-T at `archive/pre_v10_rebuild_2026_04_16/results/nb04_arcpl_opx_liq_predictions_forest.csv`) is folded into H.6 as the 16th curated locality. No separate G.4 probe.

If during execution any of these boundaries appears to be violated by the planned next step, halt and report. Do not produce per-regime RMSE numbers on natural samples under any circumstance.

## Execution plan (mirrors plan §1.5)

[H.0 - H.7 sections per source prompt]

## Halt-and-report conditions

Stop and write `results/HALT_REPORT_PHASE_H.md` if:

1. GEOROC API returns < 30,000 raw cpx rows in H.1b.
2. Twopx pair join produces < 1,000 pairs (likely sample-name collision failure).
3. Any pre-registration boundary check (Section 0 collisions 1-6) is violated by a planned step.
4. Putirka classical inference fails on > 50% of natural samples (likely Thermobar wiring issue).
5. Inference runtime exceeds 6 hours per pipeline (likely OOM or stuck process).
6. ArcPL opx archive file at `archive/pre_v10_rebuild_2026_04_16/results/nb04_arcpl_opx_liq_predictions_forest.csv` is missing (n=197 reference data lost).
7. Any natural-sample CSV exceeds 5 GB (likely a Cartesian-product bug in the join logic).

Halt report contents: git SHA, last completed step, next planned step, error traceback, recommended human action.

## Saved 2026-04-25 from chat input by Claude Code session running Phase H activation.
