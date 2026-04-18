# Manuscript + SI additions: P-regime calibration-domain characterization

**Companion to:** `opx_paper_draft_v1.md`, `opx_paper_SI_outline_v0.md`
**Source of numerical parameters:** `docs/v10_p_regime_preregistration.md` (pre-registered 2026-04-17)
**Instructions:** Drop Section 2.5.1 into main text after the existing 2.5. Add Section 4.X placeholder for results. Add Supplementary Section S8.5 entry.

---

## Main text insertion - Section 2.5.1

Insert immediately after Section 2.5 ("ArcPL natural-sample benchmark") in `draft_v1.md`:

> ### 2.5.1 Calibration-domain characterization by pressure regime
>
> Thermobarometers, whether classical or ML-based, inherit regime-dependent performance from the pressure distribution of their training data. Agreda-Lopez et al. (2024) explicitly downsampled low-pressure experiments in their clinopyroxene calibration set to address training-data imbalance, following Petrelli et al. (2020) and Branco et al. (2016). Putirka (2008) global calibrations span a wider P range but report systematic underestimation at pressures above approximately 30 kbar. A single pooled RMSE on the ArcPL Kd-equilibrated benchmark obscures these regime-dependent behaviors and conflates calibration-domain mismatches with intrinsic methodological differences.
>
> To characterize where each benchmarked method is most reliable, not to rank methods in a global sense, we partition the ArcPL Kd-equilibrated subset (n = 96) and the ExPetDB held-out test set into four pressure regimes defined on petrological grounds:
>
> | Regime label | P (kbar) | Petrological context |
> |---|---|---|
> | Shallow crustal | 0-5 | Arc storage, upper-crustal magma chambers |
> | Deep crustal / MASH | 5-15 | Melting-assimilation-storage-homogenization zone (Hildreth and Moorbath, 1988; Annen et al., 2006) |
> | Lithospheric mantle | 15-30 | Spinel-to-garnet peridotite stability (Klemme and O'Neill, 2000) |
> | Deeper mantle | >30 | Asthenospheric and below |
>
> Bin edges are pre-registered and locked prior to computation of any per-regime metric; the full registration document, including petrological justification, is archived in the reproducibility repository (`docs/v10_p_regime_preregistration.md`, registered 2026-04-17). The configuration constant `P_REGIME_BIN_EDGES_KBAR` in the released code is set to `[0, 5, 15, 30, 100]` kbar (where 100 kbar is the training-set ceiling `P_CEILING_KBAR`).
>
> Within each regime, for each benchmarked method, we report sample size, RMSE with bootstrap 95% confidence intervals (B = 1000 resamples), T bias, P bias, and 90% prediction-interval coverage fraction. Bins with fewer than 20 ArcPL samples are explicitly flagged as sample-size-limited; we do not merge bins post-hoc or drop bins with few samples. Within-bin method comparisons are reported honestly when bootstrap confidence intervals overlap: we do not interpret overlapping CIs as ranking information. This analysis is **calibration-domain characterization**, not a performance ranking. Methods that perform well in one regime and poorly in another reflect different training-data distributions and calibration strategies, not intrinsic methodological superiority.
>
> Aggregate pooled RMSE on the full ArcPL Kd-equilibrated set remains the primary benchmark reported in Section 4.3; per-regime analysis is a secondary characterization (Section 4.X, full per-bin tables in Supplementary Section S8.5).

---

## Main text placeholder - Section 4.X

Insert after Section 4.3 (primary Putirka benchmark) in `draft_v1.md`:

> ### 4.X Regime-dependent benchmark performance
>
> [V10: To draft from per-regime output. Expected structure: (1) describe the ArcPL n-per-regime distribution after applying the pre-registered bin structure of Section 2.5.1; (2) summarize per-regime RMSE patterns for our opx model, Putirka (2008) Eq. 28a/29a/29b, Agreda-Lopez et al. (2024), Jorgenson et al. (2022), and Wang et al. (2021); (3) explicitly flag sample-size-limited bins; (4) identify regimes where any single method's bootstrap 95% CI excludes zero deviation from the one-to-one line; (5) identify regimes where method CIs overlap and statistical resolution is insufficient to distinguish methods; (6) tie observed regime-dependent patterns to known training-data distributions of each method (citing Agreda-Lopez et al. 2024's explicit downsampling of low-pressure experiments, Putirka 2008's global calibration heterogeneity, and our ExPetDB training distribution). Summary table of per-regime RMSE, T bias, and P bias for all methods referred to Table X; full bootstrap CIs deferred to Supplementary Section S8.5. Draft tone: calibration-domain characterization, not performance ranking.]

---

## Discussion placeholder - Section 5.X

Insert as new subsection in Section 5 of `draft_v1.md` (before Limitations):

> ### 5.X Calibration-domain mismatch and its consequences for natural-sample inference
>
> [V10: To draft after regime results exist. Expected themes: (a) the Agreda-Lopez (2024) explicit low-pressure downsampling is a legitimate methodological choice for their cpx goals but constrains the P regime where their model should be applied; (b) the Putirka (2008) high-pressure underestimation is a documented, not hidden, feature of the global calibration; (c) users of any thermobarometer, including ours, should consult the per-regime behavior before interpretation of natural-sample P-T estimates; (d) a practical recommendation: report per-regime uncertainty alongside the aggregate RMSE. Do NOT overclaim our method's performance in any regime where our bootstrap 95% CI overlaps a competitor's. Keep the framing at the level of "these are the calibration domains of these methods" rather than "we win here."]

---

## Supplementary insertion - SI Section S8.5

Add to `SI_outline_v0.md` under Section S8:

> ### S8.5 Full per-regime benchmark tables
>
> [V10: Full tables for each (method x regime x metric) combination. Reference for main-text Section 4.X.]
>
> **Table S8.5.1** - Per-regime sample size distribution in ArcPL Kd-equilibrated subset (n = 96) and ExPetDB held-out test set.
>
> | Regime | ArcPL n | ExPetDB test n |
> |---|---|---|
> | shallow_crustal | [V10] | [V10] |
> | deep_crustal_MASH | [V10] | [V10] |
> | lithospheric_mantle | [V10] | [V10] |
> | deeper_mantle | [V10] | [V10] |
>
> Bins with ArcPL n < 20 are marked sample-size-limited in all downstream analyses.
>
> **Table S8.5.2** - Per-regime RMSE with bootstrap 95% CIs. Rows: regime. Columns: method (our opx model, Putirka 2008 iterative, Agreda-Lopez 2024 cpx-liq, Jorgenson 2022 cpx-only, Wang 2021). Separate sub-tables for T and P.
>
> **Table S8.5.3** - Per-regime T bias with bootstrap 95% CIs.
>
> **Table S8.5.4** - Per-regime P bias with bootstrap 95% CIs.
>
> **Table S8.5.5** - Per-regime 90% prediction-interval coverage fractions for methods that report uncertainty (our model, Agreda-Lopez 2024).
>
> **Figure S8.5** - Violin or box plots of per-regime residual distributions for T and P, one panel per method, color-coded by regime. Overlaid with 1:1 deviation reference line and bootstrap CI whiskers.

---

## Cross-references to update in `draft_v1.md`

The intro currently doesn't preview the regime analysis. Consider adding one sentence to the Section 1 contributions list (after contribution 4 about the ArcPL benchmark):

> We additionally report a **calibration-domain characterization by pressure regime** (Section 2.5.1, Section 4.X), pre-registered with petrologically-motivated bin boundaries at 0/5/15/30/100 kbar, which exposes where each benchmarked method is most reliable. This characterization is exploratory and secondary to the aggregate benchmark.

Keep this optional - the intro is already long. If adding it makes section 1 exceed ~900 words, skip the intro mention and keep the characterization as a Section 2.5.1 + 4.X disclosure only.

---

## Tone guardrails for results and discussion prose

When V10 delivers per-bin numbers, resist the following temptations:

| Tempting phrasing | Preferred phrasing |
|---|---|
| "Our model beats Agreda-Lopez in the 15-30 kbar bin." | "In the 15-30 kbar regime, our model's bootstrap 95% RMSE CI is [X, Y] kbar, Agreda-Lopez's is [A, B] kbar; the CIs [do/do not] overlap at the 5% level." |
| "Putirka 2008 fails at high pressure." | "Consistent with the systematic underestimation reported by Putirka (2008) above 30 kbar, the deeper_mantle bin shows P bias of [+X] kbar for Eq. 29a/29b." |
| "Our method is calibrated across a wider regime." | "Our training set spans [range] kbar with approximately uniform P coverage after filtering (Section 2.2); per-regime performance reflects this training distribution." |
| "Figure X demonstrates our model's robustness." | "Figure X presents per-regime residual distributions; readers should note the sample-size-limited status of the deeper_mantle bin before interpretation." |

Every claim of the form "we outperform" must be backed by a non-overlapping bootstrap CI at the 5% level AND a bin sample size >= 20. If either condition fails, the claim is reported as "competitive with" or "indistinguishable from."

---

## Summary: what this achieves

1. **Audit trail.** Git commit timestamp on `config.py` precedes any V10 per-bin result file. If a reviewer asks whether bins were chosen post-hoc, the commit SHA is the answer.

2. **Petrological defense.** Bin boundaries cite Putirka 2008, Blundy and Cashman 2008, and standard mantle petrology literature. Reviewers familiar with igneous petrology will recognize the regime set.

3. **Demoted claim scope.** The regime analysis is "calibration-domain characterization," not a performance ranking. You can't be accused of overclaiming because you're not claiming anything binary.

4. **Fallback for small n.** Sample-size-limited bins are flagged but not dropped. Honest reporting of statistical resolution.

5. **Tone guardrails.** Explicit list of forbidden and preferred phrasings for V10 results drafting. Reviewers cannot catch us overclaiming because we've pre-committed to honest phrasing.
