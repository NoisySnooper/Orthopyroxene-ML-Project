# Cover Letter Draft

[Submission date]

Dr. [Editor Name]
Editor, Journal of Geophysical Research: Machine Learning and Computation
American Geophysical Union

Dear Dr. [Editor Name],

We are pleased to submit our manuscript entitled "A Pre-Registered Machine-Learning Thermobarometer for Orthopyroxene: Regime-Stratified Evaluation, Regime-Piecewise Bias Correction, and a Foundation-Model Baseline" for consideration at JGR: Machine Learning and Computation.

The manuscript reports the first machine-learning thermobarometer trained specifically on orthopyroxene and evaluated under a pre-registered regime-stratified protocol. The principal finding is a 41.9% aggregate RMSE reduction on opx-only pressure (10.35 to 6.05 kbar) against the best available classical barometer (Putirka 2008 equation 29c), unanimous across 20 random-seed train/test splits and winning every pre-registered pressure regime. The pipeline also integrates TabPFN v2 (Hollmann et al., 2025, Nature) as a ninth model family, reports its one-of-eight head-to-head win against tuned ensembles, and discusses the explainability gap created by TabPFN's lack of an efficient SHAP pathway.

Four features of this work make it a natural fit for JGR:MLC rather than for a traditional petrology venue.

First, the full evaluation framework is pre-registered in version-controlled documents committed before any post-hoc correction was fit. Pre-registration in petrology ML is rare, and we believe reviewer-visible pre-registration discipline is essential for the field to mature beyond the garden-of-forking-paths concerns that have accompanied the rapid expansion of ML thermobarometer papers over the past four years.

Second, the manuscript directly addresses the regression-to-the-mean bias problem identified by Agreda-Lopez et al. (2024, Computers & Geosciences) for clinopyroxene and evaluates whether their quantile-thresholded piecewise correction transfers to orthopyroxene. It does not: Form B accepts on one of eight cells at canonical seed only and is not rescued by the 15x Gaussian augmentation protocol that enables it on cpx. We present a regime-piecewise linear correction (Form A, motivated by Zhang and Lu, 2012) that accepts on seven of eight cells under a pre-registered tolerance-band rule. The head-to-head of Form A against Form B is a substantive methodological contribution to ML-in-petrology.

Third, the nine-family model sweep with TabPFN v2 as the ninth family is the first evaluation of a pretrained tabular foundation model in an experimental-petrology ML thermobarometer paper. Our finding is honest: TabPFN wins one of eight cells, is competitive on one, loses on six, and delivers the scorecard-winning post-correction result on two opx cells where the tuned-family margin is narrow. We retain TabPFN in the main-text roster and recommend future ML thermobarometer papers adopt the same first-class-peer treatment, while flagging the real explainability gap it creates.

Fourth, the manuscript reports an unusually extensive set of honest nulls. Opx-liq temperature is effectively a null against the classical Putirka 28a thermometer. Cpx-only temperature does not accept a correction under any reasonable tolerance. Opx-liq deeper-mantle (n = 8) fails the pre-registered n >= 20 honesty bar and is reported with no directional claim. The cpx replication is modestly less favorable than the published Agreda-Lopez and Jorgenson benchmarks at most regimes. We position these nulls as findings in their own right and believe they strengthen the defensibility of the shipped claims.

The work is ready for peer review. The evaluation framework (pre-registration, amendments, 20-seed per-cell results, SHA256 dataset hashes, trained model joblibs) is archived at a tagged commit and deposited at Zenodo for reviewer verification. Reviewers can clone the repository and run `python -m pytest tests/test_preregistration.py` to verify that the pre-registered acceptance-rule constants are consistent across all source files.

A companion paper addressing out-of-training-domain validation on 327 natural two-pyroxene pairs from the LEPR compilation is in preparation and will be submitted separately. We flag this in §5.7 of the manuscript so reviewers are aware of the companion scope.

We suggest the following reviewers based on subject-matter expertise:
- Dr. Maurizio Petrelli, University of Perugia (ML thermobarometry for clinopyroxene, Petrelli et al. 2020)
- Dr. Monica Agreda-Lopez, University of Perugia (ML bias correction for cpx, Agreda-Lopez et al. 2024)
- Dr. Penny Wieser, UC Berkeley (Thermobar package, Wieser et al. 2022)
- Dr. Luca Caricchi, University of Geneva (ML thermobarometry foundational work, Jorgenson et al. 2022 co-author)

Any of the above can speak to the methodological correctness of the ML pipeline and the defensibility of our pre-registration and regime-stratification protocols. We request that Dr. Kanani K. M. Lee (USCGA, co-author) and Dr. Junjie Dong (Stony Brook, co-author) be excluded as reviewers due to co-authorship.

Thank you for your consideration.

Sincerely,

NQTa
Lead author and corresponding author
United States Coast Guard Academy, New London, CT
Incoming PhD student (Fall 2026), Stony Brook University, Department of Geosciences
