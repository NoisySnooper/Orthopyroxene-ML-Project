# Cover Letter Draft

[Submission date]

Dr. [Editor Name]
Editor, Journal of Geophysical Research: Machine Learning and Computation
American Geophysical Union

Dear Dr. [Editor Name],

We are pleased to submit our manuscript entitled "A Pre-Registered Machine-Learning Thermobarometer for Orthopyroxene: Regime-Stratified Evaluation and Regime-Piecewise Bias Correction" for consideration at JGR: Machine Learning and Computation.

The manuscript reports the first machine-learning thermobarometer trained specifically on orthopyroxene and evaluated under a pre-registered regime-stratified protocol. The principal finding is a 41.9 % aggregate RMSE reduction on opx-only pressure (10.35 to 6.05 kbar) against the best available classical barometer (Putirka 2008 equation 29c), unanimous across 20 random-seed train/test splits and winning every pre-registered pressure regime. The pipeline evaluates nine model families including TabPFN v2 (Hollmann et al., 2025) as a pretrained foundation-model peer.

Three features fit JGR:MLC's scope rather than a traditional petrology venue.

First, the full evaluation framework is pre-registered in version-controlled documents committed before any post-hoc correction was fit. Pre-registration is rare in ML thermobarometry; we adopt it here as an explicit, reviewer-visible discipline to make every analytical decision auditable, and we believe it complements the methodological progress of the past four years of ML-in-petrology work.

Second, the manuscript builds on the regression-to-the-mean bias correction introduced by Agreda-Lopez et al. (2024, Computers & Geosciences) for clinopyroxene and tests whether their quantile-thresholded piecewise correction transfers to orthopyroxene. We find that it does not transfer cleanly at opx sample sizes: Form B accepts on one of four opx cells at canonical seed only and is not rescued by the 15× Gaussian augmentation protocol that enables it on cpx. We present a regime-piecewise linear correction (Form A, motivated by Zhang and Lu, 2012) that accepts on three of four opx cells under a pre-registered tolerance-band rule. The head-to-head of Form A against Form B is a substantive methodological contribution to ML-in-petrology.

Third, the manuscript reports an extensive set of honest nulls. Opx-liq temperature is effectively a null against the classical Putirka 28a thermometer. Form B does not transfer to opx data under the pre-registered tolerance. Opx-liq deeper-mantle (n = 8) fails the pre-registered n ≥ 20 honesty bar and is reported with no directional claim. These nulls are findings in their own right and strengthen the defensibility of the shipped claims.

The work is ready for peer review. The evaluation framework (pre-registration, 20-seed per-cell results, SHA256 dataset hashes, trained model joblibs) is archived at a tagged commit and deposited at Zenodo for reviewer verification. Reviewers can clone the repository and run `python -m pytest tests/test_preregistration.py` to verify that the pre-registered acceptance-rule constants are consistent across all source files.

We suggest the following reviewers based on subject-matter expertise; affiliations are best-effort at the time of writing and should be verified by the editorial office:
- Dr. Maurizio Petrelli, University of Perugia (ML thermobarometry, Petrelli et al. 2020)
- Dr. Monica Agreda-Lopez (ML bias correction for cpx, Agreda-Lopez et al. 2024)
- Dr. Penny Wieser, UC Berkeley (Thermobar package, Wieser et al. 2022)
- Dr. Corin Jorgenson (ML thermobarometry, Jorgenson et al. 2022)

Any of the above can speak to the methodological correctness of the ML pipeline and the defensibility of our pre-registration and regime-stratification protocols. We request that Dr. Kanani K. M. Lee (USCGA, co-author) be excluded as reviewer due to co-authorship.

Thank you for your consideration.

Sincerely,

________________________________________
NQTa
Lead author and corresponding author
United States Coast Guard Academy, New London, CT
