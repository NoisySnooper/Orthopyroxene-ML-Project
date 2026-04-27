# Acknowledgments, Author Contributions, Conflict of Interest

## Acknowledgments

The lead author thanks Dr. Kanani K. M. Lee (USCGA) for mentorship during the cadet research tenure. We thank the maintainers of the Experimental Petrology Database (ExPetDB) for open-access distribution of the experimental corpus used in this study, and the developers of Thermobar (P. E. Wieser et al.) for the open-source Python package that underlies our classical-benchmark computations. We thank the authors of TabPFN v2 (Hollmann et al., 2025) for releasing the pretrained model weights and inference code under an open license that permitted our 20-seed peer evaluation. Conversations with Dr. Lee shaped the decision to pre-register the evaluation protocol before running the bias-correction fit.

This work was performed under the lead author's first-class cadet research project at the United States Coast Guard Academy. Views and conclusions in this manuscript are those of the authors and do not represent the official position of the U.S. Coast Guard or the U.S. Coast Guard Academy.

Computational resources were provided by the lead author's personal workstation (Windows 11 Pro, Python 3.13.13). No GPU was used; all TabPFN v2 inference ran on CPU. The Optuna hyperparameter search for the eight tuned families across the two opx pipelines and three feature sets took approximately 72 CPU-hours total; the 20-seed multiseed refit took approximately 40 CPU-hours; the bias-correction fit and rescore took approximately 2 CPU-hours.

## Use of Generative AI Tools

During the preparation of this manuscript and the accompanying analysis pipeline, the lead author used Anthropic's Claude (claude-opus-4 family) for code generation and refactoring assistance, draft prose generation for selected manuscript sections, and parallel review of analytical decisions. Anthropic's Claude Code (a command-line agentic coding tool built on the Claude API) was used to execute multi-step pipeline modifications under direct authorial supervision. Google Gemini (Gemini 2.5 Pro) was used as an independent parallel reviewer on a subset of methodological decisions including bias-correction form selection and feature-set choice. All AI-generated code was reviewed, executed, and validated against the pre-registered test suite by the lead author before integration into the analysis pipeline. All AI-drafted prose was edited, fact-checked against primary sources, and rewritten as needed by the lead author before inclusion in the manuscript. The authors take full responsibility for the content, methodology, results, and conclusions presented in this work. AI tools are not listed as authors, consistent with the AGU policy on generative AI in manuscripts.

## Author Contributions

Contributions follow the CRediT taxonomy (Brand et al., 2015).

**NQTa (lead author):**
Conceptualization; Data curation; Formal analysis; Investigation; Methodology; Software; Validation; Visualization; Writing (original draft); Writing (review and editing); Project administration.

**Kanani K. M. Lee:**
Conceptualization; Funding acquisition (USCGA cadet research support); Methodology (pre-registration discipline); Supervision; Resources; Writing (review and editing).

All authors read and approved the final manuscript.

## Conflict of Interest Statement

The authors declare no competing financial or non-financial interests that could have appeared to influence the work reported in this manuscript.

## Funding

This research received no external funding. The lead author's time was supported through the standard USCGA cadet research program.

## Open Research and Preprint Status

A preprint of this manuscript is deposited on ESSOAr / EarthArXiv concurrent with submission to JGR: Machine Learning and Computation. The preprint DOI will be added to the metadata block upon ESSOAr acceptance. All materials referenced in the Data Availability and Code Availability sections are publicly archived and reviewer-accessible at the time of submission.
