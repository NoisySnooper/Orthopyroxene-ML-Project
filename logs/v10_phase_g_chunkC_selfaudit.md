# Phase G Chunk C self-audit

Summary: 19/19 checks pass.

| # | Check | Pass | Detail |
|---|---|---|---|
| 1 | 1. pre-registration doc exists | PASS | C:/Users/NQTa/Documents/MLCourse/Final Project/docs/v10_p_regime_preregistration.md |
| 2 | 2. P_REGIME_BIN_EDGES_KBAR == [0, 5, 15, 30, 100] | PASS | [0.0, 5.0, 15.0, 30.0, 100.0] |
| 3 | 3. honesty-bar floor n >= 20 | PASS | 20 |
| 4 | 4. Chunk B claims audit CSV present (axis-1 only) | PASS | rows=8 |
| 5 | C1a. probe predictions CSV non-empty | PASS | rows=24360 |
| 6 | C1b. probe per-regime RMSE CSV non-empty | PASS | rows=560 |
| 7 | C1c. probe aggregate RMSE CSV non-empty | PASS | rows=140 |
| 8 | C2. probe aggregate matches existing multi-seed summary (tol 1e-6) | PASS | all cells match |
| 9 | C3. robust audit CSV has 8 rows (4 regimes x 2 targets) | PASS | rows=8 |
| 10 | C3b. robust audit CSV has all required columns | PASS | [] |
| 11 | C4. every Chunk B outperforms row is represented in robust audit | PASS | 1 row(s) matched |
| 12 | C5. at least one robust outperforms Putirka row in audit | PASS | n_rows=1: [('shallow_crustal', 'P_kbar')] |
| 13 | C6a. S8.5.4 markdown exists and non-trivial (>= 500 bytes) | PASS | bytes=3085 |
| 14 | C6b. S8.5.4 csv exists and non-trivial (>= 500 bytes) | PASS | bytes=2908 |
| 15 | C7a. latest T15 log row sourced from robust audit | PASS | source=robust |
| 16 | C7b. latest T15 log row passed | PASS | passed=True |
| 17 | C8a. autofill references Table S8.5.4 | PASS | found=True |
| 18 | C8b. autofill references results/v10_opx_per_regime_claims_audit_robust.csv | PASS | found=True |
| 19 | C8c. autofill describes the two-axis honesty bar | PASS | found=True |
