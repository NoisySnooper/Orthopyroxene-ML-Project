# Inline Prompt for Claude Code

Copy the text below and paste directly into Claude Code's input box. Hit enter.

---

Read the file at prompts/tabpfn_and_figures_rework/LEE_REVISIONS_OVERNIGHT_PROMPT.md in full. Execute every instruction in it autonomously, start to finish, without asking for confirmation between steps.

Execution order:
1. Create git branch lee_revisions_overnight from main.
2. Read and execute prompts/tabpfn_and_figures_rework/LEE_REVISIONS_PHASE_1_PROMPT.md end-to-end (bootstrap CI fix, 22-row pairing matrix, §4.7 insertion, §3.10 natural-sample protocol, Table 2 update).
3. Commit at Phase 1 boundary with message "feat: phase 1 complete - bootstrap CI + pairing matrix + §4.7". Write checkpoint to results/OVERNIGHT_RUN_LOG.md.
4. Execute Phase 2 subtasks P2.1 through P2.11 in order (best-per-cell defense, feature concordance, classical-feature ablation, classical-equivalence regression, partial dependence, surrogate trees, physics checks, Gasparik analysis, §4.5 rewrite, §5.5 discussion, bibliography).
5. Commit at P2.2 / P2.5 / P2.8 / P2.11 boundaries for incremental progress.
6. Execute Part 3 finalization (regenerate all figures including Core_13 through Core_18, terminology sweep, test suite, audit v3, Word doc regen via build_manuscript_docx.py, delta report).
7. Final commit with message "feat: phase 2 complete - interpretability bundle + best-per-cell defense".

Constraints:
- Do not push. All commits local.
- Log every checkpoint to results/OVERNIGHT_RUN_LOG.md with UTC timestamp.
- On any of the 10 halt conditions in the prompt file, write results/HALT_REPORT.md and stop.
- Manuscript prose stays formal research style. Internal logs and commit messages use caveman tone.
- Every numeric claim in manuscript prose must cite a CSV and column. No fabrication.
- If a required input CSV is missing, log NOT FOUND to results/finalization_log.md and continue unless it blocks a downstream subtask.

Expected duration 10-14 hours CPU. Budget for 16 hours. I will review in the morning via the checklist at the bottom of the overnight prompt file. Start now.
