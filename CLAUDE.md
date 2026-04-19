# Claude Code session guardrails for pyroxene ML thermobarometer project

## Context sources (read these first, every session)
1. README.md
2. PROJECT_OVERVIEW.md
3. docs/master_plan.md

## Hard rules (never violate)
- models/external/ is read-only. Never write, move, or delete.
- data/processed/, data/splits/, data/raw/, data/external/ are read-only.
- Never auto-approve REVIEW items from v10_cleanup_manifest.md.
- Never run destructive git operations (reset --hard, push --force, branch
  deletion) without explicit user confirmation.
- Always git tag before any multi-file destructive operation.
- Never run nbF_figures.ipynb until Phase G (CANONICAL_FIGURES is stale
  until then).
- Never invoke docs/archive_superseded/v9_deletion_plan.md or
  docs/archive_superseded/v9_archive_plan.md directly; both are superseded
  by docs/cleanup_manifest.md.

## User preferences
- Short, blunt, plain language
- No em dashes, no AI clichés
- Active voice, first person
- Cite sources for factual claims
- Ask before assumptions on multi-step work
- Caveman mode available on request (drops articles/filler)

## Default behavior
- Verify before action. Audit first, act second.
- Stop at phase boundaries. Don't leak Phase A work into Phase B.
- Commit each logical sub-step.
- Write execution logs to logs/v10_phase_{X}_execution_log.md
