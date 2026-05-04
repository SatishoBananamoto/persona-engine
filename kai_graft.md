# kai_graft - Branch Tracker

> Active tracker for branch: `research-ir-market-wedge`.
> `.graft` points here while this branch is active.
> `GRAFT.md` is historical/mainline context unless explicitly promoted again.

**Created**: 2026-05-04
**Base checkout before branch**: `enterprise-research-ir`
**Base HEAD before branch**: `d8d51fb003d35c8b9af6fc56e0f0a7379a9d10d0`
**Upstream before branch**: none configured
**Tracker pointer**: `.graft -> kai_graft.md`
**Current rule**: one active tracker only. Do not duplicate chunk logs in both `GRAFT.md` and `kai_graft.md`.

---

## NEXT SESSION - START HERE

### Product Direction

This branch tests one narrow product claim:

> Correct compact ResearchIR improves enterprise rollout-risk simulation more than prompt-only, no-IR, stale-IR, shuffled-IR, contradictory-IR, or current generic IR.

The wedge is enterprise rollout risk simulation, not broad market prediction.

Target output:
- likely stakeholder objections
- trust conditions
- adoption blockers
- workaround risk
- claim/evidence boundaries
- research-usefulness signals for follow-up human interviews

Non-goals for this branch:
- UI
- persona compiler
- segment-aware Layer Zero
- purchase intent or price sensitivity
- RAG/customer-data ingestion
- multi-persona panel orchestration
- public API expansion

### Repo Reality At Branch Start

Branch created from dirty `enterprise-research-ir` state.

Inherited dirty state:
- Modified tracked: `GRAFT.md`, `persona_engine/planner/stages/behavioral_metrics.py`
- Untracked pointer: `.graft`
- Untracked eval/product artifacts and scripts: `eval/product_value/`, root eval scripts/results, `experiments/`, validation artifacts
- Current `GRAFT.md` header is stale and still describes `main` as clean

Source of truth for this branch:
- repo/tool output outranks tracker text
- `kai_graft.md` tracks active branch work
- `GRAFT.md` remains historical/mainline context

### Current Chunk: Exit gate and handoff

Intent:
Run final repo checks and hand off the branch state without hiding the remaining
dirty inherited artifacts.

Files expected:
- planner/schema/generation ResearchIR plumbing
- eval/product_value harness files
- targeted tests
- docs and tracker updates

Verification:
- exit gate before handoff
- repo status confirms only inherited/unrelated files remain dirty

---

## Chunk Log

### 2026-05-04 - Chunk 0 - Branch and tracker

Status: completed

Evidence collected before edits:
- `git status --short --branch` showed branch `enterprise-research-ir` with dirty tracked and untracked work.
- `git switch -c research-ir-market-wedge` created this branch from `d8d51fb003d35c8b9af6fc56e0f0a7379a9d10d0`.
- `.graft` previously pointed to `GRAFT.md`.

Decision:
Use `kai_graft.md` as the active branch tracker and keep `GRAFT.md` as historical context.

Verification:
- `git status --short --branch` showed branch `research-ir-market-wedge`.
- `sed -n l .graft` showed `kai_graft.md$`.
- `git diff --cached --name-status` showed only tracker/doc files staged for the first commit.

### 2026-05-04 - Chunk 1 - Eval gate hardening

Status: completed

Intent:
Stop incomplete or trace-presence-only eval runs from producing a proceed signal.

Files changed:
- `eval/product_value/run_eval.py`
- `tests/test_product_value_eval.py`

Changes:
- Added required-method matrix validation.
- Made proceed gates depend on matrix completeness.
- Split trace presence from trace correctness.
- Changed product/causal scoring so bad IR is not rewarded just for citations.
- Made competence bands non-overlapping.
- Added negation-aware forbidden-marker matching.
- Added report matrix status and `--run-type`.

Verification:
- `python3 -m compileall eval/product_value/run_eval.py`
- `python3 -m pytest -q tests/test_product_value_eval.py` -> 6 passed
- `python3 -m eval.product_value.run_eval --backend template --repeats 1 --out /tmp/codex-persona-eval-hardening` -> `proceed_signal: false`, `matrix_complete: true`
- `python3 -m eval.product_value.run_eval --backend template --methods schema_driven --repeats 1 --out /tmp/codex-persona-eval-partial` -> `proceed_signal: false`, `matrix_complete: false`

Next:
Add `enterprise_research` context and compact optional `ResearchIR`.

### 2026-05-04 - Chunk 2 - Enterprise research context and ResearchIR

Status: completed

Intent:
Add the smallest useful enterprise rollout-risk contract without replacing the
generic persona IR.

Files changed:
- `persona_engine/planner/context_classifier.py`
- `persona_engine/planner/research_ir.py`
- `persona_engine/schema/ir_schema.py`
- `persona_engine/planner/stages/interpretation.py`
- `persona_engine/planner/stages/behavioral.py`
- `persona_engine/planner/stages/behavioral_metrics.py`
- `persona_engine/planner/stages/knowledge.py`
- `persona_engine/planner/stages/finalization.py`
- `persona_engine/generation/prompt_builder.py`
- `persona_engine/generation/llm_adapter.py`
- `persona_engine/__init__.py`
- `persona_engine/schema/__init__.py`
- `tests/test_context_classifier.py`
- `tests/test_product_value_eval.py`
- `docs/ir_reference.md`
- `docs/PIPELINE_FLOWCHARTS.md`
- `docs/REPO_STRUCTURE.md`
- `eval/product_value/README.md`

Changes:
- Added `enterprise_research` context detection gated on workplace actor,
  rollout/tool language, and research/adoption intent.
- Added optional `ResearchIR` with focus, stakeholder role, workflow exposure,
  claim basis, likely objections, trust conditions, adoption blockers,
  workaround risk, and evidence boundary.
- Routed ResearchIR through interpretation, behavioral confidence/competence,
  knowledge claim boundaries, final IR assembly, prompt generation, template
  generation, and eval serialization.
- Enterprise research responses now remain stakeholder hypotheses and do not
  become `domain_expert` claims by default.

Verification:
- `python3 -m compileall persona_engine/planner/research_ir.py persona_engine/schema/ir_schema.py persona_engine/planner/context_classifier.py persona_engine/planner/stages/interpretation.py persona_engine/planner/stages/behavioral.py persona_engine/planner/stages/behavioral_metrics.py persona_engine/planner/stages/knowledge.py persona_engine/planner/stages/finalization.py persona_engine/generation/prompt_builder.py persona_engine/generation/llm_adapter.py eval/product_value/run_eval.py`
- `python3 -m pytest -q tests/test_context_classifier.py tests/test_product_value_eval.py` -> 42 passed, 12 warnings
- `python3 -m eval.product_value.run_eval --backend template --repeats 1 --out /tmp/codex-persona-research-ir` -> matrix complete, correct IR beats prompt-only/no-IR/negative controls, proceed signal remains false because segment differentiation did not beat prompt-only in the template smoke

Next:
Run exit gate and decide whether the next engineering chunk should attack the
remaining segment-differentiation gate or move to a live/human review run.

### 2026-05-04 - Chunk 3 - ResearchIR commit boundary

Status: completed

Commit:
- `7bd18ca Add enterprise research IR gate`

Staging boundary:
- Committed ResearchIR code, product-value eval source, targeted tests, docs,
  and `kai_graft.md`.
- Did not stage inherited `GRAFT.md` changes.
- Did not stage unrelated root eval scripts/results, experiment folders, or
  historical `eval/product_value/results_haiku_*` output folders.

Verification after commit:
- `git status --short --branch` shows branch `research-ir-market-wedge` with
  only inherited/unrelated dirty files remaining.
