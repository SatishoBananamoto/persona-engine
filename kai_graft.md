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

### Current Chunk: Tracker Stabilization

Intent:
Create a branch-local tracker and make the active tracker unambiguous before behavior changes.

Files expected:
- `kai_graft.md`
- `.graft`
- `GRAFT.md`
- `docs/REPO_STRUCTURE.md`

Verification:
- `git status --short --branch`
- `sed -n l .graft`
- `git diff --name-status`

### Next Chunk

Harden `eval/product_value/run_eval.py` before adding ResearchIR:
- fail closed when required baselines/negative controls are missing
- separate output quality, trace correctness, causal sensitivity, safety, and segment specificity
- fix overlapping competence bands and raw forbidden-marker matching
- make negative controls invalid by construction

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
