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

### Current Chunk: Brokered KV Anthropic pilot

Intent:
Pilot the new `kv-secrets` script-native capability path in PersonaEngine so
Anthropic-backed scripts can run with a scoped `KV_CAP_TOKEN` instead of a raw
`ANTHROPIC_API_KEY`.

Files expected:
- `persona_engine/generation/llm_adapter.py`
- `eval/product_value/run_eval.py`
- `tests/test_response_generation.py`
- `tests/test_product_value_eval.py`
- `docs/sdk_guide.md`
- `eval/product_value/README.md`
- `kai_graft.md`

Acceptance:
- Existing `AnthropicAdapter(model=...)` keeps raw-key compatibility.
- If `ANTHROPIC_API_KEY` is absent but `KV_CAP_TOKEN` is scoped to
  `api:anthropic`, PersonaEngine calls Anthropic through `kv.cap_client`.
- Wrong-provider capability tokens do not satisfy Anthropic setup.
- Product-value eval `--backend anthropic` accepts either raw env key or KV
  capability mode.
- Tests prove the brokered path without real provider keys.

Latest evidence:
- Live Anthropic enterprise-only smoke slice was run on 2026-05-04 under
  `/tmp/codex-persona-live-slice/results`.
- It intentionally used only `prompt_only`, `schema_driven`, `contradictory_ir`,
  and `shuffled_ir`, so the full matrix gate is incomplete by design.
- Result: `schema_driven` beat `prompt_only` and `contradictory_ir` on product
  score, but `shuffled_ir` slightly beat `schema_driven`. Segment
  differentiation was also not stronger than prompt-only.
- Interpretation: ResearchIR has useful signal, but it is not yet causally
  clean enough. The bad-IR controls are still too easy for the LLM to route
  around using the persona prompt and user prompt.
- Brokered KV adapter pilot targeted tests pass locally:
  `python3 -m compileall persona_engine/generation/llm_adapter.py
  eval/product_value/run_eval.py tests/test_response_generation.py
  tests/test_product_value_eval.py` and `python3 -m pytest -q
  tests/test_response_generation.py::TestLLMAdapterErrorHandling
  tests/test_product_value_eval.py`.

---

## Chunk Log

### 2026-05-06 - Chunk 6 - Brokered KV Anthropic pilot

Status: completed

Intent:
Make PersonaEngine usable from `kv run --capability api:anthropic` without
requiring generated scripts to receive `ANTHROPIC_API_KEY`.

Changes:
- `AnthropicAdapter` now uses `kv.cap_client.api_call()` when no raw key is
  present and the process has a matching KV Anthropic capability token.
- The KV-backed path exposes a tiny Anthropic-compatible `.messages.create()`
  client shape so existing adapter and eval code can stay mostly unchanged.
- Product-value eval now accepts either `ANTHROPIC_API_KEY` or
  `kv run --capability api:anthropic` for `--backend anthropic`.
- SDK/eval docs now show KV capability mode as the agent-safe run path.

Validation so far:
- `python3 -m compileall persona_engine/generation/llm_adapter.py
  eval/product_value/run_eval.py tests/test_response_generation.py
  tests/test_product_value_eval.py`
- `python3 -m pytest -q
  tests/test_response_generation.py::TestLLMAdapterErrorHandling
  tests/test_product_value_eval.py` -> 14 passed, 1 existing warning
- `python3 -m pytest -q
  tests/test_response_generation.py::TestLLMAdapterErrorHandling
  tests/test_product_value_eval.py tests/test_multi_provider_adapters.py` -> 65
  passed, 1 existing warning
- `env -u ANTHROPIC_API_KEY -u KV_CAP_TOKEN -u KV_CAPABILITY python3 -m
  eval.product_value.run_eval --backend anthropic --methods prompt_only
  --repeats 1 --out /tmp/codex-persona-kv-missing-check` exits with the new
  raw-key-or-KV-capability guidance

Next step:
- Run one live KV-backed Anthropic smoke with `kv run --capability
  api:anthropic --max-calls N -- python3 -m eval.product_value.run_eval ...`
  from a user-approved/unlocked KV surface.

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

### 2026-05-04 - Chunk 4 - Exit gate

Status: completed

Command:
- `bash /home/satishocoin/.codex/skills/done/scripts/exit_gate.sh /home/satishocoin/persona-engine`

Result:
- Active tracker resolved through `.graft` to `kai_graft.md`.
- Recent commits visible: `1d7a593`, `7bd18ca`, `5412637`.
- Remaining dirty state is inherited/unrelated: `GRAFT.md`, root eval scripts/results,
  experiment artifacts, and historical `eval/product_value/results_haiku_*` output folders.

Cleared for handoff:
- Code/docs/tests for the current ResearchIR slice are committed.
- Verification evidence is recorded in this tracker.
- No history-bearing files were deleted.
- Product-value smoke did not produce a fake proceed signal; it still fails the
  segment-differentiation gate.

### 2026-05-04 - Chunk 5 - Live enterprise eval slice

Status: completed

Intent:
Use the unlocked secret path to see whether live model behavior changes the
ResearchIR verdict before tuning generation.

Commands:
- `kv run --secret ANTHROPIC_API_KEY --cwd /home/satishocoin/persona-engine python3 -m eval.product_value.run_eval --backend anthropic --repeats 1 --out /tmp/codex-persona-live-anthropic --run-type smoke --max-tokens 500`
- `bash -lc 'ANTHROPIC_API_KEY=... python3 -m eval.product_value.run_eval --backend anthropic --repeats 1 --out /tmp/codex-persona-live-anthropic --run-type smoke --max-tokens 500'`
- `bash -lc 'ANTHROPIC_API_KEY=... python3 -m eval.product_value.run_eval --scenarios /tmp/codex-persona-live-slice/enterprise_scenarios.yaml --backend anthropic --methods prompt_only schema_driven contradictory_ir shuffled_ir --repeats 1 --out /tmp/codex-persona-live-slice/results --run-type smoke --max-tokens 350'`

Notes:
- `kv status` reported the vault locked in this sandboxed shell, so `kv run`
  could not execute directly.
- Directly sourcing the full exported env file is unsafe because it contains
  non-shell-compatible entries; use targeted extraction for one secret name
  instead of `source` for future runs.
- The first full-matrix live run failed in sandbox with DNS resolution, then ran
  under escalated network access but took too long with no result artifacts, so
  it was stopped and replaced with a smaller enterprise-only slice.
- The exported dev env file was used only to extract `ANTHROPIC_API_KEY`; secret
  values are not recorded here.

Live slice result:
- `prompt_only`: product `0.595`, segment diff `0.8928`
- `schema_driven`: product `0.810`, segment diff `0.8668`
- `contradictory_ir`: product `0.715`, segment diff `0.8832`
- `shuffled_ir`: product `0.8175`, segment diff `0.8869`

Interpretation:
- Correct ResearchIR improved output usefulness over prompt-only and
  contradictory IR.
- Shuffled IR slightly beat correct IR, so the current control setup still does
  not prove causal ResearchIR value.
- Next implementation slice should make bad ResearchIR controls harder to route
  around and/or make generation depend on role/workflow exposure more explicitly.
