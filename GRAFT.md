# The Graft — Operation Tracker

> Surgically moving unique work from `review/tier1-bugfixes` (wrong base) onto `claude/analyze-test-coverage-d93F4` (correct base).
> This file is the SINGLE SOURCE OF TRUTH. Updated before every step, after every step. If it's not here, it didn't happen.

**Session**: Persona Engine's — The Graft
**Date started**: 2026-03-20
**Working branch**: `graft/merge-tier1` (to be created off `claude/analyze-test-coverage-d93F4`)

---

## Branches

| Role | Branch | Commits | Status |
|------|--------|---------|--------|
| **Target (destination)** | `claude/analyze-test-coverage-d93F4` | 108 | Most complete — mixin arch, Layer Zero, R1-R7, multi-provider, CI/CD |
| **Source (rescue from)** | `review/tier1-bugfixes` | 60 (13 unique) | Good work, wrong base (`explore-repo-KZTBj`) |
| **Working branch** | `graft/merge-tier1` | TBD | Created off target — all work lands here |

---

## Reference Files

| File | Purpose | Location |
|------|---------|----------|
| `GRAFT.md` | **This file.** Live operation tracker | `/home/satishocoin/persona-engine/GRAFT.md` |
| `HANDOFF.md` | Original cherry-pick instructions (pre-review) | `/home/satishocoin/persona-engine/HANDOFF.md` |
| `BRANCH_COMPARISON.md` | Full branch inventory + feature matrix | `/home/satishocoin/persona-engine/BRANCH_COMPARISON.md` |
| `TRACKER.md` | Work tracker from the source branch session | `/home/satishocoin/persona-engine/TRACKER.md` |
| `research/calibration_report.txt` | Psych literature calibration research (source branch) | On `review/tier1-bugfixes` |
| `eval/persona_eval.py` | Automated eval suite (source branch) | On `review/tier1-bugfixes` |

---

## Pre-Graft Review Findings

Completed 6 parallel investigations before starting work. Key findings:

### Test Baseline (Target)
- **2,822 tests — ALL PASSING.** Clean baseline.
- 668 warnings (cosmetic: Pydantic field deprecation, misnomered test helpers)
- `test_server.py` skipped (missing fastapi dependency — expected on Chromebook)

### Bug Verification
All 5 bugs from source branch confirmed to exist on target:

| Bug | Exists on target | Port method |
|-----|-----------------|-------------|
| Mood drift inversion (N → faster drift, should be slower) | YES | Direct |
| MemoryError shadows Python builtin | YES | Direct (6 lines, 2 files) |
| evolve_state_post_turn timing (called during IR, not after) | YES | Adapt (staged architecture) |
| Unbounded mood noise (no clamp after add_noise) | YES | Direct |
| Elasticity None crash in cross_turn.py | YES | Direct |

### Stance Generator
- Target's R5 "composable stance" = string prefix/suffix bolt-on. 10 flat templates, domain-agnostic.
- Ours = genuine compositional architecture. VALUE_TOPIC_TABLE (50 entries), COMPETENCE_FRAMES (12 frames), invariant safety.
- **Decision: replace target's with ours, port `_modulate_stance_by_personality()` (~25 lines) from target.**

### Calibration (trait_interpreter.py)
- Target has sigmoid activation + Dunning-Kruger curve. Our values tuned against linear pipeline.
- 4 port directly, 3 adapt, 5 skip (conflict with nonlinear transforms)
- See detailed matrix in review notes below.

### Save/Load
- Target = v2 (loses mood/fatigue/stress, stance cache, cross-turn inertia on reload)
- Ours = v3 (serializes dynamic_state, stance_cache, prior_snapshot)
- Underlying data structures identical. Manual adapt ~65 lines, not cherry-pick.

### Structured Logging
- ~70% overlap with target's existing logging.
- Unique from ours: centralized `_handle_llm_exception()`, `LLMAPIKeyError` mapping, DEBUG LLM logging, validation warnings.
- Partial port only.

---

## Graft Plan

Order: correctness → measurement infrastructure → enhancements → high-conflict items.

### Step 1: Bug Fixes (5 bugs)
- **Status**: NOT STARTED
- **Method**: 4 direct ports, 1 adaptation
- **Source commits**: `80f5ae5`, `ad0d702` (partial — only unique fixes)
- **Files affected**:
  - `persona_engine/behavioral/state_manager.py` — mood drift inversion + unbounded noise clamp
  - `persona_engine/exceptions.py` — MemoryError → PersonaMemoryError
  - `persona_engine/planner/stages/interpretation.py` — move evolve_state_post_turn (adapt)
  - `persona_engine/planner/turn_planner.py` — add evolve_state_post_turn after pipeline (adapt)
  - `persona_engine/validation/cross_turn.py` — elasticity None guard
  - `tests/test_phase3_fixes.py` — update MemoryError references
- **Test after**: Full suite. Expect 2,822 pass.

### Step 2: Eval Suite
- **Status**: NOT STARTED
- **Method**: Clean copy — new files, don't exist on target
- **Source commit**: `5b3e608`
- **Files to add**:
  - `eval/__init__.py`
  - `eval/persona_eval.py`
- **Dependencies**: scipy (check if in target's requirements.txt)
- **Test after**: Full suite + run eval suite standalone.

### Step 3: Calibration Research Report
- **Status**: NOT STARTED
- **Method**: Clean copy — new file
- **Source commit**: `de21860` (file only, not code changes)
- **Files to add**:
  - `research/calibration_report.txt`
- **Test after**: N/A (documentation only)

### Step 4: Save/Load v3
- **Status**: NOT STARTED
- **Method**: Manual adaptation (~65 lines)
- **Source commit**: `8177396`
- **Files affected**:
  - `persona_engine/engine.py` — add dynamic_state/stance_cache/prior_snapshot to save(), restore in load()
- **Extra**: Also serialize `BiasSimulator._anchor_stance` (target uses it, neither branch persists it)
- **Test after**: Full suite + test save/load round-trip explicitly.

### Step 5: Calibration Code
- **Status**: NOT STARTED
- **Method**: Selective — 4 direct, 3 adapt, 5 skip
- **Source commits**: `de21860`, `f33dbc4`
- **Files affected**:
  - `persona_engine/behavioral/trait_interpreter.py`
- **Port directly**:
  - `influences_proactivity` floor/ceiling: `0.2 + E*0.6`
  - `get_enthusiasm_baseline` floor/ceiling: `0.2 + E*0.5`
  - `influences_hedging_frequency` N co-factor: `+ N*0.2` with `min(0.8, ...)`
  - `influences_verbosity` E co-factor: `(E-0.5)*0.15`
- **Port with adaptation**:
  - `get_negative_tone_bias`: 0.7 → 0.5 (verify hostile_critic interaction)
  - `get_tone_from_mood`: additional tone branches (verify enum members exist in target schema)
  - `get_self_disclosure_modifier`: add N co-factor `+ N*0.1` onto target's sigmoid formula
- **Skip (conflict with sigmoid/DK)**:
  - `get_elasticity` O-weight (sigmoid handles this)
  - `influences_verbosity` C-multiplier (target's 0.5 intentional)
  - `influences_directness` A-multiplier (sigmoid handles this)
  - `get_confidence_modifier` C-boost and N-penalty (DK curve replaces)
  - `get_elasticity` shift-up (keep target's +0.2)
- **Test after**: Full suite.

### Step 6: Stance Generator
- **Status**: NOT STARTED
- **Method**: Replace target's stance_generator.py with ours + port personality modulation
- **Source commit**: `03e3091`
- **Files affected**:
  - `persona_engine/planner/stance_generator.py` — full replacement
- **Port from target before replacing**:
  - `_modulate_stance_by_personality(stance, traits)` — ~25 lines
  - `BigFiveTraits` import
- **Integration point**: call personality modulation after `_assemble_stance()` returns
- **Test after**: Full suite. May need test updates if target has stance-specific tests.

### Step 7: Structured Logging (Partial)
- **Status**: NOT STARTED
- **Method**: Selective port — unique improvements only
- **Source commit**: `9809104`
- **Port**:
  - `_handle_llm_exception()` centralized mapper → replace target's 3 inline try/excepts
  - `LLMAPIKeyError` mapping (auth errors currently thrown as generic LLMResponseError)
  - DEBUG-level LLM request/response logging in `llm_adapter.py`
  - Validation failure warnings in `engine.py`
- **Skip** (already on target):
  - Logger setup in engine.py, turn_planner.py, response_generator.py, memory_manager.py
  - Basic exception handling structure
- **Test after**: Full suite.

---

## Items Explicitly Skipped

| Source commit | Content | Why skipped |
|---------------|---------|-------------|
| `8cf9f6c` | README rewrite + repo cleanup | Target already rewrote README |
| `a593908` | TRACKER.md | Will rewrite for new branch state |
| `3102005` | Wiring audit | Target wired all methods in R1. Audit doc value is marginal. |
| `c92dad5` | SMALL_TALK + remove response/ | Target already removed response/. SMALL_TALK — check during bug fixes. |
| `ad0d702` | american → American casing | Check if exists on target during bug fixes. |

---

## Decision Log

| # | Date | Decision | Reasoning |
|---|------|----------|-----------|
| D1 | 2026-03-20 | Work on `graft/merge-tier1` branch, not directly on target | Safety — target stays clean if things go sideways |
| D2 | 2026-03-20 | Bug fixes first, not eval suite | Correctness before measurement — bugs affect calibration and eval baselines |
| D3 | 2026-03-20 | Replace target's stance generator, don't merge | Ours is architecturally superior. Target's R5 is just string prefix/suffix. Port personality modulation only. |
| D4 | 2026-03-20 | Skip 5 of 11 calibration values | Target's sigmoid + DK transforms are more sophisticated. Our linear values would regress them. |
| D5 | 2026-03-20 | Partial logging port, not full | ~70% overlap. Only port unique improvements (centralized handler, LLMAPIKeyError, DEBUG logging). |
| D6 | 2026-03-20 | Adapt save/load, don't cherry-pick | Different surrounding context. Same data structures. ~65 lines manual. |

---

## Progress

| Step | Item | Status | Tests After | Notes |
|------|------|--------|-------------|-------|
| 1 | Bug fixes (5) | NOT STARTED | — | — |
| 2 | Eval suite | NOT STARTED | — | — |
| 3 | Calibration report | NOT STARTED | — | — |
| 4 | Save/load v3 | NOT STARTED | — | — |
| 5 | Calibration code | NOT STARTED | — | — |
| 6 | Stance generator | NOT STARTED | — | — |
| 7 | Logging (partial) | NOT STARTED | — | — |

---

## Post-Graft Checklist

- [ ] All 7 steps complete with passing tests
- [ ] GRAFT.md fully updated with final state
- [ ] Close or retarget PR #1 (currently against wrong base)
- [ ] Create new PR: `graft/merge-tier1` → `claude/analyze-test-coverage-d93F4` (or → `main`)
- [ ] Archive stale branches: `main`, `claude-md`, `explore-repo-KZTBj`, `external-review`, `general-session-VUY6r`, `review/tier1-bugfixes`
- [ ] Update TRACKER.md for new branch state
- [ ] Verify version consistency across pyproject.toml / __init__.py
