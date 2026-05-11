# Persona Engine — Review Plan

**Status:** Historical methodology plan preserved on 2026-05-11. The resulting
review work is recorded in `REVIEW.md` and `MEMORY-project.md`; this file is
kept to show the corrected review method that avoided grep-only dead-code
claims.

**Current continuation:** Do not treat this as a fresh active review without
first checking branch/repo reality. The next active task list is at the end.

## Context

First review attempt delegated to an exploration agent that grep'd for imports and assumed "not imported = not used." Result: 3 fabricated dead code claims out of 7 critical findings. The agent didn't trace call chains through TurnPlanner's 2,284-line pipeline. This plan fixes that.

**Goal:** Produce an accurate, verified assessment of persona-engine on `claude/general-session-VUY6r`.

---

## Phase 1: Understand Before Judging

- [ ] Read README.md, ARCHITECTURE.md, ROADMAP.md
- [ ] Read IMPROVEMENT_PLAN.md and PSYCHOLOGICAL_REALISM_PLAN.md
- [ ] Run `python demo_full_pipeline.py --backend template` — see actual output
- [ ] Run `python demo_turn_planner.py` — see IR generation in action
- [ ] Read `engine.py` top to bottom — understand the SDK entry point
- [ ] Trace ONE full request manually: user input → TurnPlanner → IR → response
  - Follow the data, not the imports
  - Document the exact path through modules
- [ ] Continue

**Output:** Written summary of how the system actually works, verified by execution.

---

## Phase 2: Structural Analysis (Automated)

- [ ] `find . -name '*.py' -not -path './.git/*' | xargs wc -l | sort -n` — find monoliths
- [ ] `python -m mypy persona_engine/ --ignore-missing-imports` — type errors
- [ ] `python -m ruff check persona_engine/` — linting issues (install if needed)
- [ ] `python -m pytest tests/ -v --tb=short` — do tests pass?
- [ ] `python -m pytest tests/ --cov=persona_engine --cov-report=term-missing` — coverage gaps
- [ ] Check dependencies: `cat pyproject.toml` or `cat setup.py` — what's required?
- [ ] Continue

**Output:** Hard numbers. Line counts, test pass/fail, coverage %, type errors, lint violations.

---

## Phase 3: Trace, Don't Grep

For each module/feature, verify integration by reading the actual consumption point in TurnPlanner.

### Bias Simulator (`behavioral/bias_simulator.py`)
- [ ] Read bias_simulator.py — what bias types exist?
- [ ] Search TurnPlanner for `bias_simulator` and `_current_bias_modifiers` calls
- [ ] For each bias type, confirm: is it computed AND consumed?
- [ ] Document: which lines in TurnPlanner consume each bias
- [ ] Continue

### Linguistic Markers (`behavioral/linguistic_markers.py`)
- [ ] Read linguistic_markers.py — what does `build_personality_language_directives()` return?
- [ ] Trace where the return value goes in TurnPlanner
- [ ] Trace whether directives reach prompt building or just citations
- [ ] Verify: do they actually affect generated text, or just get logged?
- [ ] Continue

### Social Cognition (`behavioral/social_cognition.py`)
- [ ] Read social_cognition.py — what does it produce?
- [ ] Find consumption in TurnPlanner — `schema_effect`, `adaptation`
- [ ] Verify: do outputs modify IR fields or just add citations?
- [ ] Continue

### Emotional Appraisal (`behavioral/emotional_appraisal.py`)
- [ ] Read the module — how does emotion detection work?
- [ ] Test negation handling: does "I'm not happy" trigger happiness?
- [ ] Check if valence/arousal deltas actually modify state
- [ ] Continue

### Trait Interactions (`behavioral/trait_interactions.py`)
- [ ] Read all 9 patterns — are thresholds cited or arbitrary?
- [ ] Find where interaction effects are consumed
- [ ] Check: do they change IR values or just produce metadata?
- [ ] Continue

### Memory System (`memory/`)
- [ ] Trace memory reads: where does retrieved context enter the pipeline?
- [ ] Trace memory writes: when are new facts stored? Before or after response?
- [ ] Check stance cache: does it actually prevent opinion flip-flopping?
- [ ] Continue

### Response Generation (`generation/` vs `response/`)
- [ ] Confirm which module `engine.py` uses
- [ ] Confirm which module demo scripts use
- [ ] Check: is `response/` truly dead, or still referenced?
- [ ] Continue

**Output:** Verified integration map. Every claim backed by file:line references.

---

## Phase 4: Verify Known Issues

Re-examine the claims the repo handler confirmed as true:

### Confirmed True — Validate Severity
- [ ] TurnPlanner monolith (2,284 lines) — how bad is it really? Is it readable despite size?
- [ ] Keyword emotion detection — test with negation cases, quantify the gap
- [ ] Test helper duplication — count exact instances, estimate consolidation effort
- [ ] Deprecated `response/` module — what breaks if we delete it?
- [ ] Unenforced strict mode — trace `write_policy` parameter, confirm it's ignored
- [ ] Continue

### Previously Wrong — Verify Corrections
- [ ] Confirm all 8 biases are wired (already verified: lines 757, 1222, 1308, 1531)
- [ ] Confirm linguistic markers reach behavioral_directives (already verified: lines 986-997)
- [ ] Confirm social cognition outputs are consumed (already verified: lines 981-982)
- [ ] Confirm error handling exists on `message.content` (already verified: lines 132, 387)
- [ ] Continue

**Output:** Severity-ranked issue list with evidence.

---

## Phase 5: Test Quality Assessment

- [ ] Run full test suite — record pass/fail count
- [ ] Pick 3 test files at random — read them, assess what they actually verify
- [ ] Check behavioral coherence tests — do they test IR math or actual text output?
- [ ] Check counterfactual twin tests — do they prove traits make a difference?
- [ ] Check determinism tests — same seed, same IR?
- [ ] Look for integration tests that run the full pipeline (not just unit tests)
- [ ] Assess: is the 1.6:1 test-to-code ratio meaningful coverage or padding?
- [ ] Continue

**Output:** Honest test quality assessment with specific examples.

---

## Phase 6: The Hard Questions

These are the claims that require judgment, not just tracing:

- [ ] **Does personality actually come through in text?** Generate 5 responses with high-O persona and 5 with low-O persona using template backend. Compare.
- [ ] **Are research citations reflected in implementation?** Pick 2 cited papers (LIWC, Scherer) — does the code do what the paper describes?
- [ ] **Is the R-phase work (this branch) net positive?** Compare complexity added vs behavioral improvement gained.
- [ ] **What's the minimum viable cleanup?** If we had to ship this, what's the shortest path?
- [ ] Continue

**Output:** Evidence-based answers to each question.

---

## Deliverable

After all phases, produce a revised review that:
1. States only verified claims with file:line evidence
2. Separates confirmed issues from open questions
3. Ranks issues by impact, not by how impressive they sound
4. Gives actionable next steps, not vague recommendations

## Current Continuation Tasks

- [x] Preserve the uncommitted review methodology plan as repo history.
- [x] Ignore generated `.svx-audit/` output instead of committing or deleting it.
- [x] Re-run the current test suite before committing this preservation slice.
  2026-05-11: `compileall persona_engine layer_zero tests` and `git diff --check`
  passed. Full pytest is not green in this checkout: property-based collection
  needs missing `hypothesis`, and the broad non-server run failed on
  `tests/test_phase7_sdk.py::TestCLI::test_plan_json` even though that test
  passed when isolated. Treat this as a baseline follow-up, not evidence against
  the `.gitignore`/plan preservation slice.
- [ ] Follow up on the baseline pytest blockers before using this branch as a
  clean validation source.
- [ ] Continue
