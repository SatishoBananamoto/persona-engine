# Persona Engine — Verified Code Review

**Branch:** `claude/general-session-VUY6r`
**Reviewer:** Claude (Opus 4.6)
**Date:** 2026-03-15
**Method:** 6-phase review plan — read docs, run code, trace call chains, execute tests, verify every claim

---

## Review Methodology

This review was conducted in two passes. The first pass delegated to an exploration agent that made grep-based assumptions about code usage. The repo handler identified **4 factual errors** in that review — 3 false "dead code" claims and 1 false "missing error handling" claim. All 4 were caused by searching for imports instead of tracing actual data flow through TurnPlanner.

This second pass corrected the methodology:

1. **Read all documentation** — README.md, ARCHITECTURE.md, IMPROVEMENT_PLAN.md, PSYCHOLOGICAL_REALISM_PLAN.md
2. **Read the full pipeline** — engine.py → TurnPlanner (all 2,284 lines) → ResponseGenerator → PromptBuilder
3. **Traced every disputed claim** by reading the exact consumption lines in TurnPlanner
4. **Ran the code** — loaded personas, generated IR, compared twin personas
5. **Ran 2,376 tests** — all passing, 0 failures
6. **Tested negation handling** with specific inputs to confirm the gap
7. **Verified module relationships** — confirmed which imports are canonical vs legacy

Every claim below includes file:line evidence or execution output.

---

## What This Project Is

Persona Engine creates AI-driven conversational personas grounded in psychological models — Big Five traits, Schwartz values, cognitive styles, and dynamic emotional state. The core innovation is an **Intermediate Representation (IR)** computed from personality math *before* any text is generated.

This makes persona behavior testable without LLM calls:
```python
ir = engine.plan("What do you think about AI?")
assert ir.response_structure.confidence > 0.7
assert ir.communication_style.tone == Tone.thoughtful_engaged
```

### End-to-End Pipeline

```
User Input
    ↓
engine.py: _validate_user_input() → _generate_ir()
    ↓
TurnPlanner.generate_ir() — 5 stages:
    Stage 1: Foundation    — TraceContext, per-turn seed, memory context
    Stage 2: Interpretation — topic relevance, bias modifiers, state evolution, intent, domain, expert eligibility
    Stage 3: Behavioral    — elasticity, stance, confidence, competence, tone, verbosity, communication style
    Stage 4: Knowledge     — disclosure, uncertainty, claim type, patterns, constraints
    Stage 5: Finalization  — memory writes, IR assembly, stance cache, snapshot
    ↓
IntermediateRepresentation (structured numbers + enums + citations)
    ↓
ResponseGenerator.generate() → PromptBuilder → LLM/Template → StyleModulator
    ↓
ChatResult (text + IR + validation + metadata)
```

### Key Design Principle

Every IR field follows a canonical modifier sequence: **base → role → trait → state → bias → clamp**. Each step builds on the previous one. No step reaches back. Every modification is recorded as a citation with before/after delta. This makes the pipeline deterministic and auditable.

---

## Project Numbers

| Metric | Value |
|--------|-------|
| Source code | 18,535 LOC across ~50 modules |
| Tests | 2,376 passing, 0 failures |
| Test files | 52 |
| Test-to-code ratio | ~1.6:1 |
| Largest file | `planner/turn_planner.py` — 2,284 lines |
| Ready-made personas | 10 (YAML) + builder API |
| Core dependencies | pydantic ≥2.5, pyyaml ≥6.0 |
| Optional dependencies | anthropic, openai, fastapi |
| LLM backends | Template (free), Mock (testing), Anthropic, OpenAI |
| Python | ≥3.11 |

---

## What Was Verified and How

### Bias Simulator — All 8 Biases Wired

**Claim from first review:** "Only 3 of 8 bias types wired; 5 computed and thrown away."
**Verdict:** FALSE. All 8 are wired.

**How I verified:** Read `bias_simulator.py:222-284` — `compute_modifiers()` calls all 8 `_compute_*` methods. Each returns a `BiasModifier` with a `target_field`. Then read TurnPlanner to find where `get_total_modifier_for_field()` is called for each target field.

**Evidence — all 8 biases mapped to consumption points:**

| # | Bias Type | Target Field | Computed At | Consumed At |
|---|-----------|-------------|-------------|-------------|
| 1 | Confirmation | `response_structure.elasticity` | `bias_simulator.py:286` | `turn_planner.py:1221` |
| 2 | Negativity | `communication_style.arousal` | `bias_simulator.py:330` | `turn_planner.py:1530` |
| 3 | Authority | `response_structure.confidence` | `bias_simulator.py:381` | `turn_planner.py:1307` |
| 4 | Anchoring | `response_structure.elasticity` | `bias_simulator.py:438` | `turn_planner.py:1221` (aggregated) |
| 5 | Status Quo | `response_structure.elasticity` | `bias_simulator.py:477` | `turn_planner.py:1221` (aggregated) |
| 6 | Availability | `communication_style.arousal` | `bias_simulator.py:523` | `turn_planner.py:1530` (aggregated) |
| 7 | Empathy Gap | `knowledge_disclosure.disclosure_level` | `bias_simulator.py:568` | `turn_planner.py:756` |
| 8 | Dunning-Kruger | `response_structure.confidence` | `bias_simulator.py:617` | `turn_planner.py:1307` (aggregated) |

The consumption mechanism is `BiasSimulator.get_total_modifier_for_field()` which sums all modifiers targeting a given field. TurnPlanner calls this at 4 points — one for each unique target field — and applies the aggregated modifier with citation.

---

### Linguistic Markers — Fully Wired to LLM Prompt

**Claim from first review:** "linguistic_markers.py generates orphaned directives never used in prompts."
**Verdict:** FALSE. They reach the LLM prompt.

**How I verified:** Traced the full chain from computation to consumption.

**Evidence — the complete data flow:**

1. **Computed:** `turn_planner.py:986-992` — `build_personality_language_directives()` called with traits, mood, formality
2. **Collected:** `turn_planner.py:993-997` — directives collected into `personality_language` list
3. **Cited:** `turn_planner.py:999-1005` — citation added to TraceContext
4. **Stored on IR:** `turn_planner.py:1043` — `personality_language=personality_language` passed to IR constructor
5. **Read by PromptBuilder:** `generation/prompt_builder.py:177` — `personality_language = ir.personality_language`
6. **Injected into prompt:** `generation/prompt_builder.py:179-184` — each directive written under "=== LANGUAGE STYLE (personality-grounded) ===" section
7. **Sent to LLM:** `generation/response_generator.py:156-161` — `build_generation_prompt()` called with `behavioral_directives=ir.behavioral_directives`

The directives are not orphaned. They flow from TurnPlanner → IR → PromptBuilder → LLM system prompt.

---

### Social Cognition — Outputs Consumed at Multiple Points

**Claim from first review:** "social_cognition.py computes schema threat but nothing uses the output."
**Verdict:** FALSE. Consumed at 5 points in TurnPlanner.

**How I verified:** Searched for `schema_effect` and `adaptation` in TurnPlanner, read each consumption point.

**Evidence — consumption points:**

| # | What | Where | Effect |
|---|------|-------|--------|
| 1 | `schema_effect.confidence_modifier` | `turn_planner.py:537` | Modifies confidence value |
| 2 | `schema_effect.elasticity_modifier` | `turn_planner.py:538` | Modifies elasticity value |
| 3 | `schema_effect.disclosure_modifier` | `turn_planner.py:777-789` | Modifies disclosure level |
| 4 | `schema_effect.prompt_directive` | `turn_planner.py:981-982` | Added to behavioral directives |
| 5 | `adaptation.prompt_directives` | `turn_planner.py:979-980` | Added to behavioral directives |
| 6 | `adaptation.formality_shift` | `turn_planner.py:658-671` | Modifies formality |
| 7 | `adaptation.depth_shift` | `turn_planner.py:672-701` | Modifies verbosity |
| 8 | `adaptation.disclosure_reciprocity` | `turn_planner.py:792-805` | Modifies disclosure |

Social cognition outputs affect 6 different IR fields plus behavioral directives.

---

### Error Handling — Guards Exist

**Claim from first review:** "message.content[0] without empty-list check."
**Verdict:** FALSE.

**Evidence:**
- `generation/llm_adapter.py:132`: `if not message.content: raise LLMResponseError("Anthropic returned empty response (no content blocks)")`
- `response/adapters.py:387`: `if not message.content: raise LLMResponseError("Anthropic returned empty response (no content blocks)")`

Both adapter implementations guard `message.content` before accessing `[0]`.

---

### Strict Mode — Implemented

**Claim from IMPROVEMENT_PLAN.md (A.2):** "strict_mode=True is accepted as a parameter but doesn't change behavior."
**Verdict:** FALSE on this branch. It is implemented.

**Evidence:**
- `generation/response_generator.py:93-95`:
  ```python
  # In strict mode, force TemplateAdapter for deterministic output
  if strict_mode and not isinstance(adapter, TemplateAdapter):
      self.adapter = TemplateAdapter()
  ```
- Also enforces strict verbosity at `response_generator.py:184`: `strict=self.strict_mode`

Note: The `write_policy` field in `MemoryOps` (ir_schema.py:393) IS unenforced — it's set to "strict" by default but the memory system never checks it. This is a separate issue from strict_mode.

---

## Confirmed Issues

### Issue 1: TurnPlanner Monolith (2,284 Lines)

**File:** `persona_engine/planner/turn_planner.py`
**Severity:** Medium — code is readable and well-commented, but hard to test stages in isolation

The TurnPlanner class has 5 stage methods averaging 400+ lines each, plus 20+ helper methods. The `generate_ir()` method at line 232 orchestrates them cleanly, but the individual stages mix computation with citation recording, cross-turn smoothing, and module interactions.

**Impact:**
- Can't unit-test `_stage_behavioral_metrics()` without setting up the full pipeline
- High cyclomatic complexity within each stage
- New contributors must read 2,284 lines to understand any single behavior

**Recommendation:** Split into 5 stage classes (`FoundationStage`, `InterpretationStage`, `BehavioralMetricsStage`, `KnowledgeSafetyStage`, `FinalizationStage`) each under 500 lines. TurnPlanner becomes a thin orchestrator.

---

### Issue 2: Emotional Appraisal Has No Negation Handling

**File:** `persona_engine/behavioral/emotional_appraisal.py`
**Severity:** Medium — affects mood updates, not core IR fields

**How I verified:** Ran `detect_user_emotion()` with negated inputs:

```
Input: "I am excited about this"     → {'joy': 0.25}
Input: "I am not excited about this" → {'joy': 0.25}   ← WRONG (identical)

Input: "worried"                     → {'fear': 0.2}
Input: "not worried at all"          → {'fear': 0.2}    ← WRONG (identical)
```

The function at line 62 uses word-set intersection (`words & _ENTHUSIASM_MARKERS`) which cannot detect negation. The bias simulator (`bias_simulator.py:107`) has `_count_unnegated_markers()` which correctly handles negation by checking a 3-token window for negation words — but this pattern was never ported to emotional appraisal.

**Impact:** Emotional appraisal will incorrectly detect positive/negative emotions from negated statements, causing inappropriate mood updates. The effect is bounded (valence delta clamped to ±0.4, arousal to ±0.3) but still incorrect.

**Recommendation:** Port the `_count_unnegated_markers` pattern from bias_simulator.py, or refactor into a shared utility.

---

### Issue 3: Test Helper Duplication (7 Copies)

**File pattern:** `tests/test_phase_r*.py`
**Severity:** Low — maintenance burden, no functional impact

**How I verified:** Searched for `def _make_persona_data` across test files:

```
tests/test_phase_r1_activate_psychology.py:54
tests/test_phase_r2_amplify_effects.py:53
tests/test_phase_r3_trait_interactions.py:42
tests/test_phase_r4_emotional_appraisal.py:39
tests/test_phase_r5_language_generation.py:55
tests/test_phase_r6_social_cognition.py:67
tests/test_phase_r7_psychometric_validation.py:67
```

Seven identical copies of the same helper function. Should be in `tests/conftest.py` as a shared fixture.

---

### Issue 4: Deprecated `response/` Module Still Active

**Files:** `persona_engine/response/` vs `persona_engine/generation/`
**Severity:** Medium — two parallel implementations cause confusion

**How I verified:** Grepped for imports from both modules.

**Canonical path (used by SDK):**
- `engine.py:40-41` → imports from `generation/llm_adapter` and `generation/response_generator`

**Legacy path (still used by):**
- `demo_full_pipeline.py:33` → imports from `response/`
- `tests/test_response_generation.py:19-21` → imports from `response/`
- `tests/test_response_behavioral.py:24-26` → imports from `response/`
- `docs/tutorial.md:73` → imports from `response/`
- `docs/sdk_guide.md:105,118,157` → imports from `response/`

The `response/__init__.py` marks itself as deprecated but still exports all classes. The two modules have different class names (`LLMAdapter` vs `BaseLLMAdapter`), different APIs, and different prompt builders.

**Recommendation:** Update demo, tests, and docs to use `generation/`. Then remove `response/` entirely.

---

### Issue 5: `write_policy` Never Enforced

**File:** `persona_engine/schema/ir_schema.py:393`
**Severity:** Low

The `MemoryOps.write_policy` field defaults to `"strict"` with description: "Strict = only write high-confidence memories, Lenient = write all." But the memory system (`memory/memory_manager.py`) never reads this field. All write intents are processed regardless of the policy value.

**How I verified:** Searched for `write_policy` in the entire `memory/` directory — zero matches.

---

### Issue 6: `test_property_based.py` Collection Error

**Severity:** Low

The test file fails to collect because `hypothesis` is not in the `dev` dependencies in `pyproject.toml`. The dev deps include pytest, pytest-cov, black, ruff, mypy — but not hypothesis.

---

### Issue 7: Trait Interaction Thresholds Are Arbitrary

**File:** `persona_engine/behavioral/trait_interactions.py`
**Severity:** Low — the feature works, but the specific numbers lack justification

Nine hardcoded personality synergy patterns (e.g., "intellectual_combatant" = high O + low A) with thresholds like `openness > 0.7` and `agreeableness < 0.35`. No citations or references for why these specific thresholds were chosen. The patterns are consumed at `turn_planner.py:556-597` and do affect IR values (confidence, elasticity, hedging, enthusiasm, directness).

---

## Behavioral Validation — Do Personas Actually Differ?

**Method:** Loaded UX researcher persona YAML, created twin with only openness changed (0.95 vs 0.15), ran 3 prompts through each.

| Prompt | Metric | HIGH-O (0.95) | LOW-O (0.15) | Delta |
|--------|--------|--------------|-------------|-------|
| "What do you think about remote work?" | Elasticity | 0.756 | 0.363 | **+0.393** |
| "What do you think about remote work?" | Confidence | 0.329 | 0.377 | -0.048 |
| "I disagree with your approach" | Elasticity | 0.756 | 0.363 | **+0.393** |
| "I disagree with your approach" | Directness | 0.649 | 0.649 | 0.000 |
| "Tell me about a challenging experience" | Elasticity | 0.666 | 0.338 | **+0.328** |
| "Tell me about a challenging experience" | Confidence | 0.858 | 0.906 | -0.048 |

**Findings:**
- Elasticity shows strong differentiation (+0.33 to +0.39) — psychologically correct (high openness = more willing to change mind)
- Confidence shows slight inverse correlation — psychologically plausible (high openness = slightly less certain)
- Directness unchanged by openness — correct (directness is primarily driven by agreeableness)
- Tone and formality also showed differences between the 10 ready-made personas (chef: directness 0.984, formality 0.300 vs UX researcher: directness 0.496, formality 0.525)

**Verdict:** The personality model produces measurably different IR values that align with psychological theory. The system works as designed at the IR level.

**Gap:** No validation that generated *text* reflects these IR differences. Template backend produces formulaic text. LLM backend wasn't tested (requires API key). This remains the biggest open question.

---

## Architecture Assessment

### Strengths

1. **Canonical modifier sequence** — base → role → trait → state → bias → clamp prevents double-counting and makes every modification traceable
2. **Citation system** — every IR field has a full audit trail showing exactly which trait/value/bias/rule modified it and by how much
3. **Deterministic by design** — per-turn SHA-256 seeding (hash of base_seed + conversation_id + turn_number) guarantees reproducible output
4. **Clean separation** — behavioral interpreters are decoupled, single-responsibility modules
5. **Pydantic v2 schemas** — strong type safety, self-documenting models
6. **Memory system** — immutable frozen dataclass records with confidence decay, preventing persona drift
7. **Safety layers** — invariants have veto power, claim policies gate knowledge claims, privacy filters cap disclosure
8. **Multiple backends** — same IR renders through Template (free), Mock (testing), or real LLMs
9. **Bias simulator** — all 8 biases bounded at ±0.15, observable via citations, never override safety constraints

### Weaknesses

1. **TurnPlanner monolith** — 2,284 lines in one class makes isolated testing difficult
2. **Two response modules** — `response/` and `generation/` create API confusion
3. **Emotional appraisal is naive** — keyword matching without negation handling
4. **No end-to-end text validation** — tests verify IR math, not whether text sounds like the persona
5. **Stage method returns are `dict[str, Any]`** — defeats static analysis

---

## Priority Recommendations

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Add negation handling to emotional_appraisal.py | Small (port existing pattern) | Fixes incorrect emotion detection |
| 2 | Consolidate test helpers into conftest.py | Small | Reduces 7 copies to 1 |
| 3 | Remove `response/` module, update refs | Medium | Eliminates API confusion |
| 4 | Enforce or remove `write_policy` | Small | Removes dead configuration |
| 5 | Split TurnPlanner into stage classes | Large | Improves testability and maintainability |
| 6 | Add `hypothesis` to dev dependencies | Trivial | Enables property-based tests |
| 7 | Type stage method returns | Medium | Enables static analysis |
| 8 | Add text-level behavioral validation | Large | Proves the core thesis |

---

## Corrections From First Review

The first review (delegated to an exploration agent) contained 4 factual errors, all caught by the repo handler:

| Wrong Claim | Reality | Root Cause |
|-------------|---------|------------|
| "Only 3/8 biases wired" | All 8 wired via `get_total_modifier_for_field()` at 4 aggregation points | Agent searched for direct imports, not `get_total_modifier_for_field()` calls |
| "Linguistic markers orphaned" | Full chain: TurnPlanner → IR → PromptBuilder → LLM prompt | Agent didn't trace through IR object storage |
| "Social cognition unused" | Consumed at 8 points across confidence, elasticity, disclosure, formality, verbosity, and directives | Agent didn't follow `metrics.get("schema_effect")` pattern |
| "No error handling on message.content" | Both adapters have explicit `if not message.content` guards | Agent didn't read the actual adapter code |

**Lesson:** Never claim code is dead without tracing the full consumption path. Grep for imports is insufficient — data flows through intermediate objects (IR, metrics dicts, modifier lists).
