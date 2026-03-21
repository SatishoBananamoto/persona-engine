# Persona Engine — Improvement Plan (Phase 8+)

**Created:** 2026-03-14
**Status:** PROPOSED — Awaiting Approval
**Baseline:** 1,899 tests passing, Phases 1-7 complete

---

## Context

Phases 1-7 of the Implementation Plan are complete with 1,899 passing tests. This plan addresses the remaining ROADMAP gaps, newly discovered code quality issues, and the path from "impressive prototype" to "production-ready MVP."

Two independent agent audits identified the priorities below. Issues are organized into 5 phases, ordered by impact and dependency.

---

## Phase A: Critical Functional Gaps (Memory & Strict Mode)

**Goal:** Make the two biggest "wired but not working" features actually work.

### A.1 — Wire Memory Reads Into IR Generation (CRITICAL)
- **Files:** `persona_engine/planner/turn_planner.py`, `persona_engine/memory/memory_manager.py`
- **Problem:** `fulfill_read_requests()` exists in MemoryManager but is never called during `generate_ir()`. Memory writes work, but stored facts never influence stance, confidence, or disclosure decisions. The persona doesn't "remember" anything.
- **Fix:**
  1. In `_stage_foundation()`, call `memory_manager.get_context_for_turn()` to retrieve relevant facts, preferences, trust/rapport, and recent episodes
  2. In `_stage_behavioral_metrics()`, use retrieved facts to adjust confidence (if persona has prior knowledge of topic) and stance (if prior stance exists beyond stance cache)
  3. In `_stage_knowledge_safety()`, use trust level to modulate disclosure
  4. Add citations for each memory-influenced field
- **Test:** Store a fact ("User is a Python developer"), send related prompt ("What language should I learn?"), assert confidence is higher and stance references the stored fact.

### A.2 — Implement Working Strict Mode (HIGH)
- **Files:** `persona_engine/generation/response_generator.py`, `persona_engine/generation/llm_adapter.py`
- **Problem:** `strict_mode=True` is accepted as a parameter but doesn't change behavior. The system still calls the LLM with prompt constraints instead of enforcing template-based generation.
- **Fix:**
  1. When `strict_mode=True`, force `TemplateAdapter` regardless of configured backend
  2. Expand TemplateAdapter's template library: add templates for all 17 tones × 3 verbosity levels (at least a representative subset — ~20 templates)
  3. Use IR fields (tone, verbosity, confidence, stance) to select and populate templates deterministically
- **Test:** Run same prompt with strict_mode on/off using same seed — strict mode always produces identical output. Non-strict mode may vary.

### A.3 — Wire Bias Simulator Modifiers (HIGH)
- **Files:** `persona_engine/planner/turn_planner.py`
- **Problem:** `compute_modifiers()` returns `BiasModifier` objects that are stored in `self._current_bias_modifiers` but never applied to IR fields. Biases are computed but have no effect.
- **Fix:**
  1. After computing modifiers in `_stage_interpretation()`, apply them in `_stage_behavioral_metrics()`:
     - Confirmation bias modifier → elasticity adjustment
     - Negativity bias modifier → tone/arousal adjustment
     - Authority bias modifier → confidence adjustment
  2. Add citations for each applied modifier
  3. Respect existing ±0.15 bounds
- **Test:** Persona with high confirmation bias + value-aligned topic → measurably lower elasticity than same persona without bias.

### Checkpoint A
- [ ] Memory reads influence IR generation (facts affect confidence/stance)
- [ ] Strict mode produces deterministic template-based output
- [ ] Bias modifiers are applied to IR fields with citations
- [ ] All 1,899 existing tests still pass
- [ ] New tests for each fix

---

## Phase B: Robustness & Error Handling

**Goal:** Make the system production-safe — handle failures gracefully, validate inputs thoroughly.

### B.1 — Add Exception Handling in LLM Adapters (HIGH)
- **Files:** `persona_engine/generation/llm_adapter.py`
- **Problem:** AnthropicAdapter and OpenAIAdapter have no try-catch around API calls. Empty responses cause `IndexError` on `message.content[0]`. Network failures propagate unhandled.
- **Fix:**
  1. Wrap API calls in try-except, catch provider-specific errors
  2. Convert to typed exceptions (`LLMConnectionError`, `LLMResponseError`)
  3. Guard `message.content[0]` with empty check
  4. Add configurable retry with exponential backoff (max 3 retries)
  5. Optional: fallback to TemplateAdapter on persistent LLM failure
- **Test:** Mock API to return empty content → `LLMResponseError`. Mock network failure → retry then `LLMConnectionError`.

### B.2 — Fix Unsafe Field Exposure (MEDIUM)
- **Files:** `persona_engine/engine.py`, `persona_engine/conversation.py`
- **Problem:** `ChatResult._user_input` is named private (underscore) but accessed directly in multiple places. Breaks encapsulation.
- **Fix:** Rename to `user_input` (public) since it's part of the API surface. Update all references.
- **Test:** Assert `ChatResult.user_input` is accessible and contains the original input.

### B.3 — Add Input Sanitization to Domain Detection (MEDIUM)
- **Files:** `persona_engine/planner/domain_detection.py`
- **Problem:** Domain detection runs on raw user input without the sanitization applied in `engine.py`. Unicode normalization, emoji, and control characters are not handled.
- **Fix:** Apply the same `_sanitize_text()` from engine.py before keyword matching. Extract sanitizer to a shared utility.
- **Test:** Domain detection with emoji-laden input and unicode variants produces same results as clean input.

### B.4 — Validate Persona Data Completeness (MEDIUM)
- **Files:** `persona_engine/planner/turn_planner.py`, `persona_engine/schema/persona_schema.py`
- **Problem:** TurnPlanner assumes `knowledge_domains` is always iterable. If `None`, `TypeError` instead of clear validation error. `DisclosurePolicy.bounds` assumed to be 2-tuple without validation.
- **Fix:**
  1. Add Pydantic validators that ensure `knowledge_domains` defaults to `[]`
  2. Validate `DisclosurePolicy.bounds` is exactly 2 elements with min < max
  3. Add schema-level validation for all required nested structures
- **Test:** Persona with `knowledge_domains=None` → defaults to `[]`, not crash. Bounds `(0.9, 0.1)` → validation error.

### Checkpoint B
- [ ] LLM failures produce typed exceptions, not crashes
- [ ] Retry logic handles transient API failures
- [ ] All persona field access is safe with proper defaults
- [ ] Input sanitization is consistent across all paths
- [ ] All previous tests still pass

---

## Phase C: Persona Library & Validation

**Goal:** Build the persona library ROADMAP Phase 8 calls for, with counterfactual twins and benchmarks.

### C.1 — Create Counterfactual Twin Personas (HIGH)
- **New files:** `personas/twins/`
- **What:** For each Big Five dimension, create a twin pair: identical persona except one trait differs significantly.
  - `high_openness_twin.yaml` / `low_openness_twin.yaml` (O: 0.9 vs 0.2, all else equal)
  - `high_extraversion_twin.yaml` / `low_extraversion_twin.yaml`
  - `high_neuroticism_twin.yaml` / `low_neuroticism_twin.yaml`
  - `high_conscientiousness_twin.yaml` / `low_conscientiousness_twin.yaml`
  - `high_agreeableness_twin.yaml` / `low_agreeableness_twin.yaml`
- **Purpose:** Prove that changing one trait actually changes behavior. If twins produce identical IR, something is wrong.
- **Test:** Run same 5 prompts through each twin pair. Assert measurable IR differences:
  - Openness twins: different elasticity
  - Extraversion twins: different disclosure
  - Neuroticism twins: different confidence
  - Conscientiousness twins: different verbosity
  - Agreeableness twins: different directness

### C.2 — Build Benchmark Conversations (HIGH)
- **New files:** `benchmarks/`
- **What:** Pre-defined conversation scripts for 4 interaction modes:
  1. `casual_chat.json` — 5 turns of friendly conversation
  2. `interview.json` — 5 turns of probing questions
  3. `customer_support.json` — 5 turns of issue resolution
  4. `survey.json` — 5 turns of opinion gathering
- **Each benchmark includes:**
  - User inputs (the "script")
  - Expected IR ranges per persona type (expert should have confidence > 0.7, etc.)
  - Behavioral assertions (neuroticism twin should show lower confidence than stable twin)
- **Test:** Run all 8 personas through all 4 benchmarks. Assert no validation failures. Record coherence scores.

### C.3 — Add 4 More Production Personas (MEDIUM)
- **New files:** `personas/`
- **Candidates (diverse backgrounds):**
  1. **Software Engineer** — High analytical, moderate openness, domain: technology
  2. **Social Worker** — High agreeableness, high benevolence, domain: psychology/health
  3. **Entrepreneur** — High achievement, high risk tolerance, domain: business
  4. **Retired Teacher** — High conscientiousness, moderate neuroticism, domain: education
- **Each persona:** Full YAML with all fields, validated against schema, tested with 5-turn benchmark

### C.4 — Implement Trait Marker Scorer (MEDIUM)
- **New file:** `persona_engine/validation/trait_scorer.py`
- **What:** Given a generated response text and expected Big Five profile, score how well the text exhibits expected markers:
  - High openness: abstract language, metaphors, "what if" phrasing
  - High conscientiousness: structured responses, lists, qualifiers
  - High extraversion: self-disclosure, enthusiasm, exclamation
  - High agreeableness: validation, hedging, inclusive language
  - High neuroticism: hedging, worry words, qualifiers
- **Returns:** Per-trait marker score (0-1) + overall coherence score
- **Test:** Feed known high-openness response → high openness score. Feed known low-openness response → low openness score.

### Checkpoint C
- [ ] 10 counterfactual twin personas (5 pairs)
- [ ] 4 benchmark conversation scripts
- [ ] 12+ production personas total
- [ ] Trait marker scorer validates psychological realism
- [ ] All twins produce measurably different IR for their varied trait
- [ ] All personas pass all benchmarks without validation failures

---

## Phase D: Code Quality & Performance

**Goal:** Fix the code quality issues found in the audit. Clean, consistent, well-typed codebase.

### D.1 — Type Stage Method Returns (MEDIUM)
- **Files:** `persona_engine/planner/turn_planner.py`
- **Problem:** `_stage_*()` methods return `dict[str, Any]`, defeating static analysis.
- **Fix:** Create `TypedDict` or `@dataclass` for each stage's return type:
  - `FoundationResult`, `InterpretationResult`, `BehavioralMetricsResult`, `KnowledgeSafetyResult`, `FinalizationResult`
- **Test:** mypy passes with no new `type: ignore` annotations.

### D.2 — Pre-split Negative Markers (LOW)
- **Files:** `persona_engine/behavioral/bias_simulator.py`
- **Problem:** `_count_unnegated_markers()` re-splits marker strings on every call.
- **Fix:** Pre-split markers at module level: `_MARKERS_SPLIT = {m: m.split() for m in NEGATIVE_MARKERS}`
- **Test:** Behavior unchanged, marginally faster on long inputs.

### D.3 — Atomic State Mutations (LOW)
- **Files:** `persona_engine/behavioral/state_manager.py`
- **Problem:** State fields are temporarily out of bounds during mutation before clamping.
- **Fix:** Compute clamped value before assignment:
  ```python
  self.state.mood_valence = max(-1.0, min(1.0, self.state.mood_valence + delta))
  ```
- **Test:** No observable behavior change, but eliminates latent inconsistency.

### D.4 — Consolidate Duplicate Adapter Modules (LOW)
- **Files:** `persona_engine/generation/llm_adapter.py`, `persona_engine/response/adapters.py`
- **Problem:** Two adapter modules exist. If both are needed, document why. If not, consolidate.
- **Fix:** Audit imports. If `response/adapters.py` is legacy, deprecate with forwarding imports.
- **Test:** All existing imports still work.

### D.5 — Add Structured Logging (MEDIUM)
- **Files:** All core modules
- **Problem:** No structured logging. Debug prints scattered. No observability.
- **Fix:**
  1. Add `logging.getLogger(__name__)` to each module
  2. Log at key decision points: IR generation start/end, memory reads/writes, bias activations, constraint violations
  3. Use structured format: `logger.info("ir_generated", extra={"turn": n, "confidence": 0.82, "tone": "warm"})`
  4. No log output by default (user configures handler)
- **Test:** Generate IR, assert log records contain expected fields.

### Checkpoint D
- [ ] All stage methods have typed returns
- [ ] mypy passes cleanly
- [ ] Structured logging in all core modules
- [ ] No duplicate module confusion
- [ ] All previous tests still pass

---

## Phase E: Documentation & Developer Experience

**Goal:** Make the project usable by someone who hasn't read the source code.

### E.1 — SDK Usage Guide
- **New file:** `docs/sdk_guide.md`
- **Contents:**
  - Installation & setup
  - Quick start (3 lines to first conversation)
  - PersonaEngine API reference (chat, plan, reset, save/load)
  - Conversation class usage (iteration, export, summary)
  - PersonaBuilder fluent API (building personas programmatically)
  - Backend selection (mock, template, anthropic, openai)
  - Error handling (exception hierarchy, retry patterns)

### E.2 — Persona Authoring Guide
- **New file:** `docs/persona_authoring.md`
- **Contents:**
  - YAML schema reference (every field explained)
  - Big Five trait → behavior mapping table
  - Schwartz values → goal derivation
  - How to set knowledge domains and proficiency
  - Claim policy configuration
  - Invariants and safety constraints
  - Common mistakes and troubleshooting

### E.3 — IR Field Reference
- **New file:** `docs/ir_reference.md`
- **Contents:**
  - Every IR field with type, range, and semantic meaning
  - Citation system explained (source types, operations, weights)
  - Safety plan interpretation
  - How to assert on IR fields for testing
  - Example IR with annotations

### E.4 — Getting Started Tutorial
- **New file:** `docs/tutorial.md`
- **Contents:**
  - Step-by-step: install → load persona → chat → inspect IR → customize
  - 10-minute path to a working conversation
  - Common patterns and recipes

### Checkpoint E
- [ ] All docs written and reviewed
- [ ] Code examples in docs are tested (extract and run)
- [ ] New user can go from zero to working conversation in <10 minutes

---

## Excluded from This Plan (Deliberate Deferrals)

| Item | Reason |
|------|--------|
| **Async/await support** | Large initiative that breaks API stability. Own project after MVP. |
| **Event bus / hooks** | Architectural enhancement for post-MVP extensibility |
| **CI/CD setup** | Important but doesn't affect functionality. Can be added anytime. |
| **Embedding-based intent analysis** | Current keyword matching works. Embeddings add dependency complexity. |
| **Response strategy IR field** | Nice-to-have. Current IR tells "what" but not "how to structure." Post-MVP. |
| **Full pipeline middleware pattern** | 5-stage pipeline is sufficient for current needs |
| **Distributional guarantees** | Requires large-scale statistical infrastructure. Post-MVP. |
| **Confirmation bias proxy refinement** | Current topic_relevance proxy is acceptable for MVP |

---

## Estimated Effort

| Phase | Focus | Effort |
|-------|-------|--------|
| **A** | Memory reads, strict mode, bias wiring | ~8-12 hours |
| **B** | Error handling, validation, sanitization | ~6-8 hours |
| **C** | Persona library, twins, benchmarks, scorer | ~12-16 hours |
| **D** | Type safety, logging, code quality | ~6-8 hours |
| **E** | Documentation & tutorials | ~8-10 hours |
| **Total** | | **~40-54 hours** |

---

## Success Criteria

The MVP is complete when:
1. Memory reads influence IR generation (personas learn from conversations)
2. Strict mode produces deterministic, template-based responses
3. Bias modifiers are applied (not just computed)
4. 12+ personas with 5 counterfactual twin pairs
5. 4 benchmark conversations with coherence scores
6. Trait marker scorer validates psychological realism
7. LLM failures produce typed exceptions with retry
8. Comprehensive documentation enables self-service onboarding
9. All tests pass (target: 2,100+)
