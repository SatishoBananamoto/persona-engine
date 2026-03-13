# Persona Engine — Implementation Plan (Agent-Reviewed)

**Created:** 2026-03-13
**Status:** Approved after review by 3 independent agents
**Goal:** Fix all critical/major issues identified in `AGENT_REVIEWS.md` and bring the project to production quality.

---

## Guiding Principles

1. **Fix bugs before refactors** — correctness first, elegance second.
2. **Each phase ends with a checkpoint** — all existing tests must pass, plus new tests covering the fixes.
3. **No phase starts until the previous checkpoint is green.**
4. **Minimal blast radius** — each fix is a focused, reviewable commit.

---

## Phase 1: Critical Bug Fixes

**Goal:** Eliminate data corruption and logic errors that silently produce wrong results.

### Fix 1.1 — Remove Double Memory Writes (CRITICAL)
- **Files:** `engine.py:260-265`, `turn_planner.py:649`
- **Problem:** Every `chat()`/`plan()` call writes memory intents twice — once in TurnPlanner and once in PersonaEngine. Relationship trust/rapport deltas are applied double, causing artificially accelerated trust growth.
- **Fix:** Remove the memory write block in `engine.py:260-265`. TurnPlanner is the canonical write site.
- **Test:** Assert that after N turns, each memory store has exactly N entries (not 2N). Assert trust delta matches expected single-write value.

### Fix 1.2 — Fix Elasticity Formula (MAJOR)
- **Files:** `trait_interpreter.py:42-47`
- **Problem:** A persona with `openness=0.0` and `confidence=0.0` gets elasticity `0.714` (quite flexible). Low openness should produce LOW elasticity. The `(1 - confidence_penalty)` term always adds at least 0.7, dominating the openness factor.
- **Fix:** Use `elasticity = openness_factor - confidence_penalty` (no division by 1.4). This gives:
  - `openness=0.0, confidence=0.0` → elasticity ≈ 0.2 (rigid, correct)
  - `openness=1.0, confidence=0.0` → elasticity ≈ 0.7 (flexible, correct)
  - `openness=1.0, confidence=1.0` → elasticity ≈ 0.4 (tempered, correct)
- **Test:** Parametric test covering edge cases: `(0,0)→low`, `(1,0)→high`, `(0,1)→very low`, `(1,1)→moderate`.

### Fix 1.3 — Enforce `must_avoid` on Stance (CRITICAL)
- **Files:** `constraint_safety.py`
- **Problem:** `validate_stance_against_invariants()` checks `cannot_claim` but does NOT check `must_avoid`. A stance mentioning a must_avoid topic passes through unchecked.
- **Fix:** Add `must_avoid` checking in `validate_stance_against_invariants()`. If stance text contains a must_avoid topic, flag it and force stance regeneration or neutralization.
- **Test:** Create persona with `must_avoid=["politics"]`, send political prompt, assert stance does not contain political content.

### Fix 1.4 — Enforce `must_avoid`/`cannot_claim` on Generated Text (CRITICAL)
- **Files:** `style_modulator.py:138` (or add new post-generation validator)
- **Problem:** Prompt builder tells the LLM about forbidden claims, but `_check_safety` only checks `blocked_topics`, NOT `cannot_claim` or `must_avoid` against generated text.
- **Fix:** Add post-generation validation that checks final text against `cannot_claim` and `must_avoid`. On violation, either regenerate or return a safe fallback.
- **Test:** Mock LLM to return text containing a `cannot_claim` item, assert it gets caught.

### Fix 1.5 — Clamp Mood Valence After Stress (MINOR)
- **Files:** `state_manager.py:185`
- **Problem:** `apply_stress_trigger` subtracts 0.15 from mood_valence without clamping. Values can exceed [-1, 1] bounds.
- **Fix:** Add `self.mood_valence = max(-1.0, min(1.0, self.mood_valence))` after mutation. Audit all other state mutation sites for missing clamps.
- **Test:** Set mood to -0.95, apply stress trigger, assert mood ≥ -1.0.

### Fix 1.6 — Fix Conversation History Bug (MAJOR)
- **Files:** `engine.py:385`
- **Problem:** Saves `goal.value` instead of actual user input into conversation history, corrupting multi-turn context.
- **Fix:** Save the actual `user_input` string into conversation history.
- **Test:** Send "Hello world", assert conversation history contains "Hello world" not a goal enum value.

### Fix 1.7 — Fix `is_personal_experience` Always False (MAJOR)
- **Files:** `turn_planner.py:1441`
- **Problem:** Hardcoded `is_personal_experience = False`. `KnowledgeClaimType.PERSONAL_EXPERIENCE` exists but detection is stubbed.
- **Fix:** Implement heuristic detection: check if user input asks about personal experience ("have you ever", "what's your experience with", "do you like") AND persona has relevant domain knowledge or preferences.
- **Test:** Send "Have you ever used Python?" to a software engineer persona, assert `is_personal_experience=True`.

### Checkpoint 1
- [ ] All 161 existing tests pass
- [ ] New tests for each fix pass (minimum 15 new tests)
- [ ] Run full `pytest` with `-x` flag — zero failures
- [ ] Manual smoke test: 5-turn conversation with Dr. Amara persona, verify no double trust growth
- [ ] Verify: elasticity values make intuitive sense for extreme trait combinations

---

## Phase 2: Structural Fixes & Data Integrity

**Goal:** Fix architectural issues that cause state corruption, data loss, or incorrect reporting.

### Fix 2.1 — Consolidate Duplicate StanceCache (MAJOR)
- **Files:** `engine.py:145`, `memory_manager.py:84`
- **Problem:** Two separate StanceCache instances exist. Engine uses one; memory manager creates another (never used). `memory_manager.stats()` reports from the wrong cache (always 0).
- **Fix:** Remove StanceCache creation from `memory_manager.py`. Pass the engine's StanceCache instance to MemoryManager via constructor injection.
- **Test:** Assert `memory_manager.stats()` reports correct hit/miss counts after cached lookups.

### Fix 2.2 — Add Memory Store Capacity Limits (CRITICAL)
- **Files:** All memory stores (`fact_store.py`, `preference_store.py`, `episodic_store.py`, `relationship_store.py`)
- **Problem:** All stores are unbounded lists/dicts with no eviction. A 100-turn conversation accumulates 400+ records. `RelationshipStore.trust` is O(n) per access.
- **Fix:**
  - Add `max_capacity` parameter to each store (default: 500 for facts, 200 for episodic, 100 for preferences, 50 for relationships).
  - Implement LRU eviction for facts and episodic stores.
  - **For RelationshipStore:** Maintain running `base_trust` and `base_rapport` totals. When evicting old events, fold their deltas into the base values before removing them. This preserves accuracy while bounding memory.
  - Add `O(1)` trust/rapport access via cached running totals.
- **Test:** Fill stores beyond capacity, assert size stays at limit. Assert trust value after eviction matches trust value computed from full history.

### Fix 2.3 — Fix `save()`/`load()` (MAJOR)
- **Files:** `engine.py`
- **Problem:** `save()` loses all memory state, stance cache, and IR history. `load()` can't replay conversation state. Persistence is effectively broken.
- **Fix:** Serialize full engine state including memory stores, conversation history, and dynamic state. Use versioned format for forward compatibility.
- **Test:** Save engine after 5-turn conversation, load into new instance, assert memory stores and conversation history match.

### Fix 2.4 — Enforce `DisclosurePolicy.bounds` (MAJOR)
- **Files:** `turn_planner.py:86-87`
- **Problem:** Schema declares default bounds `(0.1, 0.9)` but planner uses `clamp01` which clamps to `[0.0, 1.0]`. Persona's declared disclosure bounds are never read.
- **Fix:** Replace `clamp01` call with `clamp(value, policy.bounds[0], policy.bounds[1])`. Read bounds from the persona's DisclosurePolicy.
- **Test:** Set bounds to `(0.3, 0.7)`, compute disclosure, assert result is within bounds.

### Fix 2.5 — Fix Expert Threshold Inconsistency (MAJOR)
- **Files:** `turn_planner.py:69`, `persona_compliance.py:26`
- **Problem:** Planner uses `getattr(claim_policy, 'expert_threshold', 0.7)` but `ClaimPolicy` doesn't define `expert_threshold` — always falls back. Validator hardcodes its own 0.7 separately.
- **Fix:** Add `expert_threshold: float = 0.7` to `ClaimPolicy` schema. Both planner and validator read from the same field.
- **Test:** Set custom threshold to 0.5, assert persona claims expertise at proficiency 0.6.

### Fix 2.6 — Wire Memory Read Path (MAJOR)
- **Files:** `turn_planner.py` (or memory integration point)
- **Problem:** Memory writes work but reads are never dynamically executed during IR generation. `MemoryOps.read_requests` is populated but `fulfill_read_requests()` is never called.
- **Fix:** Call `fulfill_read_requests()` at the appropriate point in IR generation, after read requests are populated and before they're needed for stance/disclosure decisions.
- **Test:** Store a fact in memory, send related prompt, assert the fact influences IR generation.

### Checkpoint 2
- [ ] All Checkpoint 1 tests still pass
- [ ] New tests for Phase 2 fixes pass (minimum 12 new tests)
- [ ] Memory stores respect capacity limits under stress test (1000-turn simulation)
- [ ] Save/load round-trip preserves conversation state
- [ ] `memory_manager.stats()` returns accurate StanceCache metrics

---

## Phase 3: Custom Exception Hierarchy & Input Validation

**Goal:** Replace generic exceptions with a typed hierarchy. Add input validation at system boundaries.

### Fix 3.1 — Implement Exception Hierarchy (HIGH)
- **New file:** `persona_engine/exceptions.py`
- **Hierarchy:**
  ```
  PersonaEngineError (base)
  ├── PersonaValidationError      # Schema/builder validation failures
  ├── LLMError                    # All LLM-related failures
  │   ├── LLMAPIKeyError          # Missing or invalid API key
  │   ├── LLMConnectionError      # Network/timeout failures
  │   └── LLMResponseError        # Malformed or empty LLM response
  ├── IRGenerationError           # Turn planner failures
  ├── MemoryError                 # Memory store failures
  │   ├── MemoryCapacityError     # Store full, eviction failed
  │   └── MemoryCorruptionError   # Inconsistent state detected
  └── ConfigurationError          # Missing deps, bad config
  ```
- **Fix:** Replace all `ValueError`, `ImportError`, and bare `Exception` raises with appropriate typed exceptions. Ensure each exception includes contextual information (which field failed, what was expected vs actual).
- **Test:** Assert specific exception types are raised for each failure mode.

### Fix 3.2 — Add Input Validation (MAJOR)
- **Files:** `engine.py` (entry points), `prompt_builder.py:103`
- **Problem:** User input goes straight to planner with no length checks, content filtering, or injection protection. Prompt injection is possible via direct string interpolation.
- **Fix:**
  - Add max input length check (configurable, default 10,000 chars).
  - Sanitize user input before prompt interpolation (escape control characters, strip prompt injection patterns).
  - Add input validation at `chat()` and `plan()` entry points.
- **Test:** Assert oversized input raises `PersonaValidationError`. Assert injection attempts are sanitized.

### Checkpoint 3
- [ ] All previous checkpoint tests pass
- [ ] Every public API method raises typed exceptions (no generic `ValueError`)
- [ ] Input validation rejects malicious/oversized input
- [ ] Exception messages are informative and actionable

---

## Phase 4: Developer Experience & SDK Polish

**Goal:** Make the SDK pleasant to use, well-documented, and IDE-friendly.

### Fix 4.1 — Fix README Quick-Start Examples (HIGH)
- **Files:** `README.md`
- **Problem:** Quick-start code references `.start_conversation()` and `.send()` which don't exist.
- **Fix:** Update all code examples to use actual API: `PersonaEngine()`, `.chat()`, `.plan()`.
- **Test:** Extract all code blocks from README, assert they parse as valid Python.

### Fix 4.2 — Add `__repr__` Methods (P2)
- **Files:** `engine.py`, `schema/ir_schema.py`
- **Fix:** Add informative `__repr__` to `PersonaEngine` and `IntermediateRepresentation`.
  ```python
  # PersonaEngine
  def __repr__(self):
      return f"PersonaEngine(persona='{self.persona.name}', turns={self.turn_count})"

  # IntermediateRepresentation
  def __repr__(self):
      return f"IR(stance='{self.stance[:50]}...', confidence={self.confidence})"
  ```
- **Test:** Assert repr output contains persona name and is human-readable.

### Fix 4.3 — Add Context Manager Support (P1)
- **Files:** `engine.py`
- **Fix:** Add `__enter__`/`__exit__` to `PersonaEngine` for resource cleanup (LLM clients, memory stores).
  ```python
  def __enter__(self):
      return self

  def __exit__(self, *exc):
      self.close()

  def close(self):
      # Clean up LLM client connections
      # Flush pending memory writes
      pass
  ```
- **Test:** Use engine in `with` block, assert cleanup runs on exit.

### Fix 4.4 — Add Docstring Examples to Public Methods
- **Files:** All public API methods in `engine.py`, `persona_builder.py`
- **Fix:** Add usage examples to all 8+ undocumented public methods.
- **Test:** Run `python -m doctest` on modules with examples.

### Fix 4.5 — Create Examples Directory
- **New directory:** `examples/`
- **Examples:**
  1. `quick_chat.py` — Minimal 3-turn conversation
  2. `multi_turn.py` — 10-turn conversation showing personality consistency
  3. `ir_debugging.py` — Inspect IR fields and citations
  4. `custom_persona.py` — Build persona from scratch with builder
  5. `persona_comparison.py` — Same prompt, different personas

### Checkpoint 4
- [ ] All previous checkpoint tests pass
- [ ] README examples execute without error
- [ ] All public methods have docstrings with examples
- [ ] Context manager properly cleans up resources
- [ ] All 5 examples run successfully

---

## Phase 5: Architectural Refactoring

**Goal:** Improve maintainability and extensibility without changing behavior.

### Fix 5.1 — Extract `EngineConfig` Dataclass (MEDIUM)
- **New file:** `persona_engine/config.py`
- **Problem:** 15+ hardcoded constants scattered across files (`EXPERT_THRESHOLD`, `MAX_BIAS_IMPACT`, `CROSS_TURN_INERTIA`, etc.).
- **Fix:** Create a single typed `EngineConfig` dataclass. All modules read from config instead of hardcoded values.
- **Test:** Assert changing config value propagates to all consumers.

### Fix 5.2 — Refactor TurnPlanner God Method (HIGH, LARGE EFFORT)
- **Files:** `turn_planner.py`
- **Problem:** `generate_ir()` is 1,526 lines with 18 sections, out-of-order numbering, mixing concerns.
- **Fix:** Extract into staged pipeline:
  ```
  Pipeline([
      IntentAnalysisStage(),
      DomainDetectionStage(),
      StanceGenerationStage(),
      CommunicationStyleStage(),
      DisclosureStage(),
      ConstraintEnforcementStage(),
      MemoryWriteStage(),
  ])
  ```
  Each stage is independently testable. Pipeline context flows through a `PlannerContext` object.
- **Approach:** Extract one stage at a time, running full test suite after each extraction. This is NOT a rewrite — it's mechanical extraction.
- **Test:** Before/after comparison: same inputs produce identical IR output. Each stage has unit tests.

### Fix 5.3 — Complete Cross-Turn Inertia Smoothing (MEDIUM)
- **Files:** `turn_planner.py` (smoothing section)
- **Problem:** Only 3 fields have inertia smoothing (confidence, formality, directness). Missing: tone, verbosity, competence, elasticity.
- **Fix:** Apply same smoothing pattern to all behavioral IR fields.
- **Test:** Assert behavioral fields don't jump more than allowed delta between turns.

### Fix 5.4 — Externalize Domain Registry (MEDIUM)
- **Files:** `domain_detection.py` (935 lines of config-as-code)
- **Fix:** Extract `DOMAIN_REGISTRY` to a YAML/JSON data file. Load at startup.
- **Test:** Assert domain detection produces identical results with externalized config.

### Fix 5.5 — Organize Test Files (LOW)
- **Problem:** 25+ test files at project root instead of `tests/`.
- **Fix:** Move all test files to `tests/` directory. Update any import paths.
- **Test:** All tests still discoverable and passing from new location.

### Checkpoint 5 (Final)
- [ ] All previous checkpoint tests pass
- [ ] TurnPlanner pipeline produces identical output to monolithic version
- [ ] EngineConfig is the single source for all tuning constants
- [ ] Domain detection works identically with externalized config
- [ ] All tests organized under `tests/`
- [ ] Full test suite: 200+ tests, zero failures

---

## Excluded from Plan (Deliberate Decisions)

| Item | Reason |
|------|--------|
| **Async/await support** | Large initiative, should be its own project after API is stable |
| **State evolution timing change** | Agent review confirmed current code is CORRECT despite misleading naming — `evolve_state_post_turn` runs pre-response but models anticipatory state, which is valid |
| **Dead schema fields** (`languages[]`, `decision_policies[]`, `response_patterns[]`) | Wire these when implementing the features that need them, not as standalone fixes |
| **Bias Simulator wiring** | `compute_modifiers()` returns modifiers that are never applied — defer until bias system is redesigned |
| **Documentation site** (Sphinx/readthedocs) | Post-MVP, after API stabilizes |
| **Event bus / hooks** | Architectural enhancement for post-MVP extensibility |
| **Pipeline middleware pattern** | Covered partially by Fix 5.2; full middleware is post-MVP |

---

## Estimated Timeline

| Phase | Effort | Cumulative |
|-------|--------|------------|
| Phase 1: Critical Bug Fixes | ~3 days | 3 days |
| Phase 2: Structural Fixes | ~4 days | 7 days |
| Phase 3: Exceptions & Validation | ~2 days | 9 days |
| Phase 4: DX & SDK Polish | ~3 days | 12 days |
| Phase 5: Architectural Refactoring | ~5-7 days | 17-19 days |

---

## Review Notes

This plan was reviewed by 3 independent agents. Key revisions from their feedback:

1. **Removed** Fix 1.6 (state evolution timing) — General-Purpose agent confirmed current code is correct; `evolve_state_post_turn` models anticipatory state changes.
2. **Added** Fix 1.7 (personal experience detection) — Explorer flagged as missing critical stub.
3. **Added** Fix 2.6 (memory read path wiring) — Explorer flagged unfulfilled read requests.
4. **Refined** Fix 1.2 (elasticity formula) — General-Purpose provided exact formula: `elasticity = openness_factor - confidence_penalty`.
5. **Refined** Fix 2.2 (memory eviction) — General-Purpose identified that evicted deltas must be folded into base values to preserve accuracy.
6. **Expanded** Phase 4 — Guide agent recommended `__repr__`, context managers, and docstring examples.
7. **Clarified** exclusions — Async support, dead schema fields, and bias simulator wiring are deliberate deferrals, not oversights.
