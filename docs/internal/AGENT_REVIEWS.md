# Persona Engine — Agent Code Reviews

Four specialized agents independently reviewed the persona-engine codebase on 2026-03-13. Each agent had a different focus area and no visibility into each other's findings. Issues flagged by multiple agents are noted in the consolidated section at the end.

---

## Agent 1: Guide (SDK & Developer Experience)

**Focus:** SDK design patterns, developer experience, packaging, comparison to industry best practices.

**Verdict:** *"Conceptually excellent, operationally incomplete."*

### P0 — Blocking Production Use

1. **Broken README examples** — Quick-start code references `.start_conversation()` and `.send()` which don't exist. First impression is broken.

2. **No async/await support** — Every modern Python SDK (Anthropic, OpenAI) has async variants. This blocks FastAPI, aiohttp, and concurrent use. Recommends `PersonaEngineAsync` with full API parity.

3. **No custom exception hierarchy** — Currently uses generic `ValueError` and `ImportError`. Developers can't distinguish between missing dependencies, configuration errors, LLM failures, and validation issues. Recommends:
   ```
   PersonaEngineError
   ├── PersonaValidationError
   ├── LLMError
   │   ├── LLMAPIKeyError
   │   └── LLMConnectionError
   ├── IRGenerationError
   ├── MemoryError
   └── ConfigurationError
   ```

### P1 — Important for Adoption

4. **No context manager support** — `PersonaEngine` holds LLM clients but has no `__enter__`/`__exit__` cleanup.

5. **Incomplete docstrings** — 8 public methods missing docs. Zero methods have usage examples in docstrings.

6. **No documentation site** — No Sphinx, no readthedocs, no API reference beyond README.

7. **No examples directory** — Recommends 5 core examples: quick chat, multi-turn, FastAPI integration, LangChain integration, IR debugging.

8. **Missing integration patterns** — No guidance for FastAPI, Django, or framework integration.

### P2 — Polish

9. **Missing `__repr__`** on `PersonaEngine` and `IntermediateRepresentation`.

10. **Builder type stubs** — IDE can't autocomplete `.age()`, `.location()`, `.traits()` etc.

11. **pyproject.toml gaps** — Missing `project.urls`, could benefit from more metadata.

### Estimated Effort

~4 weeks to production-ready SDK. The 80/20 is: fix README, add exceptions, add async, write 5 examples.

---

## Agent 2: Explorer (Codebase Quality Audit)

**Focus:** Dead code, test gaps, schema inconsistencies, architectural issues, security concerns.

**Verdict:** *"Usable for basic simulation, but gaps prevent world-class."*

**Scores:** Code Quality 7/10 · Maintainability 6/10 · Feature Completeness 6/10 · Robustness 5/10

### Critical Findings

1. **Dead Schema Fields — False Affordances** — Multiple persona fields are defined but never used in the pipeline:
   - `languages[]` — never influences communication style
   - `decision_policies[]` — loaded but never called
   - `biases[]` — defined in persona but BiasSimulator uses hardcoded traits instead
   - `response_patterns[]` — only partially wired (1 location)

2. **Conversation History Bug** — `engine.py:385` saves `goal.value` instead of actual user input into conversation history. This corrupts multi-turn context.

3. **TurnPlanner is a God Class (1,525 lines)** — `generate_ir()` is 700+ lines orchestrating 16 subsystems. Hard to test, hard to understand, mixes concerns.

4. **Incomplete Cross-Turn Smoothing** — Only 3 fields have inertia smoothing (confidence, formality, directness). Missing: tone, verbosity, competence, elasticity. Makes personas feel inconsistent across turns.

5. **Memory Read Path is Broken** — Memory writes work but reads are never dynamically executed during IR generation. `MemoryOps.read_requests` is populated but `fulfill_read_requests()` is never called.

### Major Issues

6. **Bias Simulator is Dead Code** — `compute_modifiers()` returns modifiers that are never applied to IR fields.

7. **Uncertainty Resolver Ignores Dynamic State** — Doesn't consider stress, fatigue, or recent history. A stressed persona hedges the same as a relaxed one.

8. **Success Criteria Not Implemented** — `ConversationFrame.success_criteria` is always `None`. No goal-tracking.

9. **Personal Experience Detection Missing** — `KnowledgeClaimType.PERSONAL_EXPERIENCE` exists but detection is stubbed with `is_personal_experience = False`.

### Test Coverage

55% of modules have tests. Untested critical modules:
- `engine.py` (main entry point)
- `memory_manager.py`
- `llm_adapter.py`
- `domain_detection.py`
- All memory stores (fact, preference, episodic)

### Security Concerns

10. **No Input Validation** — User input goes straight to planner with no length checks, content filtering, or injection protection.

11. **Persona Invariant Enforcement is Substring-Only** — `"doctor" in text.lower()` is trivially bypassed. No semantic similarity.

12. **State Mutations Unaudited** — `mood_valence += delta` with no bounds checking at mutation time. Corrupt state silently propagates.

---

## Agent 3: General-Purpose (Bug Hunting & Deep Code Review)

**Focus:** Concrete bugs, logic errors, psychological realism, formula correctness, invariant enforcement.

**Verdict:** Found 26 specific issues with file paths and line numbers. Three most impactful: double memory writes, unbounded memory growth, and unenforced `must_avoid` constraints.

### Critical Bugs

1. **Double Memory Writes** — Every `chat()` and `plan()` call writes memory intents twice: once in `TurnPlanner.generate_ir()` (turn_planner.py:649) and again in `PersonaEngine.chat()` (engine.py:260-265). Memory stores don't deduplicate. Relationship trust/rapport deltas applied double, causing artificially accelerated trust growth.

2. **Unbounded Memory Growth** — All 4 stores (`FactStore`, `EpisodicStore`, `PreferenceStore`, `RelationshipStore`) are unbounded lists/dicts with no capacity limits or eviction. Combined with double-writes, a 100-turn conversation accumulates 400+ records. `RelationshipStore.trust` iterates ALL events on every access — O(n) with no caching.

3. **`must_avoid` Not Checked Against Stance/Rationale** — `validate_stance_against_invariants()` checks `cannot_claim` but does NOT check `must_avoid`. A stance mentioning a must_avoid topic passes through to IR generation unchecked.

### Major Bugs

4. **Elasticity Formula is Wrong** (trait_interpreter.py:42-47) — A persona with openness=0.0 and confidence=0.0 gets elasticity 0.714 (quite flexible). Low openness should produce LOW elasticity. The `(1 - confidence_penalty)` term always adds at least 0.7, dominating the openness factor.

5. **`DisclosurePolicy.bounds` is Ignored** (turn_planner.py:86-87) — Schema declares default bounds `(0.1, 0.9)` but the planner uses `clamp01` which clamps to `[0.0, 1.0]`. The persona's declared disclosure bounds are never read.

6. **Expert Threshold Inconsistency** — Planner checks `claim_policy.expert_threshold` with fallback to 0.7, but `ClaimPolicy` doesn't even define `expert_threshold` — `getattr` always returns the fallback. Validator hardcodes its own 0.7 separately.

7. **`cannot_claim` and `must_avoid` Not Enforced on LLM Output** — Prompt builder tells the LLM about forbidden claims, but `StyleModulator._check_safety` only checks `blocked_topics`, NOT `cannot_claim` or `must_avoid` against generated text.

8. **Prompt Injection Risk** (prompt_builder.py:103) — User input is directly interpolated into the prompt with zero sanitization.

9. **Duplicate StanceCache Instances** — `PersonaEngine` creates one at line 145. `MemoryManager` creates a separate one at line 84. The engine uses its own; the memory manager's is never used. `memory_manager.stats()` reports from the wrong cache (always 0).

10. **State Evolution Runs Before Response** (turn_planner.py:283) — `evolve_state_post_turn` runs BEFORE generating the response. On turn 1, fatigue/stress/mood drift before the persona even speaks.

11. **Mood Valence Not Clamped After Stress** (state_manager.py:185) — `apply_stress_trigger` subtracts 0.15 from mood_valence without clamping. If mood was at -0.95, it goes to -1.10, breaking the [-1, 1] invariant.

### Psychological Realism Issues

12. **Schwartz Value Conflict = Simple Argmax** — When two values conflict, the system picks the higher-weighted one. No internal conflict modeling — no hesitation, no nuanced stances, no increased elasticity.

13. **Confirmation Bias Misuses Topic Relevance as Value Alignment** — `topic_relevance` (keyword overlap with domain) is passed as `value_alignment`. A UX researcher has high topic relevance for "dark patterns" but may have low value alignment. Bias fires based on expertise, not belief.

14. **Negativity Bias is Bag-of-Words** — `"I have no problem with this"` triggers on `"problem"`. No negation detection.

### Summary Table

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| 1 | Double memory writes | **Critical** | engine.py:260 + turn_planner.py:649 |
| 2 | Unbounded memory growth | **Critical** | All memory stores |
| 3 | must_avoid not checked on stance | **Critical** | constraint_safety.py |
| 4 | Elasticity formula wrong for low openness | Major | trait_interpreter.py:42 |
| 5 | DisclosurePolicy.bounds ignored | Major | turn_planner.py:86 |
| 6 | Expert threshold inconsistency | Major | turn_planner.py:69 vs persona_compliance.py:26 |
| 7 | No post-generation invariant enforcement | Major | style_modulator.py:138 |
| 8 | Prompt injection risk | Major | prompt_builder.py:103 |
| 9 | Duplicate StanceCache | Major | engine.py:145 vs memory_manager.py:84 |
| 10 | State evolution timing wrong | Major | turn_planner.py:283 |
| 11 | Mood valence unclamped | Minor | state_manager.py:185 |
| 12 | Value conflict = simple argmax | Minor | values_interpreter.py:141 |
| 13 | Confirmation bias uses wrong proxy | Minor | turn_planner.py:273 |
| 14 | Negativity bias bag-of-words | Minor | bias_simulator.py:208 |

---

## Agent 4: Planner (Architecture Review)

**Focus:** Architectural patterns, extensibility, performance, API design, data model, production readiness.

**Verdict:** *"The IR-as-central-artifact was the best architectural decision. The God Method is the worst."*

### Architectural Strengths

- **IR as the central artifact** — Enables testing without LLM calls, provides debuggable citation trails, cleanly separates "what to say" from "how to say it."
- **Canonical modifier composition sequence** (base → role → trait → state → constraints) prevents double-counting.
- **TraceContext** — Makes "forgot to cite" bugs structurally impossible by coupling every mutation to a citation.
- **Module boundaries** (`behavioral/`, `planner/`, `memory/`, `generation/`, `validation/`) are well-chosen.

### P0 — Fix Immediately

1. **Double Memory Write** (confirmed — same bug found by General-Purpose agent). Writes happen in both `TurnPlanner.generate_ir():649` AND `PersonaEngine.chat():261`.

2. **`is_personal_experience` Always False** — Hardcoded at `turn_planner.py:1441`. Personal experience claims are never identified.

### P1 — Before Production

3. **God Method: `generate_ir()` = 1,526 lines, 18 sections** — Section numbering is out of order (0, 1, 1.5, 2, 1.5, 3, 2.5...). Should be refactored into a staged pipeline:
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

4. **No `EngineConfig`** — 15+ hardcoded constants scattered across files (`EXPERT_THRESHOLD`, `MAX_BIAS_IMPACT`, `CROSS_TURN_INERTIA`, etc.). Some duplicated with different semantics. Should be a single typed `EngineConfig` dataclass.

5. **Memory Eviction / Compaction** — No eviction policy anywhere. `RelationshipStore.trust` is O(n) per access (sums all events every call). Should maintain running totals.

6. **Mutable Context Mutation** — `TurnPlanner.generate_ir()` mutates the caller's `ConversationContext` in-place (lines 305-306). Side effect acknowledged in a comment but not fixed.

7. **`save()`/`load()` is Unusable** — `save()` loses all memory state, stance cache, and IR history. `load()` can't replay conversation state. Persistence is effectively broken.

### P2 — World-Class

8. **Missing Patterns from Similar Systems:**

   | Pattern | Source | What It Would Add |
   |---------|--------|-------------------|
   | Pipeline/Middleware | Web frameworks | Testable, replaceable stages |
   | Event Bus | Game engines | `on_ir_generated`, `on_validation_complete` hooks |
   | Utility AI | Game AI | Score multiple stances, pick highest-utility |
   | Conversation Phases | State machines | Named macro-states: "warming up", "deep conversation", "adversarial" |
   | Embedding Intent | Rasa/Botpress | Fallback when keyword matching is ambiguous |

9. **Missing IR Field: `response_strategy`** — The IR tells the generator *what* to say (tone, confidence, stance) but not *how* to structure it. Should the persona lead with a question? Share an anecdote?

10. **Extensibility is Modification-Based** — Adding a new personality dimension, memory store, or bias type all require modifying core files. Should use registry patterns.

11. **`domain_detection.py` — 935 Lines of Config-as-Code** — The `DOMAIN_REGISTRY` has hardcoded keyword weights. This should be external data (YAML/JSON).

12. **Test Files Scattered** — 25+ test files sit at project root instead of `tests/`. Only `test_persona_builder.py` is in the proper location.

---

## Consolidated: Cross-Agent Agreement

Issues flagged independently by multiple agents carry the highest confidence:

| Issue | Flagged By | Priority |
|-------|-----------|----------|
| **Double memory writes** | Explorer, General, Planner | CRITICAL |
| **Unbounded memory growth** | Explorer, General, Planner | CRITICAL |
| **`must_avoid` not enforced on stance** | Explorer, General | CRITICAL |
| **God Method (1,526 lines)** | Explorer, Planner | HIGH |
| **No async support** | Guide, Planner | HIGH |
| **No custom exception hierarchy** | Guide | HIGH |
| **Broken README examples** | Guide | HIGH |
| **Elasticity formula wrong** | General | MAJOR |
| **DisclosurePolicy.bounds ignored** | General | MAJOR |
| **Duplicate StanceCache** | General, Planner | MAJOR |
| **Prompt injection risk** | General | MAJOR |
| **Dead schema fields** | Explorer | MAJOR |
| **`save()`/`load()` broken** | Planner | MAJOR |
| **Conversation history bug (saves goal not input)** | Explorer, General | MAJOR |
| **No EngineConfig** | Planner | MEDIUM |
| **No event bus / hooks** | Planner | MEDIUM |
| **State evolution timing** | General | MEDIUM |
| **55% module test coverage** | Explorer | MEDIUM |

### Recommended Fix Order

1. Remove duplicate memory write (Critical bug, easy fix)
2. Add memory store capacity limits with eviction (Critical, medium effort)
3. Enforce `must_avoid`/`cannot_claim` on both IR stance AND generated text (Critical)
4. Fix elasticity formula (Major bug, easy fix)
5. Fix conversation history bug — save actual user input (Major, easy fix)
6. Remove or consolidate duplicate StanceCache (Major, easy fix)
7. Add `DisclosurePolicy.bounds` enforcement (Major, easy fix)
8. Add custom exception hierarchy (High, small effort)
9. Add mood/state clamp after mutations (Minor, easy fix)
10. Refactor TurnPlanner God Method into pipeline stages (High, large effort)
