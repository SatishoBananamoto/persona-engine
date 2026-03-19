# Persona Engine — Improvement Plan

> Synthesized from 5 specialist agent analyses (Testing, Performance, Security, Architecture, Observability) with cross-agent debate resolution. Updated with second-pass deep-dive findings.

---

## Executive Summary

The Persona Engine has a strong foundation: a brilliant deterministic IR layer, ~1,877 tests, and clean behavioral decomposition. However, it has critical gaps in **security** (prompt injection resistance is low, blocked topics enumerated in prompts), **observability** (zero logging), **architecture** (duplicate generation layers, 1,626-line god class), **reliability** (33.7% actual line coverage, LLM adapter error handling broken), and **operational readiness** (no CI/CD).

This plan addresses these gaps across 4 phases, prioritizing items where multiple specialist agents independently converged.

---

## Consensus Items (Multi-Agent Agreement)

| Item | Agents Agreeing | Priority |
|------|----------------|----------|
| Consolidate `generation/` and `response/` into one module | Testing, Architecture | CRITICAL |
| Add structured logging (zero exists today) | Testing, Security, Observability | CRITICAL |
| LLM adapter error handling is broken (no try/except, IndexError on empty responses) | Testing, Security, Observability | CRITICAL |
| FactStore needs capacity limits + deduplication | Performance, Observability | HIGH |
| Prompt injection defenses needed (user input unescaped in prompts) | Security, Observability | HIGH |
| CLI untested (0% coverage) | Testing, Architecture | HIGH |
| CI/CD pipeline doesn't exist | Testing, Observability | HIGH |
| Memory stores have unbounded growth (PreferenceStore obs, StanceCache, ChatResult history) | Performance, Observability | HIGH |
| `save()`/`load()` reaches into private memory store attributes | Architecture, Testing | HIGH |
| `reset()` duplicates `__init__` constructor logic | Architecture, Testing | HIGH |
| API keys could leak in errors | Security, Architecture | MODERATE |
| Missing `__all__`, `__version__`, `ChatResult` export | Architecture, Testing | MODERATE |
| TurnPlanner uses untyped `dict[str, Any]` for inter-stage communication | Architecture, Testing | MODERATE |
| `GeneratedResponse` stores full system/user prompts (leak risk) | Security | MODERATE |
| StyleDriftDetector is orphaned — built but never wired into pipeline | Observability, Architecture | MODERATE |

---

## Debates & Resolutions

### Debate 1: Delete `response/` directory?

**Agents 1 & 4 agree**: Two parallel generation layers (`generation/` and `response/`) is a maintenance nightmare — bug fixes applied twice, coverage on one doesn't protect the other, new developers confused.

**Resolution: YES — Do it in Phase 1.** The risks of NOT doing it compound over time. Audit both, merge unique features into `generation/`, delete `response/`.

### Debate 2: Add async support?

**Agent 2 argues NO**: LLM call is 99%+ of latency. Making CPU code async doesn't help. Memory stores aren't thread-safe. Use `asyncio.to_thread()` at the application layer.

**Resolution: SKIP async for now.** Document that `PersonaEngine` is not thread-safe. Provide a `to_thread()` example. Revisit only if streaming LLM responses are added.

### Debate 3: Replace `_sanitize_text()`?

**Agent 3 argues**: Current sanitization catches zero real attacks (strips null bytes and control chars, but no prompt injection uses those). Gives false sense of security.

**Resolution: RENAME + add structural defenses.** Rename to `_normalize_text()` to reflect what it actually does. Add real defenses: XML delimiters around user input in prompts, LLM output validation against constraints. Skip regex-based injection detection (bypassable, causes false positives).

### Debate 6: Should blocked topics appear in system prompts?

**Agent 3 argues (second pass)**: Enumerating `must_avoid` topics in the system prompt (`"CRITICAL: Do NOT mention or discuss: {topics}"`) gives attackers the complete blocklist. It's "posting a sign that says don't look in the third drawer."

**Resolution: MOVE enforcement to output validation.** The system prompt should describe what the persona IS, not what it must avoid. Blocked topics should be enforced programmatically via post-generation output scanning (Phase 2.2), not via LLM instruction. This is a stronger defense and doesn't leak the blocklist.

### Debate 7: Delete the `Conversation` class?

**Agent 4 argues (second pass)**: `Conversation` is a stateless wrapper over `PersonaEngine.chat()`. Two `Conversation` objects on the same engine are aliases, not independent conversations. It adds cognitive overhead ("do I use `chat()` or `say()`?") without adding capability.

**Resolution: KEEP but redesign later.** Don't delete now — it's in the public API and existing users may depend on it. But in Phase 3, either give it real value (conversation branching/forking, isolated state) or deprecate it. For now, document that it's a thin wrapper.

### Debate 8: Make TraceContext/citations opt-in?

**Agent 2 argues (second pass)**: Citation generation creates 15-25+ objects per turn via f-strings, estimated 30-40% of IR generation overhead. Should be opt-in with a `trace_enabled: bool = False` flag.

**Resolution: ADD opt-in flag in Phase 1.5 (EngineConfig).** Add `trace_enabled: bool = True` (default on for backward compat). When off, replace TraceContext with a no-op stub. This is a clean optimization that doesn't change behavior for existing users.

### Debate 4: Reduce 35+ IR enums to ~15?

**Agent 4 argues**: Most distinctions meaningless to LLMs. "Building a 256-color palette for a 16-color printer."

**Resolution: KEEP the 35+ enums.** The IR granularity IS the product's core IP — it enables deterministic behavioral testing without LLMs. Instead, add a `SimplifiedIR` / `PromptDirectives` view that collapses related enums for prompt rendering. Precision in planning, practicality in generation.

### Debate 5: TraceContext vs logging?

**Agent 5 argues**: Structured behavioral traces > grep-able text for AI debugging.

**Resolution: BOTH.** Add standard Python logging first (table stakes, takes a day). Then extend TraceContext to full pipeline. They serve different purposes: logging for operational issues ("LLM call failed"), TraceContext for behavioral debugging ("why did the persona talk about cooking?").

---

## Phase 1: Foundation

> Clean up architectural debt that blocks everything else.

| # | Task | Files | Size | Notes |
|---|------|-------|------|-------|
| 1.1 | **Consolidate `response/` into `generation/`** | `persona_engine/generation/*`, `persona_engine/response/*`, all imports | L | Audit both modules, merge unique features (notably `GeneratedResponse` schema from `response/schema.py`), delete `response/`, run full test suite. **Strip `prompt_system`/`prompt_user` fields from response objects** or add `.to_safe_dict()` to prevent prompt leakage. |
| 1.2 | **Add `__all__`, `__version__`, export `ChatResult`** | `persona_engine/__init__.py` | S | Clean public API surface. Also export `EngineConfig` (1.5). |
| 1.3 | **Rename `_sanitize_text()` → `_normalize_text()`** | `persona_engine/engine.py` | S | Honest naming, no false security sense. |
| 1.4 | **Add structured logging** | All source files under `persona_engine/` | M | Key instrumentation points: `engine.py` (turn start/end, timing, validation failures at WARNING), `turn_planner.py` (interpreter contributions), `llm_adapter.py` (API calls, latency, errors), `memory_manager.py` (store ops), `pipeline_validator.py` (validation issues), `constraint_safety.py` (violations at ERROR), `state_manager.py` (state transitions with before/after values). |
| 1.5 | **Add `EngineConfig` dataclass** | New: `persona_engine/config.py`, modify `engine.py` | M | Centralize all hardcoded constants, replace scattered constructor params, add `debug` flag, add `trace_enabled: bool = True` for opt-in citation generation. Expose in public API so developers can tune behavioral parameters. |
| 1.6 | **Fix LLM adapter error handling** | `persona_engine/generation/llm_adapter.py` | M | Wrap API calls in try/except, catch provider-specific errors, re-raise as `LLMConnectionError`/`LLMResponseError`. Add defensive `if not message.content:` check. The custom exceptions exist in `exceptions.py` but nothing uses them. |
| 1.7 | **Fix `reset()` to not duplicate `__init__`** | `persona_engine/engine.py` | S | Extract component wiring into `_init_components()` called by both `__init__` and `reset()`. Prevents silent state bugs when components are added. |
| 1.8 | **Add `serialize()`/`deserialize()` to memory stores** | `persona_engine/memory/*.py`, `persona_engine/engine.py` | M | `save()`/`load()` currently reaches into private attributes (`_preferences`, `_events`, `_base_trust`). Each store should own its serialization. Validate loaded data against Pydantic models, clamp values to valid ranges. |

**Exit Criteria**: Single generation layer, logging visible, config centralized, clean public API, LLM errors handled, persistence decoupled.

---

## Phase 2: Hardening

> Security, reliability, and performance for production use.

| # | Task | Files | Size | Depends On |
|---|------|-------|------|------------|
| 2.1 | **Prompt injection defenses** | `persona_engine/generation/prompt_builder.py` | M | Phase 1.1 |
| | Wrap user input in `<user_message>` delimiters, escape markdown headers, add system-level injection warning in preamble. **Remove blocked topics from system prompt** — enforce via output validation instead (Debate 6). | | | |
| 2.2 | **LLM output validation** | New: `persona_engine/validation/output_validator.py`, modify `engine.py` | M | Phase 1.1 |
| | Check LLM response against `cannot_claim` and `must_avoid` constraints. Log violations, optionally regenerate. This is the PRIMARY enforcement mechanism for blocked topics (replaces system prompt enumeration). | | | |
| 2.3 | **Bound ALL memory stores** | `persona_engine/memory/*.py` | M | — |
| | **FactStore**: Already capped at 500, but fix O(n) `min()` + `list.remove()` eviction — use heap or deque. **PreferenceStore**: Cap observation lists per key (`max_observations_per_key=20`) — currently unbounded and the most serious memory leak. **StanceCache**: Add `max_capacity=200` and proactive expiry sweep (currently lazy-only on read). **ChatResult history** in `engine.py`: Add `max_history` sliding window (currently holds every full IR forever). | | | |
| 2.4 | **Bound CrossTurnValidator + StyleDriftDetector** | `persona_engine/validation/cross_turn.py`, `style_drift.py` | S | — |
| | Use `deque(maxlen=30)` for turn history, `deque(maxlen=50)` for style history. Use `deque` for RelationshipStore events too (`list.pop(0)` is O(n)). | | | |
| 2.5 | **Mask API keys in errors** | `persona_engine/generation/llm_adapter.py` | S | Phase 1.1 |
| | Add `__repr__` that masks keys, ensure exceptions don't include adapter repr. | | | |
| 2.6 | **Sanitize memory store content for prompts** | `persona_engine/memory/fact_store.py` | S | — |
| | Facts fed back into prompts can carry injection payloads from earlier turns. Sanitize on retrieval for prompt inclusion. | | | |
| 2.7 | **CLI tests** | New: `tests/test_cli.py` | M | Phase 1.1 |
| | Test argparse, `--list-personas`, error paths (missing file, missing API key). All 5 subcommands (validate, info, plan, chat, list) have zero coverage. | | | |
| 2.8 | **Domain detection tests** | New/extend: `tests/test_domain_detection.py` | M | — |
| | Keyword edge cases, persona boost logic, threshold boundaries. Currently at 14% coverage. | | | |
| 2.9 | **Wire StyleDriftDetector into pipeline** | `persona_engine/validation/pipeline_validator.py` | S | — |
| | The detector is built and well-designed but orphaned — nothing calls `record_turn()` or `analyze()`. Connect it to `PipelineValidator`. | | | |
| 2.10 | **Eliminate double fact scan in `chat()`** | `persona_engine/engine.py` | S | — |
| | `get_context_for_turn()` runs once in TurnPlanner and again in `engine.py` for response generation. Cache or return the memory context from `generate_ir()`. | | | |
| 2.11 | **Create shared `conftest.py`** | New: `tests/conftest.py` | S | — |
| | 6 test files use `sys.path.insert(0, '.')` hacks. Create shared fixtures for `MINIMAL_PERSONA_DATA`, `persona`, `engine`, `yaml_path`. Eliminates duplicated fixture definitions. | | | |

**Exit Criteria**: Prompt injection defenses in place, all memory stores bounded, API keys masked, critical coverage gaps filled, orphaned code wired in.

---

## Phase 3: Developer Experience

> Make the system pleasant to use and extend.

| # | Task | Files | Size | Depends On |
|---|------|-------|------|------------|
| 3.1 | **Replace untyped dicts with dataclasses in TurnPlanner** | `persona_engine/planner/turn_planner.py` | M | — |
| | Create `FoundationResult`, `MetricsResult`, `KnowledgeResult` dataclasses to replace `dict[str, Any]` inter-stage returns. Gives IDE autocomplete, type checking catches key mismatches. Also extract cross-turn inertia smoothing (copy-pasted 6 times, ~90 lines) into a single `_apply_inertia()` method. | | | |
| 3.2 | **Extract `BehavioralInterpreter` Protocol** | New: `persona_engine/behavioral/protocol.py`, modify all interpreters + `turn_planner.py` | M | 3.1 |
| | Define common `interpret()` interface. Refactor TurnPlanner (currently 1,626 lines, imports 11 modules) to use pipeline pattern. Enables easy addition of new interpreters. Cache immutable persona domain dicts on `__init__` instead of rebuilding per turn. | | | |
| 3.3 | **Debug/inspection API** | `persona_engine/engine.py` | S | Phase 1.5 |
| | `engine.state` property (mood, stress, memory sizes, drift score), `engine.last_trace` property | | | |
| 3.4 | **Extend TraceContext to full pipeline** | `persona_engine/planner/trace_context.py`, `engine.py` | M | Phase 1.4 |
| | Cover memory retrieval, LLM prompt construction + response, validation checks. Add `trace_id` (UUID) for correlation. Add `to_dict()`/`to_json()` for export. | | | |
| 3.5 | **Improve CLI** | `persona_engine/__main__.py` | M | Phase 1.1, 2.7 |
| | Add `--debug` (show IR), `--plan-only`, `--trace` (dump JSON), `--interactive` REPL mode for multi-turn conversations, better help text | | | |
| 3.6 | **`SimplifiedIR` / `PromptDirectives` view** | `persona_engine/schema/ir_schema.py`, `persona_engine/generation/prompt_builder.py` | M | Phase 1.1 |
| | Collapse related enums for prompt rendering while keeping full IR for deterministic logic | | | |
| 3.7 | **Behavioral regression/snapshot tests** | New: `tests/test_regression.py`, `tests/snapshots/` | M | Phase 1.1 |
| | Generate IR for reference personas with standard inputs, store as JSON, detect unintended changes | | | |

**Exit Criteria**: TurnPlanner decoupled and typed, debug mode available, full pipeline tracing, behavioral regression testing.

---

## Phase 4: Operations

> Production readiness, monitoring, and continuous integration.

| # | Task | Files | Size | Depends On |
|---|------|-------|------|------------|
| 4.1 | **CI/CD pipeline** | New: `.github/workflows/ci.yml` | M | — |
| | Tests on PR (Python 3.11, 3.12), ruff + black + mypy checks, coverage reporting | | | |
| 4.2 | **Makefile** | New: `Makefile` | S | — |
| | `make test`, `make lint`, `make typecheck`, `make format`, `make ci` | | | |
| 4.3 | **Turn metrics emission** | New: `persona_engine/metrics.py`, modify `engine.py` | M | Phase 1.4, 1.5 |
| | `TurnMetrics` dataclass, optional callback/handler registration | | | |
| 4.4 | **Conversation recording** | New: `persona_engine/recording.py` | M | Phase 3.3 |
| | Record full turns (input, IR, response, trace), save/load JSON lines, post-mortem debugging | | | |
| 4.5 | **Health check API** | `persona_engine/engine.py` | S | Phase 3.2 |
| | `engine.health_check()` → `HealthReport` (memory utilization, drift score, contradiction count) | | | |
| 4.6 | **State file integrity** | `persona_engine/engine.py` | S | — |
| | Optional HMAC signature on saved state, integrity verification on load | | | |
| 4.7 | **Adversarial input test suite** | New: `tests/test_adversarial.py` | M | Phase 2.1 |
| | Prompt injection attempts, unicode exploits, memory poisoning attempts | | | |

**Exit Criteria**: CI/CD running, metrics available, conversation recording enabled, adversarial testing in place.

---

## What to Skip

| Recommendation | Why Skip |
|----------------|----------|
| Async support | LLM is 99% of latency. Use `asyncio.to_thread()` at app layer. Document "not thread-safe." |
| Aho-Corasick for domain detection | 300 substring searches is sub-millisecond. Premature optimization. |
| Inverted index for FactStore | Capacity limits solve the real problem. Over-engineering for ≤500 facts. |
| Rate limiting | Application-layer concern. Library shouldn't impose limits. |
| Persona field sanitization | Low risk if personas are developer-controlled. |
| Mutation testing | Snapshot tests provide similar regression value with less overhead. |
| Concurrency/thread safety tests | Document "not thread-safe" instead. |
| Reducing 35+ enums to 15 | The enums ARE the product's core IP. Add SimplifiedIR view instead. |
| Regex-based injection detection | Bypassable and causes false positives. Structural defenses are better. |
| Deleting `Conversation` class | It's in the public API. Redesign later (give it real value or deprecate), don't break users now. |
| Token budget tracking | Application-layer concern. Can be added via metrics callback (Phase 4.3). |

---

## Critical Path

```
Phase 1.1 (consolidate generation) ──→ Phase 2.1 (prompt injection) ──→ Phase 2.2 (output validation)
Phase 1.6 (LLM error handling)     ──→ Phase 2.5 (mask API keys)
Phase 1.8 (store serialization)    ──→ Phase 2.3 (bound all stores)
Phase 1.5 (EngineConfig)           ──→ Phase 3.3 (debug API)
Phase 3.1 (typed stage results)    ──→ Phase 3.2 (interpreter protocol)
```

Phase 4.1 (CI/CD) and 4.2 (Makefile) can start immediately alongside Phase 1.
Phase 2.4, 2.8, 2.9, 2.10, 2.11 have no dependencies and can run in parallel.

---

## Task Count Summary

| Phase | Tasks | Size Breakdown |
|-------|-------|---------------|
| Phase 1: Foundation | 8 | 1L, 3M, 4S |
| Phase 2: Hardening | 11 | 4M, 7S |
| Phase 3: DX | 7 | 5M, 2S |
| Phase 4: Operations | 7 | 3M, 4S |
| **Total** | **33** | **1L, 15M, 17S** |
