# Persona Engine — Improvement Plan

> Synthesized from 5 specialist agent analyses (Testing, Performance, Security, Architecture, Observability) with cross-agent debate resolution.

---

## Executive Summary

The Persona Engine has a strong foundation: a brilliant deterministic IR layer, 1,899 passing tests, and clean behavioral decomposition. However, it has critical gaps in **security** (prompt injection resistance is low), **observability** (zero logging), **architecture** (duplicate generation layers), and **operational readiness** (no CI/CD).

This plan addresses these gaps across 4 phases, prioritizing items where multiple specialist agents independently converged.

---

## Consensus Items (Multi-Agent Agreement)

| Item | Agents Agreeing | Priority |
|------|----------------|----------|
| Consolidate `generation/` and `response/` into one module | Testing, Architecture | CRITICAL |
| Add structured logging (zero exists today) | Testing, Security, Observability | CRITICAL |
| FactStore needs capacity limits + deduplication | Performance, Observability | HIGH |
| Prompt injection defenses needed | Security, Observability | HIGH |
| CLI untested (0% coverage) | Testing, Architecture | HIGH |
| CI/CD pipeline doesn't exist | Testing, Observability | HIGH |
| CrossTurnValidator/StyleDriftDetector unbounded | Performance, Observability | HIGH |
| API keys could leak in errors | Security, Architecture | MODERATE |
| Missing `__all__`, `__version__`, `ChatResult` export | Architecture, Testing | MODERATE |

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
| 1.1 | **Consolidate `response/` into `generation/`** | `persona_engine/generation/*`, `persona_engine/response/*`, all imports | L | Audit both modules, merge unique features, delete `response/`, run full test suite |
| 1.2 | **Add `__all__`, `__version__`, export `ChatResult`** | `persona_engine/__init__.py` | S | Clean public API surface |
| 1.3 | **Rename `_sanitize_text()` → `_normalize_text()`** | `persona_engine/engine.py` | S | Honest naming, no false security sense |
| 1.4 | **Add structured logging** | All source files under `persona_engine/` | M | Key instrumentation points: `engine.py` (turn start/end, timing), `turn_planner.py` (interpreter contributions), `llm_adapter.py` (API calls, latency), `memory_manager.py` (store ops), `pipeline_validator.py` (validation issues), `constraint_safety.py` (violations) |
| 1.5 | **Add `EngineConfig` dataclass** | New: `persona_engine/config.py`, modify `engine.py` | M | Centralize all hardcoded constants, replace scattered constructor params, add `debug` flag |

**Exit Criteria**: Single generation layer, logging visible, config centralized, clean public API.

---

## Phase 2: Hardening

> Security, reliability, and performance for production use.

| # | Task | Files | Size | Depends On |
|---|------|-------|------|------------|
| 2.1 | **Prompt injection defenses** | `persona_engine/generation/prompt_builder.py` | M | Phase 1.1 |
| | Wrap user input in `<user_message>` delimiters, escape markdown headers, add system-level injection warning in preamble | | | |
| 2.2 | **LLM output validation** | New: `persona_engine/validation/output_validator.py`, modify `engine.py` | M | Phase 1.1 |
| | Check LLM response against `cannot_claim` and `must_avoid` constraints. Log violations, optionally regenerate. | | | |
| 2.3 | **FactStore capacity + deduplication** | `persona_engine/memory/fact_store.py` | M | — |
| | Add `max_facts=200`, evict by lowest access_count + oldest timestamp, near-duplicate detection on insert | | | |
| 2.4 | **Bound CrossTurnValidator + StyleDriftDetector** | `persona_engine/validation/cross_turn.py`, `style_drift.py` | S | — |
| | Use `deque(maxlen=30)` for turn history, `deque(maxlen=50)` for style history | | | |
| 2.5 | **Mask API keys in errors** | `persona_engine/generation/llm_adapter.py` | S | Phase 1.1 |
| | Add `__repr__` that masks keys, ensure exceptions don't include adapter repr | | | |
| 2.6 | **Sanitize memory store content for prompts** | `persona_engine/memory/fact_store.py` | S | — |
| 2.7 | **CLI tests** | New: `tests/test_cli.py` | M | Phase 1.1 |
| | Test argparse, `--list-personas`, error paths (missing file, missing API key) | | | |
| 2.8 | **Domain detection tests** | New/extend: `tests/test_domain_detection.py` | M | — |
| | Keyword edge cases, persona boost logic, threshold boundaries | | | |

**Exit Criteria**: Prompt injection defenses in place, memory bounded, API keys masked, critical coverage gaps filled.

---

## Phase 3: Developer Experience

> Make the system pleasant to use and extend.

| # | Task | Files | Size | Depends On |
|---|------|-------|------|------------|
| 3.1 | **Extract `BehavioralInterpreter` Protocol** | New: `persona_engine/behavioral/protocol.py`, modify all interpreters + `turn_planner.py` | M | — |
| | Define common `interpret()` interface. Refactor TurnPlanner to use pipeline pattern. Enables easy addition of new interpreters. | | | |
| 3.2 | **Debug/inspection API** | `persona_engine/engine.py` | S | Phase 1.5 |
| | `engine.state` property (mood, stress, memory sizes, drift score), `engine.last_trace` property | | | |
| 3.3 | **Extend TraceContext to full pipeline** | `persona_engine/planner/trace_context.py`, `engine.py` | M | Phase 1.4 |
| | Cover memory retrieval, LLM prompt construction + response, validation checks. Add `trace_id` (UUID) for correlation. | | | |
| 3.4 | **Improve CLI** | `persona_engine/__main__.py` | M | Phase 1.1, 2.7 |
| | Add `--debug` (show IR), `--plan-only`, `--trace` (dump JSON), better help text | | | |
| 3.5 | **`SimplifiedIR` / `PromptDirectives` view** | `persona_engine/schema/ir_schema.py`, `persona_engine/generation/prompt_builder.py` | M | Phase 1.1 |
| | Collapse related enums for prompt rendering while keeping full IR for deterministic logic | | | |
| 3.6 | **Behavioral regression/snapshot tests** | New: `tests/test_regression.py`, `tests/snapshots/` | M | Phase 1.1 |
| | Generate IR for reference personas with standard inputs, store as JSON, detect unintended changes | | | |

**Exit Criteria**: TurnPlanner decoupled, debug mode available, full pipeline tracing, behavioral regression testing.

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
| Async support | LLM is 99% of latency. Use `asyncio.to_thread()` at app layer. |
| Aho-Corasick for domain detection | 300 substring searches is sub-millisecond. Premature optimization. |
| Inverted index for FactStore | Capacity limits solve the real problem. Over-engineering for <200 facts. |
| Rate limiting | Application-layer concern. Library shouldn't impose limits. |
| Persona field sanitization | Low risk if personas are developer-controlled. |
| Mutation testing | Snapshot tests provide similar regression value with less overhead. |
| Concurrency/thread safety tests | Document "not thread-safe" instead. |
| Reducing 35+ enums to 15 | The enums ARE the product's core IP. Add SimplifiedIR view instead. |
| Regex-based injection detection | Bypassable and causes false positives. Structural defenses are better. |

---

## Critical Path

```
Phase 1.1 (consolidate generation)
    → Phase 2.1 (prompt injection defenses)
    → Phase 2.2 (output validation)
```

Everything else can be parallelized within each phase. Phase 4.1 (CI/CD) and 4.2 (Makefile) can start immediately alongside Phase 1.
