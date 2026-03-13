# Memory — Claude Session Log

## Project: persona-engine
**Repo**: SatishoBananamoto/persona-engine
**Branch**: `claude/general-session-VUY6r`
**Date started**: 2026-03-13

---

## What This Project Is

A **psychologically-grounded conversational persona engine** that creates behaviorally coherent synthetic humans. Uses Big Five traits, Schwartz values, and cognitive style models. Key flow: Persona → Turn Planner (IR) → Response Generator (LLM) → Validator.

**Stack**: Python 3.11+, Pydantic, PyYAML. Optional: Anthropic SDK, OpenAI SDK.

---

## Repo Structure (key paths)

```
persona_engine/       # Core package
  engine.py           # PersonaEngine — main orchestrator
  conversation.py     # Conversation class (Phase 7)
  persona_builder.py  # Builder API for creating personas without YAML
  schema/             # Pydantic models for persona, IR, etc.
  planner/            # Turn planner — generates IR before text
  generation/         # Response generator (renders IR → natural language)
  validation/         # Validators for consistency, knowledge bounds, trait coherence
  memory/             # Typed fact storage, episodic summaries
  behavioral/         # Behavioral coherence layer
  utils/              # Shared utilities
  __main__.py         # CLI entry point
personas/             # YAML persona definitions (chef, lawyer, musician, etc.)
tests/                # ~37 test files, extensive coverage
examples/             # Example scripts
```

---

## Implementation History (commit log)

Project has gone through **7 phases** of development:
1. Foundation & schema
2. Structural fixes (6 fixes, 27 tests)
3. Custom exceptions & input validation
4. Developer experience & SDK polish
5. Architecture & maintainability (5 fixes, 22 tests)
6. Behavioral fidelity & validation infrastructure
7. Python SDK — Conversation class, CLI tool, expanded exports

**Latest commit**: `fef33a7` — Phase 7

---

## What I've Done This Session

- [ ] Explored the repo structure, branches, commit history
- [ ] Created this memory.md file

---

## Open Questions / Things to Investigate

- No task assigned yet — waiting for user direction
- Tests: ~37 test files exist; should verify current pass rate before making changes
- 8 persona YAML files available (chef, fitness_coach, lawyer, musician, physicist, ux_researcher, + 2 test personas)

---

## Notes for Future Me

- Branch is already checked out and tracking remote
- `pyproject.toml` has full dev deps including pytest, mypy, ruff, black
- The engine supports `mock` LLM provider for testing without API keys
- `PersonaEngine.from_yaml()` and `PersonaEngine.from_description()` are the main entry points
- IR (Intermediate Representation) layer is central to the architecture — plans before generating text
- Always run `pytest` before pushing to verify nothing is broken
