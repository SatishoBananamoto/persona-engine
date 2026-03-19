# Contributing to Persona Engine

Thank you for your interest in contributing to Persona Engine. This guide covers the development workflow, code standards, and testing expectations.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/SatishoBananamoto/persona-engine.git
cd persona-engine

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

1. **Create a branch** from `main` for your work
2. **Make changes** following the code standards below
3. **Run tests** to verify nothing is broken
4. **Submit a pull request** with a clear description of your changes

## Code Standards

### Type Safety

All code must pass **mypy** with strict settings:

- All functions require type annotations (`disallow_untyped_defs: true`)
- Return types must be explicit (`warn_return_any: true`)
- Use `from __future__ import annotations` for modern syntax (`X | None` instead of `Optional[X]`)

### Style

- **Line length:** 100 characters (enforced by ruff and black)
- **Target:** Python 3.11+
- **Linting rules:** E, F, I, N, W, UP (via ruff)
- **Formatting:** black with `target-version = ["py311"]`

### Pre-commit Hooks

Pre-commit runs automatically on every commit:

- **ruff** — linting and import sorting
- **mypy** — type checking
- **trailing-whitespace** — no trailing spaces
- **end-of-file-fixer** — newline at end of files
- **check-yaml / check-json** — valid config files

To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run a specific test file
pytest tests/test_engine.py

# Run with verbose output
pytest -v tests/test_conversation.py
```

### Test Expectations

- All PRs must maintain or improve test coverage (currently ~94%)
- New features require corresponding tests
- Use **hypothesis** for property-based tests where appropriate
- Tests should use `llm_provider="mock"` or `llm_provider="template"` — never call real LLM APIs in tests

### Test Structure

```
tests/
  test_engine.py              # Core engine tests
  test_conversation.py        # Conversation wrapper tests
  test_persona_builder.py     # PersonaBuilder tests
  test_server.py              # FastAPI server tests (requires fastapi)
  test_cross_turn_dynamics.py # Multi-turn behavior tests
  test_*.py                   # Module-specific tests
```

## Architecture

The codebase follows a pipeline architecture:

```
User Input → TurnPlanner → IR (Intermediate Representation) → LLM → Validation → Response
```

Key directories:

| Directory | Purpose |
|-----------|---------|
| `persona_engine/` | Core SDK |
| `persona_engine/planner/` | IR generation pipeline (5 stages) |
| `persona_engine/behavioral/` | Psychological models (traits, biases, cognition) |
| `persona_engine/generation/` | LLM adapters and prompt building |
| `persona_engine/validation/` | IR and response validation |
| `persona_engine/memory/` | Conversation memory (facts, preferences, episodes) |
| `layer_zero/` | Persona minting from descriptions/demographics |
| `personas/` | YAML persona library |

### Making Changes to the Planner

The planner pipeline has 5 stages under `persona_engine/planner/stages/`:

1. **foundation.py** — trace setup, memory context
2. **interpretation.py** — domain detection, intent analysis
3. **behavioral.py** — metrics orchestration (delegates to mixins)
4. **knowledge.py** — disclosure, safety, claim types
5. **finalization.py** — IR assembly, memory writes

Behavioral computation is split into three mixin files:
- `behavioral_metrics.py` — elasticity, confidence, competence
- `behavioral_style.py` — tone, verbosity, formality, directness
- `behavioral_guidance.py` — trait guidance, cognitive guidance, stance

### Making Changes to Layer Zero

Layer Zero is a standalone package at the repo root. Changes to the minting pipeline should:

1. Preserve provenance tracking (every derived field needs a `FieldProvenance`)
2. Pass the 11 validation rules in `layer_zero/validator.py`
3. Include tests under `tests/` (e.g., `test_sampler.py`, `test_gap_filler.py`)

## Submitting a Pull Request

1. Ensure all tests pass: `pytest`
2. Ensure type checking passes: `mypy persona_engine`
3. Ensure linting passes: `ruff check .`
4. Write a clear PR description explaining **what** and **why**
5. Reference any related issues

## Reporting Issues

File issues at https://github.com/SatishoBananamoto/persona-engine/issues with:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant persona YAML (if applicable)
