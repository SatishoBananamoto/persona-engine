# CLAUDE.md - Persona Engine

## Project Overview

Persona Engine is a psychologically-grounded synthetic human simulator. It generates structured Intermediate Representations (IRs) that describe how a persona would behave in a conversation, based on Big Five personality traits, Schwartz values, cognitive styles, and dynamic state. The IR is deterministic and fully traceable -- every decision carries a citation back to its source.

**Status:** Alpha (v0.1.0). Phases 1-3 complete (Schema, Behavioral Interpreters, Turn Planner). Phases 4-6 planned (Memory System, LLM Response Generator, Validation Suite).

## Quick Reference

```bash
# Install
pip install -e ".[dev]"

# Run tests (must use -s flag due to stdout reassignment in test files)
python -m pytest -v -s --override-ini="addopts="

# Lint
ruff check persona_engine/
black --check persona_engine/

# Type check
mypy persona_engine/
```

## Architecture

### Core Pipeline

Every turn follows the **canonical modifier composition sequence**:

```
base value -> role adjustment -> trait modifiers -> state modifiers -> bias modifiers -> clamp
```

This ordering is a hard invariant. It prevents double-counting and ensures clear citation trails.

### Module Map

```
persona_engine/
  schema/
    persona_schema.py    # Pydantic v2 models for persona profiles
    ir_schema.py         # IR output models, enums, citation types
  behavioral/
    trait_interpreter.py       # Big Five -> behavioral parameters
    values_interpreter.py      # Schwartz values conflict resolution
    cognitive_interpreter.py   # Decision style mapping
    state_manager.py           # Dynamic mood/fatigue/stress/engagement
    rules_engine.py            # Social role adjustments, decision policies
    bias_simulator.py          # Bounded cognitive biases (max +/-0.15)
    uncertainty_resolver.py    # Uncertainty handling (single source of truth)
    constraint_safety.py       # Hard safety vetoes
  planner/
    turn_planner.py      # Central orchestrator (main entry point)
    trace_context.py     # Citation tracking (ctx.num/ctx.enum/ctx.clamp)
    intent_analyzer.py   # User intent detection
    stance_generator.py  # Stance & rationale generation
    domain_detection.py  # Topic/domain keyword scoring
  memory/
    stance_cache.py      # Per-conversation stance consistency (~10 turn decay)
  utils/
    determinism.py       # Seeded RNG (DeterminismManager)
```

### Entry Point

```python
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.schema.persona_schema import Persona
from persona_engine.memory.stance_cache import StanceCache
import yaml

persona = Persona(**yaml.safe_load(open("personas/ux_researcher.yaml")))
planner = TurnPlanner(persona)
context = ConversationContext(
    conversation_id="conv_001", turn_number=1,
    interaction_mode=InteractionMode.INTERVIEW,
    goal=ConversationGoal.GATHER_INFO,
    topic_signature="user_research",
    user_input="Tell me about your methods...",
    stance_cache=StanceCache()
)
ir = planner.generate_ir(context)
```

## Codebase Conventions

### Python

- **Version:** 3.11+ required
- **Line length:** 100 characters (black + ruff)
- **Type hints:** Mandatory on all function signatures (`mypy --disallow-untyped-defs`)
- **Data models:** Pydantic v2 `BaseModel` for schemas, `@dataclass` for internal structs
- **Naming:** PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants, `_leading_underscore` for private methods
- **Imports:** Sorted by ruff (isort rules via `I` selector)

### Determinism

All randomness flows through `DeterminismManager`. Never use `random` module directly. Same seed + same persona + same input = byte-identical IR output. Tests enforce this.

### Citations

Every mutation to an IR numeric field must go through `TraceContext` methods:
- `ctx.num()` for numeric modifiers (set, add, multiply, blend)
- `ctx.enum()` for enum value changes
- `ctx.clamp()` for bounds enforcement (also records in SafetyPlan)

Never write to a float IR field directly ("no naked writes"). The test suite has a killer contract that enforces this.

### Biases

Cognitive biases (confirmation, negativity, authority) are bounded to a maximum impact of +/-0.15. This is a hard invariant. Tests assert this bound.

### Safety Constraints

`constraint_safety.py` enforces hard vetoes. `must_avoid` topics are always blocked. Persona invariants (`can_never_claim`) are checked against generated stances.

## Testing

### Running Tests

```bash
# All tests (53 tests, ~0.7s)
python -m pytest -v -s --override-ini="addopts="

# Specific test file
python -m pytest test_bias_simulator.py -v -s --override-ini="addopts="

# Also runnable directly
python test_turn_planner_suite.py
```

The `--override-ini="addopts="` flag is needed because pyproject.toml sets `addopts` with coverage options, and the `testpaths = ["tests"]` directive doesn't match the actual test location (root). The `-s` flag avoids a pytest capture conflict caused by `sys.stdout` reassignment in `test_turn_planner_suite.py`.

### Test Structure

Tests live in the project root (not in a `tests/` directory):

| File | Purpose |
|------|---------|
| `test_turn_planner_suite.py` | Main regression gate: determinism, contracts, scenarios, killer contract |
| `test_bias_simulator.py` | Bias bounds and behavior |
| `test_ir_hardening.py` | IR schema edge cases |
| `test_trace_context.py` | Citation tracking correctness |
| `test_sparse_persona.py` | Handling incomplete persona definitions |
| `test_domain_fixes.py` | Domain detection accuracy |
| `test_scenarios.py` | Multi-turn conversation scenarios |
| `test_production_integration.py` | Integration tests |
| `test_schemas.py` | Schema validation |
| `test_turn_planner.py` | End-to-end planner tests |

### Test Philosophy

- All assertions are hard (`assert`), never warnings
- Determinism tests compare byte-level identity with same seed
- Table-driven scenarios use `@dataclass` TestScenario fixtures
- Persona fixtures loaded from `personas/*.yaml`

## Persona Definitions

Personas are YAML files in `personas/`. Key sections:

- `identity` -- age, gender, location, education, occupation, background
- `psychology.big_five` -- openness, conscientiousness, extraversion, agreeableness, neuroticism (0.0-1.0)
- `psychology.values` -- 10 Schwartz values (0.0-1.0)
- `psychology.cognitive_style` -- analytical_intuitive, risk_tolerance, need_for_closure, cognitive_complexity
- `psychology.communication` -- verbosity, formality, directness, emotional_expressiveness
- `knowledge_domains` -- domain expertise with proficiency scores and subdomains
- `social_roles` -- context-specific behavior adjustments (at_work, with_friends, etc.)
- `invariants` -- hard identity facts, `can_never_claim`, `must_avoid` topics
- `biases` -- cognitive bias types and strengths

## Key Design Invariants

1. **Canonical modifier sequence:** base -> role -> trait -> state -> bias -> clamp. Never reorder.
2. **No naked float writes:** All numeric IR mutations go through TraceContext.
3. **Bias bounds:** Max cognitive bias impact is +/-0.15.
4. **Single source of truth:** Each IR parameter is computed by exactly one process.
5. **Deterministic output:** Same seed = same IR, enforced by regression tests.
6. **Stance consistency:** StanceCache prevents flip-flopping within a conversation (~10 turn decay).
7. **Safety vetoes are absolute:** `must_avoid` and `can_never_claim` constraints cannot be overridden.

## Common Tasks

### Adding a New Behavioral Interpreter

1. Create `persona_engine/behavioral/new_interpreter.py`
2. Export from `persona_engine/behavioral/__init__.py`
3. Wire into `TurnPlanner.__init__()` and `generate_ir()` in `turn_planner.py`
4. Ensure all mutations use `TraceContext` methods for citation
5. Follow the canonical modifier sequence (place in correct position)
6. Add tests verifying bounds, citations, and determinism

### Adding a New Persona

1. Create `personas/your_persona.yaml` following the schema in `persona_schema.py`
2. All Big Five and Schwartz values are floats in [0.0, 1.0]
3. Test with `Persona(**yaml.safe_load(open("personas/your_persona.yaml")))` to validate

### Adding a New IR Field

1. Add to the appropriate Pydantic model in `ir_schema.py`
2. Compute in `turn_planner.py` following canonical sequence
3. Use `ctx.num()`/`ctx.enum()` for all mutations
4. Add to the killer contract in `test_turn_planner_suite.py` if it's a float field

## Dependencies

| Package | Purpose |
|---------|---------|
| pydantic >=2.5 | Schema validation and data models |
| anthropic >=0.40 | Claude API integration (future response generation) |
| pyyaml >=6.0 | Persona YAML parsing |
| python-dotenv >=1.0 | Environment variable management |
| pytest >=7.4 | Testing framework (dev) |
| black >=23.0 | Code formatting (dev) |
| ruff >=0.1 | Linting (dev) |
| mypy >=1.7 | Type checking (dev) |
