# Persona Engine

**Psychologically-grounded synthetic humans for testing, research, and simulation.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-2%2C100%2B%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)]()

---

## What Is This?

Persona Engine creates AI-driven conversational personas that behave like *specific* people -- not generic chatbots with a personality prompt, but synthetic humans with defined psychology, values, expertise, biases, and memories. It does this by computing a structured **Intermediate Representation (IR)** from psychological models (Big Five, Schwartz values, cognitive style) *before* any text is generated. The IR makes persona behavior **testable, debuggable, and deterministic** -- you can assert on personality math, not generated prose.

## Key Features

- **Psychologically Grounded** -- Personality computed from Big Five traits, Schwartz values, and cognitive style models, not prompt hacks
- **Structured IR Pipeline** -- Every response is planned as a traceable data structure before text generation; inspect confidence, tone, stance, and disclosure as numbers
- **Full Citation Trail** -- Every behavioral decision carries before/after deltas tracing it to its psychological source (trait, value, bias, or constraint)
- **Deterministic & Reproducible** -- Per-turn SHA-256 seeding guarantees identical output for the same persona + input + seed
- **Multi-Backend Generation** -- Same IR renders through Template (free), Mock (testing), Anthropic Claude, or OpenAI backends
- **Typed Memory System** -- Facts, preferences, relationships, and episodic summaries with confidence decay prevent persona drift across turns
- **Safety by Design** -- Invariants, claim policies, knowledge boundaries, and must-avoid rules with hard veto power and full audit trails
- **10 Ready-Made Personas** -- Diverse persona library from UX researchers to jazz musicians, all defined in human-readable YAML

## Architecture Overview

```
                         User Input
                             |
                             v
                    +------------------+
                    |  Persona Engine   |
                    +------------------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
      +------------+  +------------+  +------------+
      |   Intent   |  |  Behavioral |  |   Memory   |
      |  Analyzer  |  | Interpreters|  |   Manager  |
      +------------+  +------------+  +------------+
              |         |  |  |  |          |
              |         |  Traits, Values   |
              |         |  Cognitive Style   |
              |         |  State, Biases    |
              |         |                   |
              +---------+-------------------+
                         |
                         v
                  +-------------+
                  | Turn Planner |   16-step canonical
                  | (IR Builder) |   modifier sequence
                  +-------------+
                         |
                         v
              +---------------------+
              |  Intermediate Rep.  |   Structured, traceable,
              |  (IR)               |   assertable data
              +---------------------+
                    |           |
                    v           v
            +------------+ +-----------+
            |  Response  | | Validator |
            | Generator  | |           |
            +------------+ +-----------+
                    |           |
                    v           v
                  Text    Violations/Pass
                         |
                         v
                    ChatResult
```

## Quick Start

### Installation

```bash
pip install -e .

# With dev tools (testing, linting)
pip install -e ".[dev]"

# With REST API server
pip install -e ".[server]"
```

### Load a persona and chat

```python
from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")

result = engine.chat("What makes a perfect sauce?")
print(result.text)
print(f"Confidence: {result.confidence:.2f}")
print(f"Validation: {'PASS' if result.passed else 'FAIL'}")
```

### Builder API -- create personas in code

```python
from persona_engine import PersonaEngine, PersonaBuilder

persona = (
    PersonaBuilder("Alice", "Data Scientist")
    .archetype("analyst")
    .trait("curious")
    .trait("methodical")
    .build()
)

engine = PersonaEngine(persona, llm_provider="mock")
result = engine.chat("How would you approach this dataset?")
```

### IR inspection -- test behavior without text generation

```python
# Plan-only mode: no LLM call, no API cost
ir = engine.plan("Tell me about molecular gastronomy")

print(ir.response_structure.confidence)    # 0.93
print(ir.response_structure.competence)    # 0.85
print(ir.communication_style.tone)         # Tone.thoughtful_engaged
print(ir.communication_style.directness)   # 0.46

# Assert on behavior, not prose
assert ir.response_structure.confidence > 0.7
assert ir.knowledge_disclosure.knowledge_claim_type == "domain_expert"
```

### Multi-turn conversations

```python
from persona_engine import PersonaEngine, Conversation

engine = PersonaEngine.from_yaml("personas/ux_researcher.yaml", llm_provider="mock")
convo = Conversation(engine)

convo.say("What do you think about AI in UX research?")
convo.say("How would you test that hypothesis?")

for turn in convo:
    print(f"Turn {turn.turn_number}: {turn.text}")

print(convo.summary())
convo.export_json("conversation.json")
```

## REST API

A FastAPI reference server is included for HTTP-based integration.

```bash
uvicorn persona_engine.server:app --reload
```

```bash
# Create a session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"persona_id": "personas/chef.yaml", "llm_provider": "template"}'

# Chat  (use the session_id from the response above)
curl -X POST http://localhost:8000/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What makes a perfect sauce?"}'

# Plan only (IR, no text generation)
curl -X POST http://localhost:8000/sessions/{session_id}/plan \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about fermentation"}'
```

Endpoints: `POST /sessions`, `POST /sessions/{id}/chat`, `POST /sessions/{id}/plan`, `GET /sessions/{id}`, `POST /sessions/{id}/reset`, `DELETE /sessions/{id}`, `GET /personas`, `GET /health`.

## Persona Library

Ten ready-made personas spanning diverse backgrounds, expertise, and personality profiles:

| Persona | Occupation | Location | Key Traits |
|---------|-----------|----------|------------|
| **Sarah** | UX Researcher | London, UK | Analytical, self-directed, high openness |
| **Marcus** | Head Chef | Chicago, IL | Direct, passionate, low agreeableness |
| **Dr. Priya Nair** | Theoretical Physicist | Mumbai, India | Deeply curious, high cognitive complexity |
| **Tomas Rivera** | Jazz Musician | New Orleans, LA | Intuitive, emotionally expressive, tradition-valuing |
| **Catherine Wei** | Corporate Lawyer | Singapore | Precise, methodical, high need for closure |
| **Jordan Ellis** | Fitness Coach | Denver, CO | Encouraging, high extraversion, inclusive |
| **Alex** | Software Engineer | Seattle, WA | Systematic, risk-averse, high conscientiousness |
| **Maya** | Social Worker | Chicago, IL | Empathetic, high agreeableness, benevolence-driven |
| **Jordan** | Entrepreneur | Austin, TX | Risk-tolerant, high self-direction, optimistic |
| **Margaret** | Retired Teacher | Portland, OR | Nurturing, tradition-valuing, experience-driven |

Create your own by writing a YAML file or using the `PersonaBuilder` API. See [docs/persona_authoring.md](docs/persona_authoring.md) for the full schema.

## Testing

```bash
# Run the full test suite (2,100+ tests)
pytest

# Run with coverage
pytest --cov=persona_engine

# Run property-based tests (Hypothesis)
pytest tests/test_property_based.py

# Run specific test modules
pytest tests/test_turn_planner.py
pytest tests/test_behavioral_coherence.py
pytest tests/test_determinism.py
```

The test suite includes unit tests, integration tests, property-based tests (Hypothesis), behavioral coherence checks, counterfactual twin comparisons, and cross-turn dynamics validation.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep dive into system design, IR structure, and modifier sequence |
| [docs/tutorial.md](docs/tutorial.md) | Step-by-step tutorial for getting started |
| [docs/sdk_guide.md](docs/sdk_guide.md) | Full SDK API reference |
| [docs/ir_reference.md](docs/ir_reference.md) | IR schema and field documentation |
| [docs/persona_authoring.md](docs/persona_authoring.md) | Guide to creating custom personas |

## Project Status

**Production-ready MVP** -- all 10 phases complete.

| Phase | Description | Status |
|-------|-------------|--------|
| 1-3 | Schema, Behavioral Core, Turn Planner | Complete |
| 4-5 | Memory System, Response Generation | Complete |
| 6-7 | Validation Layer, SDK & CLI | Complete |
| 8-9 | Persona Library, Documentation | Complete |
| 10 | CI/CD, FastAPI Server, Analysis Tools | Complete |

**Test suite**: 2,100+ tests passing, 94% coverage, 0 mypy errors.

## Contributing

Contributions are welcome. To get started:

1. Fork the repository and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Make your changes with tests
4. Ensure `pytest` passes and `mypy persona_engine/` reports no errors
5. Open a pull request with a clear description of the change

## License

MIT
