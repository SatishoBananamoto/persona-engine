# Universal Conversational Persona System

A psychologically-grounded persona engine that creates behaviorally coherent synthetic humans for testing, research, and simulation.

## Features

- **Psychologically Grounded**: Based on Big Five personality traits, Schwartz values, and cognitive style models
- **Cross-Domain Coherent**: Same persona behaves consistently across casual chat, interviews, customer support, surveys
- **Testable & Debuggable**: Intermediate Representation (IR) layer makes behavior explainable and reproducible
- **Typed Memory**: Structured fact storage prevents persona drift
- **Deterministic**: Seeded randomness enables reproducible test scenarios

## Architecture

- **Persona Engine**: Core orchestrator
- **Turn Planner**: Generates IR (structured plan) before text generation
- **Response Generator**: Renders IR to natural language (Anthropic Claude)
- **Validator**: Checks IR and text for consistency, knowledge boundaries, trait coherence
- **Memory System**: Typed facts, preferences, relationships, episodic summaries

## Quick Start

```python
from persona_engine import PersonaEngine

# Load a persona from YAML
engine = PersonaEngine.from_yaml("personas/ux_researcher.yaml", llm_provider="mock")

# Full round-trip: planning → generation → validation
result = engine.chat("What do you think about AI in UX research?")
print(result.text)

# Inspect the IR for debugging
print(result.ir.citations)  # See which traits/values influenced the response
print(result.validation.passed)  # Check if response is persona-consistent
```

### Create a persona from a description (no YAML needed)

```python
from persona_engine import PersonaEngine

engine = PersonaEngine.from_description(
    "A 45-year-old French chef named Marcus, passionate and direct",
    llm_provider="mock",
)

result = engine.chat("What makes a perfect sauce?")
print(result.text)
```

### IR-only mode (no LLM calls)

```python
# Inspect the planning layer without spending API credits
ir = engine.plan("Tell me about molecular gastronomy")
print(ir.response_structure.competence)
print(ir.conversation_frame.goal)
```

### Multi-turn conversations

```python
r1 = engine.chat("Tell me about sauces.")
r2 = engine.chat("And what about soups?")  # turn 2, memory active
print(f"Turn {r2.turn_number}: {r2.text}")
```

## Installation

```bash
pip install -e .
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run validation suite
python -m persona_engine.validation.qa_suite --personas personas/
```

## Builder API

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

## Project Status

🚧 **MVP in Development** - Phase 1: Foundation & Schema Complete

## License

MIT
