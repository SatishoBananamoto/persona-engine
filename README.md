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
from persona_engine import PersonaEngine, Persona

# Load a persona
engine = PersonaEngine()
persona = engine.load_persona("personas/ux_researcher.yaml")

# Start a conversation
conversation = persona.start_conversation(
    interaction_mode="casual_chat",
    seed=42  # Deterministic
)

response = conversation.send("What do you think about AI in UX research?")
print(response.text)

# Inspect the IR for debugging
print(response.ir.citations)  # See which traits/values influenced the response
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

## Project Status

🚧 **MVP in Development** - Phase 1: Foundation & Schema Complete

## License

MIT
