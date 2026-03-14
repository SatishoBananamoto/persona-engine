# Persona Engine — SDK Guide

## Installation

```bash
pip install persona-engine
```

For development:

```bash
git clone https://github.com/SatishoBananamoto/persona-engine.git
cd persona-engine
pip install -e ".[dev]"
```

## Quick Start (3 Lines)

```python
from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/ux_researcher.yaml")
result = engine.chat("What do you think about user testing?")
print(result.text)
```

## Core API

### PersonaEngine

The main entry point. Wraps persona loading, IR generation, and response generation.

```python
engine = PersonaEngine.from_yaml("personas/chef.yaml")

# Single turn
result = engine.chat("Tell me about Italian cooking")
print(result.text)       # Generated response
print(result.ir)         # Intermediate Representation
print(result.validation) # Validation results

# Multi-turn conversation
r1 = engine.chat("What's your specialty?")
r2 = engine.chat("How did you learn?")
r3 = engine.chat("What about French cuisine?")

# Access conversation history
print(engine.conversation.turn_count)
```

### PersonaBuilder (Programmatic Creation)

Build personas in code instead of YAML:

```python
from persona_engine import PersonaBuilder

persona = (
    PersonaBuilder("My Persona")
    .identity(age=30, occupation="Designer", location="NYC")
    .big_five(openness=0.8, conscientiousness=0.7, extraversion=0.6,
              agreeableness=0.65, neuroticism=0.3)
    .domain("Design", proficiency=0.9, subdomains=["UX", "Visual"])
    .domain("Technology", proficiency=0.6)
    .build()
)
```

### TurnPlanner (IR Generation)

For direct IR generation without response text:

```python
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.utils.determinism import DeterminismManager

planner = TurnPlanner(
    persona=persona,
    determinism=DeterminismManager(seed=42),
)

ctx = ConversationContext(
    conversation_id="my_convo",
    turn_number=1,
    user_input="What do you think about AI?",
    topic_signature="ai",
    interaction_mode=InteractionMode.CASUAL_CHAT,
    goal=ConversationGoal.EXPLORE_IDEAS,
    stance_cache=StanceCache(),
)

ir = planner.generate_ir(ctx)
print(f"Confidence: {ir.response_structure.confidence}")
print(f"Tone: {ir.communication_style.tone}")
print(f"Disclosure: {ir.knowledge_disclosure.disclosure_level}")
```

### ResponseGenerator

Convert IR to natural language:

```python
from persona_engine.response.generator import ResponseGenerator

gen = ResponseGenerator(persona=persona)
response = gen.generate(ir, user_input="What do you think about AI?")
print(response.text)
print(response.backend)  # "template", "mock", or "anthropic"
```

### Strict Mode (Deterministic Output)

Force template-based responses for testing:

```python
from persona_engine.response.schema import ResponseConfig

config = ResponseConfig(strict_mode=True)
gen = ResponseGenerator(config=config, persona=persona)
# Always produces identical output for the same IR
```

### Memory System

Personas can remember facts across turns:

```python
from persona_engine.memory import MemoryManager

memory = MemoryManager()
memory.remember_fact("User is a Python developer", category="occupation")
memory.remember_preference("Prefers detailed explanations", strength=0.8)

planner = TurnPlanner(
    persona=persona,
    memory_manager=memory,
)
# Facts now influence confidence; preferences inform style
```

## Backend Selection

| Backend | Use Case | API Key Required |
|---------|----------|-----------------|
| `template` | Development, testing, deterministic output | No |
| `mock` | Unit tests, prompt inspection | No |
| `anthropic` | Production with Claude | Yes |
| `openai` | Production with GPT | Yes |

```python
# Template (default, no API key)
gen = ResponseGenerator(persona=persona)

# Mock (for testing)
from persona_engine.response.schema import ResponseConfig, GenerationBackend
config = ResponseConfig(backend=GenerationBackend.MOCK)
gen = ResponseGenerator(config=config, persona=persona)

# Anthropic (requires ANTHROPIC_API_KEY env var)
config = ResponseConfig(backend=GenerationBackend.ANTHROPIC)
gen = ResponseGenerator(config=config, persona=persona)
```

## Error Handling

```python
from persona_engine.exceptions import (
    PersonaEngineError,      # Base
    PersonaValidationError,  # Invalid persona YAML
    LLMConnectionError,      # Network/timeout/rate limit
    LLMResponseError,        # Empty or malformed LLM response
    LLMAPIKeyError,          # Missing API key
    ConfigurationError,      # Missing package or bad config
)

try:
    result = engine.chat("Hello")
except LLMConnectionError:
    # Retry or fall back to template
    pass
except LLMResponseError:
    # Log and investigate
    pass
```

## Logging

Enable debug logging to see the IR pipeline:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or target specific modules:
logging.getLogger("persona_engine.planner").setLevel(logging.DEBUG)
logging.getLogger("persona_engine.memory").setLevel(logging.DEBUG)
```
