# persona-engine

**pytest for AI personality.** Define personas with psychological depth, test their behavior deterministically, generate responses only when you're ready.

Built for developers who need synthetic humans that are consistent, testable, and grounded in real personality science -- not just prompt engineering.

## Why persona-engine?

- **Testable behavior, no LLM required.** The `plan()` method generates a full behavioral plan (IR) with zero API calls. Assert on tone, confidence, competence, and disclosure decisions the same way you'd assert on function return values.
- **Psychological grounding, not vibes.** Personas are defined by Big Five traits, Schwartz values, and cognitive style parameters -- not adjective lists. A cautious lawyer and a brash chef don't just sound different; they reason differently.
- **Deterministic and reproducible.** Seeded randomness means the same persona + input produces the same IR every time. Golden tests work. CI works. No flaky personality.
- **Prompt engineering breaks at scale.** When you need 10 distinct personas that stay consistent across 50-turn conversations, you need structure. persona-engine gives you typed memory, cross-turn validation, and delta-based citations that trace every behavioral decision back to its source trait.

## Quick Start

Two personas, same question, different behavior -- no API keys needed:

```python
from persona_engine import PersonaEngine

chef = PersonaEngine.from_description(
    "A 41-year-old Chicago chef named Marcus, blunt and opinionated",
    llm_provider="template",
)
lawyer = PersonaEngine.from_description(
    "A 44-year-old corporate lawyer named Catherine, precise and cautious",
    llm_provider="template",
)

r1 = chef.chat("What do you think about fusion cuisine?")
r2 = lawyer.chat("What do you think about fusion cuisine?")

print(r1.text)  # Direct, informal, strong opinion
print(r2.text)  # Measured, formal, hedged

# The IR proves they're actually different, not just randomly different
print(f"Chef  - formality: {r1.ir.communication_style.tone.value}, confidence: {r1.confidence:.2f}")
print(f"Lawyer - formality: {r2.ir.communication_style.tone.value}, confidence: {r2.confidence:.2f}")
```

## IR-Only Mode: Test Without Calling Any LLM

This is the killer feature. `plan()` produces the full Intermediate Representation -- a structured behavioral plan -- with zero API calls. You can write tests against it.

```python
engine = PersonaEngine.from_description(
    "A cautious junior analyst named Sam, methodical and reserved",
    llm_provider="template",
)

ir = engine.plan("Should we invest everything in crypto?")

# Assert on behavioral properties directly
assert ir.response_structure.competence < 0.5   # low domain expertise
assert ir.response_structure.confidence < 0.5   # uncertain on this topic
assert ir.knowledge_disclosure.uncertainty_action.value in ("hedge", "defer")
assert ir.communication_style.tone.value != "casual"

# Every decision traces back to a persona trait
for citation in ir.citations:
    print(f"{citation.field} -> {citation.delta:+.2f} because {citation.reason}")
```

No mocks. No API stubs. Real persona logic, fully deterministic.

## YAML Personas

For full control, define personas in YAML with 50+ psychological parameters:

```yaml
# personas/chef.yaml (abbreviated -- see full version in personas/)
persona_id: "P_002_CHEF"
label: "Marcus - Head Chef, Chicago"
identity:
  age: 41
  occupation: "Head Chef"
  location: "Chicago, IL"
psychology:
  big_five: { openness: 0.55, conscientiousness: 0.78, extraversion: 0.72, agreeableness: 0.35, neuroticism: 0.50 }
  values: { achievement: 0.80, conformity: 0.20, self_direction: 0.70 }
  cognitive_style: { risk_tolerance: 0.65, need_for_closure: 0.70 }
  communication: { directness: 0.85, formality: 0.30 }
```

```python
engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="template")
result = engine.chat("What makes a perfect sauce?")
```

## Builder API

Programmatic persona construction with a fluent interface:

```python
from persona_engine import PersonaBuilder, PersonaEngine

persona = (
    PersonaBuilder("Dr. Lee", "Physicist")
    .age(52)
    .location("Cambridge, MA")
    .traits("analytical", "reserved", "precise")
    .archetype("analyst")
    .build()
)

engine = PersonaEngine(persona, llm_provider="template")
ir = engine.plan("Explain quantum entanglement to me")
```

## Installation

```bash
git clone https://github.com/SatishoBananamoto/persona-engine.git
cd persona-engine
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
pytest
```

## What's Under the Hood

persona-engine is not a prompt template. It's a pipeline:

1. **Persona Schema** -- Big Five traits, Schwartz values, cognitive style, communication preferences, domain knowledge, disclosure policies, invariants. ~50 parameters per persona.
2. **Turn Planner** -- Maps persona + user input into a deterministic Intermediate Representation (IR). The IR specifies tone, verbosity, competence, confidence, disclosure level, and response structure before any text is generated.
3. **Delta-Based Citations** -- Every IR field traces back to the specific trait or value that influenced it, with signed deltas showing magnitude and direction.
4. **Response Generator** -- Renders the IR into natural language via the configured LLM backend (Anthropic, OpenAI, or the zero-cost template adapter).
5. **Validation** -- Checks coherence (do traits match behavior?), compliance (did the response respect invariants?), and cross-turn consistency (did the persona contradict itself?).
6. **Typed Memory** -- Facts, preferences, relationships, and episodic summaries stored per-conversation. Prevents persona drift over long interactions.

## Project Status

v0.2.0 -- 2034 tests passing. Actively developed.

## License

MIT
