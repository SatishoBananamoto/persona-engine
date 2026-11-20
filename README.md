# Persona Engine

A prototype for building psychologically-grounded conversational personas. Personality is computed from Big Five traits, Schwartz values, and cognitive style into a structured **Intermediate Representation (IR)** *before* any text is generated. The IR is deterministic, inspectable, and assertable — you can test confidence, tone, stance, and disclosure as numbers, not as prose.

This is a hands-on prototype, not a packaged product. It is not published on PyPI.

The free **mock** and **template** backends return generic filler text. Without an LLM API key, the IR is the artefact worth inspecting.

## What it does

- **Structured IR pipeline** — every turn is planned as a traceable data structure before generation
- **Citation trail** — behavioural decisions carry before/after deltas back to a trait, value, bias, or constraint
- **Deterministic seeding** — per-turn SHA-256 seeds give the same IR for the same persona, input, and seed
- **Typed memory** — facts, preferences, relationships, and episodic summaries, with confidence decay
- **Safety checks** — invariants, claim policies, and knowledge boundaries with an audit trail
- **Multiple backends** — the same IR can be rendered through Template, Mock, Anthropic, or OpenAI
- **Persona library** — ten YAML personas, plus counterfactual twins under `personas/twins/` for trait comparisons

## Install

```bash
git clone https://github.com/SatishoBananamoto/persona-engine.git
cd persona-engine
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests and development tools:

```bash
pip install -e ".[dev]"
```

## Quick start

Load a persona and chat (no API key required):

```python
from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")

result = engine.chat("What makes a perfect sauce?")
print(result.text)
print(f"Confidence: {result.confidence:.2f}")
print(f"Validation: {'PASS' if result.passed else 'FAIL'}")
```

### Builder API

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

### Inspect the IR without generating text

```python
ir = engine.plan("Tell me about molecular gastronomy")

print(ir.response_structure.confidence)    # 0.95
print(ir.response_structure.competence)    # 1.0
print(ir.communication_style.tone)         # professional_composed
print(ir.communication_style.directness)   # 0.984...

assert ir.response_structure.confidence > 0.7
assert ir.knowledge_disclosure.knowledge_claim_type == "domain_expert"
```

Those figures are a live run of the chef persona on the prompt above. They will change for a different persona, prompt, or seed.

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

A FastAPI reference server is included for local use. It binds to `127.0.0.1` by default.

```bash
pip install -e ".[server]"
uvicorn persona_engine.server:app --host 127.0.0.1 --port 8000
```

```bash
# Create a session
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"persona_id": "chef.yaml", "llm_provider": "template"}'

# Chat (use the session_id from the response above)
curl -X POST http://127.0.0.1:8000/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What makes a perfect sauce?"}'

# Plan only (IR, no text generation)
curl -X POST http://127.0.0.1:8000/sessions/{session_id}/plan \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about fermentation"}'
```

Endpoints: `POST /sessions`, `POST /sessions/{id}/chat`, `POST /sessions/{id}/plan`, `GET /sessions/{id}`, `POST /sessions/{id}/reset`, `DELETE /sessions/{id}`, `GET /personas`, `GET /health`.

Set `PERSONA_API_KEY` if you want requests to require an `X-API-Key` header. Leave it unset for local development.

## Persona library

| Persona | Occupation | Location | Key traits |
|---------|-----------|----------|------------|
| **Sarah** | UX Researcher | London, UK | Analytical, self-directed, high openness |
| **Marcus** | Head Chef | Chicago, IL | Direct, passionate, low agreeableness |
| **Dr. Priya Nair** | Theoretical Physicist | Mumbai, India | Deeply curious, high cognitive complexity |
| **Tomás Rivera** | Jazz Musician | New Orleans, LA | Intuitive, emotionally expressive, tradition-valuing |
| **Catherine Wei** | Corporate Lawyer | Singapore | Precise, methodical, high need for closure |
| **Jordan Ellis** | Fitness Coach | Denver, CO | Encouraging, high extraversion, inclusive |
| **Alex** | Software Engineer | Seattle, WA | Systematic, risk-averse, high conscientiousness |
| **Maya** | Social Worker | Chicago, IL | Empathetic, high agreeableness, benevolence-driven |
| **Jordan** | Entrepreneur | Austin, TX | Risk-tolerant, high self-direction, optimistic |
| **Margaret** | Retired Teacher | Portland, OR | Nurturing, tradition-valuing, experience-driven |

Create your own with a YAML file or `PersonaBuilder`. See [docs/persona_authoring.md](docs/persona_authoring.md).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite uses the mock and template backends and does not need API keys.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, IR structure, and planner sequence |
| [docs/tutorial.md](docs/tutorial.md) | Step-by-step getting started guide |
| [docs/sdk_guide.md](docs/sdk_guide.md) | SDK API reference |
| [docs/ir_reference.md](docs/ir_reference.md) | IR schema and field documentation |
| [docs/persona_authoring.md](docs/persona_authoring.md) | Guide to creating custom personas |

## License

MIT
