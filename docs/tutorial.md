# Getting Started Tutorial

This tutorial takes you from zero to a working persona conversation in under 10 minutes.

## Step 1: Install

```bash
pip install persona-engine
```

## Step 2: Explore an Existing Persona

Let's start with the UX Researcher persona:

```python
import yaml
from persona_engine.schema.persona_schema import Persona

with open("personas/ux_researcher.yaml") as f:
    persona = Persona(**yaml.safe_load(f))

print(f"Name: {persona.label}")
print(f"Occupation: {persona.identity.occupation}")
print(f"Openness: {persona.psychology.big_five.openness}")
print(f"Domains: {[d.domain for d in persona.knowledge_domains]}")
```

## Step 3: Generate an IR

The Intermediate Representation (IR) captures how this persona *would* respond — before generating any text:

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
    conversation_id="tutorial",
    turn_number=1,
    user_input="What makes a good user interview?",
    topic_signature="ux_research",
    interaction_mode=InteractionMode.CASUAL_CHAT,
    goal=ConversationGoal.EXPLORE_IDEAS,
    stance_cache=StanceCache(),
    domain="ux_research",
)

ir = planner.generate_ir(ctx)

print(f"Confidence: {ir.response_structure.confidence:.2f}")
print(f"Tone: {ir.communication_style.tone.value}")
print(f"Verbosity: {ir.communication_style.verbosity.value}")
print(f"Disclosure: {ir.knowledge_disclosure.disclosure_level:.2f}")
print(f"Claim type: {ir.knowledge_disclosure.knowledge_claim_type.value}")
```

Expected output for the UX researcher on her home domain:
- **High confidence** (~0.8+): She's an expert in UX research
- **Warm/confident tone**: Enthusiastic about her domain
- **Domain expert claim**: She can speak authoritatively

## Step 4: Generate a Response

Convert the IR to natural language text:

```python
from persona_engine.response.generator import ResponseGenerator

gen = ResponseGenerator(persona=persona)
response = gen.generate(ir, user_input="What makes a good user interview?")

print(response.text)
print(f"Backend: {response.backend}")
```

## Step 5: Try Different Personas

Load the Chef and ask the same question:

```python
with open("personas/chef.yaml") as f:
    chef = Persona(**yaml.safe_load(f))

chef_planner = TurnPlanner(chef, DeterminismManager(seed=42))
chef_ctx = ConversationContext(
    conversation_id="tutorial_chef",
    turn_number=1,
    user_input="What makes a good user interview?",
    topic_signature="ux_research",
    interaction_mode=InteractionMode.CASUAL_CHAT,
    goal=ConversationGoal.EXPLORE_IDEAS,
    stance_cache=StanceCache(),
)
chef_ir = chef_planner.generate_ir(chef_ctx)

print(f"Chef confidence: {chef_ir.response_structure.confidence:.2f}")
# Much lower! UX research isn't the chef's domain
```

## Step 6: Add Memory

Make the persona remember facts across turns:

```python
from persona_engine.memory import MemoryManager

memory = MemoryManager()
memory.remember_fact(
    "User is a design team lead",
    category="occupation",
    confidence=0.9,
)

planner_with_mem = TurnPlanner(
    persona=persona,
    determinism=DeterminismManager(seed=42),
    memory_manager=memory,
)

ctx2 = ConversationContext(
    conversation_id="tutorial_memory",
    turn_number=2,
    user_input="How should I structure my research plan?",
    topic_signature="ux_research",
    interaction_mode=InteractionMode.CASUAL_CHAT,
    goal=ConversationGoal.EXPLORE_IDEAS,
    stance_cache=StanceCache(),
    domain="ux_research",
)

ir2 = planner_with_mem.generate_ir(ctx2)
# Confidence gets a memory boost from the stored fact
```

## Step 7: Inspect the Citation Trail

Every decision is traced:

```python
for citation in ir.citations[:10]:
    print(f"[{citation.source_type}:{citation.source_id}]")
    print(f"  Target: {citation.target_field}")
    print(f"  Effect: {citation.effect}")
    print()
```

## Next Steps

- Read the [Persona Authoring Guide](persona_authoring.md) to create your own personas
- Read the [IR Reference](ir_reference.md) for detailed field documentation
- Read the [SDK Guide](sdk_guide.md) for the full API reference
- Run `python -m pytest tests/ -v` to see the test suite
- Try the counterfactual twins in `personas/twins/` to see how traits drive behavior
