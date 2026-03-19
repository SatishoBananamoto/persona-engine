"""
Custom Persona — Build a persona from scratch with the builder API.

No YAML file needed. Uses the fluent builder to construct a fully
parameterized persona with archetype, traits, and custom settings.
"""

from persona_engine import PersonaEngine, PersonaBuilder

# Build a persona using the fluent API
persona = (
    PersonaBuilder("Dr. Sarah Chen", "Neuroscientist")
    .archetype("expert")
    .trait("passionate", "curious", "methodical")
    .time_scarcity(0.6)
    .privacy_sensitivity(0.4)
    .lookup_behavior("hedge")
    .build()
)

# Inspect the generated persona
print(f"Persona: {persona.label}")
print(f"  ID: {persona.persona_id}")
print(f"  Big Five openness: {persona.psychology.big_five.openness:.2f}")
print(f"  Domains: {[d.domain for d in persona.knowledge_domains]}")
print(f"  Claim policy: {persona.claim_policy.lookup_behavior}")
print()

# Use it in a conversation
engine = PersonaEngine(persona, llm_provider="template")
result = engine.chat("What are the latest advances in neuroplasticity?")
print(f"Response: {result.text[:150]}...")
print(f"Competence: {result.competence:.2f}")
print(repr(engine))
