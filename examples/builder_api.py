#!/usr/bin/env python3
"""Builder API — Create personas programmatically, no YAML needed.

Shows three ways to build a persona:
1. Fluent builder with chained methods
2. One-liner from natural-language description
3. Archetype shortcut
"""

from persona_engine import PersonaEngine, PersonaBuilder

# --- Method 1: Fluent Builder API ---
persona = (
    PersonaBuilder("Amara", "Data Scientist")
    .age(29)
    .location("Toronto, Canada")
    .traits("curious", "methodical", "introverted")
    .archetype("analyst")
    .build()
)
print("=== Builder API ===")
print(f"Name/Role: {persona.label}")
print(f"Big Five: O={persona.psychology.big_five.openness:.2f} "
      f"C={persona.psychology.big_five.conscientiousness:.2f} "
      f"E={persona.psychology.big_five.extraversion:.2f}")

engine = PersonaEngine(persona, llm_provider="mock")
result = engine.chat("How would you approach cleaning a messy dataset?")
print(f"Competence: {result.competence:.2f}  Confidence: {result.confidence:.2f}")
print(f"Response: {result.text[:150]}...")
print()

# --- Method 2: From natural-language description ---
engine2 = PersonaEngine.from_description(
    "A 55-year-old retired librarian named Dorothy from Portland, "
    "gentle and thoughtful, loves mystery novels",
    llm_provider="mock",
)
print("=== From Description ===")
print(f"Persona: {engine2.persona.label}")
print(f"Age: {engine2.persona.identity.age}")
ir = engine2.plan("What's a good book for a rainy afternoon?")
print(f"Competence on books: {ir.response_structure.competence:.2f}")
print(f"Tone: {ir.communication_style.tone.value}")
print()

# --- Method 3: Archetype shortcut ---
archetype_persona = PersonaBuilder.archetype_persona(
    "expert", name="Dr. Yuki", occupation="Neuroscientist"
)
print("=== Archetype Shortcut ===")
print(f"Persona: {archetype_persona.label}")
print(f"Openness: {archetype_persona.psychology.big_five.openness:.2f}")

engine3 = PersonaEngine(archetype_persona, llm_provider="mock")
ir3 = engine3.plan("What happens in the brain during sleep?")
print(f"Knowledge claim: {ir3.knowledge_disclosure.knowledge_claim_type.value}")
print(f"Confidence: {ir3.response_structure.confidence:.2f}")
