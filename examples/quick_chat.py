"""
Quick Chat — Minimal 3-turn conversation with a persona.

Demonstrates the simplest way to use PersonaEngine: load a YAML
persona and have a multi-turn chat using the mock adapter (no API key).
"""

from persona_engine import PersonaEngine

# Load persona from YAML, use mock adapter (no API key needed)
engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")

# 3-turn conversation
for message in [
    "What makes a perfect French mother sauce?",
    "Can you teach me to make a béchamel?",
    "What if I want to make it dairy-free?",
]:
    result = engine.chat(message)
    print(f"[Turn {result.turn_number}] User: {message}")
    print(f"  Response: {result.text[:120]}...")
    print(f"  Competence: {result.competence:.2f}")
    print(f"  Validation: {'PASS' if result.passed else 'FAIL'}")
    print()
