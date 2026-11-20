#!/usr/bin/env python3
"""Quick Start — Load a persona, chat, and inspect the IR.

Uses the "mock" LLM provider so no API keys are needed.
The mock provider returns template-based text; the interesting
part is the IR (Intermediate Representation) that the planner
produces deterministically from the persona's psychology.
"""

from persona_engine import PersonaEngine

# Load a chef persona from YAML and use the mock LLM backend
engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
print(f"Loaded: {engine.persona.label}")
print(f"Occupation: {engine.persona.identity.occupation}")
print(f"Big Five openness: {engine.persona.psychology.big_five.openness}")
print()

# Full round-trip: planning -> generation -> validation
result = engine.chat("What makes a perfect French mother sauce?")
print("=== Chat Result ===")
print(f"Response text ({len(result.text)} chars):")
print(result.text[:300])
print()

# Inspect IR fields produced by the planner
ir = result.ir
print("=== IR Snapshot ===")
print(f"Mode:       {ir.conversation_frame.interaction_mode.value}")
print(f"Goal:       {ir.conversation_frame.goal.value}")
print(f"Competence: {ir.response_structure.competence:.3f}")
print(f"Confidence: {ir.response_structure.confidence:.3f}")
print(f"Tone:       {ir.communication_style.tone.value}")
print(f"Verbosity:  {ir.communication_style.verbosity.value}")
print(f"Directness: {ir.communication_style.directness:.3f}")
print(f"Formality:  {ir.communication_style.formality:.3f}")
print()

# Validation tells us if the response is persona-consistent
print("=== Validation ===")
print(f"Passed: {result.validation.passed}")
print(f"Checked: {result.validation.checked_invariants}")
if result.validation.violations:
    for v in result.validation.violations:
        print(f"  [{v.severity}] {v.message}")

# IR-only mode (no LLM call) — useful for testing the planner
ir_only = engine.plan("Tell me about molecular gastronomy")
print(f"\nIR-only plan: {ir_only!r}")
