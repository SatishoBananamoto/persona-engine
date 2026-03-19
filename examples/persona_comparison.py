"""
Persona Comparison — Same prompt, different personas.

Shows how different personas produce different IR plans for the
same user input, demonstrating personality-driven behavior.
"""

from persona_engine import PersonaEngine

prompt = "What do you think about work-life balance?"

# Compare multiple persona YAML files
persona_files = [
    "personas/chef.yaml",
    "personas/lawyer.yaml",
    "personas/musician.yaml",
    "personas/fitness_coach.yaml",
]

for path in persona_files:
    engine = PersonaEngine.from_yaml(path, llm_provider="template")
    ir = engine.plan(prompt)
    rs = ir.response_structure
    cs = ir.communication_style

    print(f"=== {engine.persona.label} ===")
    print(f"  Competence: {rs.competence:.2f} | Confidence: {rs.confidence:.2f}")
    print(f"  Tone: {cs.tone.value} | Formality: {cs.formality:.2f}")
    print(f"  Directness: {cs.directness:.2f} | Verbosity: {cs.verbosity.value}")
    print(f"  Stance: {rs.stance[:80]}...")
    print()
