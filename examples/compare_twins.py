#!/usr/bin/env python3
"""Counterfactual Twins — Compare how different personas respond to the same input.

Loads two personas with contrasting psychology (chef vs physicist) and
shows how the IR diverges on identical questions. This is the core
value of the persona engine: same prompt, different structured behavior.
"""

from persona_engine import PersonaEngine

# Load two contrasting personas
chef = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
physicist = PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock")

twins = {"Chef (Marcus)": chef, "Physicist (Dr. Nair)": physicist}

# Same questions, different personas
questions = [
    "What do you think about AI replacing human expertise?",
    "How do you handle making mistakes at work?",
    "What's something most people get wrong about your field?",
]

for question in questions:
    print(f"Q: {question}")
    print("-" * 70)

    for label, engine in twins.items():
        ir = engine.plan(question)
        rs = ir.response_structure
        cs = ir.communication_style

        print(f"  {label}:")
        print(f"    Competence:  {rs.competence:.2f}   Confidence: {rs.confidence:.2f}")
        print(f"    Tone:        {cs.tone.value}")
        print(f"    Directness:  {cs.directness:.2f}   Formality:  {cs.formality:.2f}")
        print(f"    Verbosity:   {cs.verbosity.value}")
        print(f"    Stance:      {rs.stance or '(none)'}")
        print(f"    Claim type:  {ir.knowledge_disclosure.knowledge_claim_type.value}")

    # Show the numeric gaps between the two personas
    ir_c = chef.plan(question)
    ir_p = physicist.plan(question)
    delta_dir = abs(ir_c.communication_style.directness
                     - ir_p.communication_style.directness)
    delta_comp = abs(ir_c.response_structure.competence
                      - ir_p.response_structure.competence)
    print(f"  >> Divergence: directness={delta_dir:.2f}, competence={delta_comp:.2f}")
    print()
