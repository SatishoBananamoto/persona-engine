"""
IR Debugging — Inspect IR fields and citation trails.

Demonstrates using plan() to generate IR without LLM calls,
then inspecting the structured plan for debugging.
"""

from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/ux_researcher.yaml", llm_provider="template")

# plan() generates IR without calling the LLM
ir = engine.plan("What do you think about dark patterns in UX?")

# Inspect conversation frame
print("=== Conversation Frame ===")
print(f"  Mode: {ir.conversation_frame.interaction_mode.value}")
print(f"  Goal: {ir.conversation_frame.goal.value}")

# Inspect response structure
rs = ir.response_structure
print("\n=== Response Structure ===")
print(f"  Competence: {rs.competence:.3f}")
print(f"  Confidence: {rs.confidence:.3f}")
print(f"  Stance: {rs.stance[:80]}...")

# Inspect communication style
cs = ir.communication_style
print("\n=== Communication Style ===")
print(f"  Tone: {cs.tone.value}")
print(f"  Formality: {cs.formality:.3f}")
print(f"  Verbosity: {cs.verbosity.value}")
print(f"  Directness: {cs.directness:.3f}")

# Inspect knowledge & disclosure
kd = ir.knowledge_disclosure
print("\n=== Knowledge & Disclosure ===")
print(f"  Claim type: {kd.knowledge_claim_type.value}")
print(f"  Uncertainty action: {kd.uncertainty_action.value}")
print(f"  Disclosure level: {kd.disclosure_level:.3f}")

# Inspect citations (trait/value changes)
print(f"\n=== Citations ({len(ir.citations)}) ===")
for c in ir.citations[:5]:
    print(f"  [{c.target_field}] {c.value_before} → {c.value_after} (source: {c.source_type}/{c.source_id})")

# Safety plan
sp = ir.safety_plan
print(f"\n=== Safety Plan ===")
print(f"  Active constraints: {sp.active_constraints}")
print(f"  Blocked topics: {sp.blocked_topics}")

# Repr
print(f"\n=== Repr ===")
print(repr(ir))
