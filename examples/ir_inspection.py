#!/usr/bin/env python3
"""IR Deep Dive — Inspect citations, safety plan, and memory ops.

The Intermediate Representation is what makes persona behavior
auditable. This script walks through every section of the IR
so you can see exactly why a persona behaves the way it does.
"""

from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
result = engine.chat("Can you share the secret recipe from your restaurant?")
ir = result.ir

# --- 1. Conversation Frame ---
print("=== Conversation Frame ===")
print(f"  Mode: {ir.conversation_frame.interaction_mode.value}")
print(f"  Goal: {ir.conversation_frame.goal.value}")
print(f"  Success criteria: {ir.conversation_frame.success_criteria}")

# --- 2. Response Structure ---
rs = ir.response_structure
print("\n=== Response Structure ===")
print(f"  Intent:     {rs.intent}")
print(f"  Stance:     {rs.stance}")
print(f"  Rationale:  {rs.rationale}")
print(f"  Competence: {rs.competence:.3f}")
print(f"  Confidence: {rs.confidence:.3f}")
print(f"  Elasticity: {rs.elasticity}")

# --- 3. Communication Style ---
cs = ir.communication_style
print("\n=== Communication Style ===")
print(f"  Tone:       {cs.tone.value}")
print(f"  Verbosity:  {cs.verbosity.value}")
print(f"  Formality:  {cs.formality:.3f}")
print(f"  Directness: {cs.directness:.3f}")

# --- 4. Knowledge & Disclosure ---
kd = ir.knowledge_disclosure
print("\n=== Knowledge & Disclosure ===")
print(f"  Disclosure level:  {kd.disclosure_level:.3f}")
print(f"  Uncertainty:       {kd.uncertainty_action.value}")
print(f"  Claim type:        {kd.knowledge_claim_type.value}")

# --- 5. Citations (the audit trail) ---
print(f"\n=== Citations ({len(ir.citations)} total) ===")
for i, c in enumerate(ir.citations[:8]):  # Show first 8
    delta_str = f" delta={c.delta:+.3f}" if c.delta is not None else ""
    print(f"  [{i}] {c.source_type}/{c.source_id} -> {c.target_field or '(general)'}"
          f"{delta_str}")
    print(f"      {c.effect}")
if len(ir.citations) > 8:
    print(f"  ... and {len(ir.citations) - 8} more citations")

# --- 6. Safety Plan ---
sp = ir.safety_plan
print("\n=== Safety Plan ===")
print(f"  Active constraints: {sp.active_constraints}")
print(f"  Blocked topics:     {sp.blocked_topics}")
print(f"  Cannot claim:       {sp.cannot_claim}")
print(f"  Must avoid:         {sp.must_avoid}")
if sp.clamped_fields:
    for field, clamps in sp.clamped_fields.items():
        for cl in clamps:
            print(f"  Clamped {field}: {cl.proposed:.3f} -> {cl.actual:.3f} ({cl.reason})")

# --- 7. Memory Operations ---
mo = ir.memory_ops
print(f"\n=== Memory Ops ===")
print(f"  Write policy: {mo.write_policy}")
print(f"  Read requests: {len(mo.read_requests)}")
for r in mo.read_requests:
    print(f"    [{r.query_type}] {r.query}")
print(f"  Write intents: {len(mo.write_intents)}")
for w in mo.write_intents:
    print(f"    [{w.content_type}] {w.content} (conf={w.confidence:.2f})")
