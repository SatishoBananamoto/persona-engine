"""
End-to-End Demo: Persona Engine Turn Planner

This demonstrates the current production-ready pipeline:
1. Load a persona from YAML
2. Generate an IR for a user message
3. Inspect the full citation trail and behavioral outputs

Run: python demo_turn_planner.py
"""

import sys
import json
sys.path.insert(0, '.')

from persona_engine.schema.persona_schema import Persona
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal


def load_persona(yaml_path: str) -> Persona:
    """Load persona from YAML file"""
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return Persona(**data)


def demo():
    print("=" * 70)
    print("PERSONA ENGINE - TURN PLANNER DEMO")
    print("=" * 70)
    
    # 1. Load persona
    print("\n[1] Loading persona from YAML...")
    persona = load_persona("personas/ux_researcher.yaml")
    print(f"    [OK] Loaded: {persona.persona_id}")
    print(f"    [OK] Occupation: {persona.identity.occupation}")
    print(f"    [OK] Big 5: O={persona.psychology.big_five.openness:.2f}, "
          f"C={persona.psychology.big_five.conscientiousness:.2f}, "
          f"E={persona.psychology.big_five.extraversion:.2f}, "
          f"A={persona.psychology.big_five.agreeableness:.2f}, "
          f"N={persona.psychology.big_five.neuroticism:.2f}")
    
    # 2. Initialize Turn Planner
    print("\n[2] Initializing Turn Planner...")
    planner = TurnPlanner(persona)
    print("    [OK] Trait Interpreter initialized")
    print("    [OK] Values Interpreter initialized") 
    print("    [OK] Cognitive Interpreter initialized")
    print("    [OK] State Manager initialized")
    print("    [OK] Bias Simulator initialized")
    
    # 3. Create conversation context
    print("\n[3] Creating conversation context...")
    context = ConversationContext(
        conversation_id="demo_001",
        turn_number=1,
        interaction_mode=InteractionMode.INTERVIEW,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="ai_in_ux_research",
        user_input="What do you think about using AI tools in UX research? Research shows they can improve efficiency.",
        stance_cache=StanceCache(),
        domain=None
    )
    print(f"    [OK] Mode: {context.interaction_mode.value}")
    print(f"    [OK] Goal: {context.goal.value}")
    print(f"    [OK] Input: \"{context.user_input[:50]}...\"")
    
    # 4. Generate IR
    print("\n[4] Generating Intermediate Representation...")
    ir = planner.generate_ir(context)
    print("    [OK] IR generated successfully!")
    
    # 5. Display results using model_dump for safe access
    ir_dict = ir.model_dump()
    
    print("\n" + "=" * 70)
    print("GENERATED IR")
    print("=" * 70)
    
    print("\n--- Conversation Frame ---")
    cf = ir_dict.get("conversation_frame", {})
    print(f"  Mode: {cf.get('interaction_mode', 'N/A')}")
    print(f"  Goal: {cf.get('goal', 'N/A')}")
    
    print("\n--- Response Structure ---")
    rs = ir_dict.get("response_structure", {})
    print(f"  Stance: {rs.get('stance', 'N/A')}")
    rationale = rs.get('rationale', 'N/A')
    print(f"  Rationale: {rationale[:80] if rationale else 'N/A'}...")
    print(f"  Elasticity: {rs.get('elasticity', 'N/A')}")
    print(f"  Confidence: {rs.get('confidence', 'N/A')}")
    print(f"  Intent: {rs.get('intent', 'N/A')}")
    
    print("\n--- Communication Style ---")
    cs = ir_dict.get("communication_style", {})
    print(f"  Tone: {cs.get('tone', 'N/A')}")
    print(f"  Verbosity: {cs.get('verbosity', 'N/A')}")
    print(f"  Formality: {cs.get('formality', 'N/A')}")
    print(f"  Directness: {cs.get('directness', 'N/A')}")
    
    print("\n--- Knowledge & Disclosure ---")
    kd = ir_dict.get("knowledge_disclosure", {})
    print(f"  Claim Type: {kd.get('knowledge_claim_type', 'N/A')}")
    print(f"  Uncertainty Action: {kd.get('uncertainty_action', 'N/A')}")
    print(f"  Disclosure Level: {kd.get('disclosure_level', 'N/A')}")
    
    # 6. Show citations
    citations = ir_dict.get("citations", [])
    print("\n" + "=" * 70)
    print(f"CITATIONS ({len(citations)} total)")
    print("=" * 70)
    
    for i, cite in enumerate(citations[:15], 1):  # Show first 15
        source = f"[{cite.get('source_type', '?')}:{cite.get('source_id', '?')}]"
        effect = cite.get('effect', '')[:50]
        print(f"  {i:2}. {source:35} -> {effect}...")
    
    if len(citations) > 15:
        print(f"  ... and {len(citations) - 15} more citations")
    
    # 7. Check for bias effects
    print("\n" + "=" * 70)
    print("BIAS EFFECTS")
    print("=" * 70)
    
    bias_citations = [c for c in citations if 'bias' in c.get('source_id', '').lower()]
    if bias_citations:
        for cite in bias_citations:
            print(f"  [OK] {cite.get('source_id')}: {cite.get('effect')}")
    else:
        print("  (No biases triggered for this input)")
    
    # 8. Safety plan
    print("\n" + "=" * 70)
    print("SAFETY PLAN")
    print("=" * 70)
    safety = ir_dict.get("safety_plan", {})
    clamps = safety.get("clamps", {}) if safety else {}
    if clamps:
        for field, records in clamps.items():
            for rec in records:
                print(f"  • {field}: clamped to [{rec.get('min_applied')}, {rec.get('max_applied')}]")
    else:
        print("  (No clamps applied)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe Turn Planner is production-ready for IR generation.")
    print("To get natural language output, Phase 5 (Response Generator) is needed.")
    
    # 9. Optional: dump full IR as JSON
    print("\n--- Full IR JSON (truncated) ---")
    full_json = json.dumps(ir_dict, indent=2, default=str)
    if len(full_json) > 2000:
        print(full_json[:2000] + "\n... (truncated)")
    else:
        print(full_json)
    
    return ir


if __name__ == "__main__":
    demo()
