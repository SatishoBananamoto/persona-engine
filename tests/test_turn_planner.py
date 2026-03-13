"""
End-to-end test of Turn Planner

Verifies that all components work together to generate a complete IR.
"""

import yaml
from pathlib import Path

from persona_engine.schema import Persona, IntermediateRepresentation
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal
from persona_engine.planner import TurnPlanner, ConversationContext, create_turn_planner
from persona_engine.memory import StanceCache
from persona_engine.utils import DeterminismManager


def test_turn_planner_end_to_end():
    """Test Turn Planner generates complete IR"""
    
    # Load persona
    persona_path = Path("personas/ux_researcher.yaml")
    with open(persona_path, 'r') as f:
        persona_data = yaml.safe_load(f)
    
    persona = Persona(**persona_data)
    print(f"✓ Loaded persona: {persona.label}")
    
    # Create planner with deterministic seed
    determinism = DeterminismManager(seed=42)
    planner = create_turn_planner(persona, determinism)
    print("✓ Created turn planner")
    
    # Create conversation context
    context = ConversationContext(
        conversation_id="test_conv_001",
        turn_number=1,
        interaction_mode=InteractionMode.INTERVIEW,
        goal=ConversationGoal.GATHER_INFO,
        topic_signature="user_research_methods",
        user_input="What's your approach to conducting user interviews?",
        stance_cache=StanceCache(),
        domain="UX research"
    )
    print(f"✓ Created context: {context.topic_signature}")
    
    # Generate IR
    ir = planner.generate_ir(context)
    print("\n" + "="*60)
    print("GENERATED IR:")
    print("="*60)
    
    # Display IR fields
    print(f"\n📋 Conversation Frame:")
    print(f"   Mode: {ir.conversation_frame.interaction_mode.value}")
    print(f"   Goal: {ir.conversation_frame.goal.value}")
    
    print(f"\n💭 Response Structure:")
    print(f"   Intent: {ir.response_structure.intent}")
    print(f"   Stance: {ir.response_structure.stance}")
    print(f"   Rationale: {ir.response_structure.rationale}")
    print(f"   Elasticity: {ir.response_structure.elasticity:.2f}")
    print(f"   Confidence: {ir.response_structure.confidence:.2f}")
    
    print(f"\n🎨 Communication Style:")
    print(f"   Tone: {ir.communication_style.tone.value}")
    print(f"   Verbosity: {ir.communication_style.verbosity.value}")
    print(f"   Formality: {ir.communication_style.formality:.2f}")
    print(f"   Directness: {ir.communication_style.directness:.2f}")
    
    print(f"\n🔍 Knowledge & Disclosure:")
    print(f"   Disclosure Level: {ir.knowledge_disclosure.disclosure_level:.2f}")
    print(f"   Uncertainty Action: {ir.knowledge_disclosure.uncertainty_action.value}")
    print(f"   Claim Type: {ir.knowledge_disclosure.knowledge_claim_type.value}")
    
    print(f"\n📌 Citations ({len(ir.citations)}):")
    for i, citation in enumerate(ir.citations[:10], 1):  # Show first 10
        print(f"   {i}. [{citation.source_type}] {citation.source_id}: {citation.effect}")
    
    if len(ir.citations) > 10:
        print(f"   ... and {len(ir.citations) - 10} more citations")
    
    print("\n" + "="*60)
    print("✓ IR generation successful!")
    print("="*60)
    
    # Verify IR structure
    assert ir.response_structure.stance is not None
    assert ir.response_structure.elasticity > 0
    assert ir.response_structure.confidence > 0
    assert 0 <= ir.communication_style.formality <= 1
    assert 0 <= ir.communication_style.directness <= 1
    assert 0 <= ir.knowledge_disclosure.disclosure_level <= 1
    assert len(ir.citations) > 0
    
    print("\n✓ All assertions passed")
    
    # Test determinism
    print("\n🔬 Testing determinism...")
    planner2 = create_turn_planner(persona, DeterminismManager(seed=42))
    ir2 = planner2.generate_ir(context)
    
    # Should be identical
    assert ir.response_structure.elasticity == ir2.response_structure.elasticity
    assert ir.response_structure.confidence == ir2.response_structure.confidence
    assert ir.communication_style.tone == ir2.communication_style.tone
    assert ir.communication_style.formality == ir2.communication_style.formality
    
    print("✓ Determinism verified: same seed → identical IR")


if __name__ == "__main__":
    test_turn_planner_end_to_end()
    print("\n🎉 Turn Planner end-to-end test PASSED!")
