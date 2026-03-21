"""
Quick test to verify persona schema works correctly
"""

import yaml
from pathlib import Path
from persona_engine.schema import Persona, IntermediateRepresentation, ConversationFrame, InteractionMode, ConversationGoal

def test_load_persona():
    """Test loading the example persona from YAML"""
    persona_path = Path("personas/ux_researcher.yaml")
    
    with open(persona_path, 'r') as f:
        persona_data = yaml.safe_load(f)
    
    # Validate with Pydantic
    persona = Persona(**persona_data)
    
    print("✓ Persona loaded successfully!")
    print(f"  ID: {persona.persona_id}")
    print(f"  Label: {persona.label}")
    print(f"  Age: {persona.identity.age}")
    print(f"  Occupation: {persona.identity.occupation}")
    print(f"\nPersonality traits:")
    print(f"  Openness: {persona.psychology.big_five.openness}")
    print(f"  Conscientiousness: {persona.psychology.big_five.conscientiousness}")
    print(f"  Extraversion: {persona.psychology.big_five.extraversion}")
    print(f"\nInvariants:")
    print(f"  Identity facts: {len(persona.invariants.identity_facts)}")
    print(f"  Cannot claim: {len(persona.invariants.cannot_claim)}")
    
    return persona


def test_create_ir():
    """Test creating an IR structure"""
    ir = IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS
        ),
        response_structure={
            "intent": "Test intent",
            "confidence": 0.75
        },
        communication_style={
            "tone": "neutral_calm",
            "verbosity": "medium",
            "formality": 0.5,
            "directness": 0.6
        },
        knowledge_disclosure={
            "disclosure_level": 0.5,
            "uncertainty_action": "answer",
            "knowledge_claim_type": "personal_experience"
        },
        citations=[]
    )
    
    print("\n✓ IR created successfully!")
    print(f"  Mode: {ir.conversation_frame.interaction_mode}")
    print(f"  Goal: {ir.conversation_frame.goal}")
    print(f"  Confidence: {ir.response_structure.confidence}")
    
    return ir


if __name__ == "__main__":
    print("Testing Persona Engine schemas...\n")
    print("=" * 60)
    
    try:
        persona = test_load_persona()
        ir = test_create_ir()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
