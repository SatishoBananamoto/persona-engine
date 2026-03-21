from persona_engine.planner.turn_planner import create_turn_planner, ConversationContext
from persona_engine.schema.persona_schema import Persona
from persona_engine.schema.ir_schema import IntermediateRepresentation, InteractionMode, ConversationGoal
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import IntermediateRepresentation


def mock_persona():
    import yaml
    from pathlib import Path
    
    # Load default minimal persona for testing
    # Assuming one exists or we create a minimal dict
    # Let's try to load valid one from examples/persona.yaml if strictly necessary
    # Or just construct one.
    # Construction is verbose. Let's try to load existing.
    
    # Fallback: Use the mock objects if possible? 
    # TurnPlanner requires a full Persona object.
    # Let's try to load a known yaml.
    
    path = Path("personas/ux_researcher.yaml")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Persona(**data)
        
    raise FileNotFoundError("examples/persona.yaml not found")

def test_citations_presence():
    persona = mock_persona()
    planner = create_turn_planner(persona)
    cache = StanceCache()
    
    # 1. Test Domain Detection Citation
    ctx = ConversationContext(
        conversation_id="test_conv",
        turn_number=1,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.GATHER_INFO,
        topic_signature="tech_topic",
        user_input="I want to discuss quantum computing technology",
        stance_cache=cache
    )
    
    ir = planner.generate_ir(ctx)
    
    # Check for domain citation
    domain_cites = [c for c in ir.citations if c.source_id == "domain_detection"]
    assert len(domain_cites) >= 1, "Missing domain_detection citation"
    print(f"✓ Found domain citation: {domain_cites[0].effect}")
    
    # Check for topic relevance citation
    topic_cites = [c for c in ir.citations if c.source_id == "topic_relevance"]
    assert len(topic_cites) >= 1, "Missing topic_relevance citation"
    print(f"✓ Found topic relevance citation: {topic_cites[0].effect}")

    # 2. Test Evidence Strength Citation (Challenge)
    ctx_challenge = ConversationContext(
        conversation_id="test_conv",
        turn_number=2,
        interaction_mode=InteractionMode.DEBATE,
        goal=ConversationGoal.RESOLVE_ISSUE,
        topic_signature="tech_topic",
        user_input="Actually I disagree, that's wrong.",
        stance_cache=cache
    )
    
    ir_challenge = planner.generate_ir(ctx_challenge)
    
    # Check for evidence strength citation
    evidence_cites = [c for c in ir_challenge.citations if c.source_id == "evidence_strength"]
    assert len(evidence_cites) >= 1, "Missing evidence_strength citation"
    print(f"✓ Found evidence strength citation: {evidence_cites[0].effect}")
    
    # Check for stress trigger citation (if strength was high enough)
    stress_cites = [c for c in ir_challenge.citations if c.source_id == "stress_trigger"]
    if "Strong challenge" in evidence_cites[0].effect:
        assert len(stress_cites) >= 1, "Expected stress trigger for strong challenge"
        print(f"✓ Found stress trigger citation: {stress_cites[0].effect}")

if __name__ == "__main__":
    test_citations_presence()
    print("\nSUCCESS: All expected citations confirmed in IR.")
