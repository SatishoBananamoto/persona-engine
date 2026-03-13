"""
Test TraceContext and production fixes:
- Multiple clamps per field (List[ClampRecord])
- Constraint name tracking
- Per-turn seed generation
- Double clamp scenario
"""

from persona_engine.planner.trace_context import TraceContext, clamp01, create_turn_seed
from persona_engine.schema.ir_schema import ClampRecord


def test_multiple_clamps_same_field():
    """Test that multiple clamps on same field are all recorded"""
    print("\n" + "="*60)
    print("TEST: Multiple Clamps on Same Field")
    print("="*60)
    
    ctx = TraceContext()
    
    # Scenario: disclosure clamped by privacy, then by topic sensitivity
    disclosure = 0.8
    
    # First clamp: privacy filter
    disclosure = ctx.clamp(
        field_name="disclosure_level",
        target_field="knowledge_disclosure.disclosure_level",
        proposed=disclosure,
        minimum=None,
        maximum=0.6,
        constraint_name="privacy_filter",
        reason="Privacy filter (0.4) limits disclosure"
    )
    print(f"After privacy clamp: {disclosure:.2f}")
    
    # Second clamp: topic sensitivity
    disclosure = ctx.clamp(
        field_name="disclosure_level",
        target_field="knowledge_disclosure.disclosure_level",
        proposed=disclosure,
        minimum=None,
        maximum=0.4,
        constraint_name="topic_sensitivity",
        reason="Topic 'mental_health' is sensitive (0.6)"
    )
    print(f"After topic clamp: {disclosure:.2f}")
    
    # Verify both clamps recorded
    clamp_records = ctx.safety_plan.clamped_fields.get("disclosure_level", [])
    print(f"\nClamp records: {len(clamp_records)}")
    
    assert len(clamp_records) == 2, f"Expected 2 clamp records, got {len(clamp_records)}"
    
    # Verify first clamp
    assert clamp_records[0].proposed == 0.8
    assert clamp_records[0].actual == 0.6
    assert "privacy_filter" in clamp_records[0].reason
    
    # Verify second clamp
    assert clamp_records[1].proposed == 0.6
    assert clamp_records[1].actual == 0.4
    assert "topic_sensitivity" in clamp_records[1].reason
    
    # Verify active constraints
    assert "privacy_filter" in ctx.safety_plan.active_constraints
    assert "topic_sensitivity" in ctx.safety_plan.active_constraints
    
    # Verify citations (2 clamp citations)
    clamp_citations = [c for c in ctx.citations if c.operation == "clamp"]
    assert len(clamp_citations) == 2
    
    print("\n✓ Multiple clamps recorded correctly")
    print(f"  Clamp 1: {clamp_records[0].proposed:.2f} → {clamp_records[0].actual:.2f} ({clamp_records[0].reason})")
    print(f"  Clamp 2: {clamp_records[1].proposed:.2f} → {clamp_records[1].actual:.2f} ({clamp_records[1].reason})")


def test_constraint_name_tracking():
    """Test that different constraints are tracked separately"""
    print("\n" + "="*60)
    print("TEST: Constraint Name Tracking")
    print("="*60)
    
    ctx = TraceContext()
    
    # Clamp with bounds_check
    value1 = clamp01(ctx, "formality", "communication_style.formality", 1.2)
    
    # Clamp with privacy
    value2 = ctx.clamp(
        field_name="disclosure",
        target_field="knowledge_disclosure.disclosure_level",
        proposed=0.9,
        minimum=None,
        maximum=0.5,
        constraint_name="privacy_filter",
        reason="Privacy settings"
    )
    
    # Verify different constraints tracked
    assert "bounds_check" in ctx.safety_plan.active_constraints
    assert "privacy_filter" in ctx.safety_plan.active_constraints
    
    # Verify citations have correct source_id
    bounds_citations = [c for c in ctx.citations if c.source_id == "bounds_check"]
    privacy_citations = [c for c in ctx.citations if c.source_id == "privacy_filter"]
    
    assert len(bounds_citations) == 1
    assert len(privacy_citations) == 1
    
    print("\n✓ Constraint names tracked separately")
    print(f"  Active constraints: {ctx.safety_plan.active_constraints}")


def test_no_clamp_no_record():
    """Test that values within bounds don't generate clamp records"""
    print("\n" + "="*60)
    print("TEST: No Clamp = No Record")
    print("="*60)
    
    ctx = TraceContext()
    
    # Value already within bounds
    value = clamp01(ctx, "directness", "communication_style.directness", 0.5)
    
    assert value == 0.5
    assert len(ctx.citations) == 0, "Should not generate citation if no clamp"
    assert len(ctx.safety_plan.clamped_fields) == 0, "Should not record if no clamp"
    assert "bounds_check" not in ctx.safety_plan.active_constraints
    
    print("\n✓ No unnecessary clamp records")


def test_per_turn_seed():
    """Test deterministic per-turn seed generation"""
    print("\n" + "="*60)
    print("TEST: Per-Turn Seed Generation")
    print("="*60)
    
    # Same inputs → same seed
    seed1 = create_turn_seed(42, "conv_123", 5)
    seed2 = create_turn_seed(42, "conv_123", 5)
    assert seed1 == seed2, "Same inputs should produce same seed"
    
    # Different turn → different seed
    seed3 = create_turn_seed(42, "conv_123", 6)
    assert seed3 != seed1, "Different turn should produce different seed"
    
    # Different conversation → different seed
    seed4 = create_turn_seed(42, "conv_456", 5)
    assert seed4 != seed1, "Different conversation should produce different seed"
    
    print(f"\n✓ Per-turn seeds deterministic")
    print(f"  Base=42, conv=conv_123, turn=5: {seed1}")
    print(f"  Base=42, conv=conv_123, turn=6: {seed3}")
    print(f"  Base=42, conv=conv_456, turn=5: {seed4}")


def test_pattern_block_recording():
    """Test pattern blocking populates safety plan"""
    print("\n" + "="*60)
    print("TEST: Pattern Block Recording")
    print("="*60)
    
    ctx = TraceContext()
    
    ctx.block_pattern("share_work_story", "mentions must_avoid 'employer_name'")
    
    assert len(ctx.safety_plan.pattern_blocks) == 1
    assert "pattern_safety" in ctx.safety_plan.active_constraints
    assert "share_work_story" in ctx.safety_plan.pattern_blocks[0]
    assert "employer_name" in ctx.safety_plan.pattern_blocks[0]
    
    print("\n✓ Pattern blocks recorded in safety plan")
    print(f"  Block: {ctx.safety_plan.pattern_blocks[0]}")


def test_base_citations():
    """Test base citation initialization"""
    print("\n" + "="*60)
    print("TEST: Base Citations")
    print("="*60)
    
    ctx = TraceContext()
    
    # Initialize with base citation
    formality = ctx.base(
        field_name="communication.formality",
        target_field="communication_style.formality",
        value=0.6,
        effect="Base formality from persona"
    )
    
    assert formality == 0.6
    assert len(ctx.citations) == 1
    
    cit = ctx.citations[0]
    assert cit.source_type == "base"
    assert cit.operation == "set"
    assert cit.value_before == 0.0
    assert cit.value_after == 0.6
    assert cit.delta == 0.6
    
    print("\n✓ Base citations initialized correctly")
    print(f"  {cit.source_type}/{cit.source_id}: {cit.value_before} → {cit.value_after} ({cit.delta:+.1f})")


if __name__ == "__main__":
    test_multiple_clamps_same_field()
    test_constraint_name_tracking()
    test_no_clamp_no_record()
    test_per_turn_seed()
    test_pattern_block_recording()
    test_base_citations()
    
    print("\n" + "="*60)
    print("🎉 ALL TRACECONTEXT TESTS PASSED!")
    print("="*60)
    print("\nProduction fixes validated:")
    print("✓ List[ClampRecord] supports multiple clamps per field")
    print("✓ Constraint names tracked separately")
    print("✓ No unnecessary records for values within bounds")
    print("✓ Per-turn seeds are deterministic")
    print("✓ Pattern blocks populate safety plan")
    print("✓ Base citations initialize cleanly")
    print()
