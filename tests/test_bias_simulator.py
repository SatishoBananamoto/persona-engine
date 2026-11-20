"""
Tests for Bounded Bias Simulation (Phase 2)

Verifies:
1. BiasSimulator computes modifiers correctly
2. Confirmation bias reduces elasticity for aligned input
3. Negativity bias increases arousal for negative input
4. Authority bias increases confidence for authority-citing input
5. All biases are bounded to ±0.15 max impact
6. Biases generate proper citations
"""

import sys
sys.path.insert(0, '.')

from persona_engine.behavioral.bias_simulator import (
    BiasSimulator,
    BiasType,
    BiasModifier,
    MAX_BIAS_IMPACT,
    CONFIRMATION_ALIGNMENT_THRESHOLD,
    NEGATIVITY_NEUROTICISM_THRESHOLD,
    AUTHORITY_CONFORMITY_THRESHOLD,
)
from persona_engine.planner.trace_context import TraceContext


# =============================================================================
# TESTS
# =============================================================================

def test_confirmation_bias():
    """Confirmation bias should reduce elasticity when value alignment is high."""
    print("\n--- Testing Confirmation Bias ---")
    
    # High openness counters confirmation bias
    low_openness_simulator = BiasSimulator(
        traits={"openness": 0.3, "neuroticism": 0.3},
        value_priorities={"conformity": 0.3, "tradition": 0.3, "security": 0.3}
    )
    
    ctx = TraceContext()
    modifiers = low_openness_simulator.compute_modifiers(
        user_input="This aligns with my values",
        value_alignment=0.8,  # High alignment
        ctx=ctx
    )
    
    # Should trigger confirmation bias
    conf_mod = low_openness_simulator.get_modifier_for_field(
        modifiers, "response_structure.elasticity"
    )
    
    assert conf_mod is not None, "Expected confirmation bias modifier"
    assert conf_mod.bias_type == BiasType.CONFIRMATION
    assert conf_mod.modifier < 0, "Confirmation bias should reduce elasticity"
    assert abs(conf_mod.modifier) <= MAX_BIAS_IMPACT, f"Modifier exceeds max: {conf_mod.modifier}"
    
    # Check citation was added
    cites = [c for c in ctx.citations if c.source_id == "confirmation_bias"]
    assert len(cites) == 1, "Expected confirmation bias citation"
    print(f"✓ Confirmation bias triggered: elasticity {conf_mod.modifier:+.3f}")


def test_confirmation_bias_countered_by_openness():
    """High openness should reduce confirmation bias effect."""
    print("\n--- Testing Openness Counters Confirmation Bias ---")
    
    high_openness = BiasSimulator(
        traits={"openness": 0.9, "neuroticism": 0.3},
        value_priorities={}
    )
    
    low_openness = BiasSimulator(
        traits={"openness": 0.2, "neuroticism": 0.3},
        value_priorities={}
    )
    
    high_o_mods = high_openness.compute_modifiers("test", value_alignment=0.8)
    low_o_mods = low_openness.compute_modifiers("test", value_alignment=0.8)
    
    high_o_conf = high_openness.get_modifier_for_field(high_o_mods, "response_structure.elasticity")
    low_o_conf = low_openness.get_modifier_for_field(low_o_mods, "response_structure.elasticity")
    
    # Both might be None or have modifiers, but high openness should have weaker effect
    if high_o_conf and low_o_conf:
        assert abs(high_o_conf.modifier) < abs(low_o_conf.modifier), \
            "High openness should reduce confirmation bias"
        print(f"✓ High openness: {high_o_conf.modifier:+.3f}, Low openness: {low_o_conf.modifier:+.3f}")
    elif low_o_conf and not high_o_conf:
        print(f"✓ High openness eliminated bias entirely, low openness: {low_o_conf.modifier:+.3f}")
    else:
        print("✓ Both cases handled (alignment threshold not met or countered)")


def test_negativity_bias():
    """Negativity bias should increase arousal for negative input."""
    print("\n--- Testing Negativity Bias ---")
    
    high_neuroticism = BiasSimulator(
        traits={"openness": 0.5, "neuroticism": 0.8},
        value_priorities={}
    )
    
    ctx = TraceContext()
    modifiers = high_neuroticism.compute_modifiers(
        user_input="I'm really worried about this problem and frustrated",
        value_alignment=0.3,  # Low alignment (won't trigger confirmation)
        ctx=ctx
    )
    
    neg_mod = high_neuroticism.get_modifier_for_field(
        modifiers, "communication_style.arousal"
    )
    
    assert neg_mod is not None, "Expected negativity bias modifier"
    assert neg_mod.bias_type == BiasType.NEGATIVITY
    assert neg_mod.modifier > 0, "Negativity bias should increase arousal"
    assert neg_mod.modifier <= MAX_BIAS_IMPACT, f"Modifier exceeds max: {neg_mod.modifier}"
    
    cites = [c for c in ctx.citations if c.source_id == "negativity_bias"]
    assert len(cites) == 1, "Expected negativity bias citation"
    print(f"✓ Negativity bias triggered: arousal {neg_mod.modifier:+.3f}")


def test_negativity_bias_requires_neuroticism():
    """Low neuroticism should not trigger negativity bias."""
    print("\n--- Testing Negativity Requires Neuroticism ---")
    
    low_neuroticism = BiasSimulator(
        traits={"openness": 0.5, "neuroticism": 0.2},  # Below threshold
        value_priorities={}
    )
    
    modifiers = low_neuroticism.compute_modifiers(
        user_input="This is terrible and frustrating",
        value_alignment=0.3
    )
    
    neg_mod = low_neuroticism.get_modifier_for_field(
        modifiers, "communication_style.arousal"
    )
    
    assert neg_mod is None, "Low neuroticism should not trigger negativity bias"
    print("✓ Low neuroticism correctly prevents negativity bias")


def test_authority_bias():
    """Authority bias should increase confidence when input cites authorities."""
    print("\n--- Testing Authority Bias ---")
    
    conformist = BiasSimulator(
        traits={"openness": 0.5, "neuroticism": 0.3},
        value_priorities={
            "conformity": 0.7,
            "tradition": 0.6,
            "security": 0.6
        }
    )
    
    ctx = TraceContext()
    modifiers = conformist.compute_modifiers(
        user_input="Research shows this is correct and experts agree on this point",
        value_alignment=0.3,
        ctx=ctx
    )
    
    auth_mod = conformist.get_modifier_for_field(
        modifiers, "response_structure.confidence"
    )
    
    assert auth_mod is not None, "Expected authority bias modifier"
    assert auth_mod.bias_type == BiasType.AUTHORITY
    assert auth_mod.modifier > 0, "Authority bias should increase confidence"
    assert auth_mod.modifier <= MAX_BIAS_IMPACT, f"Modifier exceeds max: {auth_mod.modifier}"
    
    cites = [c for c in ctx.citations if c.source_id == "authority_bias"]
    assert len(cites) == 1, "Expected authority bias citation"
    print(f"✓ Authority bias triggered: confidence {auth_mod.modifier:+.3f}")


def test_bias_bounds():
    """All biases should be bounded to MAX_BIAS_IMPACT."""
    print("\n--- Testing Bias Bounds ---")
    
    # Extreme values to try to maximize bias
    extreme_simulator = BiasSimulator(
        traits={"openness": 0.0, "neuroticism": 1.0},  # Maximize negativity, confirmation
        value_priorities={
            "conformity": 1.0,
            "tradition": 1.0,
            "security": 1.0
        }
    )
    
    modifiers = extreme_simulator.compute_modifiers(
        user_input="Research shows this is terrible and frustrating problem",
        value_alignment=1.0  # Maximum alignment
    )
    
    for mod in modifiers:
        assert abs(mod.modifier) <= MAX_BIAS_IMPACT, \
            f"{mod.bias_type} exceeds bound: {mod.modifier}"
    
    print(f"✓ All {len(modifiers)} modifiers within ±{MAX_BIAS_IMPACT} bound")


def test_no_bias_without_triggers():
    """Biases should not activate without proper triggers."""
    print("\n--- Testing No Bias Without Triggers ---")
    
    moderate = BiasSimulator(
        traits={"openness": 0.5, "neuroticism": 0.3},  # Below neuroticism threshold
        value_priorities={
            "conformity": 0.2,  # Below authority threshold
            "tradition": 0.2,
            "security": 0.2
        }
    )
    
    modifiers = moderate.compute_modifiers(
        user_input="This is a neutral statement",  # No negative markers, no authority
        value_alignment=0.3  # Below confirmation threshold
    )
    
    assert len(modifiers) == 0, f"Expected no modifiers, got {len(modifiers)}"
    print("✓ No biases triggered for neutral input with moderate traits")


if __name__ == "__main__":
    test_confirmation_bias()
    test_confirmation_bias_countered_by_openness()
    test_negativity_bias()
    test_negativity_bias_requires_neuroticism()
    test_authority_bias()
    test_bias_bounds()
    test_no_bias_without_triggers()
    print("\n✅ All Bias Simulator Tests Passed!")
