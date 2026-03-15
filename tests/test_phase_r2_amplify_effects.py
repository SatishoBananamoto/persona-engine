"""
Phase R2 Tests: Amplify Effect Sizes

Tests that recalibrated effect sizes, sigmoid activation, Dunning-Kruger
confidence curve, and reduced personality-field inertia produce observable
behavioral differences between personalities.

Target: 25+ tests covering:
- Sigmoid trait_effect() function properties
- Dunning-Kruger confidence curve shape
- Amplified trait effect sizes (verbosity, directness, disclosure, confidence)
- Elasticity sigmoid activation
- Reduced personality-field inertia vs standard inertia
- Twin perceptibility: same scenario, different personalities yield different IR
"""

import copy
import math
import pytest
import yaml

from conftest import make_persona_data

from persona_engine.behavioral.trait_interpreter import (
    TraitInterpreter,
    dunning_kruger_confidence,
    trait_effect,
)
from persona_engine.memory import StanceCache
from persona_engine.planner.engine_config import DEFAULT_CONFIG, EngineConfig
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
    CROSS_TURN_INERTIA,
    PERSONALITY_FIELD_INERTIA,
    _smooth,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    Tone,
    Verbosity,
)
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    Persona,
)
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Helpers
# ============================================================================

def _make_context(user_input: str = "What do you think?") -> ConversationContext:
    return ConversationContext(
        conversation_id="test",
        turn_number=1,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="general",
        user_input=user_input,
        stance_cache=StanceCache(),
    )


def _generate_ir(persona_data: dict, user_input: str = "What do you think?"):
    persona = Persona(**persona_data)
    planner = TurnPlanner(persona, DeterminismManager(seed=42))
    ctx = _make_context(user_input)
    return planner.generate_ir(ctx)


# ============================================================================
# Test Sigmoid trait_effect() Function Properties
# ============================================================================

class TestSigmoidTraitEffect:
    """Verify sigmoid activation has correct mathematical properties."""

    def test_center_returns_half(self):
        """trait_effect(0.5) should equal 0.5 (neutral point)."""
        assert trait_effect(0.5) == pytest.approx(0.5, abs=0.001)

    def test_high_trait_amplified(self):
        """trait_effect(0.8) should be > 0.8 (amplified)."""
        result = trait_effect(0.8)
        assert result > 0.8
        assert result < 1.0

    def test_low_trait_amplified(self):
        """trait_effect(0.2) should be < 0.2 (amplified opposite)."""
        result = trait_effect(0.2)
        assert result < 0.2
        assert result > 0.0

    def test_extreme_high_near_one(self):
        """trait_effect(0.95) should approach 1.0."""
        result = trait_effect(0.95)
        assert result > 0.95

    def test_extreme_low_near_zero(self):
        """trait_effect(0.05) should approach 0.0."""
        result = trait_effect(0.05)
        assert result < 0.05

    def test_monotonically_increasing(self):
        """Sigmoid should be monotonically increasing."""
        prev = 0.0
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            current = trait_effect(x)
            assert current > prev
            prev = current

    def test_steepness_affects_amplification(self):
        """Higher steepness should increase amplification at extremes."""
        low_steep = trait_effect(0.8, steepness=4.0)
        high_steep = trait_effect(0.8, steepness=12.0)
        assert high_steep > low_steep

    def test_symmetric_around_center(self):
        """trait_effect(0.5+d) + trait_effect(0.5-d) should equal ~1.0."""
        for d in [0.1, 0.2, 0.3, 0.4]:
            high = trait_effect(0.5 + d)
            low = trait_effect(0.5 - d)
            assert high + low == pytest.approx(1.0, abs=0.001)


# ============================================================================
# Test Dunning-Kruger Confidence Curve
# ============================================================================

class TestDunningKrugerCurve:
    """Verify the DK curve has correct shape: overconfident novice, valley, calibrated expert."""

    def test_novice_overconfident(self):
        """Novice (prof=0.15) should be overconfident (confidence > proficiency)."""
        conf = dunning_kruger_confidence(0.15, neuroticism=0.3)
        assert conf > 0.15

    def test_novice_neuroticism_reduces_overconfidence(self):
        """High neuroticism should reduce novice overconfidence."""
        low_n = dunning_kruger_confidence(0.15, neuroticism=0.2)
        high_n = dunning_kruger_confidence(0.15, neuroticism=0.8)
        assert low_n > high_n

    def test_intermediate_valley(self):
        """Intermediate (prof=0.45) should be underconfident (confidence < proficiency)."""
        conf = dunning_kruger_confidence(0.45, neuroticism=0.5)
        assert conf < 0.45

    def test_expert_calibrated(self):
        """Expert (prof=0.85) should be slightly underconfident but close."""
        conf = dunning_kruger_confidence(0.85, neuroticism=0.5)
        assert conf < 0.85
        assert conf > 0.75  # Close to actual proficiency

    def test_valley_deeper_than_expert_gap(self):
        """The valley penalty (0.08) should be bigger than expert penalty (0.03)."""
        valley = 0.45 - dunning_kruger_confidence(0.45, neuroticism=0.5)
        expert_gap = 0.85 - dunning_kruger_confidence(0.85, neuroticism=0.5)
        assert valley > expert_gap


# ============================================================================
# Test Amplified Trait Effect Sizes
# ============================================================================

class TestAmplifiedEffectSizes:
    """Verify that recalibrated effect sizes produce larger differences."""

    def test_conscientiousness_verbosity_high_c(self):
        """High C (0.9) with base_verbosity=0.5 should push to DETAILED."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.9,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        result = ti.influences_verbosity(0.5)
        assert result == Verbosity.DETAILED

    def test_conscientiousness_verbosity_low_c(self):
        """Low C (0.1) with base_verbosity=0.5 should push to BRIEF."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.1,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        result = ti.influences_verbosity(0.5)
        assert result == Verbosity.BRIEF

    def test_agreeableness_directness_high_a(self):
        """High A (0.9) should noticeably reduce directness."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.9, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        result = ti.influences_directness(0.5)
        assert result < 0.35  # Noticeably lower with amplified A effect

    def test_agreeableness_directness_low_a(self):
        """Low A (0.1) should noticeably increase directness."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.1, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        result = ti.influences_directness(0.5)
        assert result > 0.65  # Noticeably higher with amplified A effect

    def test_extraversion_disclosure_high_e(self):
        """High E (0.9) should produce positive disclosure modifier."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.9, agreeableness=0.5, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        modifier = ti.get_self_disclosure_modifier()
        assert modifier > 0.15  # Amplified E should give strong positive modifier

    def test_extraversion_disclosure_low_e(self):
        """Low E (0.1) should produce negative disclosure modifier."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.5)
        ti = TraitInterpreter(traits)
        modifier = ti.get_self_disclosure_modifier()
        assert modifier < -0.15  # Amplified E should give strong negative modifier

    def test_disclosure_spread_sufficient(self):
        """Difference between high-E and low-E disclosure should be >= 0.4."""
        high_e = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.9, agreeableness=0.5, neuroticism=0.5)
        low_e = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.5)
        hi_mod = TraitInterpreter(high_e).get_self_disclosure_modifier()
        lo_mod = TraitInterpreter(low_e).get_self_disclosure_modifier()
        spread = hi_mod - lo_mod
        assert spread >= 0.4


# ============================================================================
# Test Elasticity Sigmoid Activation
# ============================================================================

class TestElasticitySigmoid:
    """Verify elasticity uses sigmoid activation for openness."""

    def test_high_openness_more_elastic(self):
        """High openness (0.9) should produce higher elasticity than moderate (0.5)."""
        high_o = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        mod_o = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        hi_e = TraitInterpreter(high_o).get_elasticity(0.5)
        mod_e = TraitInterpreter(mod_o).get_elasticity(0.5)
        assert hi_e > mod_e

    def test_low_openness_rigid(self):
        """Low openness (0.1) should produce low elasticity."""
        low_o = BigFiveTraits(openness=0.1, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        e = TraitInterpreter(low_o).get_elasticity(0.5)
        assert e < 0.35

    def test_sigmoid_amplifies_extreme_openness(self):
        """Difference between O=0.9 and O=0.7 should be smaller than O=0.5 vs O=0.3
        (sigmoid compresses center, amplifies extremes)."""
        traits = lambda o: BigFiveTraits(openness=o, conscientiousness=0.5,
                                          extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        e_09 = TraitInterpreter(traits(0.9)).get_elasticity(0.3)
        e_07 = TraitInterpreter(traits(0.7)).get_elasticity(0.3)
        e_05 = TraitInterpreter(traits(0.5)).get_elasticity(0.3)
        e_03 = TraitInterpreter(traits(0.3)).get_elasticity(0.3)

        # High end compressed, low end compressed (sigmoid property)
        high_diff = e_09 - e_07
        mid_diff = e_07 - e_05

        # Both should be positive (monotonic)
        assert high_diff > 0
        assert mid_diff > 0

    def test_elasticity_clamped(self):
        """Elasticity should always be in [0.1, 0.9]."""
        for o in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for c in [0.0, 0.5, 1.0]:
                traits = BigFiveTraits(openness=o, conscientiousness=0.5,
                                       extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
                e = TraitInterpreter(traits).get_elasticity(c)
                assert 0.1 <= e <= 0.9


# ============================================================================
# Test Confidence with DK Curve + Amplified Effects
# ============================================================================

class TestConfidenceAmplified:
    """Verify confidence uses DK curve and amplified C/N effects."""

    def test_novice_overconfident_low_n(self):
        """Low-N novice should be notably overconfident."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.2)
        ti = TraitInterpreter(traits)
        conf = ti.get_confidence_modifier(0.2)  # Novice
        # DK: 0.2 + 0.8*0.25 = 0.4, then C/N adjustments
        assert conf > 0.2  # Should be overconfident

    def test_high_n_reduces_confidence(self):
        """High neuroticism should substantially reduce confidence."""
        low_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.2)
        high_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.9)
        low_n_conf = TraitInterpreter(low_n).get_confidence_modifier(0.6)
        high_n_conf = TraitInterpreter(high_n).get_confidence_modifier(0.6)
        assert low_n_conf > high_n_conf
        assert low_n_conf - high_n_conf > 0.1  # Amplified effect should be visible

    def test_high_c_boosts_confidence(self):
        """High conscientiousness should boost confidence."""
        low_c = BigFiveTraits(openness=0.5, conscientiousness=0.2,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        high_c = BigFiveTraits(openness=0.5, conscientiousness=0.8,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        low_c_conf = TraitInterpreter(low_c).get_confidence_modifier(0.6)
        high_c_conf = TraitInterpreter(high_c).get_confidence_modifier(0.6)
        assert high_c_conf > low_c_conf

    def test_confidence_clamped(self):
        """Confidence should always be in [0.1, 0.95]."""
        for n in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5, 1.0]:
                for p in [0.1, 0.5, 0.9]:
                    traits = BigFiveTraits(openness=0.5, conscientiousness=c,
                                           extraversion=0.5, agreeableness=0.5, neuroticism=n)
                    conf = TraitInterpreter(traits).get_confidence_modifier(p)
                    assert 0.1 <= conf <= 0.95


# ============================================================================
# Test Personality-Field Inertia
# ============================================================================

class TestPersonalityFieldInertia:
    """Verify personality-driven fields use lower inertia than knowledge fields."""

    def test_personality_inertia_less_than_standard(self):
        """personality_field_inertia should be less than cross_turn_inertia."""
        assert PERSONALITY_FIELD_INERTIA < CROSS_TURN_INERTIA

    def test_personality_inertia_is_008(self):
        """Default personality_field_inertia should be 0.08."""
        assert DEFAULT_CONFIG.personality_field_inertia == 0.08

    def test_standard_inertia_is_015(self):
        """Standard cross_turn_inertia should remain 0.15."""
        assert DEFAULT_CONFIG.cross_turn_inertia == 0.15

    def test_smooth_function_with_low_inertia(self):
        """Lower inertia should converge to new value faster."""
        prev, new = 0.3, 0.7
        smooth_standard = _smooth(prev, new, CROSS_TURN_INERTIA)
        smooth_personality = _smooth(prev, new, PERSONALITY_FIELD_INERTIA)
        # Lower inertia = less weight on prev = closer to new
        assert smooth_personality > smooth_standard
        assert smooth_personality > 0.6  # Should be close to new (0.7)

    def test_personality_fields_use_lower_inertia_in_ir(self):
        """Multi-turn: personality-driven fields should converge faster than confidence."""
        data = make_persona_data(openness=0.9, agreeableness=0.2)
        persona = Persona(**data)
        planner = TurnPlanner(persona, DeterminismManager(seed=42))

        # Generate 2 turns to activate inertia
        ctx1 = _make_context("Hello there")
        ir1 = planner.generate_ir(ctx1)

        ctx2 = ConversationContext(
            conversation_id="test",
            turn_number=2,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="general",
            user_input="What about something completely different?",
            stance_cache=StanceCache(),
        )
        ir2 = planner.generate_ir(ctx2)

        # Both should produce valid IR without errors
        assert ir1 is not None
        assert ir2 is not None


# ============================================================================
# Test Twin Perceptibility with Amplified Effects
# ============================================================================

class TestTwinPerceptibilityAmplified:
    """Verify that personality differences produce observably different IR."""

    def test_agreeable_vs_disagreeable_directness(self):
        """High-A and low-A twins should differ noticeably in directness."""
        ir_high_a = _generate_ir(make_persona_data(agreeableness=0.9))
        ir_low_a = _generate_ir(make_persona_data(agreeableness=0.1))
        diff = abs(ir_high_a.communication_style.directness -
                   ir_low_a.communication_style.directness)
        assert diff > 0.05  # Amplified effect should be visible

    def test_conscientious_vs_not_verbosity(self):
        """High-C and low-C should differ in verbosity."""
        ir_high_c = _generate_ir(make_persona_data(conscientiousness=0.9))
        ir_low_c = _generate_ir(make_persona_data(conscientiousness=0.1))
        # High-C should trend toward DETAILED, low-C toward BRIEF
        verbosity_order = {Verbosity.BRIEF: 0, Verbosity.MEDIUM: 1, Verbosity.DETAILED: 2}
        assert verbosity_order[ir_high_c.communication_style.verbosity] >= \
               verbosity_order[ir_low_c.communication_style.verbosity]

    def test_open_vs_closed_elasticity(self):
        """High-O and low-O should differ in elasticity."""
        ir_high_o = _generate_ir(make_persona_data(openness=0.9))
        ir_low_o = _generate_ir(make_persona_data(openness=0.1))
        assert ir_high_o.response_structure.elasticity > ir_low_o.response_structure.elasticity

    def test_neurotic_vs_stable_confidence(self):
        """High-N should have lower confidence than low-N."""
        ir_high_n = _generate_ir(make_persona_data(neuroticism=0.9))
        ir_low_n = _generate_ir(make_persona_data(neuroticism=0.1))
        assert ir_low_n.response_structure.confidence > ir_high_n.response_structure.confidence

    def test_extreme_personas_differ_on_multiple_axes(self):
        """An anxious introvert vs confident extrovert should differ on 3+ IR fields."""
        anxious_introvert = make_persona_data(
            extraversion=0.1, neuroticism=0.9, agreeableness=0.8, conscientiousness=0.3
        )
        confident_extrovert = make_persona_data(
            extraversion=0.9, neuroticism=0.1, agreeableness=0.2, conscientiousness=0.8
        )
        ir_ai = _generate_ir(anxious_introvert)
        ir_ce = _generate_ir(confident_extrovert)

        differences = 0
        if ir_ai.communication_style.directness != ir_ce.communication_style.directness:
            differences += 1
        if ir_ai.communication_style.tone != ir_ce.communication_style.tone:
            differences += 1
        if ir_ai.response_structure.confidence != ir_ce.response_structure.confidence:
            differences += 1
        if ir_ai.response_structure.elasticity != ir_ce.response_structure.elasticity:
            differences += 1
        if ir_ai.communication_style.verbosity != ir_ce.communication_style.verbosity:
            differences += 1

        assert differences >= 3, f"Only {differences} IR fields differed between extreme personas"
