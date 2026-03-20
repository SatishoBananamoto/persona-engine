"""
Tests for TraitInterpreter

Comprehensive coverage of all methods in trait_interpreter.py,
targeting 95%+ line/branch coverage.  Covers every Big Five trait
group (Openness, Conscientiousness, Extraversion, Agreeableness,
Neuroticism), multi-trait interactions (tone, confidence, validation
markers), and the factory function.
"""

import pytest

from persona_engine.schema.persona_schema import BigFiveTraits
from persona_engine.schema.ir_schema import Tone, Verbosity
from persona_engine.behavioral.trait_interpreter import (
    TraitInterpreter,
    create_trait_interpreter,
)


# ============================================================================
# Helpers
# ============================================================================

def make_traits(
    openness: float = 0.5,
    conscientiousness: float = 0.5,
    extraversion: float = 0.5,
    agreeableness: float = 0.5,
    neuroticism: float = 0.5,
) -> BigFiveTraits:
    """Create BigFiveTraits with sensible defaults, overriding as needed."""
    return BigFiveTraits(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
    )


def make_interpreter(**kwargs) -> TraitInterpreter:
    """Shortcut to build a TraitInterpreter from keyword trait params."""
    return TraitInterpreter(make_traits(**kwargs))


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    def test_stores_traits(self):
        traits = make_traits(openness=0.8)
        interp = TraitInterpreter(traits)
        assert interp.traits is traits

    def test_trait_fields_accessible(self):
        interp = make_interpreter(neuroticism=0.42)
        assert interp.traits.neuroticism == pytest.approx(0.42)


# ============================================================================
# OPENNESS: get_elasticity
# ============================================================================

class TestGetElasticity:
    """
    Actual formula (from trait_interpreter.py — Phase R2: sigmoid activation):
        openness_activated = trait_effect(openness)
        openness_factor = openness_activated * 0.7
        confidence_penalty = base_confidence * 0.3
        elasticity = openness_factor - confidence_penalty + 0.2
        return max(0.1, min(0.9, elasticity))
    """

    def test_mid_values(self):
        interp = make_interpreter(openness=0.5)
        # sigmoid(0.5)=0.5, 0.5*0.7 - 0.5*0.3 + 0.2 = 0.4
        result = interp.get_elasticity(0.5)
        assert result == pytest.approx(0.4)

    def test_high_openness_low_confidence(self):
        interp = make_interpreter(openness=1.0)
        # sigmoid(1.0)≈0.9997, 0.9997*0.7 - 0 + 0.2 ≈ 0.887
        result = interp.get_elasticity(0.0)
        assert result == pytest.approx(0.887, abs=0.01)

    def test_low_openness_high_confidence(self):
        interp = make_interpreter(openness=0.0)
        # sigmoid(0.0)≈0.0003, 0.0003*0.7 - 0.3 + 0.2 ≈ -0.1 → clamped to 0.1
        result = interp.get_elasticity(1.0)
        assert result == pytest.approx(0.1)

    def test_zero_openness_zero_confidence(self):
        interp = make_interpreter(openness=0.0)
        # sigmoid(0.0)≈0.0003, 0.0003*0.7 + 0.2 ≈ 0.213
        result = interp.get_elasticity(0.0)
        assert result == pytest.approx(0.213, abs=0.01)

    def test_max_openness_max_confidence(self):
        interp = make_interpreter(openness=1.0)
        # sigmoid(1.0)*0.7 - 0.3 + 0.2 ≈ 0.587
        result = interp.get_elasticity(1.0)
        assert result == pytest.approx(0.587, abs=0.01)

    def test_upper_clamp_fires(self):
        """High openness and low confidence produces high elasticity."""
        interp = make_interpreter(openness=0.9)
        # sigmoid(0.9)*0.7 + 0.2 ≈ 0.873
        result = interp.get_elasticity(0.0)
        assert result == pytest.approx(0.873, abs=0.01)

    def test_lower_clamp_fires_with_extreme_confidence(self):
        """
        With base_confidence much greater than 1, the raw value can go
        below 0.1 and should be clamped.
        """
        interp = make_interpreter(openness=0.0)
        # 0 - 10*0.3 + 0.2 = -2.8 → clamped to 0.1
        result = interp.get_elasticity(10.0)
        assert result == pytest.approx(0.1)

    def test_result_always_in_bounds(self):
        """Sweep across a range of inputs to verify [0.1, 0.9] invariant."""
        for o in [0.0, 0.3, 0.5, 0.7, 1.0]:
            interp = make_interpreter(openness=o)
            for bc in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = interp.get_elasticity(bc)
                assert 0.1 <= result <= 0.9, (
                    f"Out of bounds: openness={o}, base_confidence={bc} -> {result}"
                )

    def test_return_type_is_float(self):
        result = make_interpreter(openness=0.5).get_elasticity(0.5)
        assert isinstance(result, float)


# ============================================================================
# OPENNESS: influences_abstract_reasoning
# ============================================================================

class TestInfluencesAbstractReasoning:
    """Threshold: openness > 0.7"""

    def test_high_openness(self):
        assert make_interpreter(openness=0.9).influences_abstract_reasoning() is True

    def test_just_above_threshold(self):
        assert make_interpreter(openness=0.71).influences_abstract_reasoning() is True

    def test_at_threshold(self):
        # 0.7 is NOT > 0.7
        assert make_interpreter(openness=0.7).influences_abstract_reasoning() is False

    def test_below_threshold(self):
        assert make_interpreter(openness=0.5).influences_abstract_reasoning() is False

    def test_zero(self):
        assert make_interpreter(openness=0.0).influences_abstract_reasoning() is False


# ============================================================================
# OPENNESS: get_novelty_seeking
# ============================================================================

class TestGetNoveltySeeking:
    def test_returns_openness(self):
        assert make_interpreter(openness=0.73).get_novelty_seeking() == pytest.approx(0.73)

    def test_zero(self):
        assert make_interpreter(openness=0.0).get_novelty_seeking() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(openness=1.0).get_novelty_seeking() == pytest.approx(1.0)


# ============================================================================
# CONSCIENTIOUSNESS: influences_verbosity
# ============================================================================

class TestInfluencesVerbosity:
    """
    adjusted = base_verbosity + (C - 0.5) * 0.2, clamped [0, 1]
    <0.35 -> BRIEF, <0.65 -> MEDIUM, else -> DETAILED
    """

    # --- Three return values ---

    def test_brief(self):
        # base=0.2, C=0.3: adjusted = 0.2 + (0.3-0.5)*0.2 = 0.2 - 0.04 = 0.16
        interp = make_interpreter(conscientiousness=0.3)
        assert interp.influences_verbosity(0.2) == Verbosity.BRIEF

    def test_medium(self):
        # base=0.5, C=0.5: adjusted = 0.5 + 0 = 0.5
        interp = make_interpreter(conscientiousness=0.5)
        assert interp.influences_verbosity(0.5) == Verbosity.MEDIUM

    def test_detailed(self):
        # base=0.8, C=0.8: adjusted = 0.8 + 0.06 = 0.86
        interp = make_interpreter(conscientiousness=0.8)
        assert interp.influences_verbosity(0.8) == Verbosity.DETAILED

    # --- Boundary: BRIEF / MEDIUM at 0.35 ---

    def test_at_035_is_medium(self):
        # adjusted exactly 0.35 -> NOT < 0.35 -> MEDIUM
        # Need: base + (C - 0.5)*0.2 = 0.35
        # With C=0.5: base = 0.35
        interp = make_interpreter(conscientiousness=0.5)
        assert interp.influences_verbosity(0.35) == Verbosity.MEDIUM

    def test_just_below_035_is_brief(self):
        # adjusted = 0.349
        interp = make_interpreter(conscientiousness=0.5)
        assert interp.influences_verbosity(0.349) == Verbosity.BRIEF

    # --- Boundary: MEDIUM / DETAILED at 0.65 ---

    def test_at_065_is_detailed(self):
        # adjusted exactly 0.65 -> NOT < 0.65 -> DETAILED
        interp = make_interpreter(conscientiousness=0.5)
        assert interp.influences_verbosity(0.65) == Verbosity.DETAILED

    def test_just_below_065_is_medium(self):
        interp = make_interpreter(conscientiousness=0.5)
        assert interp.influences_verbosity(0.649) == Verbosity.MEDIUM

    # --- Clamping of adjusted value ---

    def test_clamped_to_zero(self):
        # Very low base, very low C
        # base=0.0, C=0.0: adjusted = 0.0 + (0-0.5)*0.2 = -0.1 -> clamped to 0
        interp = make_interpreter(conscientiousness=0.0)
        result = interp.influences_verbosity(0.0)
        assert result == Verbosity.BRIEF

    def test_clamped_to_one(self):
        # Very high base, very high C
        # base=1.0, C=1.0: adjusted = 1.0 + 0.1 = 1.1 -> clamped to 1.0
        interp = make_interpreter(conscientiousness=1.0)
        result = interp.influences_verbosity(1.0)
        assert result == Verbosity.DETAILED

    # --- High C shifts verbosity up ---

    def test_high_c_pushes_brief_to_medium(self):
        # base=0.3 alone would be BRIEF; high C bumps it
        # C=1.0: adjusted = 0.3 + 0.1 = 0.4 -> MEDIUM
        interp = make_interpreter(conscientiousness=1.0)
        assert interp.influences_verbosity(0.3) == Verbosity.MEDIUM

    # --- Low C shifts verbosity down ---

    def test_low_c_pushes_medium_to_brief(self):
        # base=0.4, C=0.0: adjusted = 0.4 + (-0.5)*0.2 = 0.4 - 0.1 = 0.3 -> BRIEF
        interp = make_interpreter(conscientiousness=0.0)
        assert interp.influences_verbosity(0.4) == Verbosity.BRIEF


# ============================================================================
# CONSCIENTIOUSNESS: get_planning_language_tendency
# ============================================================================

class TestGetPlanningLanguageTendency:
    def test_returns_conscientiousness(self):
        assert make_interpreter(conscientiousness=0.82).get_planning_language_tendency() == pytest.approx(0.82)

    def test_zero(self):
        assert make_interpreter(conscientiousness=0.0).get_planning_language_tendency() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(conscientiousness=1.0).get_planning_language_tendency() == pytest.approx(1.0)


# ============================================================================
# CONSCIENTIOUSNESS: get_follow_through_likelihood
# ============================================================================

class TestGetFollowThroughLikelihood:
    def test_returns_conscientiousness(self):
        assert make_interpreter(conscientiousness=0.65).get_follow_through_likelihood() == pytest.approx(0.65)

    def test_zero(self):
        assert make_interpreter(conscientiousness=0.0).get_follow_through_likelihood() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(conscientiousness=1.0).get_follow_through_likelihood() == pytest.approx(1.0)


# ============================================================================
# EXTRAVERSION: influences_proactivity
# ============================================================================

class TestInfluencesProactivity:
    def test_returns_extraversion(self):
        assert make_interpreter(extraversion=0.77).influences_proactivity() == pytest.approx(0.662)

    def test_zero(self):
        assert make_interpreter(extraversion=0.0).influences_proactivity() == pytest.approx(0.2)

    def test_one(self):
        assert make_interpreter(extraversion=1.0).influences_proactivity() == pytest.approx(0.8)


# ============================================================================
# EXTRAVERSION: get_self_disclosure_modifier
# ============================================================================

class TestGetSelfDisclosureModifier:
    """Phase R2: (trait_effect(extraversion) - 0.5) * 0.45, sigmoid-amplified"""

    def test_high_extraversion(self):
        # trait_effect(1.0)≈0.982, (0.982-0.5)*0.45 ≈ 0.217, + N(0.5)*0.1 = 0.267
        assert make_interpreter(extraversion=1.0).get_self_disclosure_modifier() == pytest.approx(0.267, abs=0.01)

    def test_low_extraversion(self):
        # trait_effect(0.0)≈0.018, (0.018-0.5)*0.45 ≈ -0.217, + N(0.5)*0.1 = -0.167
        assert make_interpreter(extraversion=0.0).get_self_disclosure_modifier() == pytest.approx(-0.167, abs=0.01)

    def test_midpoint_extraversion(self):
        # trait_effect(0.5)=0.5, (0.5-0.5)*0.45 = 0.0, + N(0.5)*0.1 = 0.05
        assert make_interpreter(extraversion=0.5).get_self_disclosure_modifier() == pytest.approx(0.05)

    def test_slightly_above_mid(self):
        # trait_effect(0.75)≈0.881, (0.881-0.5)*0.45 ≈ 0.171, + N(0.5)*0.1 = 0.221
        assert make_interpreter(extraversion=0.75).get_self_disclosure_modifier() == pytest.approx(0.221, abs=0.01)

    def test_slightly_below_mid(self):
        # trait_effect(0.25)≈0.119, (0.119-0.5)*0.45 ≈ -0.171, + N(0.5)*0.1 = -0.121
        assert make_interpreter(extraversion=0.25).get_self_disclosure_modifier() == pytest.approx(-0.121, abs=0.01)

    def test_range_lower_bound(self):
        result = make_interpreter(extraversion=0.0).get_self_disclosure_modifier()
        assert result >= -0.225

    def test_range_upper_bound(self):
        result = make_interpreter(extraversion=1.0).get_self_disclosure_modifier()
        assert result <= 0.275


# ============================================================================
# EXTRAVERSION: influences_response_length_social
# ============================================================================

class TestInfluencesResponseLengthSocial:
    def test_returns_extraversion(self):
        assert make_interpreter(extraversion=0.6).influences_response_length_social() == pytest.approx(0.6)

    def test_zero(self):
        assert make_interpreter(extraversion=0.0).influences_response_length_social() == pytest.approx(0.0)


# ============================================================================
# EXTRAVERSION: get_enthusiasm_baseline
# ============================================================================

class TestGetEnthusiasmBaseline:
    def test_returns_extraversion(self):
        assert make_interpreter(extraversion=0.88).get_enthusiasm_baseline() == pytest.approx(0.64)

    def test_zero(self):
        assert make_interpreter(extraversion=0.0).get_enthusiasm_baseline() == pytest.approx(0.2)


# ============================================================================
# AGREEABLENESS: influences_directness
# ============================================================================

class TestInfluencesDirectness:
    """
    Phase R2: Sigmoid-amplified agreeableness effect on directness.
    modifier = (0.5 - trait_effect(agreeableness)) * 0.5
    adjusted = base_directness + modifier, clamped [0, 1]
    """

    def test_high_agreeableness_reduces_directness(self):
        # A=0.9: trait_effect≈0.961, modifier≈-0.230
        interp = make_interpreter(agreeableness=0.9)
        assert interp.influences_directness(0.6) == pytest.approx(0.370, abs=0.01)

    def test_low_agreeableness_increases_directness(self):
        # A=0.1: trait_effect≈0.039, modifier≈+0.230
        interp = make_interpreter(agreeableness=0.1)
        assert interp.influences_directness(0.6) == pytest.approx(0.830, abs=0.01)

    def test_mid_agreeableness_no_change(self):
        # A=0.5: trait_effect=0.5, modifier=0
        interp = make_interpreter(agreeableness=0.5)
        assert interp.influences_directness(0.6) == pytest.approx(0.6)

    def test_clamped_to_zero(self):
        # A=1.0: trait_effect≈0.982, modifier≈-0.241
        interp = make_interpreter(agreeableness=1.0)
        assert interp.influences_directness(0.1) == pytest.approx(0.0)

    def test_clamped_to_one(self):
        # A=0.0: trait_effect≈0.018, modifier≈+0.241
        interp = make_interpreter(agreeableness=0.0)
        assert interp.influences_directness(0.9) == pytest.approx(1.0)

    def test_zero_base_low_agreeableness(self):
        # A=0.0: modifier≈+0.241
        interp = make_interpreter(agreeableness=0.0)
        assert interp.influences_directness(0.0) == pytest.approx(0.241, abs=0.01)

    def test_one_base_high_agreeableness(self):
        # A=1.0: trait_effect≈0.982, modifier≈-0.241, base=1.0 → 0.759
        interp = make_interpreter(agreeableness=1.0)
        assert interp.influences_directness(1.0) == pytest.approx(0.759, abs=0.01)


# ============================================================================
# AGREEABLENESS: get_validation_tendency
# ============================================================================

class TestGetValidationTendency:
    def test_returns_agreeableness(self):
        assert make_interpreter(agreeableness=0.62).get_validation_tendency() == pytest.approx(0.62)

    def test_zero(self):
        assert make_interpreter(agreeableness=0.0).get_validation_tendency() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(agreeableness=1.0).get_validation_tendency() == pytest.approx(1.0)


# ============================================================================
# AGREEABLENESS: get_conflict_avoidance
# ============================================================================

class TestGetConflictAvoidance:
    def test_returns_agreeableness(self):
        assert make_interpreter(agreeableness=0.44).get_conflict_avoidance() == pytest.approx(0.44)

    def test_zero(self):
        assert make_interpreter(agreeableness=0.0).get_conflict_avoidance() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(agreeableness=1.0).get_conflict_avoidance() == pytest.approx(1.0)


# ============================================================================
# AGREEABLENESS: influences_hedging_frequency
# ============================================================================

class TestInfluencesHedgingFrequency:
    """Formula: agreeableness * 0.6"""

    def test_high_agreeableness(self):
        assert make_interpreter(agreeableness=1.0).influences_hedging_frequency() == pytest.approx(0.7)

    def test_low_agreeableness(self):
        assert make_interpreter(agreeableness=0.0).influences_hedging_frequency() == pytest.approx(0.1)

    def test_mid_agreeableness(self):
        assert make_interpreter(agreeableness=0.5).influences_hedging_frequency() == pytest.approx(0.4)

    def test_arbitrary_value(self):
        # 0.8 * 0.6 + 0.5 * 0.2 = 0.58
        assert make_interpreter(agreeableness=0.8).influences_hedging_frequency() == pytest.approx(0.58)


# ============================================================================
# NEUROTICISM: get_stress_sensitivity
# ============================================================================

class TestGetStressSensitivity:
    def test_returns_neuroticism(self):
        assert make_interpreter(neuroticism=0.55).get_stress_sensitivity() == pytest.approx(0.55)

    def test_zero(self):
        assert make_interpreter(neuroticism=0.0).get_stress_sensitivity() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(neuroticism=1.0).get_stress_sensitivity() == pytest.approx(1.0)


# ============================================================================
# NEUROTICISM: influences_mood_stability
# ============================================================================

class TestInfluencesMoodStability:
    """Formula: 1.0 - neuroticism"""

    def test_high_neuroticism_low_stability(self):
        assert make_interpreter(neuroticism=0.9).influences_mood_stability() == pytest.approx(0.1)

    def test_low_neuroticism_high_stability(self):
        assert make_interpreter(neuroticism=0.1).influences_mood_stability() == pytest.approx(0.9)

    def test_zero_neuroticism(self):
        assert make_interpreter(neuroticism=0.0).influences_mood_stability() == pytest.approx(1.0)

    def test_full_neuroticism(self):
        assert make_interpreter(neuroticism=1.0).influences_mood_stability() == pytest.approx(0.0)

    def test_mid_neuroticism(self):
        assert make_interpreter(neuroticism=0.5).influences_mood_stability() == pytest.approx(0.5)


# ============================================================================
# NEUROTICISM: get_anxiety_baseline
# ============================================================================

class TestGetAnxietyBaseline:
    def test_returns_neuroticism(self):
        assert make_interpreter(neuroticism=0.38).get_anxiety_baseline() == pytest.approx(0.38)

    def test_zero(self):
        assert make_interpreter(neuroticism=0.0).get_anxiety_baseline() == pytest.approx(0.0)

    def test_one(self):
        assert make_interpreter(neuroticism=1.0).get_anxiety_baseline() == pytest.approx(1.0)


# ============================================================================
# NEUROTICISM: get_negative_tone_bias
# ============================================================================

class TestGetNegativeToneBias:
    """Formula: neuroticism * 0.7"""

    def test_high_neuroticism(self):
        assert make_interpreter(neuroticism=1.0).get_negative_tone_bias() == pytest.approx(0.5)

    def test_low_neuroticism(self):
        assert make_interpreter(neuroticism=0.0).get_negative_tone_bias() == pytest.approx(0.0)

    def test_mid_neuroticism(self):
        assert make_interpreter(neuroticism=0.5).get_negative_tone_bias() == pytest.approx(0.25)

    def test_arbitrary_value(self):
        # 0.8 * 0.5 = 0.4
        assert make_interpreter(neuroticism=0.8).get_negative_tone_bias() == pytest.approx(0.4)


# ============================================================================
# MULTI-TRAIT: get_tone_from_mood  --  Complete decision-tree coverage
# ============================================================================

class TestGetToneFromMood:
    """
    Decision tree:
      1. stress > 0.6 AND N > 0.6:
         - arousal > 0.6  -> ANXIOUS_STRESSED
         - else           -> CONCERNED_EMPATHETIC
      2. e_bonus = 0.2 if E > 0.7 else 0
      3. valence > 0.3 (positive):
         - arousal > 0.7:
             e_bonus > 0 -> EXCITED_ENGAGED
             else        -> WARM_ENTHUSIASTIC
         - arousal > 0.4:
             O > 0.6     -> THOUGHTFUL_ENGAGED
             else        -> WARM_CONFIDENT
         - else          -> CONTENT_CALM
      4. valence > -0.3 (neutral):
         - arousal > 0.5 -> PROFESSIONAL_COMPOSED
         - else          -> NEUTRAL_CALM
      5. valence <= -0.3 (negative):
         - arousal > 0.6 -> FRUSTRATED_TENSE
         - arousal > 0.3 -> DISAPPOINTED_RESIGNED
         - else          -> SAD_SUBDUED
    """

    # --- Branch 1: High stress + high neuroticism ---

    def test_anxious_stressed(self):
        """stress>0.6, N>0.6, arousal>0.6 -> ANXIOUS_STRESSED"""
        interp = make_interpreter(neuroticism=0.8)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.8, stress=0.8
        )
        assert result == Tone.ANXIOUS_STRESSED

    def test_concerned_empathetic(self):
        """stress>0.6, N>0.6, arousal<=0.6 -> CONCERNED_EMPATHETIC"""
        interp = make_interpreter(neuroticism=0.8)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.4, stress=0.8
        )
        assert result == Tone.CONCERNED_EMPATHETIC

    def test_stress_boundary_not_triggered(self):
        """stress=0.6 (NOT >0.6) should NOT enter the stress branch."""
        interp = make_interpreter(neuroticism=0.8, extraversion=0.3, openness=0.3)
        # Falls through to valence-based logic instead
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.8, stress=0.6
        )
        # valence=0 is neutral, arousal>0.5 -> PROFESSIONAL_COMPOSED
        assert result == Tone.PROFESSIONAL_COMPOSED

    def test_neuroticism_boundary_not_triggered(self):
        """N=0.6 (NOT >0.6) should NOT enter the stress branch."""
        interp = make_interpreter(neuroticism=0.6, extraversion=0.3, openness=0.3)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.8, stress=0.8
        )
        assert result == Tone.PROFESSIONAL_COMPOSED

    def test_stressed_arousal_boundary(self):
        """stress>0.6, N>0.6, arousal=0.6 (NOT >0.6) -> CONCERNED_EMPATHETIC"""
        interp = make_interpreter(neuroticism=0.7)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.6, stress=0.7
        )
        assert result == Tone.CONCERNED_EMPATHETIC

    # --- Branch 3: Positive valence (>0.3) ---

    def test_excited_engaged(self):
        """valence>0.3, arousal>0.7, E>0.7 -> EXCITED_ENGAGED"""
        interp = make_interpreter(extraversion=0.8, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.9, stress=0.2
        )
        assert result == Tone.EXCITED_ENGAGED

    def test_warm_enthusiastic(self):
        """valence>0.3, arousal>0.7, E<=0.7 -> WARM_ENTHUSIASTIC"""
        interp = make_interpreter(extraversion=0.5, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.9, stress=0.2
        )
        assert result == Tone.WARM_ENTHUSIASTIC

    def test_warm_enthusiastic_extraversion_boundary(self):
        """E=0.7 (NOT >0.7), so e_bonus=0 -> WARM_ENTHUSIASTIC"""
        interp = make_interpreter(extraversion=0.7, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.9, stress=0.2
        )
        assert result == Tone.WARM_ENTHUSIASTIC

    def test_thoughtful_engaged(self):
        """valence>0.3, 0.4<arousal<=0.7, O>0.6 -> THOUGHTFUL_ENGAGED"""
        interp = make_interpreter(openness=0.8, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.6, stress=0.2
        )
        assert result == Tone.THOUGHTFUL_ENGAGED

    def test_warm_confident(self):
        """valence>0.3, 0.4<arousal<=0.7, O<=0.6 -> WARM_CONFIDENT"""
        interp = make_interpreter(openness=0.4, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.6, stress=0.2
        )
        assert result == Tone.WARM_CONFIDENT

    def test_warm_confident_openness_boundary(self):
        """O=0.6 (NOT >0.6) -> WARM_CONFIDENT"""
        interp = make_interpreter(openness=0.6, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.6, stress=0.2
        )
        assert result == Tone.WARM_CONFIDENT

    def test_content_calm(self):
        """valence>0.3, arousal<=0.4 -> CONTENT_CALM"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.3, stress=0.2
        )
        assert result == Tone.CONTENT_CALM

    def test_content_calm_arousal_boundary(self):
        """arousal=0.4 (NOT >0.4) -> CONTENT_CALM"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.4, stress=0.2
        )
        assert result == Tone.CONTENT_CALM

    def test_positive_arousal_boundary_at_07(self):
        """arousal=0.7 (NOT >0.7) falls to 'arousal > 0.4' branch"""
        interp = make_interpreter(openness=0.8, neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.5, mood_arousal=0.7, stress=0.2
        )
        assert result == Tone.THOUGHTFUL_ENGAGED

    def test_positive_valence_boundary(self):
        """valence=0.3 (NOT >0.3) falls to neutral branch"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.3, mood_arousal=0.6, stress=0.2
        )
        assert result == Tone.PROFESSIONAL_COMPOSED

    # --- Branch 4: Neutral valence (-0.3 < valence <= 0.3) ---

    def test_professional_composed(self):
        """-0.3<valence<=0.3, arousal>0.5 -> PROFESSIONAL_COMPOSED"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.7, stress=0.2
        )
        assert result == Tone.PROFESSIONAL_COMPOSED

    def test_neutral_calm(self):
        """-0.3<valence<=0.3, arousal<=0.5 -> NEUTRAL_CALM"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.3, stress=0.2
        )
        assert result == Tone.NEUTRAL_CALM

    def test_neutral_calm_arousal_boundary(self):
        """arousal=0.5 (NOT >0.5) -> NEUTRAL_CALM"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=0.0, mood_arousal=0.5, stress=0.2
        )
        assert result == Tone.NEUTRAL_CALM

    def test_neutral_valence_boundary_negative(self):
        """valence=-0.3 (NOT >-0.3) falls to negative branch"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.3, mood_arousal=0.8, stress=0.2
        )
        assert result == Tone.FRUSTRATED_TENSE

    # --- Branch 5: Negative valence (valence <= -0.3) ---

    def test_frustrated_tense(self):
        """valence<=-0.3, arousal>0.6 -> FRUSTRATED_TENSE"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.5, mood_arousal=0.8, stress=0.2
        )
        assert result == Tone.FRUSTRATED_TENSE

    def test_disappointed_resigned(self):
        """valence<=-0.3, 0.3<arousal<=0.6 -> DISAPPOINTED_RESIGNED"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.5, mood_arousal=0.5, stress=0.2
        )
        assert result == Tone.DISAPPOINTED_RESIGNED

    def test_sad_subdued(self):
        """valence<=-0.3, arousal<=0.3 -> SAD_SUBDUED"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.5, mood_arousal=0.2, stress=0.2
        )
        assert result == Tone.SAD_SUBDUED

    def test_negative_arousal_boundary_06(self):
        """arousal=0.6 (NOT >0.6) -> DISAPPOINTED_RESIGNED"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.5, mood_arousal=0.6, stress=0.2
        )
        assert result == Tone.DISAPPOINTED_RESIGNED

    def test_negative_arousal_boundary_03(self):
        """arousal=0.3 (NOT >0.3) -> SAD_SUBDUED"""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-0.5, mood_arousal=0.3, stress=0.2
        )
        assert result == Tone.SAD_SUBDUED

    # --- Stress branch takes priority over positive valence ---

    def test_stress_overrides_positive_valence(self):
        """Even with very positive valence, high stress + N takes priority."""
        interp = make_interpreter(neuroticism=0.9, extraversion=0.9)
        result = interp.get_tone_from_mood(
            mood_valence=0.9, mood_arousal=0.9, stress=0.9
        )
        assert result == Tone.ANXIOUS_STRESSED

    # --- Return type ---

    def test_return_type_is_tone(self):
        interp = make_interpreter()
        result = interp.get_tone_from_mood(0.0, 0.5, 0.2)
        assert isinstance(result, Tone)


# ============================================================================
# MULTI-TRAIT: get_confidence_modifier
# ============================================================================

class TestGetConfidenceModifier:
    """
    Phase R2: DK curve + sigmoid-amplified C/N effects.
    dk = dunning_kruger_confidence(prof, n)
    c_boost = (c - 0.5) * 0.3
    n_penalty = trait_effect(n) * 0.25
    adjusted = dk + c_boost - n_penalty, clamped [0.1, 0.95]
    """

    def test_mid_traits_mid_proficiency(self):
        # DK(0.5, 0.5)=0.42, c_boost=0, n_penalty=0.125 → 0.295
        interp = make_interpreter(conscientiousness=0.5, neuroticism=0.5)
        assert interp.get_confidence_modifier(0.5) == pytest.approx(0.295, abs=0.01)

    def test_high_c_boost(self):
        # DK(0.5, 0.0)=0.42, c_boost=0.15, n_penalty≈0.005 → 0.566
        interp = make_interpreter(conscientiousness=1.0, neuroticism=0.0)
        assert interp.get_confidence_modifier(0.5) == pytest.approx(0.566, abs=0.01)

    def test_low_c_penalty(self):
        # DK(0.5, 0.0)=0.42, c_boost=-0.15, n_penalty≈0.005 → 0.266
        interp = make_interpreter(conscientiousness=0.0, neuroticism=0.0)
        assert interp.get_confidence_modifier(0.5) == pytest.approx(0.266, abs=0.01)

    def test_high_n_penalty(self):
        # DK(0.5, 1.0)=0.42, c_boost=0, n_penalty≈0.246 → 0.175
        interp = make_interpreter(conscientiousness=0.5, neuroticism=1.0)
        assert interp.get_confidence_modifier(0.5) == pytest.approx(0.175, abs=0.01)

    def test_combined_boost_and_penalty(self):
        # DK(0.7, 0.4)=0.67, c_boost=0.09, n_penalty≈0.078 → 0.683
        interp = make_interpreter(conscientiousness=0.8, neuroticism=0.4)
        assert interp.get_confidence_modifier(0.7) == pytest.approx(0.683, abs=0.01)

    # --- Clamping ---

    def test_clamped_to_min(self):
        interp = make_interpreter(conscientiousness=0.0, neuroticism=1.0)
        assert interp.get_confidence_modifier(0.0) == pytest.approx(0.1)

    def test_clamped_to_max(self):
        interp = make_interpreter(conscientiousness=1.0, neuroticism=0.0)
        assert interp.get_confidence_modifier(1.0) == pytest.approx(0.95)

    def test_dk_novice_overconfident(self):
        """Novice (prof=0.1, N=0.3) should be overconfident due to DK effect."""
        interp = make_interpreter(conscientiousness=0.5, neuroticism=0.3)
        conf = interp.get_confidence_modifier(0.1)
        # DK: 0.1 + 0.7*0.25 = 0.275, minus n_penalty
        assert conf > 0.1  # Should exceed raw proficiency

    def test_dk_valley_of_despair(self):
        """Intermediate (prof=0.45) should be underconfident (valley)."""
        interp = make_interpreter(conscientiousness=0.5, neuroticism=0.3)
        conf = interp.get_confidence_modifier(0.45)
        # DK: 0.45 - 0.08 = 0.37 (underconfident)
        assert conf < 0.45

    def test_result_always_in_bounds(self):
        """Sweep across inputs to verify [0.1, 0.95] invariant."""
        for c in [0.0, 0.5, 1.0]:
            for n in [0.0, 0.5, 1.0]:
                interp = make_interpreter(conscientiousness=c, neuroticism=n)
                for prof in [0.0, 0.1, 0.5, 0.9, 1.0]:
                    result = interp.get_confidence_modifier(prof)
                    assert 0.1 <= result <= 0.95, (
                        f"Out of bounds: C={c}, N={n}, prof={prof} -> {result}"
                    )

    def test_return_type_is_float(self):
        result = make_interpreter().get_confidence_modifier(0.5)
        assert isinstance(result, float)


# ============================================================================
# MULTI-TRAIT: get_trait_markers_for_validation
# ============================================================================

class TestGetTraitMarkersForValidation:
    """Checks structure and values of the validation marker dict."""

    def test_returns_dict(self):
        result = make_interpreter().get_trait_markers_for_validation()
        assert isinstance(result, dict)

    def test_has_all_five_trait_keys(self):
        result = make_interpreter().get_trait_markers_for_validation()
        expected_keys = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        assert set(result.keys()) == expected_keys

    # --- Openness markers ---

    def test_openness_high(self):
        result = make_interpreter(openness=0.9).get_trait_markers_for_validation()
        o = result["openness"]
        assert o["level"] == "high"
        assert o["expect_abstract_reasoning"] is True
        assert o["expect_novelty_seeking"] is True
        assert o["elasticity_range"] == (0.5, 0.9)

    def test_openness_moderate(self):
        result = make_interpreter(openness=0.5).get_trait_markers_for_validation()
        o = result["openness"]
        assert o["level"] == "moderate"
        assert o["expect_abstract_reasoning"] is False
        assert o["expect_novelty_seeking"] is False
        assert o["elasticity_range"] == (0.1, 0.5)

    def test_openness_low(self):
        result = make_interpreter(openness=0.2).get_trait_markers_for_validation()
        o = result["openness"]
        assert o["level"] == "low"
        assert o["expect_abstract_reasoning"] is False
        assert o["expect_novelty_seeking"] is False
        assert o["elasticity_range"] == (0.1, 0.5)

    def test_openness_boundary_high(self):
        # 0.7 is NOT > 0.7, so level should be moderate
        result = make_interpreter(openness=0.7).get_trait_markers_for_validation()
        assert result["openness"]["level"] == "moderate"

    def test_openness_boundary_low(self):
        # 0.3 is NOT < 0.3, so level should be moderate
        result = make_interpreter(openness=0.3).get_trait_markers_for_validation()
        assert result["openness"]["level"] == "moderate"

    def test_openness_novelty_boundary(self):
        # 0.6 is NOT > 0.6, so expect_novelty_seeking is False
        result = make_interpreter(openness=0.6).get_trait_markers_for_validation()
        assert result["openness"]["expect_novelty_seeking"] is False

    def test_openness_novelty_above_threshold(self):
        result = make_interpreter(openness=0.61).get_trait_markers_for_validation()
        assert result["openness"]["expect_novelty_seeking"] is True

    # --- Conscientiousness markers ---

    def test_conscientiousness_high(self):
        result = make_interpreter(conscientiousness=0.9).get_trait_markers_for_validation()
        c = result["conscientiousness"]
        assert c["level"] == "high"
        assert c["expect_planning_language"] is True
        assert c["expect_detail_orientation"] is True
        assert c["verbosity_tendency"] == "detailed"

    def test_conscientiousness_moderate(self):
        result = make_interpreter(conscientiousness=0.5).get_trait_markers_for_validation()
        c = result["conscientiousness"]
        assert c["level"] == "moderate"
        assert c["expect_planning_language"] is False
        assert c["expect_detail_orientation"] is False
        assert c["verbosity_tendency"] == "brief"

    def test_conscientiousness_low(self):
        result = make_interpreter(conscientiousness=0.2).get_trait_markers_for_validation()
        c = result["conscientiousness"]
        assert c["level"] == "low"
        assert c["expect_planning_language"] is False
        assert c["verbosity_tendency"] == "brief"

    def test_conscientiousness_boundary_high(self):
        # 0.7 is NOT > 0.7
        result = make_interpreter(conscientiousness=0.7).get_trait_markers_for_validation()
        assert result["conscientiousness"]["level"] == "moderate"
        assert result["conscientiousness"]["expect_planning_language"] is False

    def test_conscientiousness_detail_boundary(self):
        # 0.6 is NOT > 0.6
        result = make_interpreter(conscientiousness=0.6).get_trait_markers_for_validation()
        assert result["conscientiousness"]["expect_detail_orientation"] is False

    def test_conscientiousness_detail_above(self):
        result = make_interpreter(conscientiousness=0.61).get_trait_markers_for_validation()
        assert result["conscientiousness"]["expect_detail_orientation"] is True

    # --- Extraversion markers ---

    def test_extraversion_high(self):
        result = make_interpreter(extraversion=0.9).get_trait_markers_for_validation()
        e = result["extraversion"]
        assert e["level"] == "high"
        assert e["expect_proactive_engagement"] is True
        assert e["response_length_modifier"] == "longer"
        # Phase R2 sigmoid: (trait_effect(0.9) - 0.5) * 0.45 ≈ 0.207, + N(0.5)*0.1 = 0.257
        assert e["disclosure_modifier"] == pytest.approx(0.257, abs=0.01)

    def test_extraversion_moderate(self):
        result = make_interpreter(extraversion=0.5).get_trait_markers_for_validation()
        e = result["extraversion"]
        assert e["level"] == "moderate"
        assert e["expect_proactive_engagement"] is False
        assert e["response_length_modifier"] == "shorter"
        assert e["disclosure_modifier"] == pytest.approx(0.05)

    def test_extraversion_low(self):
        result = make_interpreter(extraversion=0.2).get_trait_markers_for_validation()
        e = result["extraversion"]
        assert e["level"] == "low"
        assert e["expect_proactive_engagement"] is False
        assert e["response_length_modifier"] == "shorter"

    def test_extraversion_boundary_high(self):
        # 0.7 is NOT > 0.7
        result = make_interpreter(extraversion=0.7).get_trait_markers_for_validation()
        assert result["extraversion"]["level"] == "moderate"
        assert result["extraversion"]["expect_proactive_engagement"] is False

    def test_extraversion_length_boundary(self):
        # 0.6 is NOT > 0.6
        result = make_interpreter(extraversion=0.6).get_trait_markers_for_validation()
        assert result["extraversion"]["response_length_modifier"] == "shorter"

    def test_extraversion_length_above(self):
        result = make_interpreter(extraversion=0.61).get_trait_markers_for_validation()
        assert result["extraversion"]["response_length_modifier"] == "longer"

    # --- Agreeableness markers ---

    def test_agreeableness_high(self):
        result = make_interpreter(agreeableness=0.9).get_trait_markers_for_validation()
        a = result["agreeableness"]
        assert a["level"] == "high"
        assert a["expect_validation_before_disagreement"] is True
        assert a["hedging_tendency"] is True
        # directness_reduction = (0.5 - 0.9) * 0.3 = -0.12
        assert a["directness_reduction"] == pytest.approx(-0.12)

    def test_agreeableness_moderate(self):
        result = make_interpreter(agreeableness=0.5).get_trait_markers_for_validation()
        a = result["agreeableness"]
        assert a["level"] == "moderate"
        assert a["expect_validation_before_disagreement"] is False
        assert a["hedging_tendency"] is False
        assert a["directness_reduction"] == pytest.approx(0.0)

    def test_agreeableness_low(self):
        result = make_interpreter(agreeableness=0.2).get_trait_markers_for_validation()
        a = result["agreeableness"]
        assert a["level"] == "low"
        assert a["expect_validation_before_disagreement"] is False
        # directness_reduction = (0.5 - 0.2) * 0.3 = 0.09
        assert a["directness_reduction"] == pytest.approx(0.09)

    def test_agreeableness_boundary_high(self):
        # 0.7 is NOT > 0.7
        result = make_interpreter(agreeableness=0.7).get_trait_markers_for_validation()
        assert result["agreeableness"]["level"] == "moderate"
        assert result["agreeableness"]["expect_validation_before_disagreement"] is False

    def test_agreeableness_hedging_boundary(self):
        # 0.6 is NOT > 0.6
        result = make_interpreter(agreeableness=0.6).get_trait_markers_for_validation()
        assert result["agreeableness"]["hedging_tendency"] is False

    def test_agreeableness_hedging_above(self):
        result = make_interpreter(agreeableness=0.61).get_trait_markers_for_validation()
        assert result["agreeableness"]["hedging_tendency"] is True

    # --- Neuroticism markers ---

    def test_neuroticism_high(self):
        result = make_interpreter(neuroticism=0.8).get_trait_markers_for_validation()
        n = result["neuroticism"]
        assert n["level"] == "high"
        assert n["stress_sensitivity"] == pytest.approx(0.8)
        assert n["mood_stability"] == pytest.approx(0.2)
        assert n["anxiety_baseline"] == pytest.approx(0.8)

    def test_neuroticism_moderate(self):
        result = make_interpreter(neuroticism=0.5).get_trait_markers_for_validation()
        n = result["neuroticism"]
        assert n["level"] == "moderate"
        assert n["stress_sensitivity"] == pytest.approx(0.5)
        assert n["mood_stability"] == pytest.approx(0.5)
        assert n["anxiety_baseline"] == pytest.approx(0.5)

    def test_neuroticism_low(self):
        result = make_interpreter(neuroticism=0.2).get_trait_markers_for_validation()
        n = result["neuroticism"]
        assert n["level"] == "low"
        assert n["stress_sensitivity"] == pytest.approx(0.2)
        assert n["mood_stability"] == pytest.approx(0.8)
        assert n["anxiety_baseline"] == pytest.approx(0.2)

    def test_neuroticism_boundary_high(self):
        # NOTE: neuroticism uses 0.6 (not 0.7) for "high"
        # 0.6 is NOT > 0.6
        result = make_interpreter(neuroticism=0.6).get_trait_markers_for_validation()
        assert result["neuroticism"]["level"] == "moderate"

    def test_neuroticism_just_above_high(self):
        result = make_interpreter(neuroticism=0.61).get_trait_markers_for_validation()
        assert result["neuroticism"]["level"] == "high"

    def test_neuroticism_boundary_low(self):
        # 0.3 is NOT < 0.3
        result = make_interpreter(neuroticism=0.3).get_trait_markers_for_validation()
        assert result["neuroticism"]["level"] == "moderate"

    def test_neuroticism_just_below_low(self):
        result = make_interpreter(neuroticism=0.29).get_trait_markers_for_validation()
        assert result["neuroticism"]["level"] == "low"


# ============================================================================
# Factory function: create_trait_interpreter
# ============================================================================

class TestCreateTraitInterpreter:
    """Test the factory function that creates a TraitInterpreter from a Persona."""

    @staticmethod
    def _make_minimal_persona(**trait_kwargs):
        """Build the smallest valid Persona for testing the factory."""
        from persona_engine.schema.persona_schema import (
            Persona,
            Identity,
            PersonalityProfile,
            BigFiveTraits,
            SchwartzValues,
            CognitiveStyle,
            CommunicationPreferences,
            SocialRole,
            UncertaintyPolicy,
            ClaimPolicy,
            PersonaInvariants,
            DisclosurePolicy,
            DynamicState,
        )

        traits = make_traits(**trait_kwargs)
        return Persona(
            persona_id="test-001",
            version="1.0",
            label="Test Persona",
            identity=Identity(
                age=30,
                location="London, UK",
                education="BSc",
                occupation="Tester",
                background="Testing background",
            ),
            psychology=PersonalityProfile(
                big_five=traits,
                values=SchwartzValues(
                    self_direction=0.5,
                    stimulation=0.5,
                    hedonism=0.5,
                    achievement=0.5,
                    power=0.5,
                    security=0.5,
                    conformity=0.5,
                    tradition=0.5,
                    benevolence=0.5,
                    universalism=0.5,
                ),
                cognitive_style=CognitiveStyle(
                    analytical_intuitive=0.5,
                    systematic_heuristic=0.5,
                    risk_tolerance=0.5,
                    need_for_closure=0.5,
                    cognitive_complexity=0.5,
                ),
                communication=CommunicationPreferences(
                    verbosity=0.5,
                    formality=0.5,
                    directness=0.5,
                    emotional_expressiveness=0.5,
                ),
            ),
            social_roles={
                "default": SocialRole(
                    formality=0.5,
                    directness=0.5,
                    emotional_expressiveness=0.5,
                ),
            },
            uncertainty=UncertaintyPolicy(
                admission_threshold=0.5,
                hedging_frequency=0.5,
                clarification_tendency=0.5,
                knowledge_boundary_strictness=0.5,
            ),
            claim_policy=ClaimPolicy(),
            invariants=PersonaInvariants(
                identity_facts=["Lives in London, UK"],
            ),
            time_scarcity=0.5,
            privacy_sensitivity=0.5,
            disclosure_policy=DisclosurePolicy(
                base_openness=0.5,
                factors={"topic_sensitivity": -0.3},
            ),
            initial_state=DynamicState(
                mood_valence=0.0,
                mood_arousal=0.5,
                fatigue=0.2,
                stress=0.2,
                engagement=0.7,
            ),
        )

    def test_returns_trait_interpreter(self):
        persona = self._make_minimal_persona(openness=0.8)
        interp = create_trait_interpreter(persona)
        assert isinstance(interp, TraitInterpreter)

    def test_uses_persona_big_five(self):
        persona = self._make_minimal_persona(openness=0.85, neuroticism=0.3)
        interp = create_trait_interpreter(persona)
        assert interp.traits.openness == pytest.approx(0.85)
        assert interp.traits.neuroticism == pytest.approx(0.3)

    def test_factory_interpreter_methods_work(self):
        persona = self._make_minimal_persona(
            openness=0.9,
            conscientiousness=0.8,
            extraversion=0.2,
            agreeableness=0.7,
            neuroticism=0.4,
        )
        interp = create_trait_interpreter(persona)
        assert interp.influences_abstract_reasoning() is True
        assert interp.get_novelty_seeking() == pytest.approx(0.9)
        assert interp.influences_proactivity() == pytest.approx(0.32)
        assert interp.get_conflict_avoidance() == pytest.approx(0.7)
        assert interp.get_stress_sensitivity() == pytest.approx(0.4)

    def test_factory_shares_same_traits_object(self):
        """The factory should use the same BigFiveTraits from the persona."""
        persona = self._make_minimal_persona()
        interp = create_trait_interpreter(persona)
        assert interp.traits is persona.psychology.big_five


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases:

    def test_all_zeros(self):
        """All traits at 0.0 -- extremes should not crash."""
        interp = make_interpreter(
            openness=0.0,
            conscientiousness=0.0,
            extraversion=0.0,
            agreeableness=0.0,
            neuroticism=0.0,
        )
        assert interp.get_novelty_seeking() == pytest.approx(0.0)
        assert interp.influences_abstract_reasoning() is False
        assert interp.get_planning_language_tendency() == pytest.approx(0.0)
        assert interp.get_follow_through_likelihood() == pytest.approx(0.0)
        assert interp.influences_proactivity() == pytest.approx(0.2)
        assert interp.get_self_disclosure_modifier() == pytest.approx(-0.217, abs=0.01)
        assert interp.get_enthusiasm_baseline() == pytest.approx(0.2)
        assert interp.get_validation_tendency() == pytest.approx(0.0)
        assert interp.get_conflict_avoidance() == pytest.approx(0.0)
        assert interp.influences_hedging_frequency() == pytest.approx(0.0)
        assert interp.get_stress_sensitivity() == pytest.approx(0.0)
        assert interp.influences_mood_stability() == pytest.approx(1.0)
        assert interp.get_anxiety_baseline() == pytest.approx(0.0)
        assert interp.get_negative_tone_bias() == pytest.approx(0.0)

    def test_all_ones(self):
        """All traits at 1.0 -- extremes should not crash."""
        interp = make_interpreter(
            openness=1.0,
            conscientiousness=1.0,
            extraversion=1.0,
            agreeableness=1.0,
            neuroticism=1.0,
        )
        assert interp.get_novelty_seeking() == pytest.approx(1.0)
        assert interp.influences_abstract_reasoning() is True
        assert interp.get_planning_language_tendency() == pytest.approx(1.0)
        assert interp.get_follow_through_likelihood() == pytest.approx(1.0)
        assert interp.influences_proactivity() == pytest.approx(0.8)
        assert interp.get_self_disclosure_modifier() == pytest.approx(0.317, abs=0.01)
        assert interp.get_enthusiasm_baseline() == pytest.approx(0.7)
        assert interp.get_validation_tendency() == pytest.approx(1.0)
        assert interp.get_conflict_avoidance() == pytest.approx(1.0)
        assert interp.influences_hedging_frequency() == pytest.approx(0.8)
        assert interp.get_stress_sensitivity() == pytest.approx(1.0)
        assert interp.influences_mood_stability() == pytest.approx(0.0)
        assert interp.get_anxiety_baseline() == pytest.approx(1.0)
        assert interp.get_negative_tone_bias() == pytest.approx(0.5)

    def test_all_midpoint(self):
        """All traits at 0.5 -- midpoint sanity check."""
        interp = make_interpreter(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        assert interp.get_novelty_seeking() == pytest.approx(0.5)
        assert interp.get_self_disclosure_modifier() == pytest.approx(0.05)
        assert interp.influences_hedging_frequency() == pytest.approx(0.4)
        assert interp.influences_mood_stability() == pytest.approx(0.5)
        assert interp.get_negative_tone_bias() == pytest.approx(0.25)

    def test_tone_from_mood_with_extreme_negative_valence(self):
        """Very low valence, very low arousal -> SAD_SUBDUED."""
        interp = make_interpreter(neuroticism=0.2)
        result = interp.get_tone_from_mood(
            mood_valence=-1.0, mood_arousal=0.0, stress=0.0
        )
        assert result == Tone.SAD_SUBDUED

    def test_tone_from_mood_with_extreme_positive_valence(self):
        """Very high valence, very high arousal, high E -> EXCITED_ENGAGED."""
        interp = make_interpreter(extraversion=0.9, neuroticism=0.0)
        result = interp.get_tone_from_mood(
            mood_valence=1.0, mood_arousal=1.0, stress=0.0
        )
        assert result == Tone.EXCITED_ENGAGED

    def test_directness_preserves_relative_ordering(self):
        """Higher agreeableness should always produce lower directness for same base."""
        low_a = make_interpreter(agreeableness=0.1)
        high_a = make_interpreter(agreeableness=0.9)
        base = 0.5
        assert high_a.influences_directness(base) < low_a.influences_directness(base)

    def test_elasticity_higher_openness_means_higher_value(self):
        """Higher openness should yield higher elasticity (same confidence)."""
        low_o = make_interpreter(openness=0.2)
        high_o = make_interpreter(openness=0.8)
        base = 0.5
        assert low_o.get_elasticity(base) < high_o.get_elasticity(base)

    def test_confidence_modifier_higher_neuroticism_means_lower_confidence(self):
        """Higher neuroticism should lower confidence modifier."""
        low_n = make_interpreter(neuroticism=0.1, conscientiousness=0.5)
        high_n = make_interpreter(neuroticism=0.9, conscientiousness=0.5)
        prof = 0.5
        assert high_n.get_confidence_modifier(prof) < low_n.get_confidence_modifier(prof)

    def test_trait_markers_all_present_for_varied_persona(self):
        """Markers dict should always contain exactly 5 top-level keys."""
        for _ in range(5):
            import random
            interp = make_interpreter(
                openness=random.random(),
                conscientiousness=random.random(),
                extraversion=random.random(),
                agreeableness=random.random(),
                neuroticism=random.random(),
            )
            markers = interp.get_trait_markers_for_validation()
            assert len(markers) == 5
            assert all(
                k in markers
                for k in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
            )
