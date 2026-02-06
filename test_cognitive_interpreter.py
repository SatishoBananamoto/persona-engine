"""
Tests for CognitiveStyleInterpreter

Comprehensive coverage of all methods in cognitive_interpreter.py,
with boundary-value testing at key thresholds (0.3, 0.6, 0.7).
"""

import pytest

from persona_engine.schema.persona_schema import CognitiveStyle
from persona_engine.behavioral.cognitive_interpreter import (
    CognitiveStyleInterpreter,
    create_cognitive_interpreter,
)


# ============================================================================
# Helpers
# ============================================================================

def make_style(
    analytical_intuitive: float = 0.5,
    systematic_heuristic: float = 0.5,
    risk_tolerance: float = 0.5,
    need_for_closure: float = 0.5,
    cognitive_complexity: float = 0.5,
) -> CognitiveStyle:
    """Create a CognitiveStyle with sensible defaults, overriding as needed."""
    return CognitiveStyle(
        analytical_intuitive=analytical_intuitive,
        systematic_heuristic=systematic_heuristic,
        risk_tolerance=risk_tolerance,
        need_for_closure=need_for_closure,
        cognitive_complexity=cognitive_complexity,
    )


def make_interpreter(**kwargs) -> CognitiveStyleInterpreter:
    """Shortcut to build an interpreter from keyword style params."""
    return CognitiveStyleInterpreter(make_style(**kwargs))


# ============================================================================
# __init__
# ============================================================================

class TestInit:
    def test_stores_style(self):
        style = make_style(analytical_intuitive=0.8)
        interp = CognitiveStyleInterpreter(style)
        assert interp.style is style

    def test_style_fields_accessible(self):
        interp = make_interpreter(risk_tolerance=0.42)
        assert interp.style.risk_tolerance == pytest.approx(0.42)


# ============================================================================
# get_reasoning_approach
# ============================================================================

class TestGetReasoningApproach:
    """Threshold at 0.7 (analytical) and 0.3 (intuitive)."""

    def test_high_analytical(self):
        assert make_interpreter(analytical_intuitive=0.9).get_reasoning_approach() == "analytical"

    def test_exactly_above_0_7(self):
        # 0.71 > 0.7 -> analytical
        assert make_interpreter(analytical_intuitive=0.71).get_reasoning_approach() == "analytical"

    def test_at_0_7_boundary(self):
        # 0.7 is NOT > 0.7, falls through to mixed
        assert make_interpreter(analytical_intuitive=0.7).get_reasoning_approach() == "mixed"

    def test_low_intuitive(self):
        assert make_interpreter(analytical_intuitive=0.1).get_reasoning_approach() == "intuitive"

    def test_exactly_below_0_3(self):
        # 0.29 < 0.3 -> intuitive
        assert make_interpreter(analytical_intuitive=0.29).get_reasoning_approach() == "intuitive"

    def test_at_0_3_boundary(self):
        # 0.3 is NOT < 0.3, falls through to mixed
        assert make_interpreter(analytical_intuitive=0.3).get_reasoning_approach() == "mixed"

    def test_mixed_mid_range(self):
        assert make_interpreter(analytical_intuitive=0.5).get_reasoning_approach() == "mixed"

    def test_mixed_low_end(self):
        assert make_interpreter(analytical_intuitive=0.3).get_reasoning_approach() == "mixed"

    def test_mixed_high_end(self):
        assert make_interpreter(analytical_intuitive=0.7).get_reasoning_approach() == "mixed"


# ============================================================================
# get_rationale_depth
# ============================================================================

class TestGetRationaleDepth:
    """
    Four distinct paths:
      1. analytical (>0.7) + systematic (>0.7) -> 4
      2. analytical (>0.7) + not-systematic (<=0.7) -> 3
      3. intuitive (<0.3) + heuristic (<0.3) -> 1
      4. intuitive (<0.3) + not-heuristic (>=0.3) -> 2
      5. mixed -> 2
    """

    def test_analytical_and_systematic(self):
        # Both high -> depth 4
        interp = make_interpreter(analytical_intuitive=0.9, systematic_heuristic=0.9)
        assert interp.get_rationale_depth() == 4

    def test_analytical_and_heuristic(self):
        # Analytical but not systematic -> depth 3
        interp = make_interpreter(analytical_intuitive=0.8, systematic_heuristic=0.5)
        assert interp.get_rationale_depth() == 3

    def test_analytical_boundary_systematic_boundary(self):
        # analytical_intuitive=0.71 (>0.7) + systematic_heuristic=0.71 (>0.7) -> 4
        interp = make_interpreter(analytical_intuitive=0.71, systematic_heuristic=0.71)
        assert interp.get_rationale_depth() == 4

    def test_analytical_boundary_systematic_at_boundary(self):
        # analytical_intuitive=0.71 (>0.7) + systematic_heuristic=0.7 (NOT >0.7) -> 3
        interp = make_interpreter(analytical_intuitive=0.71, systematic_heuristic=0.7)
        assert interp.get_rationale_depth() == 3

    def test_intuitive_and_heuristic(self):
        # Both low -> depth 1
        interp = make_interpreter(analytical_intuitive=0.1, systematic_heuristic=0.1)
        assert interp.get_rationale_depth() == 1

    def test_intuitive_and_systematic(self):
        # Intuitive but systematic -> depth 2
        interp = make_interpreter(analytical_intuitive=0.2, systematic_heuristic=0.5)
        assert interp.get_rationale_depth() == 2

    def test_intuitive_boundary_heuristic_boundary(self):
        # analytical_intuitive=0.29 (<0.3) + systematic_heuristic=0.29 (<0.3) -> 1
        interp = make_interpreter(analytical_intuitive=0.29, systematic_heuristic=0.29)
        assert interp.get_rationale_depth() == 1

    def test_intuitive_boundary_heuristic_at_boundary(self):
        # analytical_intuitive=0.29 (<0.3) + systematic_heuristic=0.3 (NOT <0.3) -> 2
        interp = make_interpreter(analytical_intuitive=0.29, systematic_heuristic=0.3)
        assert interp.get_rationale_depth() == 2

    def test_mixed_always_2(self):
        # In mixed range, always returns 2 regardless of systematic_heuristic
        for sh in [0.0, 0.5, 1.0]:
            interp = make_interpreter(analytical_intuitive=0.5, systematic_heuristic=sh)
            assert interp.get_rationale_depth() == 2

    def test_range_is_1_to_5(self):
        # Document that valid range is 1-5 (though 5 is never reached by implementation)
        # Min: intuitive + heuristic -> 1
        assert make_interpreter(analytical_intuitive=0.0, systematic_heuristic=0.0).get_rationale_depth() == 1
        # Max: analytical + systematic -> 4
        assert make_interpreter(analytical_intuitive=1.0, systematic_heuristic=1.0).get_rationale_depth() == 4


# ============================================================================
# prefers_systematic_processing
# ============================================================================

class TestPrefersSystematicProcessing:
    """Threshold at 0.6."""

    def test_high_systematic(self):
        assert make_interpreter(systematic_heuristic=0.9).prefers_systematic_processing() is True

    def test_above_threshold(self):
        assert make_interpreter(systematic_heuristic=0.61).prefers_systematic_processing() is True

    def test_at_threshold(self):
        # 0.6 is NOT > 0.6
        assert make_interpreter(systematic_heuristic=0.6).prefers_systematic_processing() is False

    def test_below_threshold(self):
        assert make_interpreter(systematic_heuristic=0.3).prefers_systematic_processing() is False

    def test_zero(self):
        assert make_interpreter(systematic_heuristic=0.0).prefers_systematic_processing() is False


# ============================================================================
# get_decision_time_modifier
# ============================================================================

class TestGetDecisionTimeModifier:
    """Thresholds at 0.7 (extended) and 0.3 (quick)."""

    def test_extended(self):
        assert make_interpreter(systematic_heuristic=0.9).get_decision_time_modifier() == "extended"

    def test_extended_boundary(self):
        assert make_interpreter(systematic_heuristic=0.71).get_decision_time_modifier() == "extended"

    def test_at_0_7_is_moderate(self):
        # 0.7 is NOT > 0.7
        assert make_interpreter(systematic_heuristic=0.7).get_decision_time_modifier() == "moderate"

    def test_quick(self):
        assert make_interpreter(systematic_heuristic=0.1).get_decision_time_modifier() == "quick"

    def test_quick_boundary(self):
        assert make_interpreter(systematic_heuristic=0.29).get_decision_time_modifier() == "quick"

    def test_at_0_3_is_moderate(self):
        # 0.3 is NOT < 0.3
        assert make_interpreter(systematic_heuristic=0.3).get_decision_time_modifier() == "moderate"

    def test_moderate_mid(self):
        assert make_interpreter(systematic_heuristic=0.5).get_decision_time_modifier() == "moderate"


# ============================================================================
# get_risk_stance_modifier
# ============================================================================

class TestGetRiskStanceModifier:
    """Threshold at base_confidence < 0.4."""

    def test_low_confidence_high_risk(self):
        # base < 0.4: return base + risk * 0.3
        interp = make_interpreter(risk_tolerance=1.0)
        result = interp.get_risk_stance_modifier(0.2)
        assert result == pytest.approx(0.2 + 1.0 * 0.3)  # 0.5

    def test_low_confidence_low_risk(self):
        interp = make_interpreter(risk_tolerance=0.0)
        result = interp.get_risk_stance_modifier(0.2)
        assert result == pytest.approx(0.2)

    def test_low_confidence_mid_risk(self):
        interp = make_interpreter(risk_tolerance=0.5)
        result = interp.get_risk_stance_modifier(0.3)
        assert result == pytest.approx(0.3 + 0.5 * 0.3)  # 0.45

    def test_high_confidence_high_risk(self):
        # base >= 0.4: return base + risk * 0.1
        interp = make_interpreter(risk_tolerance=1.0)
        result = interp.get_risk_stance_modifier(0.8)
        assert result == pytest.approx(0.8 + 1.0 * 0.1)  # 0.9

    def test_high_confidence_low_risk(self):
        interp = make_interpreter(risk_tolerance=0.0)
        result = interp.get_risk_stance_modifier(0.8)
        assert result == pytest.approx(0.8)

    def test_at_boundary_0_4(self):
        # 0.4 is NOT < 0.4, so uses the high-confidence branch
        interp = make_interpreter(risk_tolerance=0.5)
        result = interp.get_risk_stance_modifier(0.4)
        assert result == pytest.approx(0.4 + 0.5 * 0.1)  # 0.45

    def test_just_below_boundary(self):
        interp = make_interpreter(risk_tolerance=0.5)
        result = interp.get_risk_stance_modifier(0.39)
        assert result == pytest.approx(0.39 + 0.5 * 0.3)  # 0.54

    def test_zero_base_zero_risk(self):
        interp = make_interpreter(risk_tolerance=0.0)
        result = interp.get_risk_stance_modifier(0.0)
        assert result == pytest.approx(0.0)


# ============================================================================
# influences_uncertainty_action
# ============================================================================

class TestInfluencesUncertaintyAction:
    """
    Three confidence tiers:
      >0.7: answer
      >0.4: risk>0.6 -> answer, else -> hedge
      <=0.4: closure>0.6 -> ask, risk<0.3 -> refuse, else -> hedge
    """

    # --- High confidence (>0.7) ---

    def test_high_confidence_returns_answer(self):
        interp = make_interpreter(risk_tolerance=0.0, need_for_closure=0.0)
        assert interp.influences_uncertainty_action(0.9) == "answer"

    def test_at_0_71_returns_answer(self):
        interp = make_interpreter()
        assert interp.influences_uncertainty_action(0.71) == "answer"

    def test_at_0_7_is_not_high_confidence(self):
        # 0.7 is NOT > 0.7, falls to middle tier
        interp = make_interpreter(risk_tolerance=0.0)
        assert interp.influences_uncertainty_action(0.7) == "hedge"

    # --- Medium confidence (>0.4 and <=0.7) ---

    def test_medium_confidence_high_risk_returns_answer(self):
        interp = make_interpreter(risk_tolerance=0.8)
        assert interp.influences_uncertainty_action(0.5) == "answer"

    def test_medium_confidence_risk_above_0_6(self):
        interp = make_interpreter(risk_tolerance=0.61)
        assert interp.influences_uncertainty_action(0.5) == "answer"

    def test_medium_confidence_risk_at_0_6(self):
        # 0.6 is NOT > 0.6
        interp = make_interpreter(risk_tolerance=0.6)
        assert interp.influences_uncertainty_action(0.5) == "hedge"

    def test_medium_confidence_low_risk_returns_hedge(self):
        interp = make_interpreter(risk_tolerance=0.3)
        assert interp.influences_uncertainty_action(0.5) == "hedge"

    def test_at_0_41_is_medium_confidence(self):
        interp = make_interpreter(risk_tolerance=0.0)
        assert interp.influences_uncertainty_action(0.41) == "hedge"

    # --- Low confidence (<=0.4) ---

    def test_low_confidence_high_closure_returns_ask(self):
        interp = make_interpreter(need_for_closure=0.8, risk_tolerance=0.5)
        assert interp.influences_uncertainty_action(0.2) == "ask"

    def test_low_confidence_closure_above_0_6(self):
        interp = make_interpreter(need_for_closure=0.61, risk_tolerance=0.5)
        assert interp.influences_uncertainty_action(0.3) == "ask"

    def test_low_confidence_closure_at_0_6(self):
        # 0.6 is NOT > 0.6, falls through to risk check
        interp = make_interpreter(need_for_closure=0.6, risk_tolerance=0.1)
        assert interp.influences_uncertainty_action(0.3) == "refuse"

    def test_low_confidence_low_risk_returns_refuse(self):
        interp = make_interpreter(need_for_closure=0.3, risk_tolerance=0.1)
        assert interp.influences_uncertainty_action(0.2) == "refuse"

    def test_low_confidence_risk_below_0_3(self):
        interp = make_interpreter(need_for_closure=0.3, risk_tolerance=0.29)
        assert interp.influences_uncertainty_action(0.2) == "refuse"

    def test_low_confidence_risk_at_0_3_returns_hedge(self):
        # 0.3 is NOT < 0.3, falls through to hedge
        interp = make_interpreter(need_for_closure=0.3, risk_tolerance=0.3)
        assert interp.influences_uncertainty_action(0.2) == "hedge"

    def test_low_confidence_moderate_risk_moderate_closure_returns_hedge(self):
        interp = make_interpreter(need_for_closure=0.4, risk_tolerance=0.5)
        assert interp.influences_uncertainty_action(0.2) == "hedge"

    def test_at_confidence_0_4(self):
        # 0.4 is NOT > 0.4, so low-confidence branch
        interp = make_interpreter(need_for_closure=0.8, risk_tolerance=0.5)
        assert interp.influences_uncertainty_action(0.4) == "ask"

    def test_at_confidence_0_0(self):
        interp = make_interpreter(need_for_closure=0.8, risk_tolerance=0.5)
        assert interp.influences_uncertainty_action(0.0) == "ask"

    # --- Priority: closure check happens before risk check ---

    def test_closure_takes_priority_over_low_risk(self):
        # Both closure>0.6 and risk<0.3, closure should win
        interp = make_interpreter(need_for_closure=0.9, risk_tolerance=0.1)
        assert interp.influences_uncertainty_action(0.2) == "ask"


# ============================================================================
# get_ambiguity_tolerance
# ============================================================================

class TestGetAmbiguityTolerance:
    def test_high_closure_low_tolerance(self):
        interp = make_interpreter(need_for_closure=0.9)
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.1)

    def test_low_closure_high_tolerance(self):
        interp = make_interpreter(need_for_closure=0.1)
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.9)

    def test_zero_closure(self):
        interp = make_interpreter(need_for_closure=0.0)
        assert interp.get_ambiguity_tolerance() == pytest.approx(1.0)

    def test_full_closure(self):
        interp = make_interpreter(need_for_closure=1.0)
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.0)

    def test_mid_closure(self):
        interp = make_interpreter(need_for_closure=0.5)
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.5)


# ============================================================================
# prefers_definite_answers
# ============================================================================

class TestPrefersDefiniteAnswers:
    """Threshold at 0.6."""

    def test_high_closure(self):
        assert make_interpreter(need_for_closure=0.9).prefers_definite_answers() is True

    def test_above_threshold(self):
        assert make_interpreter(need_for_closure=0.61).prefers_definite_answers() is True

    def test_at_threshold(self):
        # 0.6 is NOT > 0.6
        assert make_interpreter(need_for_closure=0.6).prefers_definite_answers() is False

    def test_low_closure(self):
        assert make_interpreter(need_for_closure=0.2).prefers_definite_answers() is False

    def test_zero(self):
        assert make_interpreter(need_for_closure=0.0).prefers_definite_answers() is False


# ============================================================================
# get_nuance_capacity
# ============================================================================

class TestGetNuanceCapacity:
    """Thresholds at 0.7 (high) and 0.3 (low)."""

    def test_high(self):
        assert make_interpreter(cognitive_complexity=0.9).get_nuance_capacity() == "high"

    def test_above_0_7(self):
        assert make_interpreter(cognitive_complexity=0.71).get_nuance_capacity() == "high"

    def test_at_0_7_is_moderate(self):
        assert make_interpreter(cognitive_complexity=0.7).get_nuance_capacity() == "moderate"

    def test_low(self):
        assert make_interpreter(cognitive_complexity=0.1).get_nuance_capacity() == "low"

    def test_below_0_3(self):
        assert make_interpreter(cognitive_complexity=0.29).get_nuance_capacity() == "low"

    def test_at_0_3_is_moderate(self):
        assert make_interpreter(cognitive_complexity=0.3).get_nuance_capacity() == "moderate"

    def test_moderate_mid(self):
        assert make_interpreter(cognitive_complexity=0.5).get_nuance_capacity() == "moderate"


# ============================================================================
# should_acknowledge_tradeoffs
# ============================================================================

class TestShouldAcknowledgeTradeoffs:
    """Threshold at 0.6."""

    def test_high_complexity(self):
        assert make_interpreter(cognitive_complexity=0.9).should_acknowledge_tradeoffs() is True

    def test_above_threshold(self):
        assert make_interpreter(cognitive_complexity=0.61).should_acknowledge_tradeoffs() is True

    def test_at_threshold(self):
        # 0.6 is NOT > 0.6
        assert make_interpreter(cognitive_complexity=0.6).should_acknowledge_tradeoffs() is False

    def test_low_complexity(self):
        assert make_interpreter(cognitive_complexity=0.2).should_acknowledge_tradeoffs() is False

    def test_zero(self):
        assert make_interpreter(cognitive_complexity=0.0).should_acknowledge_tradeoffs() is False


# ============================================================================
# get_stance_complexity_level
# ============================================================================

class TestGetStanceComplexityLevel:
    """Thresholds at 0.7 (level 3) and 0.3 (level 1)."""

    def test_level_3(self):
        assert make_interpreter(cognitive_complexity=0.9).get_stance_complexity_level() == 3

    def test_above_0_7(self):
        assert make_interpreter(cognitive_complexity=0.71).get_stance_complexity_level() == 3

    def test_at_0_7_is_level_2(self):
        assert make_interpreter(cognitive_complexity=0.7).get_stance_complexity_level() == 2

    def test_level_1(self):
        assert make_interpreter(cognitive_complexity=0.1).get_stance_complexity_level() == 1

    def test_below_0_3(self):
        assert make_interpreter(cognitive_complexity=0.29).get_stance_complexity_level() == 1

    def test_at_0_3_is_level_2(self):
        assert make_interpreter(cognitive_complexity=0.3).get_stance_complexity_level() == 2

    def test_level_2_mid(self):
        assert make_interpreter(cognitive_complexity=0.5).get_stance_complexity_level() == 2


# ============================================================================
# get_elasticity_from_cognitive_style
# ============================================================================

class TestGetElasticityFromCognitiveStyle:
    """Formula: complexity * 0.6 + (1 - closure) * 0.4"""

    def test_high_complexity_low_closure(self):
        interp = make_interpreter(cognitive_complexity=1.0, need_for_closure=0.0)
        expected = 1.0 * 0.6 + (1.0 - 0.0) * 0.4  # 0.6 + 0.4 = 1.0
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_low_complexity_high_closure(self):
        interp = make_interpreter(cognitive_complexity=0.0, need_for_closure=1.0)
        expected = 0.0 * 0.6 + (1.0 - 1.0) * 0.4  # 0.0
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_mid_values(self):
        interp = make_interpreter(cognitive_complexity=0.5, need_for_closure=0.5)
        expected = 0.5 * 0.6 + (1.0 - 0.5) * 0.4  # 0.3 + 0.2 = 0.5
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_high_both(self):
        interp = make_interpreter(cognitive_complexity=0.8, need_for_closure=0.8)
        expected = 0.8 * 0.6 + (1.0 - 0.8) * 0.4  # 0.48 + 0.08 = 0.56
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_low_both(self):
        interp = make_interpreter(cognitive_complexity=0.2, need_for_closure=0.2)
        expected = 0.2 * 0.6 + (1.0 - 0.2) * 0.4  # 0.12 + 0.32 = 0.44
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_zero_both(self):
        interp = make_interpreter(cognitive_complexity=0.0, need_for_closure=0.0)
        expected = 0.0 + 1.0 * 0.4  # 0.4
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(expected)

    def test_result_is_float(self):
        interp = make_interpreter(cognitive_complexity=0.5, need_for_closure=0.5)
        result = interp.get_elasticity_from_cognitive_style()
        assert isinstance(result, float)


# ============================================================================
# get_confidence_adjustment
# ============================================================================

class TestGetConfidenceAdjustment:
    """
    analytical_penalty: -0.05 if analytical >= 0.5 else 0
    closure_boost: +0.08 if closure >= 0.7 else 0
    Clamped to [0.1, 0.95]
    """

    # --- Analytical penalty ---

    def test_analytical_penalty_applied(self):
        # analytical=0.8 (>=0.5), closure=0.3 (<0.7)
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5 - 0.05)  # 0.45

    def test_no_analytical_penalty(self):
        # analytical=0.3 (<0.5), closure=0.3 (<0.7)
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5)

    def test_analytical_at_boundary_0_5(self):
        # analytical=0.5 (>=0.5) -> penalty applies
        interp = make_interpreter(analytical_intuitive=0.5, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5 - 0.05)

    def test_analytical_just_below_0_5(self):
        # analytical=0.49 (<0.5) -> no penalty
        interp = make_interpreter(analytical_intuitive=0.49, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5)

    # --- Closure boost ---

    def test_closure_boost_applied(self):
        # analytical=0.3 (<0.5), closure=0.8 (>=0.7)
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.8)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5 + 0.08)  # 0.58

    def test_no_closure_boost(self):
        # analytical=0.3 (<0.5), closure=0.5 (<0.7)
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.5)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5)

    def test_closure_at_boundary_0_7(self):
        # closure=0.7 (>=0.7) -> boost applies
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.7)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5 + 0.08)

    def test_closure_just_below_0_7(self):
        # closure=0.69 (<0.7) -> no boost
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.69)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5)

    # --- Both applied ---

    def test_both_penalty_and_boost(self):
        # analytical=0.8 (>=0.5), closure=0.8 (>=0.7)
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.8)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5 - 0.05 + 0.08)  # 0.53

    def test_neither_penalty_nor_boost(self):
        interp = make_interpreter(analytical_intuitive=0.2, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.5)
        assert result == pytest.approx(0.5)

    # --- Clamping ---

    def test_clamped_to_min(self):
        # Very low base + penalty should clamp to 0.1
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.0)
        # 0.0 - 0.05 = -0.05, clamped to 0.1
        assert result == pytest.approx(0.1)

    def test_clamped_to_max(self):
        # Very high base + boost should clamp to 0.95
        interp = make_interpreter(analytical_intuitive=0.3, need_for_closure=0.8)
        result = interp.get_confidence_adjustment(1.0)
        # 1.0 + 0.08 = 1.08, clamped to 0.95
        assert result == pytest.approx(0.95)

    def test_max_clamp_with_both(self):
        # High base with net positive adjustment
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.8)
        result = interp.get_confidence_adjustment(0.95)
        # 0.95 - 0.05 + 0.08 = 0.98, clamped to 0.95
        assert result == pytest.approx(0.95)

    def test_min_clamp_exact(self):
        # base=0.15 with penalty: 0.15 - 0.05 = 0.10 -> exactly at min, not clamped
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.15)
        assert result == pytest.approx(0.1)

    def test_just_above_min(self):
        # base=0.16 with penalty: 0.16 - 0.05 = 0.11 -> above min
        interp = make_interpreter(analytical_intuitive=0.8, need_for_closure=0.3)
        result = interp.get_confidence_adjustment(0.16)
        assert result == pytest.approx(0.11)


# ============================================================================
# get_cognitive_markers_for_validation
# ============================================================================

class TestGetCognitiveMarkersForValidation:

    def test_returns_dict(self):
        interp = make_interpreter()
        result = interp.get_cognitive_markers_for_validation()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        interp = make_interpreter()
        result = interp.get_cognitive_markers_for_validation()
        expected_keys = {
            "reasoning_approach",
            "rationale_depth",
            "systematic_preference",
            "risk_tolerance",
            "ambiguity_tolerance",
            "nuance_capacity",
            "should_acknowledge_tradeoffs",
            "stance_complexity_level",
        }
        assert set(result.keys()) == expected_keys

    def test_reasoning_approach_value(self):
        interp = make_interpreter(analytical_intuitive=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["reasoning_approach"] == "analytical"

    def test_rationale_depth_value(self):
        interp = make_interpreter(analytical_intuitive=0.9, systematic_heuristic=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["rationale_depth"] == 4

    def test_systematic_preference_value(self):
        interp = make_interpreter(systematic_heuristic=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["systematic_preference"] is True

    def test_risk_tolerance_structure(self):
        interp = make_interpreter(risk_tolerance=0.8)
        result = interp.get_cognitive_markers_for_validation()
        rt = result["risk_tolerance"]
        assert isinstance(rt, dict)
        assert "level" in rt
        assert "affects_low_confidence_stances" in rt
        assert rt["level"] == pytest.approx(0.8)
        assert rt["affects_low_confidence_stances"] is True

    def test_risk_tolerance_low(self):
        interp = make_interpreter(risk_tolerance=0.3)
        result = interp.get_cognitive_markers_for_validation()
        rt = result["risk_tolerance"]
        assert rt["level"] == pytest.approx(0.3)
        assert rt["affects_low_confidence_stances"] is False

    def test_risk_tolerance_at_boundary(self):
        # 0.6 is NOT > 0.6
        interp = make_interpreter(risk_tolerance=0.6)
        result = interp.get_cognitive_markers_for_validation()
        assert result["risk_tolerance"]["affects_low_confidence_stances"] is False

    def test_ambiguity_tolerance_value(self):
        interp = make_interpreter(need_for_closure=0.3)
        result = interp.get_cognitive_markers_for_validation()
        assert result["ambiguity_tolerance"] == pytest.approx(0.7)

    def test_nuance_capacity_value(self):
        interp = make_interpreter(cognitive_complexity=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["nuance_capacity"] == "high"

    def test_should_acknowledge_tradeoffs_value(self):
        interp = make_interpreter(cognitive_complexity=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["should_acknowledge_tradeoffs"] is True

    def test_stance_complexity_level_value(self):
        interp = make_interpreter(cognitive_complexity=0.9)
        result = interp.get_cognitive_markers_for_validation()
        assert result["stance_complexity_level"] == 3

    def test_all_markers_consistent_analytical_profile(self):
        """Verify internal consistency for a fully analytical persona."""
        interp = make_interpreter(
            analytical_intuitive=0.9,
            systematic_heuristic=0.9,
            risk_tolerance=0.2,
            need_for_closure=0.8,
            cognitive_complexity=0.8,
        )
        markers = interp.get_cognitive_markers_for_validation()
        assert markers["reasoning_approach"] == "analytical"
        assert markers["rationale_depth"] == 4
        assert markers["systematic_preference"] is True
        assert markers["risk_tolerance"]["affects_low_confidence_stances"] is False
        assert markers["ambiguity_tolerance"] == pytest.approx(0.2)
        assert markers["nuance_capacity"] == "high"
        assert markers["should_acknowledge_tradeoffs"] is True
        assert markers["stance_complexity_level"] == 3

    def test_all_markers_consistent_intuitive_profile(self):
        """Verify internal consistency for a fully intuitive persona."""
        interp = make_interpreter(
            analytical_intuitive=0.1,
            systematic_heuristic=0.1,
            risk_tolerance=0.9,
            need_for_closure=0.2,
            cognitive_complexity=0.2,
        )
        markers = interp.get_cognitive_markers_for_validation()
        assert markers["reasoning_approach"] == "intuitive"
        assert markers["rationale_depth"] == 1
        assert markers["systematic_preference"] is False
        assert markers["risk_tolerance"]["affects_low_confidence_stances"] is True
        assert markers["ambiguity_tolerance"] == pytest.approx(0.8)
        assert markers["nuance_capacity"] == "low"
        assert markers["should_acknowledge_tradeoffs"] is False
        assert markers["stance_complexity_level"] == 1


# ============================================================================
# create_cognitive_interpreter factory function
# ============================================================================

class TestCreateCognitiveInterpreter:
    """Test the factory function that creates an interpreter from a Persona."""

    def _make_minimal_persona(self, **style_kwargs):
        """Build the smallest valid Persona that lets us test the factory."""
        from persona_engine.schema.persona_schema import (
            Persona,
            Identity,
            PersonalityProfile,
            BigFiveTraits,
            SchwartzValues,
            CommunicationPreferences,
            SocialRole,
            UncertaintyPolicy,
            ClaimPolicy,
            PersonaInvariants,
            DisclosurePolicy,
            DynamicState,
        )

        style = make_style(**style_kwargs)
        persona = Persona(
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
                big_five=BigFiveTraits(
                    openness=0.5,
                    conscientiousness=0.5,
                    extraversion=0.5,
                    agreeableness=0.5,
                    neuroticism=0.5,
                ),
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
                cognitive_style=style,
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
        return persona

    def test_returns_interpreter(self):
        persona = self._make_minimal_persona(analytical_intuitive=0.8)
        interp = create_cognitive_interpreter(persona)
        assert isinstance(interp, CognitiveStyleInterpreter)

    def test_uses_persona_cognitive_style(self):
        persona = self._make_minimal_persona(analytical_intuitive=0.85)
        interp = create_cognitive_interpreter(persona)
        assert interp.style.analytical_intuitive == pytest.approx(0.85)

    def test_factory_interpreter_methods_work(self):
        persona = self._make_minimal_persona(
            analytical_intuitive=0.9,
            systematic_heuristic=0.9,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            cognitive_complexity=0.5,
        )
        interp = create_cognitive_interpreter(persona)
        assert interp.get_reasoning_approach() == "analytical"
        assert interp.get_rationale_depth() == 4
        assert interp.prefers_systematic_processing() is True


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases:

    def test_all_zeros(self):
        """All cognitive style fields at 0.0."""
        interp = make_interpreter(
            analytical_intuitive=0.0,
            systematic_heuristic=0.0,
            risk_tolerance=0.0,
            need_for_closure=0.0,
            cognitive_complexity=0.0,
        )
        assert interp.get_reasoning_approach() == "intuitive"
        assert interp.get_rationale_depth() == 1
        assert interp.prefers_systematic_processing() is False
        assert interp.get_decision_time_modifier() == "quick"
        assert interp.get_ambiguity_tolerance() == pytest.approx(1.0)
        assert interp.prefers_definite_answers() is False
        assert interp.get_nuance_capacity() == "low"
        assert interp.should_acknowledge_tradeoffs() is False
        assert interp.get_stance_complexity_level() == 1
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(0.4)

    def test_all_ones(self):
        """All cognitive style fields at 1.0."""
        interp = make_interpreter(
            analytical_intuitive=1.0,
            systematic_heuristic=1.0,
            risk_tolerance=1.0,
            need_for_closure=1.0,
            cognitive_complexity=1.0,
        )
        assert interp.get_reasoning_approach() == "analytical"
        assert interp.get_rationale_depth() == 4
        assert interp.prefers_systematic_processing() is True
        assert interp.get_decision_time_modifier() == "extended"
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.0)
        assert interp.prefers_definite_answers() is True
        assert interp.get_nuance_capacity() == "high"
        assert interp.should_acknowledge_tradeoffs() is True
        assert interp.get_stance_complexity_level() == 3
        assert interp.get_elasticity_from_cognitive_style() == pytest.approx(0.6)

    def test_all_midpoint(self):
        """All fields at 0.5 should produce moderate/mixed results."""
        interp = make_interpreter(
            analytical_intuitive=0.5,
            systematic_heuristic=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            cognitive_complexity=0.5,
        )
        assert interp.get_reasoning_approach() == "mixed"
        assert interp.get_rationale_depth() == 2
        assert interp.prefers_systematic_processing() is False
        assert interp.get_decision_time_modifier() == "moderate"
        assert interp.get_ambiguity_tolerance() == pytest.approx(0.5)
        assert interp.prefers_definite_answers() is False
        assert interp.get_nuance_capacity() == "moderate"
        assert interp.should_acknowledge_tradeoffs() is False
        assert interp.get_stance_complexity_level() == 2

    def test_risk_stance_modifier_preserves_relative_ordering(self):
        """Higher risk tolerance should always produce higher modifier."""
        low_risk = make_interpreter(risk_tolerance=0.1)
        high_risk = make_interpreter(risk_tolerance=0.9)
        base = 0.3
        assert low_risk.get_risk_stance_modifier(base) < high_risk.get_risk_stance_modifier(base)

    def test_confidence_adjustment_return_type(self):
        interp = make_interpreter()
        result = interp.get_confidence_adjustment(0.5)
        assert isinstance(result, float)

    def test_confidence_adjustment_always_in_bounds(self):
        """No matter what inputs, result is in [0.1, 0.95]."""
        for ai in [0.0, 0.5, 1.0]:
            for nfc in [0.0, 0.7, 1.0]:
                interp = make_interpreter(analytical_intuitive=ai, need_for_closure=nfc)
                for base in [0.0, 0.1, 0.5, 0.9, 1.0]:
                    result = interp.get_confidence_adjustment(base)
                    assert 0.1 <= result <= 0.95, (
                        f"Out of bounds: ai={ai}, nfc={nfc}, base={base} -> {result}"
                    )
