"""
Comprehensive tests for persona_engine.behavioral.constraint_safety module.

Covers:
- apply_response_pattern_safely (must_avoid, topic sensitivity, privacy filter, disclosure detection)
- validate_stance_against_invariants (valid, single violation, multiple violations)
- clamp_disclosure_to_constraints (within bounds, base privacy, topic privacy, equal)
"""

import pytest

from persona_engine.schema.persona_schema import ResponsePattern, TopicSensitivity
from persona_engine.behavioral.constraint_safety import (
    apply_response_pattern_safely,
    validate_stance_against_invariants,
    clamp_disclosure_to_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pattern(trigger: str, response: str, emotionality: float = 0.5) -> ResponsePattern:
    """Shortcut to build a ResponsePattern."""
    return ResponsePattern(trigger=trigger, response=response, emotionality=emotionality)


def _sensitivity(topic: str, sensitivity: float) -> TopicSensitivity:
    """Shortcut to build a TopicSensitivity."""
    return TopicSensitivity(topic=topic, sensitivity=sensitivity)


# ===========================================================================
# apply_response_pattern_safely
# ===========================================================================


class TestMustAvoidHardBlock:
    """must_avoid triggers an immediate hard block."""

    def test_pattern_blocked_when_response_contains_avoided_topic(self):
        pattern = _pattern("greeting", "Let me tell you about politics today", 0.6)
        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=["politics"],
            topic_context="",
        )

        assert result["pattern_blocked"] is True
        assert "politics" in result["block_reason"]
        assert result["pattern_trigger"] == "greeting"
        # Hard block means NO tone/content keys
        assert "pattern_triggered" not in result
        assert "tone_adjustment" not in result
        assert "suggested_content" not in result

    def test_must_avoid_is_case_insensitive(self):
        pattern = _pattern("ask", "My EMPLOYER is great", 0.5)
        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=["employer"],
        )
        assert result["pattern_blocked"] is True

    def test_first_matching_must_avoid_wins(self):
        """When multiple must_avoid topics match, the first one is reported."""
        pattern = _pattern("chat", "Politics and religion discussion", 0.5)
        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=["politics", "religion"],
        )
        assert result["pattern_blocked"] is True
        assert "politics" in result["block_reason"]

    def test_no_match_in_must_avoid_allows_pattern(self):
        pattern = _pattern("greet", "Hello, how are you?", 0.4)
        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=["politics", "religion"],
        )
        assert "pattern_blocked" not in result
        assert result["pattern_triggered"] is True


class TestTopicSensitivityWithDisclosure:
    """Sensitive topic detected AND pattern suggests personal disclosure."""

    def test_disclosure_boost_constrained_by_sensitivity(self):
        """High sensitivity topic should reduce disclosure boost."""
        pattern = _pattern("empathy", "Let me share my personal experience", 0.8)
        sensitivity = _sensitivity("mental health", 0.7)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sensitivity],
            must_avoid=[],
            topic_context="We are discussing mental health issues",
        )

        # max_boost = (1.0 - 0.7) * 0.3 = 0.09
        # emotionality * 0.2 = 0.8 * 0.2 = 0.16
        # actual_boost = min(0.16, 0.09) = 0.09
        assert result["disclosure_boost"] == pytest.approx(0.09)
        assert result["sensitivity_constraint"] == "mental health"
        assert result["sensitivity_level"] == 0.7
        # pattern_triggered should always be present when not blocked
        assert result["pattern_triggered"] is True
        assert result["trigger"] == "empathy"
        assert result["tone_adjustment"] == 0.8
        assert result["arousal_boost"] == pytest.approx(0.8 * 0.3)
        assert result["suggested_content"] == "Let me share my personal experience"

    def test_high_sensitivity_sets_pattern_constrained_flag(self):
        """When actual_boost < emotionality * 0.1, pattern_constrained is set."""
        # Need: actual_boost < emotionality * 0.1
        # actual_boost = min(emotionality*0.2, (1-sensitivity)*0.3)
        # With sensitivity=0.9, emotionality=0.8:
        #   max_boost = (1-0.9)*0.3 = 0.03
        #   emotionality*0.2 = 0.16  -> actual = 0.03
        #   threshold = 0.8*0.1 = 0.08
        #   0.03 < 0.08 -> True
        pattern = _pattern("deep", "I want to share my story with you", 0.8)
        sensitivity = _sensitivity("trauma", 0.9)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sensitivity],
            must_avoid=[],
            topic_context="Let's talk about trauma recovery",
        )

        assert result["pattern_constrained"] is True
        assert result["disclosure_boost"] == pytest.approx(0.03)

    def test_low_sensitivity_no_pattern_constrained_flag(self):
        """Low sensitivity should NOT set pattern_constrained."""
        # sensitivity=0.1, emotionality=0.5:
        #   max_boost = (1-0.1)*0.3 = 0.27
        #   emotionality*0.2 = 0.10  -> actual = 0.10
        #   threshold = 0.5*0.1 = 0.05
        #   0.10 >= 0.05 -> no constrained flag
        pattern = _pattern("relate", "Let me share my experience", 0.5)
        sensitivity = _sensitivity("career", 0.1)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sensitivity],
            must_avoid=[],
            topic_context="Discussing career changes",
        )

        assert "pattern_constrained" not in result
        assert result["disclosure_boost"] == pytest.approx(0.10)
        assert result["sensitivity_constraint"] == "career"

    def test_non_disclosure_pattern_on_sensitive_topic_no_boost(self):
        """Sensitive topic, but pattern doesn't suggest disclosure -> no disclosure_boost."""
        pattern = _pattern("inform", "Here are some facts about the topic", 0.6)
        sensitivity = _sensitivity("finances", 0.5)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sensitivity],
            must_avoid=[],
            topic_context="Let's discuss finances and budgets",
        )

        assert "disclosure_boost" not in result
        assert "sensitivity_constraint" not in result
        # Standard modifications still present
        assert result["pattern_triggered"] is True
        assert result["tone_adjustment"] == 0.6

    def test_only_first_matching_sensitivity_is_used(self):
        """The loop breaks after the first matching sensitivity topic."""
        pattern = _pattern("share", "I want to share my personal thoughts", 0.7)
        sens1 = _sensitivity("health", 0.8)
        sens2 = _sensitivity("health", 0.3)  # same topic, different level

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sens1, sens2],
            must_avoid=[],
            topic_context="Talking about health matters",
        )

        # Should use sens1 (sensitivity=0.8), not sens2
        assert result["sensitivity_level"] == 0.8


class TestPrivacyFilterNoSensitiveTopic:
    """No sensitive topic matches -> privacy filter path (else branch of for/else)."""

    def test_disclosure_pattern_exceeds_privacy_filter(self):
        """Proposed disclosure exceeds max allowed -> constrained."""
        # base_disclosure=0.6, emotionality=0.8
        # disclosure_boost = 0.8 * 0.2 = 0.16
        # proposed = 0.6 + 0.16 = 0.76
        # max_allowed = 1.0 - 0.5 = 0.5
        # 0.76 > 0.5 -> constrained
        # actual_boost = max(0.0, 0.5 - 0.6) = max(0.0, -0.1) = 0.0
        pattern = _pattern("open_up", "Let me tell you my personal story", 0.8)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.6,
            privacy_filter=0.5,
            topic_sensitivities=[],
            must_avoid=[],
            topic_context="general chat",
        )

        assert result["pattern_constrained"] is True
        assert result["disclosure_boost"] == pytest.approx(0.0)
        assert "Privacy filter" in result["constraint_reason"]
        assert "0.50" in result["constraint_reason"]
        assert result["pattern_triggered"] is True

    def test_disclosure_pattern_within_privacy_bounds(self):
        """Proposed disclosure within bounds -> allowed with full boost."""
        # base_disclosure=0.2, emotionality=0.5
        # disclosure_boost = 0.5 * 0.2 = 0.1
        # proposed = 0.2 + 0.1 = 0.3
        # max_allowed = 1.0 - 0.2 = 0.8
        # 0.3 <= 0.8 -> allowed
        pattern = _pattern("connect", "I want to share my experience with you", 0.5)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.2,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=[],
            topic_context="casual conversation",
        )

        assert result["disclosure_boost"] == pytest.approx(0.1)
        assert "pattern_constrained" not in result
        assert "constraint_reason" not in result
        assert result["pattern_triggered"] is True

    def test_non_disclosure_pattern_standard_modifications_only(self):
        """Non-disclosure pattern -> no disclosure_boost, only standard modifications."""
        pattern = _pattern("explain", "Here is a detailed explanation", 0.6)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.4,
            topic_sensitivities=[],
            must_avoid=[],
            topic_context="technical question",
        )

        assert "disclosure_boost" not in result
        assert "pattern_constrained" not in result
        assert result["pattern_triggered"] is True
        assert result["trigger"] == "explain"
        assert result["tone_adjustment"] == 0.6
        assert result["arousal_boost"] == pytest.approx(0.6 * 0.3)
        assert result["suggested_content"] == "Here is a detailed explanation"

    def test_privacy_filter_constrains_even_small_boost(self):
        """When base_disclosure is already at max, any boost gets clamped to 0."""
        # base_disclosure=0.8, privacy_filter=0.3
        # max_allowed = 1.0 - 0.3 = 0.7
        # base_disclosure (0.8) already > max_allowed (0.7)
        # actual_boost = max(0.0, 0.7 - 0.8) = 0.0
        pattern = _pattern("open", "Tell you about my family traditions", 0.4)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.8,
            privacy_filter=0.3,
            topic_sensitivities=[],
            must_avoid=[],
        )

        assert result["pattern_constrained"] is True
        assert result["disclosure_boost"] == pytest.approx(0.0)


class TestStandardModificationsAlwaysPresent:
    """Verify non-blocked patterns always get the standard modification keys."""

    def test_all_standard_keys_present(self):
        pattern = _pattern("neutral", "A neutral response about weather", 0.3)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[],
            must_avoid=[],
        )

        assert result["pattern_triggered"] is True
        assert result["trigger"] == "neutral"
        assert result["tone_adjustment"] == 0.3
        assert result["arousal_boost"] == pytest.approx(0.3 * 0.3)
        assert result["suggested_content"] == "A neutral response about weather"

    def test_default_topic_context_is_empty_string(self):
        """topic_context defaults to '' when not provided."""
        pattern = _pattern("default", "Just a normal response", 0.5)
        sensitivity = _sensitivity("politics", 0.8)

        # topic_context defaults to "" -> "politics" not in "" -> no match
        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.3,
            privacy_filter=0.2,
            topic_sensitivities=[sensitivity],
            must_avoid=[],
        )

        assert "sensitivity_constraint" not in result
        assert result["pattern_triggered"] is True


class TestDisclosureKeywordDetection:
    """Verify that various disclosure keywords are correctly detected through the public API."""

    @pytest.mark.parametrize("keyword,response_text", [
        ("personal", "This is a personal matter"),
        ("story", "Let me tell a story about it"),
        ("experience", "In my experience working here"),
        ("my", "Based on my understanding"),
        ("i ", "I think this is important"),
        ("share", "I'd like to share some thoughts"),
        ("tell you", "Let me tell you something"),
        ("family", "My family always said"),
        ("relationship", "The relationship between us"),
    ])
    def test_each_disclosure_keyword_triggers_boost(self, keyword, response_text):
        """Each disclosure keyword should lead to a disclosure_boost in the result."""
        pattern = _pattern("trigger", response_text, 0.5)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.2,
            privacy_filter=0.1,
            topic_sensitivities=[],
            must_avoid=[],
        )

        assert "disclosure_boost" in result, f"keyword '{keyword}' not detected in '{response_text}'"

    def test_no_disclosure_keyword_means_no_boost(self):
        """Response without any disclosure keywords -> no disclosure_boost."""
        pattern = _pattern("explain", "The algorithm uses a recursive approach", 0.5)

        result = apply_response_pattern_safely(
            pattern=pattern,
            base_disclosure=0.2,
            privacy_filter=0.1,
            topic_sensitivities=[],
            must_avoid=[],
        )

        assert "disclosure_boost" not in result


# ===========================================================================
# validate_stance_against_invariants
# ===========================================================================


class TestValidateStanceAgainstInvariants:
    """Tests for validate_stance_against_invariants."""

    def test_valid_stance_no_violations(self):
        result = validate_stance_against_invariants(
            stance="I believe this approach is reasonable",
            rationale="Based on common sense and general knowledge",
            identity_facts=["Lives in London", "Age 34"],
            cannot_claim=["medical doctor", "licensed therapist", "lawyer"],
        )

        assert result["valid"] is True
        assert result["violations"] == []

    def test_single_violation_in_stance(self):
        result = validate_stance_against_invariants(
            stance="As a medical doctor, I recommend this treatment",
            rationale="Based on clinical experience",
            identity_facts=["Lives in London", "UX Researcher"],
            cannot_claim=["medical doctor", "licensed therapist"],
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 1
        violation = result["violations"][0]
        assert violation["type"] == "forbidden_claim"
        assert violation["claim"] == "medical doctor"
        assert violation["severity"] == "error"
        assert "medical doctor" in violation["message"]

    def test_single_violation_in_rationale(self):
        result = validate_stance_against_invariants(
            stance="This is a safe approach",
            rationale="As a licensed therapist, I can confirm this",
            identity_facts=[],
            cannot_claim=["licensed therapist"],
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 1
        assert result["violations"][0]["claim"] == "licensed therapist"

    def test_multiple_violations(self):
        result = validate_stance_against_invariants(
            stance="As a medical doctor and lawyer, I advise caution",
            rationale="My experience as a licensed therapist confirms this",
            identity_facts=[],
            cannot_claim=["medical doctor", "licensed therapist", "lawyer"],
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 3
        claims = {v["claim"] for v in result["violations"]}
        assert claims == {"medical doctor", "licensed therapist", "lawyer"}

    def test_case_insensitive_matching(self):
        result = validate_stance_against_invariants(
            stance="As a MEDICAL DOCTOR I know this",
            rationale="",
            identity_facts=[],
            cannot_claim=["medical doctor"],
        )

        assert result["valid"] is False
        assert len(result["violations"]) == 1

    def test_empty_cannot_claim_always_valid(self):
        result = validate_stance_against_invariants(
            stance="I am a brain surgeon and a rocket scientist",
            rationale="Trust me",
            identity_facts=["Age 25"],
            cannot_claim=[],
        )

        assert result["valid"] is True
        assert result["violations"] == []

    def test_partial_match_in_combined_text(self):
        """Substring match: 'lawyer' in 'the lawyer said' should trigger."""
        result = validate_stance_against_invariants(
            stance="the lawyer said this is fine",
            rationale="",
            identity_facts=[],
            cannot_claim=["lawyer"],
        )

        assert result["valid"] is False

    def test_no_match_when_claim_absent(self):
        result = validate_stance_against_invariants(
            stance="I think the weather is nice",
            rationale="Just an observation",
            identity_facts=["Lives in London"],
            cannot_claim=["medical doctor", "lawyer"],
        )

        assert result["valid"] is True


# ===========================================================================
# clamp_disclosure_to_constraints
# ===========================================================================


class TestClampDisclosureToConstraints:
    """Tests for clamp_disclosure_to_constraints."""

    def test_within_bounds_no_clamping(self):
        """Disclosure below both constraints -> unchanged, no reason."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.3,
            privacy_sensitivity=0.2,
            topic_privacy_filter=0.1,
        )
        # max_from_base = 0.8, max_from_topic = 0.9
        # max_allowed = 0.8
        # 0.3 <= 0.8 -> no clamping
        assert level == pytest.approx(0.3)
        assert reason is None

    def test_base_privacy_wins(self):
        """Base privacy more restrictive than topic -> clamps to base, reason='privacy_sensitivity'."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.8,
            privacy_sensitivity=0.6,
            topic_privacy_filter=0.2,
        )
        # max_from_base = 0.4, max_from_topic = 0.8
        # max_allowed = 0.4
        # 0.8 > 0.4 -> clamp
        # reason: max_from_base (0.4) < max_from_topic (0.8) -> "privacy_sensitivity"
        assert level == pytest.approx(0.4)
        assert reason == "privacy_sensitivity"

    def test_topic_privacy_wins(self):
        """Topic privacy more restrictive -> clamps to topic, reason='topic_privacy'."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.7,
            privacy_sensitivity=0.1,
            topic_privacy_filter=0.8,
        )
        # max_from_base = 0.9, max_from_topic = 0.2
        # max_allowed = 0.2
        # 0.7 > 0.2 -> clamp
        # reason: max_from_base (0.9) >= max_from_topic (0.2) -> "topic_privacy"
        assert level == pytest.approx(0.2)
        assert reason == "topic_privacy"

    def test_equal_constraints_returns_topic_privacy(self):
        """When both constraints are equal, the else branch gives 'topic_privacy'."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.8,
            privacy_sensitivity=0.5,
            topic_privacy_filter=0.5,
        )
        # max_from_base = 0.5, max_from_topic = 0.5
        # max_allowed = 0.5
        # 0.8 > 0.5 -> clamp
        # reason: max_from_base (0.5) NOT < max_from_topic (0.5) -> "topic_privacy"
        assert level == pytest.approx(0.5)
        assert reason == "topic_privacy"

    def test_exactly_at_boundary_no_clamping(self):
        """Disclosure exactly at the limit -> no clamping."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.5,
            privacy_sensitivity=0.5,
            topic_privacy_filter=0.3,
        )
        # max_from_base = 0.5, max_from_topic = 0.7
        # max_allowed = 0.5
        # 0.5 <= 0.5 -> no clamping
        assert level == pytest.approx(0.5)
        assert reason is None

    def test_zero_privacy_no_constraint(self):
        """Zero privacy sensitivity and filter -> max_allowed is 1.0."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.99,
            privacy_sensitivity=0.0,
            topic_privacy_filter=0.0,
        )
        assert level == pytest.approx(0.99)
        assert reason is None

    def test_full_privacy_sensitivity_clamps_to_zero(self):
        """privacy_sensitivity=1.0 -> max_from_base=0.0."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.5,
            privacy_sensitivity=1.0,
            topic_privacy_filter=0.0,
        )
        # max_from_base = 0.0, max_from_topic = 1.0
        # max_allowed = 0.0
        assert level == pytest.approx(0.0)
        assert reason == "privacy_sensitivity"

    def test_full_topic_filter_clamps_to_zero(self):
        """topic_privacy_filter=1.0 -> max_from_topic=0.0."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.5,
            privacy_sensitivity=0.0,
            topic_privacy_filter=1.0,
        )
        # max_from_base = 1.0, max_from_topic = 0.0
        # max_allowed = 0.0
        assert level == pytest.approx(0.0)
        assert reason == "topic_privacy"

    def test_zero_disclosure_always_within_bounds(self):
        """Disclosure of 0 should never be clamped regardless of constraints."""
        level, reason = clamp_disclosure_to_constraints(
            disclosure_level=0.0,
            privacy_sensitivity=0.9,
            topic_privacy_filter=0.9,
        )
        assert level == pytest.approx(0.0)
        assert reason is None
