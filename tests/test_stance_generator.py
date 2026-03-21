"""
Unit tests for the compositional stance generator internal functions.

Tests cover:
- _extract_topic: topic hint extraction, input classification
- _check_invariants: must_avoid blocking, cannot_claim caveats
- _get_value_topic_mapping: value x domain lookups with fallback
- _select_competence_frame: tier + input type selection
- _assemble_stance: fragment composition
- _modulate_stance_by_personality: Big Five personality modulation
"""

import pytest

from persona_engine.planner.stance_generator import (
    TopicExtraction,
    InvariantCheck,
    _extract_topic,
    _check_invariants,
    _get_value_topic_mapping,
    _select_competence_frame,
    _assemble_stance,
    _modulate_stance_by_personality,
)
from persona_engine.schema.persona_schema import BigFiveTraits


# =============================================================================
# _extract_topic
# =============================================================================


class TestExtractTopic:

    def test_question_detected(self):
        result = _extract_topic("What do you think about Python?", "technology")
        assert result.is_question is True
        assert result.domain == "technology"

    def test_opinion_share_detected(self):
        result = _extract_topic("I think functional programming is overrated", "technology")
        assert result.is_opinion_share is True
        assert result.is_question is False

    def test_request_detected(self):
        result = _extract_topic("Could you explain microservices?", "technology")
        assert result.is_request is True

    def test_plain_statement(self):
        result = _extract_topic("Rust has zero-cost abstractions", "technology")
        assert result.is_question is False
        assert result.is_opinion_share is False
        assert result.is_request is False

    def test_topic_hint_about_pattern(self):
        result = _extract_topic("What do you think about container orchestration?", "technology")
        assert result.topic_hint is not None
        assert "container" in result.topic_hint

    def test_topic_hint_how_to_pattern(self):
        result = _extract_topic("How do you handle legacy codebases?", "technology")
        assert result.topic_hint is not None
        assert "legacy" in result.topic_hint

    def test_topic_hint_none_for_short_input(self):
        """No regex match when input is too short/vague for capture groups."""
        result = _extract_topic("Hi", "general")
        assert result.topic_hint is None

    def test_empty_input(self):
        result = _extract_topic("", "general")
        assert result.topic_hint is None
        assert result.is_question is False
        assert result.is_opinion_share is False
        assert result.is_request is False
        assert result.domain == "general"

    def test_frozen_dataclass(self):
        result = _extract_topic("test", "general")
        with pytest.raises(AttributeError):
            result.domain = "other"


# =============================================================================
# _check_invariants
# =============================================================================


class TestCheckInvariants:

    def _make_topic(self, hint=None, domain="general"):
        return TopicExtraction(
            topic_hint=hint,
            domain=domain,
            is_question=False,
            is_opinion_share=False,
            is_request=False,
        )

    def test_no_invariants_pass(self):
        topic = self._make_topic("cooking pasta")
        result = _check_invariants(topic, cannot_claim=[], must_avoid=[])
        assert result.is_blocked is False
        assert result.reason is None
        assert result.caveat is None

    def test_must_avoid_blocks(self):
        topic = self._make_topic("sharing confidential product roadmaps")
        result = _check_invariants(
            topic,
            cannot_claim=[],
            must_avoid=["confidential product roadmaps"],
        )
        assert result.is_blocked is True
        assert result.reason == "confidential product roadmaps"

    def test_must_avoid_matches_domain(self):
        topic = self._make_topic(hint=None, domain="gambling")
        result = _check_invariants(
            topic,
            cannot_claim=[],
            must_avoid=["gambling"],
        )
        assert result.is_blocked is True

    def test_cannot_claim_caveat(self):
        """cannot_claim produces a soft caveat, not a block."""
        topic = self._make_topic("medical doctor advice")
        result = _check_invariants(
            topic,
            cannot_claim=["medical doctor"],
            must_avoid=[],
        )
        assert result.is_blocked is False
        assert result.caveat is not None
        assert "medical doctor" in result.caveat

    def test_cannot_claim_no_overlap(self):
        """No caveat when topic doesn't overlap with cannot_claim."""
        topic = self._make_topic("cooking pasta")
        result = _check_invariants(
            topic,
            cannot_claim=["licensed therapist"],
            must_avoid=[],
        )
        assert result.is_blocked is False
        assert result.caveat is None

    def test_must_avoid_takes_precedence(self):
        """must_avoid is checked first, producing a block even if cannot_claim would match."""
        topic = self._make_topic("gambling strategies")
        result = _check_invariants(
            topic,
            cannot_claim=["gambling expert"],
            must_avoid=["gambling"],
        )
        assert result.is_blocked is True

    def test_none_topic_hint(self):
        topic = self._make_topic(hint=None, domain="general")
        result = _check_invariants(
            topic,
            cannot_claim=["doctor"],
            must_avoid=["violence"],
        )
        assert result.is_blocked is False
        assert result.caveat is None


# =============================================================================
# _get_value_topic_mapping
# =============================================================================


class TestGetValueTopicMapping:

    def test_exact_value_domain_match(self):
        stance, rationale = _get_value_topic_mapping("achievement", "technology")
        assert "performant" in stance or "ship" in stance
        assert "Achievement" in rationale

    def test_fallback_to_general(self):
        stance, rationale = _get_value_topic_mapping("achievement", "underwater_basket_weaving")
        assert "competence" in stance
        assert "general" in rationale

    def test_unknown_value_fallback(self):
        """Completely unknown value falls back to the hardcoded default."""
        stance, rationale = _get_value_topic_mapping("nonexistent_value", "technology")
        assert "considered perspective" in stance
        assert "nonexistent_value" in rationale

    def test_all_known_values_have_general(self):
        """Every value in the table has a 'general' fallback."""
        from persona_engine.planner.stance_generator import VALUE_TOPIC_TABLE
        for value_name, domains in VALUE_TOPIC_TABLE.items():
            assert "general" in domains, f"{value_name} missing 'general' fallback"

    def test_returns_tuple_of_strings(self):
        result = _get_value_topic_mapping("benevolence", "food")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# =============================================================================
# _select_competence_frame
# =============================================================================


class TestSelectCompetenceFrame:

    def _make_topic(self, hint="machine learning", is_question=False,
                    is_opinion_share=False, is_request=False):
        return TopicExtraction(
            topic_hint=hint,
            domain="technology",
            is_question=is_question,
            is_opinion_share=is_opinion_share,
            is_request=is_request,
        )

    # -- Tier thresholds --

    def test_expert_tier(self):
        topic = self._make_topic(is_question=True)
        result = _select_competence_frame(0.8, True, topic, "analytical")
        assert "experience" in result.lower()

    def test_expert_blocked_when_not_allowed(self):
        """High proficiency but expert_allowed=False falls to moderate."""
        topic = self._make_topic(is_question=True)
        result = _select_competence_frame(0.8, False, topic, "analytical")
        assert "encountered" in result.lower() or "seen" in result.lower()

    def test_moderate_tier(self):
        topic = self._make_topic(is_question=True)
        result = _select_competence_frame(0.5, False, topic, "analytical")
        assert "encountered" in result.lower() or "seen" in result.lower()

    def test_low_tier(self):
        topic = self._make_topic(is_question=True)
        result = _select_competence_frame(0.2, False, topic, "analytical")
        assert "not deeply" in result.lower() or "can't speak" in result.lower()

    # -- Boundary: expert threshold at exactly 0.6 --

    def test_expert_boundary_at_06(self):
        topic = self._make_topic()
        result = _select_competence_frame(0.6, True, topic, "analytical")
        # 0.6 >= 0.6 with expert_allowed -> expert tier
        assert "know" in result.lower() or "experience" in result.lower()

    def test_moderate_boundary_at_04(self):
        topic = self._make_topic()
        result = _select_competence_frame(0.4, False, topic, "analytical")
        # 0.4 >= 0.4 -> moderate tier
        assert "seen" in result.lower() or "familiarity" in result.lower()

    def test_low_boundary_just_below_04(self):
        topic = self._make_topic()
        result = _select_competence_frame(0.39, False, topic, "analytical")
        assert "not deeply" in result.lower() or "can't speak" in result.lower() or "without" in result.lower()

    # -- Input type selection --

    def test_opinion_share_frame(self):
        topic = self._make_topic(is_opinion_share=True)
        result = _select_competence_frame(0.8, True, topic, "analytical")
        assert "worked extensively" in result.lower()

    def test_request_frame(self):
        topic = self._make_topic(is_request=True)
        result = _select_competence_frame(0.8, True, topic, "analytical")
        assert "background" in result.lower() or "drawing" in result.lower()

    # -- None topic hint --

    def test_none_topic_hint_uses_fallback(self):
        topic = self._make_topic(hint=None)
        result = _select_competence_frame(0.8, True, topic, "analytical")
        assert "this area" in result


# =============================================================================
# _assemble_stance
# =============================================================================


class TestAssembleStance:

    def test_basic_assembly(self):
        result = _assemble_stance(
            competence_frame="From my experience with APIs,",
            value_position="I favor well-tested solutions",
            nuance=None,
            invariant_caveat=None,
        )
        assert result == "From my experience with APIs, i favor well-tested solutions"

    def test_with_nuance(self):
        result = _assemble_stance(
            competence_frame="From my experience,",
            value_position="I favor tested solutions",
            nuance="though context matters here",
            invariant_caveat=None,
        )
        assert "though context matters here" in result

    def test_with_invariant_caveat(self):
        result = _assemble_stance(
            competence_frame="From my experience,",
            value_position="I favor tested solutions",
            nuance=None,
            invariant_caveat="Speaking as a non-medical doctor",
        )
        assert result.startswith("Speaking as a non-medical doctor:")

    def test_all_parts(self):
        result = _assemble_stance(
            competence_frame="From my experience,",
            value_position="I favor tested solutions",
            nuance="though I also value autonomy",
            invariant_caveat="Speaking as a non-doctor",
        )
        assert "Speaking as a non-doctor:" in result
        assert "From my experience," in result
        assert "though I also value autonomy" in result


# =============================================================================
# _modulate_stance_by_personality
# =============================================================================


class TestModulateStanceByPersonality:

    def _make_traits(self, **overrides):
        defaults = dict(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        defaults.update(overrides)
        return BigFiveTraits(**defaults)

    def test_neutral_traits_no_modulation(self):
        """Mid-range traits should not trigger any prefix or suffix."""
        traits = self._make_traits()
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert result == "I favor tested solutions"

    def test_high_neuroticism_prefix(self):
        traits = self._make_traits(neuroticism=0.8)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert result.startswith("I worry")

    def test_high_neuroticism_lowercases_original(self):
        traits = self._make_traits(neuroticism=0.8)
        result = _modulate_stance_by_personality("Something here", traits)
        assert "something here" in result

    def test_low_neuroticism_low_agreeableness_assertive(self):
        traits = self._make_traits(neuroticism=0.2, agreeableness=0.3)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert result.startswith("Let me be clear:")

    def test_high_openness_exploratory(self):
        """High openness without high neuroticism -> exploratory prefix."""
        traits = self._make_traits(openness=0.8)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert "broader perspective" in result

    def test_high_neuroticism_overrides_high_openness(self):
        """Neuroticism prefix takes priority over openness prefix."""
        traits = self._make_traits(neuroticism=0.8, openness=0.9)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert result.startswith("I worry")
        assert "broader perspective" not in result

    def test_high_agreeableness_suffix(self):
        traits = self._make_traits(agreeableness=0.8)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert "others may see this differently" in result

    def test_high_agreeableness_combines_with_prefix(self):
        """Both prefix (neuroticism) and suffix (agreeableness) can apply."""
        traits = self._make_traits(neuroticism=0.8, agreeableness=0.8)
        result = _modulate_stance_by_personality("I favor tested solutions", traits)
        assert result.startswith("I worry")
        assert "others may see this differently" in result

    # -- Boundary tests --

    def test_neuroticism_boundary_at_065(self):
        """Exactly 0.65 should NOT trigger (threshold is > 0.65)."""
        traits = self._make_traits(neuroticism=0.65)
        result = _modulate_stance_by_personality("Test stance", traits)
        assert not result.startswith("I worry")

    def test_neuroticism_boundary_at_066(self):
        """0.66 > 0.65 should trigger cautious prefix."""
        traits = self._make_traits(neuroticism=0.66)
        result = _modulate_stance_by_personality("Test stance", traits)
        assert result.startswith("I worry")

    def test_agreeableness_boundary_at_075(self):
        """Exactly 0.75 should NOT trigger suffix (threshold is > 0.75)."""
        traits = self._make_traits(agreeableness=0.75)
        result = _modulate_stance_by_personality("Test stance", traits)
        assert "others may see this differently" not in result

    def test_openness_boundary_at_07(self):
        """Exactly 0.7 should NOT trigger (threshold is > 0.7)."""
        traits = self._make_traits(openness=0.7)
        result = _modulate_stance_by_personality("Test stance", traits)
        assert "broader perspective" not in result
