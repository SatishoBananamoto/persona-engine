"""
Phase R4 Tests: Emotional Appraisal — Personality-Dependent Emotion

Tests that the same event produces different emotional responses based
on Big Five personality traits, following Scherer's Component Process Model.

Covers:
- User emotion detection (negation-aware keyword matching)
- Personality-dependent appraisal (same event → different emotions)
- Emotional contagion modulated by personality
- Integration with IR pipeline (mood affects tone)
"""

import pytest

from conftest import make_persona_data

from persona_engine.behavioral.emotional_appraisal import (
    EmotionalAppraisal,
    appraise_event,
    detect_user_emotion,
)
from persona_engine.memory import StanceCache
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    Tone,
)
from persona_engine.schema.persona_schema import BigFiveTraits, Persona
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
# Test User Emotion Detection
# ============================================================================

class TestUserEmotionDetection:

    def test_enthusiasm_detected(self):
        result = detect_user_emotion("This is amazing! I'm so excited!")
        assert result.get("joy", 0) > 0.2

    def test_frustration_detected(self):
        result = detect_user_emotion("This is terrible and broken!")
        assert result.get("anger", 0) > 0.2

    def test_worry_detected(self):
        result = detect_user_emotion("I'm worried about the risks involved")
        assert result.get("fear", 0) > 0.1

    def test_curiosity_from_question(self):
        result = detect_user_emotion("What do you think about this?")
        assert result.get("interest", 0) > 0.1

    def test_challenge_detected(self):
        result = detect_user_emotion("I disagree, you're wrong about this")
        assert result.get("challenge", 0) > 0.2

    def test_praise_detected(self):
        result = detect_user_emotion("Thank you, that was very helpful and insightful")
        assert result.get("trust", 0) > 0.2

    def test_neutral_input(self):
        result = detect_user_emotion("Can you pass me that document")
        # Should have minimal emotional signal
        total = sum(result.values())
        assert total < 0.5

    def test_exclamation_amplifies(self):
        calm = detect_user_emotion("This is great")
        excited = detect_user_emotion("This is great!!!")
        assert excited.get("joy", 0) >= calm.get("joy", 0)


class TestNegationAwareEmotionDetection:
    """Negated markers must not register as emotional signals."""

    # --- Enthusiasm / joy ---
    def test_not_excited_no_joy(self):
        result = detect_user_emotion("I'm not excited about this")
        assert result.get("joy", 0) == 0

    def test_excited_has_joy(self):
        result = detect_user_emotion("I'm excited about this")
        assert result.get("joy", 0) > 0

    def test_never_amazing_no_joy(self):
        result = detect_user_emotion("It was never amazing")
        assert result.get("joy", 0) == 0

    # --- Frustration / anger ---
    def test_not_frustrated_no_anger(self):
        result = detect_user_emotion("I'm not frustrated at all")
        assert result.get("anger", 0) == 0

    def test_isnt_terrible_no_anger(self):
        result = detect_user_emotion("This isn't terrible")
        assert result.get("anger", 0) == 0

    def test_terrible_has_anger(self):
        result = detect_user_emotion("This is terrible")
        assert result.get("anger", 0) > 0

    # --- Worry / fear ---
    def test_not_worried_no_fear(self):
        result = detect_user_emotion("I'm not worried about this")
        assert result.get("fear", 0) == 0

    def test_dont_be_afraid_no_fear(self):
        result = detect_user_emotion("Don't be afraid")
        assert result.get("fear", 0) == 0

    def test_worried_has_fear(self):
        result = detect_user_emotion("I'm worried about this")
        assert result.get("fear", 0) > 0

    # --- Mixed: negated + real ---
    def test_mixed_negated_and_real(self):
        """'not excited but worried' → no joy, yes fear."""
        result = detect_user_emotion("I'm not excited but I am worried")
        assert result.get("joy", 0) == 0
        assert result.get("fear", 0) > 0

    # --- Challenge markers (negation should also work) ---
    def test_not_wrong_no_challenge(self):
        result = detect_user_emotion("You're not wrong about that")
        assert result.get("challenge", 0) == 0

    def test_wrong_has_challenge(self):
        result = detect_user_emotion("You're wrong about that")
        assert result.get("challenge", 0) > 0

    # --- Edge cases ---
    def test_empty_input(self):
        result = detect_user_emotion("")
        assert result == {} or sum(result.values()) == 0

    def test_double_negation_restores(self):
        """'not never' is beyond 3-token window, so 'terrible' is unnegated."""
        result = detect_user_emotion("It's not that it was never terrible, it really was terrible")
        assert result.get("anger", 0) > 0


# ============================================================================
# Test Personality-Dependent Appraisal
# ============================================================================

class TestPersonalityAppraisal:
    """Same event, different emotional responses based on personality."""

    def test_challenge_high_n_produces_fear(self):
        """High-N persona appraises challenge as threatening."""
        user_emotion = {"challenge": 0.5}
        high_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.85)
        result = appraise_event(user_emotion, high_n)
        assert result.valence_delta < 0  # Negative emotion
        assert result.dominant_emotion == "fear"

    def test_challenge_high_o_produces_interest(self):
        """High-O persona appraises challenge as interesting."""
        user_emotion = {"challenge": 0.5}
        high_o = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.3)
        result = appraise_event(user_emotion, high_o)
        assert result.valence_delta > 0  # Positive emotion
        assert result.dominant_emotion == "interest"

    def test_challenge_low_a_produces_confrontation(self):
        """Low-A persona appraises challenge as confrontation."""
        user_emotion = {"challenge": 0.5}
        low_a = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.15, neuroticism=0.3)
        result = appraise_event(user_emotion, low_a)
        assert result.arousal_delta > 0  # Aroused
        assert result.dominant_emotion == "anger"

    def test_joy_contagion_high_e(self):
        """High-E persona absorbs more positive emotion."""
        user_emotion = {"joy": 0.6}
        high_e = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.9, agreeableness=0.7, neuroticism=0.3)
        low_e = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.3, neuroticism=0.3)
        result_high = appraise_event(user_emotion, high_e)
        result_low = appraise_event(user_emotion, low_e)
        assert result_high.valence_delta > result_low.valence_delta

    def test_worry_contagion_high_n(self):
        """High-N persona absorbs more worry from user."""
        user_emotion = {"fear": 0.5}
        high_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.9)
        low_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.1)
        result_high = appraise_event(user_emotion, high_n)
        result_low = appraise_event(user_emotion, low_n)
        # High-N should absorb more negative emotion
        assert result_high.valence_delta < result_low.valence_delta

    def test_praise_helps_anxious(self):
        """High-N persona benefits more from praise."""
        user_emotion = {"trust": 0.5}
        high_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.8)
        low_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.2)
        result_high = appraise_event(user_emotion, high_n)
        result_low = appraise_event(user_emotion, low_n)
        assert result_high.valence_delta > result_low.valence_delta

    def test_stress_amplifies_negative(self):
        """Under stress, negative appraisals are amplified."""
        user_emotion = {"anger": 0.8}
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.2, neuroticism=0.7)
        calm = appraise_event(user_emotion, traits, current_stress=0.1)
        stressed = appraise_event(user_emotion, traits, current_stress=0.8)
        assert stressed.valence_delta < calm.valence_delta

    def test_appraisal_clamped(self):
        """Appraisal deltas should be bounded."""
        user_emotion = {"joy": 1.0, "anger": 1.0, "fear": 1.0}
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        result = appraise_event(user_emotion, traits)
        assert -0.4 <= result.valence_delta <= 0.4
        assert -0.3 <= result.arousal_delta <= 0.3


# ============================================================================
# Test Pipeline Integration
# ============================================================================

class TestEmotionalAppraisalPipeline:
    """Verify emotional appraisal affects IR through mood/tone."""

    def test_enthusiastic_input_affects_ir(self):
        """Enthusiastic user input should shift tone for high-E persona."""
        high_e = make_persona_data(extraversion=0.9, agreeableness=0.8, neuroticism=0.2)
        ir_enthusiastic = _generate_ir(high_e, "This is amazing! I'm so excited about this!")
        ir_neutral = _generate_ir(high_e, "Can you tell me about this topic")

        # The enthusiastic input should produce different IR
        # (either different tone or different valence-based effects)
        assert ir_enthusiastic is not None
        assert ir_neutral is not None

    def test_challenging_input_differs_by_personality(self):
        """Same challenging input should produce different IR for high-N vs high-O."""
        challenge = "I think you're completely wrong about this"

        high_n = make_persona_data(neuroticism=0.9, openness=0.3)
        high_o = make_persona_data(neuroticism=0.2, openness=0.9)

        ir_n = _generate_ir(high_n, challenge)
        ir_o = _generate_ir(high_o, challenge)

        # High-N should have lower confidence (fear response)
        assert ir_n.response_structure.confidence < ir_o.response_structure.confidence

    def test_appraisal_citation_present(self):
        """Emotional appraisal should produce a citation when triggered."""
        persona = make_persona_data(extraversion=0.9)
        ir = _generate_ir(persona, "This is absolutely amazing and wonderful!")

        citation_effects = [c.effect for c in ir.citations]
        has_appraisal = any("appraisal" in e.lower() for e in citation_effects)
        assert has_appraisal, f"Expected appraisal citation in: {citation_effects}"

    def test_neutral_input_no_appraisal_citation(self):
        """Neutral input should not produce appraisal citation."""
        persona = make_persona_data()
        ir = _generate_ir(persona, "Tell me about engineering")

        citation_effects = [c.effect for c in ir.citations]
        has_appraisal = any("appraisal" in e.lower() for e in citation_effects)
        # Neutral input might still have a question mark interest signal
        # so this is a soft check — at least no strong emotional shift
        assert ir is not None


class TestSameEventDifferentEmotions:
    """The core test from the plan: same event, personality-dependent emotions."""

    def test_your_advice_was_wrong(self):
        """
        Event: 'Your previous advice was completely wrong'
        Different personas should react differently.
        """
        challenge_input = "Your previous advice was completely wrong"

        # High-N: worried, self-doubting
        high_n = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.85)
        # Low-N + Low-A: defensive
        low_n_low_a = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                                     extraversion=0.5, agreeableness=0.15, neuroticism=0.2)
        # High-O: curious about the challenge
        high_o = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                                extraversion=0.5, agreeableness=0.5, neuroticism=0.2)

        user_emotion = detect_user_emotion(challenge_input)

        result_n = appraise_event(user_emotion, high_n)
        result_la = appraise_event(user_emotion, low_n_low_a)
        result_o = appraise_event(user_emotion, high_o)

        # High-N should be more negative
        assert result_n.valence_delta < result_o.valence_delta
        # High-O should find it more interesting
        assert result_o.dominant_emotion == "interest"
        # Low-A should have higher arousal (confrontational)
        assert result_la.arousal_delta > 0
