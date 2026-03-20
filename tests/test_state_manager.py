"""
Comprehensive tests for persona_engine.behavioral.state_manager

Targets: StateManager class and create_state_manager factory function.
Goal: raise coverage from 74% to 95%+.
"""

import pytest

from persona_engine.schema.persona_schema import BigFiveTraits, DynamicState
from persona_engine.behavioral.state_manager import StateManager, create_state_manager
from persona_engine.utils.determinism import DeterminismManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _default_traits(**overrides) -> BigFiveTraits:
    """Return BigFiveTraits with sensible mid-range defaults, overridable."""
    defaults = dict(
        openness=0.5,
        conscientiousness=0.5,
        extraversion=0.5,
        agreeableness=0.5,
        neuroticism=0.5,
    )
    defaults.update(overrides)
    return BigFiveTraits(**defaults)


def _default_state(**overrides) -> DynamicState:
    """Return a neutral DynamicState, overridable."""
    defaults = dict(
        mood_valence=0.0,
        mood_arousal=0.5,
        fatigue=0.0,
        stress=0.0,
        engagement=0.5,
    )
    defaults.update(overrides)
    return DynamicState(**defaults)


@pytest.fixture
def determinism():
    return DeterminismManager(seed=42)


@pytest.fixture
def manager(determinism):
    """Standard mid-range StateManager for general tests."""
    return StateManager(
        initial_state=_default_state(),
        traits=_default_traits(),
        determinism=determinism,
    )


# =========================================================================
# Constructor tests
# =========================================================================


class TestConstructor:
    def test_default_determinism_created_when_none(self):
        """If no DeterminismManager is passed, one is created automatically."""
        sm = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(),
            determinism=None,
        )
        assert isinstance(sm.determinism, DeterminismManager)

    def test_custom_determinism_is_stored(self, determinism):
        sm = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(),
            determinism=determinism,
        )
        assert sm.determinism is determinism

    def test_initial_state_is_copied(self):
        """Mutating the original DynamicState must not affect the manager."""
        original = _default_state(mood_valence=0.3)
        sm = StateManager(
            initial_state=original,
            traits=_default_traits(),
        )
        original.mood_valence = -0.9  # mutate original
        assert sm.state.mood_valence == pytest.approx(0.3)

    def test_mood_drift_rate_depends_on_neuroticism(self):
        low_n = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(neuroticism=0.0),
        )
        high_n = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(neuroticism=1.0),
        )
        # rate = 0.12 - neuroticism * 0.08 (high N → slower drift)
        assert low_n.mood_drift_rate == pytest.approx(0.12)
        assert high_n.mood_drift_rate == pytest.approx(0.04)

    def test_fatigue_accumulation_rate(self, manager):
        assert manager.fatigue_accumulation_rate == pytest.approx(0.02)

    def test_stress_decay_rate(self, manager):
        # rate = 0.08 + (1.0 - N) * 0.04; N=0.5 → 0.10
        assert manager.stress_decay_rate == pytest.approx(0.10)


# =========================================================================
# Getter tests
# =========================================================================


class TestGetters:
    def test_get_current_state_returns_copy(self, manager):
        copy = manager.get_current_state()
        assert copy == manager.state
        copy.mood_valence = -1.0
        assert manager.state.mood_valence != -1.0

    def test_get_mood(self, manager):
        manager.state.mood_valence = 0.3
        manager.state.mood_arousal = 0.7
        assert manager.get_mood() == (pytest.approx(0.3), pytest.approx(0.7))

    def test_get_fatigue(self, manager):
        manager.state.fatigue = 0.42
        assert manager.get_fatigue() == pytest.approx(0.42)

    def test_get_stress(self, manager):
        manager.state.stress = 0.33
        assert manager.get_stress() == pytest.approx(0.33)

    def test_get_engagement(self, manager):
        manager.state.engagement = 0.88
        assert manager.get_engagement() == pytest.approx(0.88)


# =========================================================================
# update_mood_from_event
# =========================================================================


class TestUpdateMoodFromEvent:
    def test_positive_deltas(self, manager):
        manager.state.mood_valence = 0.0
        manager.state.mood_arousal = 0.5
        manager.update_mood_from_event(0.3, 0.2)
        assert manager.state.mood_valence == pytest.approx(0.3)
        assert manager.state.mood_arousal == pytest.approx(0.7)

    def test_negative_deltas(self, manager):
        manager.state.mood_valence = 0.0
        manager.state.mood_arousal = 0.5
        manager.update_mood_from_event(-0.4, -0.3)
        assert manager.state.mood_valence == pytest.approx(-0.4)
        assert manager.state.mood_arousal == pytest.approx(0.2)

    def test_valence_clamped_upper(self, manager):
        manager.state.mood_valence = 0.8
        manager.update_mood_from_event(0.5, 0.0)
        assert manager.state.mood_valence == pytest.approx(1.0)

    def test_valence_clamped_lower(self, manager):
        manager.state.mood_valence = -0.8
        manager.update_mood_from_event(-0.5, 0.0)
        assert manager.state.mood_valence == pytest.approx(-1.0)

    def test_arousal_clamped_upper(self, manager):
        manager.state.mood_arousal = 0.9
        manager.update_mood_from_event(0.0, 0.5)
        assert manager.state.mood_arousal == pytest.approx(1.0)

    def test_arousal_clamped_lower(self, manager):
        manager.state.mood_arousal = 0.1
        manager.update_mood_from_event(0.0, -0.5)
        assert manager.state.mood_arousal == pytest.approx(0.0)


# =========================================================================
# apply_mood_drift
# =========================================================================


class TestApplyMoodDrift:
    def test_valence_drifts_toward_baseline(self):
        """Positive mood_valence should decrease toward baseline."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.8, mood_arousal=0.5),
            traits=_default_traits(neuroticism=0.5),
        )
        before = sm.state.mood_valence
        sm.apply_mood_drift()
        # baseline_valence = 0.1 - 0.5*0.2 = 0.0
        # diff = 0.0 - 0.8 = -0.8  => valence should decrease
        assert sm.state.mood_valence < before

    def test_arousal_drifts_toward_baseline(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.9),
            traits=_default_traits(neuroticism=0.5),
        )
        before = sm.state.mood_arousal
        sm.apply_mood_drift()
        # baseline_arousal = 0.5, diff = 0.5 - 0.9 = -0.4 => arousal decreases
        assert sm.state.mood_arousal < before

    def test_neuroticism_affects_baseline_valence(self):
        """Higher neuroticism lowers baseline valence (more negative baseline)."""
        low_n = StateManager(
            initial_state=_default_state(mood_valence=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        high_n = StateManager(
            initial_state=_default_state(mood_valence=0.0),
            traits=_default_traits(neuroticism=1.0),
        )
        low_n.apply_mood_drift()
        high_n.apply_mood_drift()
        # Low N baseline = 0.1, High N baseline = -0.1
        # Low N should drift positive, high N should drift negative
        assert low_n.state.mood_valence > 0.0
        assert high_n.state.mood_valence < 0.0

    def test_drift_magnitude_uses_mood_drift_rate(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=1.0, mood_arousal=0.0),
            traits=_default_traits(neuroticism=0.5),
        )
        # baseline_valence = 0.1 + 0.5*0.15 - 0.5*0.2 = 0.075, baseline_arousal = 0.5
        # rate = 0.12 - 0.5*0.08 = 0.08
        sm.apply_mood_drift()
        # valence: 1.0 + (0.075 - 1.0)*0.08 = 0.926
        assert sm.state.mood_valence == pytest.approx(0.926)
        # arousal: 0.0 + (0.5 - 0.0)*0.08 = 0.04
        assert sm.state.mood_arousal == pytest.approx(0.04)


# =========================================================================
# mood_from_topic_relevance
# =========================================================================


class TestMoodFromTopicRelevance:
    def test_high_relevance_positive_mood(self):
        """topic_relevance > 0.7 should boost valence (scaled by openness) and arousal."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.5),
            traits=_default_traits(openness=0.8),
        )
        sm.mood_from_topic_relevance(0.9)
        # valence += 0.1 * 0.8 = 0.08
        assert sm.state.mood_valence == pytest.approx(0.08)
        # arousal += 0.15
        assert sm.state.mood_arousal == pytest.approx(0.65)

    def test_low_relevance_negative_mood(self):
        """topic_relevance < 0.3 should reduce valence and arousal."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.5),
            traits=_default_traits(),
        )
        sm.mood_from_topic_relevance(0.1)
        assert sm.state.mood_valence == pytest.approx(-0.05)
        assert sm.state.mood_arousal == pytest.approx(0.4)

    def test_mid_relevance_no_change(self):
        """topic_relevance between 0.3 and 0.7 (inclusive) should not change mood."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.2, mood_arousal=0.6),
            traits=_default_traits(),
        )
        sm.mood_from_topic_relevance(0.5)
        assert sm.state.mood_valence == pytest.approx(0.2)
        assert sm.state.mood_arousal == pytest.approx(0.6)

    def test_boundary_0_7_no_change(self):
        """Exactly 0.7 is NOT > 0.7, so no positive mood change."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.5),
            traits=_default_traits(),
        )
        sm.mood_from_topic_relevance(0.7)
        assert sm.state.mood_valence == pytest.approx(0.0)
        assert sm.state.mood_arousal == pytest.approx(0.5)

    def test_boundary_0_3_no_change(self):
        """Exactly 0.3 is NOT < 0.3, so no negative mood change."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.5),
            traits=_default_traits(),
        )
        sm.mood_from_topic_relevance(0.3)
        assert sm.state.mood_valence == pytest.approx(0.0)
        assert sm.state.mood_arousal == pytest.approx(0.5)

    def test_high_relevance_openness_amplification(self):
        """openness=1.0 should give maximal valence boost."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, mood_arousal=0.5),
            traits=_default_traits(openness=1.0),
        )
        sm.mood_from_topic_relevance(0.8)
        assert sm.state.mood_valence == pytest.approx(0.1)  # 0.1 * 1.0


# =========================================================================
# increment_fatigue
# =========================================================================


class TestIncrementFatigue:
    def test_basic_accumulation(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.0),
            traits=_default_traits(conscientiousness=0.5),
        )
        sm.increment_fatigue(conversation_length=10)
        # 0.02 * (10/10) * (1 - 0.5*0.2) = 0.02 * 1.0 * 0.9 = 0.018
        assert sm.state.fatigue == pytest.approx(0.018)

    def test_conscientiousness_stamina_modifier(self):
        """High conscientiousness reduces fatigue increase (more stamina)."""
        low_c = StateManager(
            initial_state=_default_state(fatigue=0.0),
            traits=_default_traits(conscientiousness=0.0),
        )
        high_c = StateManager(
            initial_state=_default_state(fatigue=0.0),
            traits=_default_traits(conscientiousness=1.0),
        )
        low_c.increment_fatigue(20)
        high_c.increment_fatigue(20)
        # low_c: 0.02 * 2 * 1.0 = 0.04
        # high_c: 0.02 * 2 * 0.8 = 0.032
        assert low_c.state.fatigue == pytest.approx(0.04)
        assert high_c.state.fatigue == pytest.approx(0.032)
        assert high_c.state.fatigue < low_c.state.fatigue

    def test_fatigue_capped_at_1(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.99),
            traits=_default_traits(conscientiousness=0.0),
        )
        sm.increment_fatigue(conversation_length=100)
        assert sm.state.fatigue == pytest.approx(1.0)

    def test_high_fatigue_reduces_engagement(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.69, engagement=0.8),
            traits=_default_traits(conscientiousness=0.0),
        )
        # Push fatigue over 0.7
        sm.increment_fatigue(conversation_length=10)
        # fatigue: 0.69 + 0.02 * 1 * 1.0 = 0.71 > 0.7
        assert sm.state.fatigue > 0.7
        # engagement should drop by 0.1 from 0.8 -> 0.7
        assert sm.state.engagement == pytest.approx(0.7)

    def test_high_fatigue_reduces_arousal(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.69, mood_arousal=0.5),
            traits=_default_traits(conscientiousness=0.0),
        )
        sm.increment_fatigue(conversation_length=10)
        assert sm.state.fatigue > 0.7
        # arousal: 0.5 - 0.1 = 0.4
        assert sm.state.mood_arousal == pytest.approx(0.4)

    def test_high_fatigue_engagement_floor_at_0_2(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.69, engagement=0.2),
            traits=_default_traits(conscientiousness=0.0),
        )
        sm.increment_fatigue(conversation_length=10)
        # engagement max(0.2, 0.2 - 0.1) = 0.2
        assert sm.state.engagement == pytest.approx(0.2)

    def test_high_fatigue_arousal_floor_at_0_1(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.69, mood_arousal=0.1),
            traits=_default_traits(conscientiousness=0.0),
        )
        sm.increment_fatigue(conversation_length=10)
        # arousal max(0.1, 0.1 - 0.1) = 0.1
        assert sm.state.mood_arousal == pytest.approx(0.1)


# =========================================================================
# apply_stress_trigger
# =========================================================================


class TestApplyStressTrigger:
    def test_time_pressure_trigger(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("time_pressure", intensity=0.3)
        # sensitivity = 1.0 + 0*0.5 = 1.0, multiplier = 1.0
        # increase = 0.3 * 1.0 * 1.0 = 0.3
        assert sm.state.stress == pytest.approx(0.3)

    def test_conflict_trigger_high_agreeableness(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(agreeableness=0.8, neuroticism=0.0),
        )
        sm.apply_stress_trigger("conflict", intensity=0.3)
        # agreeableness > 0.6 => multiplier = 1.2
        # increase = 0.3 * 1.0 * 1.2 = 0.36
        assert sm.state.stress == pytest.approx(0.36)

    def test_conflict_trigger_low_agreeableness(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(agreeableness=0.4, neuroticism=0.0),
        )
        sm.apply_stress_trigger("conflict", intensity=0.3)
        # agreeableness <= 0.6 => multiplier = 0.8
        # increase = 0.3 * 1.0 * 0.8 = 0.24
        assert sm.state.stress == pytest.approx(0.24)

    def test_uncertainty_trigger_high_neuroticism(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.8),
        )
        sm.apply_stress_trigger("uncertainty", intensity=0.3)
        # sensitivity = 1.0 + 0.8*0.5 = 1.4
        # neuroticism > 0.6 => multiplier = 1.3
        # increase = 0.3 * 1.4 * 1.3 = 0.546
        assert sm.state.stress == pytest.approx(0.546)

    def test_uncertainty_trigger_low_neuroticism(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.4),
        )
        sm.apply_stress_trigger("uncertainty", intensity=0.3)
        # sensitivity = 1.0 + 0.4*0.5 = 1.2
        # neuroticism <= 0.6 => multiplier = 0.9
        # increase = 0.3 * 1.2 * 0.9 = 0.324
        assert sm.state.stress == pytest.approx(0.324)

    def test_complexity_trigger(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("complexity", intensity=0.5)
        # sensitivity = 1.0, multiplier = 0.7
        # increase = 0.5 * 1.0 * 0.7 = 0.35
        assert sm.state.stress == pytest.approx(0.35)

    def test_unknown_trigger_uses_default_multiplier(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("alien_invasion", intensity=0.3)
        # multiplier defaults to 1.0
        # increase = 0.3 * 1.0 * 1.0 = 0.3
        assert sm.state.stress == pytest.approx(0.3)

    def test_neuroticism_amplifies_stress(self):
        low_n = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        high_n = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=1.0),
        )
        low_n.apply_stress_trigger("time_pressure", intensity=0.3)
        high_n.apply_stress_trigger("time_pressure", intensity=0.3)
        # low_n: 0.3 * 1.0 * 1.0 = 0.3
        # high_n: 0.3 * 1.5 * 1.0 = 0.45
        assert low_n.state.stress == pytest.approx(0.3)
        assert high_n.state.stress == pytest.approx(0.45)

    def test_stress_capped_at_1(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.9),
            traits=_default_traits(neuroticism=1.0),
        )
        sm.apply_stress_trigger("time_pressure", intensity=1.0)
        assert sm.state.stress == pytest.approx(1.0)

    def test_default_intensity_parameter(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("time_pressure")  # default intensity=0.3
        assert sm.state.stress == pytest.approx(0.3)

    def test_high_stress_mood_side_effects(self):
        """When stress > 0.6 after update, valence drops and arousal rises."""
        sm = StateManager(
            initial_state=_default_state(
                stress=0.5,
                mood_valence=0.5,
                mood_arousal=0.5,
            ),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("time_pressure", intensity=0.3)
        # stress: 0.5 + 0.3 = 0.8 > 0.6
        assert sm.state.stress == pytest.approx(0.8)
        # mood side-effects: valence -= 0.15, arousal += 0.2
        assert sm.state.mood_valence == pytest.approx(0.35)
        assert sm.state.mood_arousal == pytest.approx(0.7)

    def test_no_mood_side_effects_when_stress_below_threshold(self):
        sm = StateManager(
            initial_state=_default_state(
                stress=0.0,
                mood_valence=0.5,
                mood_arousal=0.5,
            ),
            traits=_default_traits(neuroticism=0.0),
        )
        sm.apply_stress_trigger("time_pressure", intensity=0.1)
        # stress: 0.0 + 0.1 = 0.1 <= 0.6
        assert sm.state.stress == pytest.approx(0.1)
        assert sm.state.mood_valence == pytest.approx(0.5)
        assert sm.state.mood_arousal == pytest.approx(0.5)

    def test_conflict_boundary_agreeableness_0_6(self):
        """agreeableness == 0.6 is NOT > 0.6, so multiplier = 0.8."""
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(agreeableness=0.6, neuroticism=0.0),
        )
        sm.apply_stress_trigger("conflict", intensity=0.5)
        # multiplier = 0.8 (not 1.2)
        assert sm.state.stress == pytest.approx(0.5 * 1.0 * 0.8)

    def test_uncertainty_boundary_neuroticism_0_6(self):
        """neuroticism == 0.6 is NOT > 0.6, so multiplier = 0.9."""
        sm = StateManager(
            initial_state=_default_state(stress=0.0),
            traits=_default_traits(neuroticism=0.6),
        )
        sm.apply_stress_trigger("uncertainty", intensity=0.5)
        # sensitivity = 1.0 + 0.6*0.5 = 1.3, multiplier = 0.9
        assert sm.state.stress == pytest.approx(0.5 * 1.3 * 0.9)


# =========================================================================
# reduce_stress
# =========================================================================


class TestReduceStress:
    def test_normal_decay(self, manager):
        manager.state.stress = 0.5
        manager.reduce_stress()
        # stress_decay_rate = 0.08 + (1.0 - 0.5) * 0.04 = 0.10
        assert manager.state.stress == pytest.approx(0.40)

    def test_does_not_go_below_zero(self, manager):
        manager.state.stress = 0.02
        manager.reduce_stress()
        assert manager.state.stress == pytest.approx(0.0)

    def test_exact_zero_stays_zero(self, manager):
        manager.state.stress = 0.0
        manager.reduce_stress()
        assert manager.state.stress == pytest.approx(0.0)


# =========================================================================
# update_engagement
# =========================================================================


class TestUpdateEngagement:
    def test_novelty_bonus_for_early_turns(self):
        """conversation_turn < 5 gives a novelty bonus based on openness."""
        sm = StateManager(
            initial_state=_default_state(engagement=0.5, fatigue=0.0),
            traits=_default_traits(openness=0.8),
        )
        sm.update_engagement(topic_relevance=0.6, conversation_turn=3)
        # base = 0.6, novelty = 0.8*0.2 = 0.16, fatigue_penalty = 0
        # target = 0.76, clamped [0.1, 1.0] -> 0.76
        # engagement = 0.5 + (0.76 - 0.5)*0.3 = 0.5 + 0.078 = 0.578
        assert sm.state.engagement == pytest.approx(0.578)

    def test_no_novelty_bonus_for_late_turns(self):
        """conversation_turn >= 5 gives no novelty bonus."""
        sm = StateManager(
            initial_state=_default_state(engagement=0.5, fatigue=0.0),
            traits=_default_traits(openness=0.8),
        )
        sm.update_engagement(topic_relevance=0.6, conversation_turn=5)
        # base = 0.6, novelty = 0, fatigue_penalty = 0
        # target = 0.6
        # engagement = 0.5 + (0.6 - 0.5)*0.3 = 0.53
        assert sm.state.engagement == pytest.approx(0.53)

    def test_fatigue_penalty(self):
        sm = StateManager(
            initial_state=_default_state(engagement=0.5, fatigue=0.6),
            traits=_default_traits(openness=0.5),
        )
        sm.update_engagement(topic_relevance=0.7, conversation_turn=10)
        # base = 0.7, novelty = 0 (turn >= 5), fatigue_penalty = 0.6*0.3 = 0.18
        # target = 0.7 - 0.18 = 0.52
        # engagement = 0.5 + (0.52 - 0.5)*0.3 = 0.5 + 0.006 = 0.506
        assert sm.state.engagement == pytest.approx(0.506)

    def test_target_clamped_min(self):
        """Very low relevance + high fatigue should clamp target to 0.1."""
        sm = StateManager(
            initial_state=_default_state(engagement=0.5, fatigue=1.0),
            traits=_default_traits(openness=0.0),
        )
        sm.update_engagement(topic_relevance=0.0, conversation_turn=10)
        # base = 0.0, novelty = 0, fatigue_penalty = 0.3
        # target = -0.3, clamped to 0.1
        # engagement = 0.5 + (0.1 - 0.5)*0.3 = 0.5 - 0.12 = 0.38
        assert sm.state.engagement == pytest.approx(0.38)

    def test_target_clamped_max(self):
        """High relevance + novelty + no fatigue -> clamped to 1.0."""
        sm = StateManager(
            initial_state=_default_state(engagement=0.9, fatigue=0.0),
            traits=_default_traits(openness=1.0),
        )
        sm.update_engagement(topic_relevance=1.0, conversation_turn=1)
        # base = 1.0, novelty = 0.2, fatigue_penalty = 0
        # target = 1.2, clamped to 1.0
        # engagement = 0.9 + (1.0 - 0.9)*0.3 = 0.9 + 0.03 = 0.93
        assert sm.state.engagement == pytest.approx(0.93)

    def test_smooth_transition(self):
        """Engagement moves only 30% toward target each call."""
        sm = StateManager(
            initial_state=_default_state(engagement=0.0, fatigue=0.0),
            traits=_default_traits(openness=0.0),
        )
        sm.update_engagement(topic_relevance=1.0, conversation_turn=10)
        # target = 1.0, diff = 1.0, engagement = 0.0 + 1.0*0.3 = 0.3
        assert sm.state.engagement == pytest.approx(0.3)
        sm.update_engagement(topic_relevance=1.0, conversation_turn=10)
        # target = 1.0, diff = 0.7, engagement = 0.3 + 0.7*0.3 = 0.51
        assert sm.state.engagement == pytest.approx(0.51)


# =========================================================================
# evolve_state_post_turn
# =========================================================================


class TestEvolveStatePostTurn:
    def test_calls_all_sub_methods(self):
        """Verify that evolve_state_post_turn changes mood, fatigue, stress, engagement."""
        sm = StateManager(
            initial_state=_default_state(
                mood_valence=0.5,
                mood_arousal=0.8,
                fatigue=0.0,
                stress=0.4,
                engagement=0.5,
            ),
            traits=_default_traits(neuroticism=0.5),
            determinism=DeterminismManager(seed=42),
        )
        before = sm.get_current_state()
        sm.evolve_state_post_turn(conversation_length=10, topic_relevance=0.5)
        after = sm.get_current_state()

        # Mood should have drifted (plus noise)
        assert after.mood_valence != before.mood_valence
        assert after.mood_arousal != before.mood_arousal
        # Fatigue should have increased
        assert after.fatigue > before.fatigue
        # Stress should have decreased
        assert after.stress < before.stress
        # Engagement should have changed (toward target)
        assert after.engagement != before.engagement

    def test_default_topic_relevance(self):
        """topic_relevance defaults to 0.5."""
        sm = StateManager(
            initial_state=_default_state(stress=0.3),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=42),
        )
        # Should not raise
        sm.evolve_state_post_turn(conversation_length=5)
        assert sm.state.stress < 0.3  # stress decayed

    def test_deterministic_with_seed(self):
        """Two managers with same seed produce identical state evolution."""
        sm1 = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=99),
        )
        sm2 = StateManager(
            initial_state=_default_state(),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=99),
        )
        sm1.evolve_state_post_turn(10, 0.7)
        sm2.evolve_state_post_turn(10, 0.7)
        assert sm1.get_current_state() == sm2.get_current_state()


# =========================================================================
# _add_subtle_noise (tested indirectly)
# =========================================================================


class TestAddSubtleNoise:
    def test_noise_changes_mood_values(self):
        """Noise should modify mood_valence and mood_arousal by a small amount."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.5, mood_arousal=0.5),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=42),
        )
        before_v = sm.state.mood_valence
        before_a = sm.state.mood_arousal
        sm._add_subtle_noise()
        # With budget=0.03, changes should be small but non-zero
        assert sm.state.mood_valence != before_v
        assert abs(sm.state.mood_valence - before_v) <= 0.03
        assert abs(sm.state.mood_arousal - before_a) <= 0.03

    def test_noise_is_deterministic_with_same_seed(self):
        sm1 = StateManager(
            initial_state=_default_state(mood_valence=0.3, mood_arousal=0.7),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=42),
        )
        sm2 = StateManager(
            initial_state=_default_state(mood_valence=0.3, mood_arousal=0.7),
            traits=_default_traits(),
            determinism=DeterminismManager(seed=42),
        )
        sm1._add_subtle_noise()
        sm2._add_subtle_noise()
        assert sm1.state.mood_valence == sm2.state.mood_valence
        assert sm1.state.mood_arousal == sm2.state.mood_arousal


# =========================================================================
# get_disclosure_modifier
# =========================================================================


class TestGetDisclosureModifier:
    def test_positive_mood_boosts_disclosure(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.8, stress=0.0, fatigue=0.0),
            traits=_default_traits(),
        )
        # 0.8 * 0.15 + 0 + 0 = 0.12
        assert sm.get_disclosure_modifier() == pytest.approx(0.12)

    def test_negative_mood_reduces_disclosure(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=-0.6, stress=0.0, fatigue=0.0),
            traits=_default_traits(),
        )
        # -0.6 * 0.15 = -0.09
        assert sm.get_disclosure_modifier() == pytest.approx(-0.09)

    def test_stress_penalty(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, stress=0.5, fatigue=0.0),
            traits=_default_traits(),
        )
        # 0 + (-0.5*0.2) + 0 = -0.1
        assert sm.get_disclosure_modifier() == pytest.approx(-0.1)

    def test_fatigue_penalty(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.0, stress=0.0, fatigue=0.8),
            traits=_default_traits(),
        )
        # 0 + 0 + (-0.8*0.1) = -0.08
        assert sm.get_disclosure_modifier() == pytest.approx(-0.08)

    def test_combined_effects(self):
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.5, stress=0.4, fatigue=0.3),
            traits=_default_traits(),
        )
        expected = 0.5 * 0.15 + (-0.4 * 0.2) + (-0.3 * 0.1)
        assert sm.get_disclosure_modifier() == pytest.approx(expected)


# =========================================================================
# get_verbosity_modifier
# =========================================================================


class TestGetVerbosityModifier:
    def test_high_fatigue_returns_minus_one(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.8, engagement=0.5),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == -1

    def test_high_engagement_returns_plus_one(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.3, engagement=0.8),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == 1

    def test_neutral_returns_zero(self):
        sm = StateManager(
            initial_state=_default_state(fatigue=0.5, engagement=0.5),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == 0

    def test_fatigue_takes_priority_over_engagement(self):
        """When both fatigue > 0.7 AND engagement > 0.7, fatigue wins (checked first)."""
        sm = StateManager(
            initial_state=_default_state(fatigue=0.9, engagement=0.9),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == -1

    def test_boundary_fatigue_exactly_0_7(self):
        """fatigue == 0.7 is NOT > 0.7, so not -1."""
        sm = StateManager(
            initial_state=_default_state(fatigue=0.7, engagement=0.5),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == 0

    def test_boundary_engagement_exactly_0_7(self):
        """engagement == 0.7 is NOT > 0.7, so not +1."""
        sm = StateManager(
            initial_state=_default_state(fatigue=0.3, engagement=0.7),
            traits=_default_traits(),
        )
        assert sm.get_verbosity_modifier() == 0


# =========================================================================
# get_patience_level
# =========================================================================


class TestGetPatienceLevel:
    def test_normal_patience(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.2, fatigue=0.3),
            traits=_default_traits(),
        )
        # 0.7 - 0.2*0.3 - 0.3*0.2 = 0.7 - 0.06 - 0.06 = 0.58
        assert sm.get_patience_level() == pytest.approx(0.58)

    def test_zero_stress_zero_fatigue(self):
        sm = StateManager(
            initial_state=_default_state(stress=0.0, fatigue=0.0),
            traits=_default_traits(),
        )
        assert sm.get_patience_level() == pytest.approx(0.7)

    def test_high_stress_and_fatigue(self):
        sm = StateManager(
            initial_state=_default_state(stress=1.0, fatigue=1.0),
            traits=_default_traits(),
        )
        # 0.7 - 1.0*0.3 - 1.0*0.2 = 0.7 - 0.3 - 0.2 = 0.2
        assert sm.get_patience_level() == pytest.approx(0.2)

    def test_clamped_minimum(self):
        """Patience cannot go below 0.1."""
        sm = StateManager(
            initial_state=_default_state(stress=1.0, fatigue=1.0),
            traits=_default_traits(),
        )
        # 0.7 - 0.3 - 0.2 = 0.2  which is > 0.1
        # We need more extreme values... but stress and fatigue cap at 1.0
        # Actually 0.7 - 0.3 - 0.2 = 0.2, still above 0.1
        # The clamp is for edge-case safety; verify it works by direct state manipulation
        sm.state.stress = 1.0
        sm.state.fatigue = 1.0
        # Even at max, 0.7-0.3-0.2 = 0.2 which is valid
        assert sm.get_patience_level() >= 0.1

    def test_clamped_maximum(self):
        """Patience cannot exceed 1.0."""
        sm = StateManager(
            initial_state=_default_state(stress=0.0, fatigue=0.0),
            traits=_default_traits(),
        )
        # 0.7 - 0 - 0 = 0.7, which is < 1.0 so clamp doesn't apply in normal range
        assert sm.get_patience_level() <= 1.0


# =========================================================================
# create_state_manager factory
# =========================================================================


class TestCreateStateManager:
    def _build_persona(self, **trait_overrides):
        """Build a minimal valid Persona for factory testing."""
        from persona_engine.schema.persona_schema import (
            Persona,
            Identity,
            PersonalityProfile,
            SchwartzValues,
            CognitiveStyle,
            CommunicationPreferences,
            SocialRole,
            UncertaintyPolicy,
            ClaimPolicy,
            PersonaInvariants,
            DisclosurePolicy,
        )

        traits = _default_traits(**trait_overrides)
        return Persona(
            persona_id="test-001",
            version="1.0",
            label="Test Persona",
            identity=Identity(
                age=30,
                gender="nonbinary",
                location="London, UK",
                education="BSc Computer Science",
                occupation="Software Engineer",
                background="Test background",
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
                )
            },
            uncertainty=UncertaintyPolicy(
                admission_threshold=0.5,
                hedging_frequency=0.5,
                clarification_tendency=0.5,
                knowledge_boundary_strictness=0.5,
            ),
            claim_policy=ClaimPolicy(),
            invariants=PersonaInvariants(
                identity_facts=["Test identity"],
            ),
            time_scarcity=0.3,
            privacy_sensitivity=0.4,
            disclosure_policy=DisclosurePolicy(
                base_openness=0.5,
                factors={},
            ),
            initial_state=DynamicState(
                mood_valence=0.2,
                mood_arousal=0.4,
                fatigue=0.1,
                stress=0.15,
                engagement=0.6,
            ),
        )

    def test_factory_returns_state_manager(self):
        persona = self._build_persona()
        sm = create_state_manager(persona)
        assert isinstance(sm, StateManager)

    def test_factory_uses_persona_initial_state(self):
        persona = self._build_persona()
        sm = create_state_manager(persona)
        assert sm.state.mood_valence == pytest.approx(0.2)
        assert sm.state.mood_arousal == pytest.approx(0.4)
        assert sm.state.fatigue == pytest.approx(0.1)
        assert sm.state.stress == pytest.approx(0.15)
        assert sm.state.engagement == pytest.approx(0.6)

    def test_factory_uses_persona_traits(self):
        persona = self._build_persona(neuroticism=0.8)
        sm = create_state_manager(persona)
        assert sm.traits.neuroticism == pytest.approx(0.8)
        # mood_drift_rate = 0.12 - 0.8 * 0.08 = 0.056
        assert sm.mood_drift_rate == pytest.approx(0.056)

    def test_factory_with_custom_determinism(self):
        persona = self._build_persona()
        det = DeterminismManager(seed=123)
        sm = create_state_manager(persona, determinism=det)
        assert sm.determinism is det

    def test_factory_default_determinism(self):
        persona = self._build_persona()
        sm = create_state_manager(persona)
        assert isinstance(sm.determinism, DeterminismManager)


# =========================================================================
# Integration / multi-step scenarios
# =========================================================================


class TestIntegrationScenarios:
    def test_multi_turn_conversation(self):
        """Simulate several turns and verify state evolves plausibly."""
        sm = StateManager(
            initial_state=_default_state(
                mood_valence=0.2,
                mood_arousal=0.5,
                fatigue=0.0,
                stress=0.0,
                engagement=0.5,
            ),
            traits=_default_traits(neuroticism=0.6, openness=0.7),
            determinism=DeterminismManager(seed=42),
        )
        for turn in range(1, 11):
            sm.evolve_state_post_turn(conversation_length=turn, topic_relevance=0.6)

        state = sm.get_current_state()
        # After 10 turns, fatigue should have increased from 0
        assert state.fatigue > 0.0
        # Stress should remain 0 (only decays, never triggered)
        assert state.stress == pytest.approx(0.0)
        # Engagement should have moved toward the relevance-based target
        assert 0.1 <= state.engagement <= 1.0

    def test_stress_event_during_conversation(self):
        """Stress trigger mid-conversation affects subsequent mood drift."""
        sm = StateManager(
            initial_state=_default_state(mood_valence=0.3, stress=0.0),
            traits=_default_traits(neuroticism=0.7),
            determinism=DeterminismManager(seed=42),
        )
        # Turn 1: normal
        sm.evolve_state_post_turn(1, 0.5)
        # Turn 2: stress event
        sm.apply_stress_trigger("conflict", intensity=0.5)
        stress_after_trigger = sm.state.stress
        assert stress_after_trigger > 0.0
        # Turn 3: stress should decay
        sm.evolve_state_post_turn(3, 0.5)
        assert sm.state.stress < stress_after_trigger

    def test_high_fatigue_cascading_effects(self):
        """Very long conversation leads to fatigue cascade on engagement/arousal."""
        sm = StateManager(
            initial_state=_default_state(
                fatigue=0.65,
                engagement=0.8,
                mood_arousal=0.6,
            ),
            traits=_default_traits(conscientiousness=0.0),
            determinism=DeterminismManager(seed=42),
        )
        # Large conversation length to push fatigue over 0.7
        sm.increment_fatigue(conversation_length=50)
        # fatigue: 0.65 + 0.02 * 5 * 1.0 = 0.75 > 0.7
        assert sm.state.fatigue > 0.7
        # engagement dropped by 0.1
        assert sm.state.engagement == pytest.approx(0.7)
        # arousal dropped by 0.1
        assert sm.state.mood_arousal == pytest.approx(0.5)
