"""Tests for persona evolution — event-driven trait updates."""

import pytest
import numpy as np

from persona_engine.schema.persona_schema import Persona
import layer_zero
from layer_zero.evolution import evolve, list_events, LIFE_EVENTS


@pytest.fixture
def nurse_persona():
    return layer_zero.mint(occupation="nurse", age=35, count=1, seed=42)[0]


class TestLifeEvents:
    def test_all_events_defined(self):
        events = list_events()
        assert len(events) >= 10
        for e in events:
            assert "name" in e
            assert "description" in e

    def test_job_loss_increases_neuroticism(self, nurse_persona):
        original_n = nurse_persona.persona.psychology.big_five.neuroticism
        evolved = evolve(nurse_persona, event="job_loss")
        assert evolved.persona.psychology.big_five.neuroticism > original_n

    def test_promotion_decreases_neuroticism(self, nurse_persona):
        original_n = nurse_persona.persona.psychology.big_five.neuroticism
        evolved = evolve(nurse_persona, event="promotion")
        assert evolved.persona.psychology.big_five.neuroticism < original_n

    def test_new_child_increases_benevolence(self, nurse_persona):
        original_b = nurse_persona.persona.psychology.values.benevolence
        evolved = evolve(nurse_persona, event="new_child")
        assert evolved.persona.psychology.values.benevolence > original_b

    def test_retirement_decreases_achievement(self, nurse_persona):
        original_a = nurse_persona.persona.psychology.values.achievement
        evolved = evolve(nurse_persona, event="retirement")
        assert evolved.persona.psychology.values.achievement < original_a

    def test_bereavement_decreases_mood(self, nurse_persona):
        original_mood = nurse_persona.persona.initial_state.mood_valence
        evolved = evolve(nurse_persona, event="bereavement")
        assert evolved.persona.initial_state.mood_valence < original_mood


class TestEvolutionProperties:
    def test_original_not_modified(self, nurse_persona):
        original_n = nurse_persona.persona.psychology.big_five.neuroticism
        evolve(nurse_persona, event="job_loss")
        assert nurse_persona.persona.psychology.big_five.neuroticism == original_n

    def test_evolved_is_valid_persona(self, nurse_persona):
        evolved = evolve(nurse_persona, event="job_loss")
        assert isinstance(evolved.persona, Persona)

    def test_traits_stay_bounded(self, nurse_persona):
        # Apply extreme event with high intensity
        evolved = evolve(nurse_persona, event="health_crisis", intensity=2.0)
        bf = evolved.persona.psychology.big_five
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val = getattr(bf, trait)
            assert 0.0 <= val <= 1.0, f"{trait}={val} out of bounds"

    def test_intensity_scales_effect(self, nurse_persona):
        mild = evolve(nurse_persona, event="job_loss", intensity=0.5)
        strong = evolve(nurse_persona, event="job_loss", intensity=1.5)
        # Stronger intensity → bigger N shift
        original_n = nurse_persona.persona.psychology.big_five.neuroticism
        mild_delta = abs(mild.persona.psychology.big_five.neuroticism - original_n)
        strong_delta = abs(strong.persona.psychology.big_five.neuroticism - original_n)
        assert strong_delta > mild_delta

    def test_deterministic_with_seed(self, nurse_persona):
        e1 = evolve(nurse_persona, event="job_loss", seed=42)
        e2 = evolve(nurse_persona, event="job_loss", seed=42)
        assert e1.persona.psychology.big_five.neuroticism == e2.persona.psychology.big_five.neuroticism

    def test_evolution_note_in_warnings(self, nurse_persona):
        evolved = evolve(nurse_persona, event="promotion")
        assert any("Evolved" in w for w in evolved.warnings)


class TestNaturalAging:
    def test_aging_increases_agreeableness(self, nurse_persona):
        original_a = nurse_persona.persona.psychology.big_five.agreeableness
        aged = evolve(nurse_persona, years=20)
        assert aged.persona.psychology.big_five.agreeableness > original_a

    def test_aging_decreases_neuroticism(self, nurse_persona):
        original_n = nurse_persona.persona.psychology.big_five.neuroticism
        aged = evolve(nurse_persona, years=20)
        assert aged.persona.psychology.big_five.neuroticism < original_n

    def test_aging_updates_age(self, nurse_persona):
        aged = evolve(nurse_persona, years=10)
        assert aged.persona.identity.age == 45  # 35 + 10

    def test_aging_increases_tradition(self, nurse_persona):
        original_t = nurse_persona.persona.psychology.values.tradition
        aged = evolve(nurse_persona, years=20)
        assert aged.persona.psychology.values.tradition > original_t

    def test_combined_event_and_aging(self, nurse_persona):
        """Event + aging applied together."""
        evolved = evolve(nurse_persona, event="promotion", years=5)
        assert evolved.persona.identity.age == 40
        assert isinstance(evolved.persona, Persona)
