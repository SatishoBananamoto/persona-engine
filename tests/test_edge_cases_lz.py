"""Edge case tests for Layer Zero robustness."""

import pytest
import numpy as np
from persona_engine.schema.persona_schema import Persona
import layer_zero


class TestMinimalInput:
    def test_no_inputs_at_all(self):
        """Mint with zero information — should still produce valid persona."""
        personas = layer_zero.mint(count=1)
        assert isinstance(personas[0].persona, Persona)

    def test_only_count(self):
        personas = layer_zero.mint(count=5, seed=42)
        assert len(personas) == 5

    def test_only_occupation(self):
        personas = layer_zero.mint(occupation="nurse", count=1)
        assert personas[0].persona.identity.occupation == "nurse"

    def test_only_age(self):
        personas = layer_zero.mint(age=25, count=1)
        assert personas[0].persona.identity.age == 25


class TestExtremeInputs:
    def test_age_18(self):
        personas = layer_zero.mint(age=18, occupation="student", count=1)
        assert personas[0].persona.identity.age == 18

    def test_age_100(self):
        personas = layer_zero.mint(age=100, count=1)
        assert personas[0].persona.identity.age == 100

    def test_all_big_five_at_zero(self):
        personas = layer_zero.mint(
            big_five={"openness": 0.01, "conscientiousness": 0.01, "extraversion": 0.01,
                      "agreeableness": 0.01, "neuroticism": 0.01},
            count=1,
        )
        bf = personas[0].persona.psychology.big_five
        assert bf.openness < 0.1

    def test_all_big_five_at_one(self):
        personas = layer_zero.mint(
            big_five={"openness": 0.99, "conscientiousness": 0.99, "extraversion": 0.99,
                      "agreeableness": 0.99, "neuroticism": 0.99},
            count=1,
        )
        bf = personas[0].persona.psychology.big_five
        assert bf.openness > 0.9

    def test_contradictory_traits(self):
        """Introverted salesperson — should produce valid persona with warning."""
        personas = layer_zero.mint(
            occupation="salesperson",
            big_five={"extraversion": 0.1},
            count=1,
            validate="warn",
        )
        assert isinstance(personas[0].persona, Persona)

    def test_large_batch(self):
        """50 personas should all be valid and distinct."""
        personas = layer_zero.mint(occupation="nurse", count=50, seed=42)
        assert len(personas) == 50
        ids = {p.persona_id for p in personas}
        assert len(ids) == 50  # all unique


class TestUnknownOccupation:
    def test_unknown_still_works(self):
        personas = layer_zero.mint(occupation="astronaut", count=1)
        assert isinstance(personas[0].persona, Persona)

    def test_unknown_gets_general_domain(self):
        personas = layer_zero.mint(occupation="astronaut", count=1)
        domains = personas[0].persona.knowledge_domains
        assert any(d.domain == "General" for d in domains)


class TestDescriptionEdgeCases:
    def test_short_description(self):
        personas = layer_zero.from_description("engineer", count=1)
        assert isinstance(personas[0].persona, Persona)

    def test_long_description(self):
        desc = "A 42-year-old senior software engineer based in Berlin, Germany who is analytical, methodical, and values precision in everything"
        personas = layer_zero.from_description(desc, count=1)
        assert personas[0].persona.identity.age == 42

    def test_description_with_no_occupation(self):
        personas = layer_zero.from_description("A 30-year-old person from Tokyo", count=1)
        assert isinstance(personas[0].persona, Persona)


class TestValueOverrides:
    def test_single_value_override_pinned(self):
        personas = layer_zero.mint(
            values={"power": 0.9},
            count=5, seed=42,
        )
        for p in personas:
            assert p.persona.psychology.values.power == pytest.approx(0.9, abs=0.01)

    def test_cognitive_override_pinned(self):
        personas = layer_zero.mint(
            cognitive={"risk_tolerance": 0.9},
            count=1,
        )
        assert personas[0].persona.psychology.cognitive_style.risk_tolerance == pytest.approx(0.9, abs=0.01)


class TestReproducibility:
    def test_identical_across_runs(self):
        """Same parameters → identical personas across separate calls."""
        p1 = layer_zero.mint(occupation="chef", age=40, count=3, seed=123)
        p2 = layer_zero.mint(occupation="chef", age=40, count=3, seed=123)
        for a, b in zip(p1, p2):
            assert a.persona.psychology.big_five.openness == b.persona.psychology.big_five.openness
            assert a.persona.psychology.values.achievement == b.persona.psychology.values.achievement
