"""Tests for the public API — mint() and from_description()."""

import pytest
from persona_engine import PersonaEngine
from persona_engine.schema.persona_schema import Persona

import layer_zero


class TestMint:
    def test_basic_mint(self):
        personas = layer_zero.mint(occupation="nurse", count=5)
        assert len(personas) == 5
        for p in personas:
            assert isinstance(p.persona, Persona)

    def test_mint_with_age(self):
        personas = layer_zero.mint(occupation="nurse", age=35, count=1)
        assert personas[0].persona.identity.age == 35

    def test_mint_with_big_five(self):
        personas = layer_zero.mint(
            occupation="researcher",
            big_five={"openness": 0.9},
            count=1,
        )
        assert personas[0].persona.psychology.big_five.openness == pytest.approx(0.9, abs=0.05)

    def test_deterministic_with_seed(self):
        p1 = layer_zero.mint(occupation="nurse", count=3, seed=42)
        p2 = layer_zero.mint(occupation="nurse", count=3, seed=42)
        for a, b in zip(p1, p2):
            assert a.persona_id == b.persona_id

    def test_different_seeds_differ(self):
        p1 = layer_zero.mint(occupation="nurse", count=1, seed=42)
        p2 = layer_zero.mint(occupation="nurse", count=1, seed=99)
        # Big Five should differ
        bf1 = p1[0].persona.psychology.big_five
        bf2 = p2[0].persona.psychology.big_five
        assert bf1.openness != pytest.approx(bf2.openness, abs=0.01)

    def test_10_nurses_are_distinct(self):
        personas = layer_zero.mint(occupation="nurse", count=10)
        ids = {p.persona_id for p in personas}
        assert len(ids) == 10
        # Big Five should vary
        openness_vals = [p.persona.psychology.big_five.openness for p in personas]
        import numpy as np
        assert np.std(openness_vals) > 0.02

    def test_mint_with_location(self):
        personas = layer_zero.mint(occupation="nurse", location="Tokyo, Japan", count=1)
        assert personas[0].persona.identity.location == "Tokyo, Japan"

    def test_mint_with_traits(self):
        personas = layer_zero.mint(occupation="nurse", traits=["analytical", "warm"], count=1)
        assert isinstance(personas[0].persona, Persona)

    def test_mint_minimal_input(self):
        personas = layer_zero.mint(count=3)
        assert len(personas) == 3
        for p in personas:
            assert isinstance(p.persona, Persona)


class TestFromDescription:
    def test_basic_description(self):
        personas = layer_zero.from_description(
            "A 35-year-old product manager in fintech",
            count=3,
        )
        assert len(personas) == 3
        assert personas[0].persona.identity.age == 35

    def test_description_with_traits(self):
        personas = layer_zero.from_description(
            "A cautious nurse from Chicago",
            count=1,
        )
        assert personas[0].persona.identity.occupation == "nurse"

    def test_description_count(self):
        personas = layer_zero.from_description("An engineer", count=5)
        assert len(personas) == 5


class TestEngineIntegration:
    def test_mint_and_plan(self):
        """Full pipeline: mint → engine → IR."""
        personas = layer_zero.mint(occupation="nurse", age=30, count=3)
        for mp in personas:
            engine = PersonaEngine(mp.persona, llm_provider="template")
            ir = engine.plan("What do you think about patient care?")
            assert ir.response_structure.confidence > 0
            assert ir.communication_style.tone is not None

    def test_mint_and_chat(self):
        """Full pipeline: mint → engine → chat."""
        personas = layer_zero.mint(occupation="chef", age=40, count=1)
        engine = PersonaEngine(personas[0].persona, llm_provider="template")
        result = engine.chat("What makes a good sauce?")
        assert len(result.text) > 0
        assert result.passed

    def test_different_occupations_different_ir(self):
        """Different occupations should produce measurably different IR."""
        nurse = layer_zero.mint(occupation="nurse", count=1, seed=42)
        artist = layer_zero.mint(occupation="artist", count=1, seed=42)

        e1 = PersonaEngine(nurse[0].persona, llm_provider="template")
        e2 = PersonaEngine(artist[0].persona, llm_provider="template")

        ir1 = e1.plan("What do you think about creativity?")
        ir2 = e2.plan("What do you think about creativity?")

        # Artist should have higher openness → higher elasticity
        assert ir1.response_structure.elasticity != pytest.approx(
            ir2.response_structure.elasticity, abs=0.05
        )


class TestProvenance:
    def test_provenance_present(self):
        personas = layer_zero.mint(occupation="nurse", count=1)
        mp = personas[0]
        assert len(mp.provenance) > 0
        assert "psychology.big_five.openness" in mp.provenance

    def test_provenance_confidence_range(self):
        personas = layer_zero.mint(occupation="nurse", count=1)
        for key, prov in personas[0].provenance.items():
            assert 0.0 <= prov.confidence <= 1.0


class TestValidation:
    def test_warn_mode_returns_warnings(self):
        personas = layer_zero.mint(occupation="nurse", count=1, validate="warn")
        # May or may not have warnings — just shouldn't crash
        assert isinstance(personas[0].warnings, list)

    def test_silent_mode_no_warnings(self):
        personas = layer_zero.mint(occupation="nurse", count=1, validate="silent")
        assert personas[0].warnings == []
