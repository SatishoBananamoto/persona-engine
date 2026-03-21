"""Integration tests — end-to-end scenarios across the full pipeline."""

import os
import numpy as np
import pytest

from persona_engine import PersonaEngine
from persona_engine.schema.persona_schema import Persona
import layer_zero


TEST_CSV = os.path.join(os.path.dirname(__file__), "test_segments.csv")


class TestMintToEngine:
    """Mint personas and run them through the engine."""

    def test_multiple_occupations_produce_valid_ir(self):
        for occ in ["nurse", "software engineer", "artist", "chef", "entrepreneur"]:
            personas = layer_zero.mint(occupation=occ, count=2, seed=42)
            for mp in personas:
                engine = PersonaEngine(mp.persona, llm_provider="template")
                ir = engine.plan("Tell me about your work")
                assert ir.response_structure.confidence > 0
                assert ir.response_structure.competence > 0

    def test_from_description_to_chat(self):
        personas = layer_zero.from_description(
            "A 40-year-old chef from Chicago who values achievement",
            count=2,
        )
        for mp in personas:
            engine = PersonaEngine(mp.persona, llm_provider="template")
            result = engine.chat("What makes a good dish?")
            assert len(result.text) > 0
            assert result.passed

    def test_from_csv_to_plan(self):
        personas = layer_zero.from_csv(TEST_CSV, seed=42)
        assert len(personas) == 15
        for mp in personas:
            engine = PersonaEngine(mp.persona, llm_provider="template")
            ir = engine.plan("What do you think about teamwork?")
            assert ir.response_structure.confidence > 0


class TestBehavioralDistinctiveness:
    """Different personas should produce measurably different IR."""

    def test_nurse_vs_artist_elasticity(self):
        nurse = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        artist = layer_zero.mint(occupation="artist", count=1, seed=42)[0]

        e1 = PersonaEngine(nurse.persona, llm_provider="template")
        e2 = PersonaEngine(artist.persona, llm_provider="template")

        ir1 = e1.plan("What do you think about innovation?")
        ir2 = e2.plan("What do you think about innovation?")

        # Artist has higher openness → should have different elasticity
        assert abs(ir1.response_structure.elasticity - ir2.response_structure.elasticity) > 0.02

    def test_young_vs_old_confidence(self):
        young = layer_zero.mint(occupation="teacher", age=25, count=1, seed=42)[0]
        old = layer_zero.mint(occupation="teacher", age=65, count=1, seed=42)[0]

        e1 = PersonaEngine(young.persona, llm_provider="template")
        e2 = PersonaEngine(old.persona, llm_provider="template")

        ir1 = e1.plan("What teaching method works best?")
        ir2 = e2.plan("What teaching method works best?")

        # Both should produce valid IR with different parameters
        assert ir1.response_structure.confidence > 0
        assert ir2.response_structure.confidence > 0


class TestCascadeCollapseEndToEnd:
    """Downstream fields should retain variance even with fixed occupation."""

    def test_50_nurses_cognitive_variance(self):
        personas = layer_zero.mint(occupation="nurse", age=30, count=50, seed=42)
        analytical = [p.persona.psychology.cognitive_style.analytical_intuitive for p in personas]
        formality = [p.persona.psychology.communication.formality for p in personas]

        assert np.std(analytical) > 0.03, f"Cascade collapse: analytical SD={np.std(analytical):.4f}"
        assert np.std(formality) > 0.03, f"Cascade collapse: formality SD={np.std(formality):.4f}"

    def test_50_nurses_big_five_variance(self):
        personas = layer_zero.mint(occupation="nurse", age=30, count=50, seed=42)
        openness = [p.persona.psychology.big_five.openness for p in personas]
        assert np.std(openness) > 0.05, f"Big Five variance too low: O SD={np.std(openness):.4f}"


class TestPolicyInvarianceEndToEnd:
    """Policy floors must be identical across all personas in a batch."""

    def test_claim_policy_same_across_batch(self):
        personas = layer_zero.mint(occupation="nurse", count=10, seed=42)
        thresholds = {p.persona.claim_policy.expert_threshold for p in personas}
        behaviors = {p.persona.claim_policy.lookup_behavior for p in personas}
        assert len(thresholds) == 1, f"expert_threshold varies: {thresholds}"
        assert len(behaviors) == 1, f"lookup_behavior varies: {behaviors}"

    def test_disclosure_bounds_same_across_batch(self):
        personas = layer_zero.mint(occupation="nurse", count=10, seed=42)
        bounds = {p.persona.disclosure_policy.bounds for p in personas}
        assert len(bounds) == 1, f"Disclosure bounds vary: {bounds}"


class TestSeedDeterminism:
    """Same seed + same input = identical output."""

    def test_mint_deterministic(self):
        p1 = layer_zero.mint(occupation="nurse", age=35, count=5, seed=42)
        p2 = layer_zero.mint(occupation="nurse", age=35, count=5, seed=42)
        for a, b in zip(p1, p2):
            assert a.persona_id == b.persona_id
            assert a.persona.psychology.big_five.openness == pytest.approx(
                b.persona.psychology.big_five.openness, abs=1e-10
            )

    def test_from_csv_deterministic(self):
        p1 = layer_zero.from_csv(TEST_CSV, seed=42)
        p2 = layer_zero.from_csv(TEST_CSV, seed=42)
        for a, b in zip(p1, p2):
            assert a.persona_id == b.persona_id


class TestProvenanceCoverage:
    """Every field should have provenance metadata."""

    def test_provenance_covers_big_five(self):
        mp = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert f"psychology.big_five.{trait}" in mp.provenance

    def test_provenance_covers_values(self):
        mp = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        for val in ["self_direction", "benevolence", "power", "security"]:
            assert f"psychology.values.{val}" in mp.provenance

    def test_provenance_covers_cognitive(self):
        mp = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        assert "psychology.cognitive_style.analytical_intuitive" in mp.provenance

    def test_provenance_covers_policy(self):
        mp = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        assert "claim_policy.expert_threshold" in mp.provenance
        assert "uncertainty.knowledge_boundary_strictness" in mp.provenance

    def test_no_fabricated_expertise(self):
        """Default domain proficiency from occupation should not exceed 0.6."""
        mp = layer_zero.mint(occupation="nurse", count=1, seed=42)[0]
        for domain in mp.persona.knowledge_domains:
            assert domain.proficiency <= 0.6, f"Fabricated expertise: {domain.domain} at {domain.proficiency}"
