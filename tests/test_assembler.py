"""Tests for persona assembler — engine integration."""

import pytest
import numpy as np

from layer_zero.models import MintRequest, BIG_FIVE_TRAITS, SCHWARTZ_VALUES, TraitPrior
from layer_zero.gap_filler import fill_gaps
from layer_zero.policy import apply_policy_defaults
from layer_zero.assembler import assemble_persona


def _make_assembled(occupation="nurse", age=35, seed=42, persona_index=0):
    """Helper: run full gap_filler + policy + assembler pipeline."""
    bf = {t: 0.5 for t in BIG_FIVE_TRAITS}
    sv = {v: 0.5 for v in SCHWARTZ_VALUES}
    priors = {t: TraitPrior(mean=0.5, std_dev=0.15) for t in BIG_FIVE_TRAITS}
    req = MintRequest(occupation=occupation, age=age)

    filled = fill_gaps(bf, sv, req, priors, seed=seed, persona_index=persona_index)
    policy, policy_prov = apply_policy_defaults(filled, occupation=occupation)
    if "_provenance" not in filled:
        filled["_provenance"] = {}
    filled["_provenance"].update(policy_prov)

    return assemble_persona(
        big_five=bf, schwartz=sv, filled=filled, policy=policy,
        request_occupation=occupation, request_age=age,
        seed=seed, persona_index=persona_index,
    )


class TestAssemblyBasic:
    def test_returns_minted_persona(self):
        mp = _make_assembled()
        assert mp.persona is not None
        assert mp.provenance is not None

    def test_persona_has_valid_id(self):
        mp = _make_assembled()
        assert mp.persona_id.startswith("P_GEN_")
        assert "_0" in mp.persona_id  # batch index

    def test_persona_has_label(self):
        mp = _make_assembled()
        assert "nurse" in mp.label.lower() or "Nurse" in mp.label

    def test_unique_ids_in_batch(self):
        ids = set()
        for i in range(10):
            mp = _make_assembled(persona_index=i)
            ids.add(mp.persona_id)
        assert len(ids) == 10

    def test_same_seed_same_id(self):
        mp1 = _make_assembled(seed=42, persona_index=0)
        mp2 = _make_assembled(seed=42, persona_index=0)
        assert mp1.persona_id == mp2.persona_id


class TestEngineCompatibility:
    def test_persona_is_engine_type(self):
        from persona_engine.schema.persona_schema import Persona
        mp = _make_assembled()
        assert isinstance(mp.persona, Persona)

    def test_persona_loads_into_engine(self):
        from persona_engine import PersonaEngine
        mp = _make_assembled()
        engine = PersonaEngine(mp.persona, llm_provider="template")
        assert engine.persona.persona_id == mp.persona_id

    def test_engine_can_plan(self):
        from persona_engine import PersonaEngine
        mp = _make_assembled()
        engine = PersonaEngine(mp.persona, llm_provider="template")
        ir = engine.plan("What do you think about healthcare?")
        assert ir.response_structure.confidence > 0
        assert ir.communication_style.tone is not None


class TestProvenance:
    def test_big_five_provenance_exists(self):
        mp = _make_assembled()
        for trait in BIG_FIVE_TRAITS:
            key = f"psychology.big_five.{trait}"
            assert key in mp.provenance, f"Missing provenance for {key}"

    def test_schwartz_provenance_exists(self):
        mp = _make_assembled()
        for val in SCHWARTZ_VALUES:
            key = f"psychology.values.{val}"
            assert key in mp.provenance, f"Missing provenance for {key}"

    def test_cognitive_provenance_exists(self):
        mp = _make_assembled()
        assert "psychology.cognitive_style.analytical_intuitive" in mp.provenance

    def test_provenance_confidence_hierarchy(self):
        mp = _make_assembled()
        # Sampled fields should have higher confidence than derived
        sampled_conf = mp.provenance["psychology.big_five.openness"].confidence
        derived_conf = mp.provenance["psychology.cognitive_style.analytical_intuitive"].confidence
        assert sampled_conf > derived_conf
