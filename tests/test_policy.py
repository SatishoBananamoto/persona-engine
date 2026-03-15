"""Tests for policy applier — system-governed defaults."""

import pytest
from layer_zero.policy import apply_policy_defaults, SYSTEM_POLICY_DEFAULTS


def _persona_fields(**overrides):
    base = {
        "privacy_sensitivity": 0.5,
        "age": 35,
        "occupation": "nurse",
        "location": "Chicago",
        "_provenance": {},
    }
    base.update(overrides)
    return base


class TestPolicyInvariance:
    def test_claim_policy_same_regardless_of_personality(self):
        """High-openness and low-openness personas get same claim policy."""
        fields_high_o = _persona_fields(privacy_sensitivity=0.2)
        fields_low_o = _persona_fields(privacy_sensitivity=0.8)
        p1 = apply_policy_defaults(fields_high_o, "nurse")
        p2 = apply_policy_defaults(fields_low_o, "nurse")
        assert p1["claim_policy"]["expert_threshold"] == p2["claim_policy"]["expert_threshold"]
        assert p1["claim_policy"]["lookup_behavior"] == p2["claim_policy"]["lookup_behavior"]
        assert p1["claim_policy"]["allowed_claim_types"] == p2["claim_policy"]["allowed_claim_types"]

    def test_uncertainty_policy_same(self):
        p1 = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.2), "nurse")
        p2 = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.8), "nurse")
        assert p1["uncertainty"] == p2["uncertainty"]

    def test_disclosure_bounds_same(self):
        p1 = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.2), "nurse")
        p2 = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.8), "nurse")
        assert p1["disclosure_policy"]["bounds"] == p2["disclosure_policy"]["bounds"]

    def test_expert_threshold_is_system_default(self):
        p = apply_policy_defaults(_persona_fields(), "nurse")
        assert p["claim_policy"]["expert_threshold"] == 0.7

    def test_lookup_behavior_is_hedge(self):
        p = apply_policy_defaults(_persona_fields(), "nurse")
        assert p["claim_policy"]["lookup_behavior"] == "hedge"


class TestDisclosurePersonaInfluence:
    def test_low_privacy_higher_openness(self):
        p = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.2), "nurse")
        assert p["disclosure_policy"]["base_openness"] > 0.7

    def test_high_privacy_lower_openness(self):
        p = apply_policy_defaults(_persona_fields(privacy_sensitivity=0.8), "nurse")
        assert p["disclosure_policy"]["base_openness"] < 0.3


class TestInvariants:
    def test_identity_facts_generated(self):
        p = apply_policy_defaults(_persona_fields(), "nurse")
        facts = p["invariants"]["identity_facts"]
        assert any("35" in f for f in facts)  # age
        assert any("Nurse" in f for f in facts)  # occupation
        assert any("Chicago" in f for f in facts)  # location

    def test_cannot_claim_for_nurse(self):
        p = apply_policy_defaults(_persona_fields(), "nurse")
        assert "licensed physician" in p["invariants"]["cannot_claim"]

    def test_must_avoid_includes_default(self):
        p = apply_policy_defaults(_persona_fields(), "nurse")
        assert "revealing private personal information" in p["invariants"]["must_avoid"]

    def test_no_occupation_empty_cannot_claim(self):
        p = apply_policy_defaults(_persona_fields(occupation=None), None)
        assert p["invariants"]["cannot_claim"] == []


class TestProvenance:
    def test_policy_fields_have_high_confidence(self):
        fields = _persona_fields()
        p = apply_policy_defaults(fields, "nurse")
        prov = fields["_provenance"]
        assert prov["uncertainty.knowledge_boundary_strictness"].confidence >= 0.9
        assert prov["claim_policy.expert_threshold"].confidence >= 0.9

    def test_disclosure_openness_has_lower_confidence(self):
        fields = _persona_fields()
        apply_policy_defaults(fields, "nurse")
        prov = fields["_provenance"]
        assert prov["disclosure_policy.base_openness"].confidence < 0.5
        assert prov["disclosure_policy.base_openness"].source == "derived"
