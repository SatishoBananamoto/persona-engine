"""Tests for gap filler with calibrated residuals."""

import numpy as np
import pytest

from layer_zero.models import MintRequest, BIG_FIVE_TRAITS, SCHWARTZ_VALUES, TraitPrior
from layer_zero.gap_filler import fill_gaps


def _default_big_five(o=0.5, c=0.5, e=0.5, a=0.5, n=0.5):
    return {"openness": o, "conscientiousness": c, "extraversion": e,
            "agreeableness": a, "neuroticism": n}


def _default_values():
    return {v: 0.5 for v in SCHWARTZ_VALUES}


def _default_priors():
    return {t: TraitPrior(mean=0.5, std_dev=0.15) for t in BIG_FIVE_TRAITS}


def _fill(**kwargs):
    big_five = kwargs.pop("big_five", _default_big_five())
    values = kwargs.pop("values", _default_values())
    priors = kwargs.pop("priors", _default_priors())
    seed = kwargs.pop("seed", 42)
    idx = kwargs.pop("persona_index", 0)
    req = MintRequest(**kwargs)
    return fill_gaps(big_five, values, req, priors, seed=seed, persona_index=idx)


class TestBasicGapFilling:
    def test_returns_all_required_keys(self):
        result = _fill()
        assert "name" in result
        assert "education" in result
        assert "cognitive_style" in result
        assert "communication" in result
        assert "knowledge_domains" in result
        assert "goals" in result
        assert "social_roles" in result
        assert "initial_state" in result
        assert "decision_tendencies" in result
        assert "privacy_sensitivity" in result

    def test_cognitive_style_has_all_fields(self):
        result = _fill()
        cs = result["cognitive_style"]
        assert "analytical_intuitive" in cs
        assert "systematic_heuristic" in cs
        assert "risk_tolerance" in cs
        assert "need_for_closure" in cs
        assert "cognitive_complexity" in cs

    def test_communication_has_all_fields(self):
        result = _fill()
        comm = result["communication"]
        assert "verbosity" in comm
        assert "formality" in comm
        assert "directness" in comm
        assert "emotional_expressiveness" in comm

    def test_all_values_bounded(self):
        result = _fill()
        for field in result["cognitive_style"].values():
            assert 0.0 <= field <= 1.0
        for field in result["communication"].values():
            assert 0.0 <= field <= 1.0
        assert 0.0 <= result["privacy_sensitivity"] <= 1.0


class TestCascadePrevention:
    def test_same_big_five_different_seeds_differ(self):
        """Two personas with identical Big Five but different seeds should differ."""
        bf = _default_big_five(o=0.7, c=0.6)
        r1 = fill_gaps(bf, _default_values(), MintRequest(), _default_priors(), seed=42, persona_index=0)
        r2 = fill_gaps(bf, _default_values(), MintRequest(), _default_priors(), seed=42, persona_index=1)
        # Cognitive styles should differ
        cs1 = r1["cognitive_style"]["analytical_intuitive"]
        cs2 = r2["cognitive_style"]["analytical_intuitive"]
        assert cs1 != pytest.approx(cs2, abs=0.001), "Cascade collapse: identical downstream despite different index"

    def test_batch_cognitive_has_variance(self):
        """50 personas from same Big Five should show variance in cognitive style."""
        bf = _default_big_five(o=0.6, c=0.5, e=0.5, a=0.5, n=0.5)
        vals = [
            fill_gaps(bf, _default_values(), MintRequest(), _default_priors(), seed=42, persona_index=i)["cognitive_style"]["analytical_intuitive"]
            for i in range(50)
        ]
        std = np.std(vals)
        assert std > 0.03, f"Cascade collapse: cognitive SD={std:.4f} too low for 50 personas"

    def test_batch_communication_has_variance(self):
        """50 personas from same Big Five should show variance in communication."""
        bf = _default_big_five()
        vals = [
            fill_gaps(bf, _default_values(), MintRequest(), _default_priors(), seed=42, persona_index=i)["communication"]["formality"]
            for i in range(50)
        ]
        std = np.std(vals)
        assert std > 0.03, f"Cascade collapse: formality SD={std:.4f} too low"


class TestTraitInfluence:
    def test_high_openness_more_analytical(self):
        high_o = [fill_gaps(_default_big_five(o=0.9), _default_values(), MintRequest(),
                           _default_priors(), seed=42, persona_index=i)["cognitive_style"]["analytical_intuitive"]
                  for i in range(50)]
        low_o = [fill_gaps(_default_big_five(o=0.1), _default_values(), MintRequest(),
                          _default_priors(), seed=42, persona_index=i)["cognitive_style"]["analytical_intuitive"]
                 for i in range(50)]
        assert np.mean(high_o) > np.mean(low_o)

    def test_high_extraversion_more_expressive(self):
        high_e = [fill_gaps(_default_big_five(e=0.9), _default_values(), MintRequest(),
                           _default_priors(), seed=42, persona_index=i)["communication"]["emotional_expressiveness"]
                  for i in range(50)]
        low_e = [fill_gaps(_default_big_five(e=0.1), _default_values(), MintRequest(),
                          _default_priors(), seed=42, persona_index=i)["communication"]["emotional_expressiveness"]
                 for i in range(50)]
        assert np.mean(high_e) > np.mean(low_e)

    def test_high_neuroticism_more_stress(self):
        high_n = [fill_gaps(_default_big_five(n=0.9), _default_values(), MintRequest(),
                           _default_priors(), seed=42, persona_index=i)["initial_state"]["stress"]
                  for i in range(50)]
        low_n = [fill_gaps(_default_big_five(n=0.1), _default_values(), MintRequest(),
                          _default_priors(), seed=42, persona_index=i)["initial_state"]["stress"]
                 for i in range(50)]
        assert np.mean(high_n) > np.mean(low_n)


class TestOverrides:
    def test_cognitive_override_respected(self):
        result = _fill(cognitive_overrides={"analytical_intuitive": 0.95})
        assert result["cognitive_style"]["analytical_intuitive"] == pytest.approx(0.95, abs=0.01)

    def test_communication_override_respected(self):
        result = _fill(communication_overrides={"formality": 0.9})
        assert result["communication"]["formality"] == pytest.approx(0.9, abs=0.01)

    def test_explicit_name_kept(self):
        result = _fill(name="Satish")
        assert result["name"] == "Satish"

    def test_explicit_goals_kept(self):
        result = _fill(goals=["Cure cancer"])
        assert result["goals"] == ["Cure cancer"]

    def test_explicit_domains_kept(self):
        domains = [{"domain": "Astrophysics", "proficiency": 0.9}]
        result = _fill(domains=domains)
        assert result["knowledge_domains"] == domains


class TestDomainInference:
    def test_nurse_gets_healthcare(self):
        result = _fill(occupation="nurse")
        assert result["knowledge_domains"][0]["domain"] == "Healthcare"
        assert result["knowledge_domains"][0]["proficiency"] <= 0.6

    def test_unknown_occupation_gets_general(self):
        result = _fill(occupation="astronaut")
        assert result["knowledge_domains"][0]["domain"] == "General"

    def test_no_occupation_gets_general(self):
        result = _fill()
        assert result["knowledge_domains"][0]["domain"] == "General"


class TestProvenance:
    def test_provenance_attached(self):
        result = _fill()
        assert "_provenance" in result
        prov = result["_provenance"]
        assert "identity.name" in prov
        assert "psychology.cognitive_style.analytical_intuitive" in prov

    def test_explicit_name_has_high_confidence(self):
        result = _fill(name="Alex")
        prov = result["_provenance"]["identity.name"]
        assert prov.source == "explicit"
        assert prov.confidence > 0.9

    def test_derived_field_has_lower_confidence(self):
        result = _fill()
        prov = result["_provenance"]["psychology.cognitive_style.analytical_intuitive"]
        assert prov.source == "derived"
        assert prov.confidence < 0.5
