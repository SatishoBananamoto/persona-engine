"""Tests for consistency validator."""

import numpy as np
import pytest

from layer_zero.models import SCHWARTZ_VALUES
from layer_zero.validator import (
    validate_persona,
    validate_batch_diversity,
    validate_cascade_collapse,
    ValidationResult,
)


def _bf(**overrides):
    base = {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
            "agreeableness": 0.5, "neuroticism": 0.5}
    base.update(overrides)
    return base


def _vals(**overrides):
    base = {v: 0.5 for v in SCHWARTZ_VALUES}
    base.update(overrides)
    return base


def _cog(**overrides):
    base = {"analytical_intuitive": 0.5, "systematic_heuristic": 0.5,
            "risk_tolerance": 0.5, "need_for_closure": 0.5, "cognitive_complexity": 0.5}
    base.update(overrides)
    return base


def _comm():
    return {"verbosity": 0.5, "formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5}


class TestRule1MetatraitCoherence:
    def test_no_warning_normal_profile(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm())
        assert not any(w.rule == "metatrait_stability" for w in r.warnings)

    def test_warning_all_high_stability(self):
        r = validate_persona(_bf(agreeableness=0.9, conscientiousness=0.9, neuroticism=0.9),
                            _vals(), _cog(), _comm())
        assert any(w.rule == "metatrait_stability" for w in r.warnings)


class TestRule2CognitiveCompatibility:
    def test_no_warning_normal(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm())
        assert not any(w.rule == "cognitive_compatibility" for w in r.warnings)

    def test_warning_risk_closure_both_high(self):
        r = validate_persona(_bf(), _vals(), _cog(risk_tolerance=0.9, need_for_closure=0.9), _comm())
        assert any(w.rule == "cognitive_compatibility" for w in r.warnings)


class TestRule4OpposingValues:
    def test_no_warning_normal(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm())
        assert not any(w.rule == "opposing_value_conflict" for w in r.warnings)

    def test_warning_power_universalism_both_high(self):
        r = validate_persona(_bf(), _vals(power=0.9, universalism=0.9), _cog(), _comm())
        assert any(w.rule == "opposing_value_conflict" for w in r.warnings)


class TestRule7DomainExpertise:
    def test_no_warning_matching_domain(self):
        domains = [{"domain": "Healthcare", "proficiency": 0.8}]
        r = validate_persona(_bf(), _vals(), _cog(), _comm(),
                            knowledge_domains=domains, occupation="healthcare nurse")
        assert not any(w.rule == "domain_expertise" for w in r.warnings)

    def test_warning_mismatched_domain(self):
        domains = [{"domain": "Astrophysics", "proficiency": 0.9}]
        r = validate_persona(_bf(), _vals(), _cog(), _comm(),
                            knowledge_domains=domains, occupation="nurse")
        assert any(w.rule == "domain_expertise" for w in r.warnings)


class TestRule8DisclosurePrivacy:
    def test_no_warning_normal(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm(),
                            disclosure_base_openness=0.5, privacy_sensitivity=0.5)
        assert not any(w.rule == "disclosure_privacy" for w in r.warnings)

    def test_warning_both_high(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm(),
                            disclosure_base_openness=0.9, privacy_sensitivity=0.9)
        assert any(w.rule == "disclosure_privacy" for w in r.warnings)


class TestRule10CulturalConfidence:
    def test_western_no_warning(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm(), culture_region="western")
        assert not any(w.rule == "cultural_confidence" for w in r.warnings)

    def test_sub_saharan_african_info(self):
        r = validate_persona(_bf(), _vals(), _cog(), _comm(), culture_region="sub_saharan_african")
        assert any(w.rule == "cultural_confidence" for w in r.warnings)


class TestBatchDiversity:
    def test_diverse_batch_no_warning(self):
        batch = np.random.default_rng(42).uniform(0, 1, (10, 5))
        warnings = validate_batch_diversity(batch)
        assert len(warnings) == 0 or not any(w.rule == "batch_diversity" for w in warnings)

    def test_identical_batch_warning(self):
        batch = np.tile([0.5, 0.5, 0.5, 0.5, 0.5], (10, 1))
        warnings = validate_batch_diversity(batch)
        assert any(w.rule == "batch_diversity" for w in warnings)


class TestCascadeCollapse:
    def test_varied_downstream_no_warning(self):
        values = list(np.random.default_rng(42).normal(0.5, 0.1, 50))
        warnings = validate_cascade_collapse(values, "cognitive_style.analytical")
        assert not any(w.rule == "cascade_collapse" for w in warnings)

    def test_flat_downstream_warning(self):
        values = [0.5] * 50
        warnings = validate_cascade_collapse(values, "cognitive_style.analytical")
        assert any(w.rule == "cascade_collapse" for w in warnings)


class TestModes:
    def test_silent_mode_no_warnings(self):
        r = validate_persona(
            _bf(agreeableness=0.9, conscientiousness=0.9, neuroticism=0.9),
            _vals(power=0.9, universalism=0.9),
            _cog(risk_tolerance=0.9, need_for_closure=0.9),
            _comm(),
            mode="silent",
        )
        assert r.passed
        assert len(r.warnings) == 0

    def test_strict_mode_raises(self):
        with pytest.raises(ValueError, match="strict mode"):
            validate_persona(
                _bf(agreeableness=0.9, conscientiousness=0.9, neuroticism=0.9),
                _vals(power=0.9, universalism=0.9),
                _cog(risk_tolerance=0.9, need_for_closure=0.9),
                _comm(),
                disclosure_base_openness=0.9,
                privacy_sensitivity=0.9,
                mode="strict",
            )

    def test_warn_mode_passes_with_warnings(self):
        r = validate_persona(
            _bf(agreeableness=0.9, conscientiousness=0.9, neuroticism=0.9),
            _vals(), _cog(risk_tolerance=0.9, need_for_closure=0.9), _comm(),
            mode="warn",
        )
        assert r.passed  # warn mode always passes
        assert len(r.warnings) > 0
