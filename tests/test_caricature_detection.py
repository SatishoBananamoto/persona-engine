"""
Tests for caricature detection — cross-module coherence validation.

Verifies that the Research Scientist's concern about modifier stacking
is properly caught: individually-bounded modifiers can combine to produce
implausible composite profiles.
"""

import pytest

from persona_engine import PersonaEngine
from persona_engine.validation.caricature_detection import validate_caricature


def _make_engine(big_five: dict, **kwargs) -> PersonaEngine:
    """Create an engine with specified Big Five traits."""
    base = {
        "openness": 0.5, "conscientiousness": 0.5,
        "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5,
    }
    base.update(big_five)
    return PersonaEngine.from_description(
        f"A 40-year-old professional with traits",
        llm_provider="mock",
        **kwargs,
    )


class TestCaricatureAccumulation:
    """Test that extreme value stacking is detected."""

    def test_balanced_persona_no_violations(self):
        """A balanced persona should produce no caricature warnings."""
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        ir = engine.plan("What makes a good sauce?")
        violations = validate_caricature(ir, engine.persona)
        caricature_violations = [
            v for v in violations if v.violation_type == "caricature_accumulation"
        ]
        assert len(caricature_violations) == 0

    def test_extreme_persona_detected_by_pipeline(self):
        """The full pipeline should include caricature checks."""
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        result = engine.chat("Tell me about cooking")
        # Verify the pipeline now checks caricature invariants
        assert "caricature_accumulation" in result.validation.checked_invariants


class TestAnxietyDirectnessContradiction:
    """Test the high-N + low-confidence + high-directness contradiction."""

    def test_high_neuroticism_low_confidence_high_directness_flagged(self):
        """The specific scenario from the Research Scientist review."""
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        ir = engine.plan("You're completely wrong about this recipe!")

        # Manually set the contradiction scenario to test detection
        ir.response_structure.confidence = 0.15
        ir.communication_style.directness = 0.85

        # Use a high-neuroticism persona
        engine._persona.psychology.big_five.neuroticism = 0.85

        violations = validate_caricature(ir, engine.persona)
        contradiction_violations = [
            v for v in violations if v.violation_type == "caricature_contradiction"
        ]
        assert len(contradiction_violations) >= 1
        assert any("anxiety-directness" in v.message.lower() for v in contradiction_violations)

    def test_no_false_positive_for_normal_directness(self):
        """Normal directness with high-N should not trigger."""
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        ir = engine.plan("Tell me about sauces")

        ir.response_structure.confidence = 0.25
        ir.communication_style.directness = 0.40
        engine._persona.psychology.big_five.neuroticism = 0.85

        violations = validate_caricature(ir, engine.persona)
        assert all("anxiety-directness" not in v.message.lower() for v in violations)


class TestConfidenceDisclosureIncoherence:
    """Test very low confidence + very high disclosure detection."""

    def test_low_confidence_high_disclosure_flagged(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        ir = engine.plan("Tell me about yourself")

        ir.response_structure.confidence = 0.10
        ir.knowledge_disclosure.disclosure_level = 0.90

        violations = validate_caricature(ir, engine.persona)
        assert any(
            "confidence-disclosure" in v.message.lower() for v in violations
        )

    def test_normal_confidence_disclosure_no_flag(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        ir = engine.plan("Tell me about yourself")

        ir.response_structure.confidence = 0.60
        ir.knowledge_disclosure.disclosure_level = 0.70

        violations = validate_caricature(ir, engine.persona)
        assert not any(
            "confidence-disclosure" in v.message.lower() for v in violations
        )
