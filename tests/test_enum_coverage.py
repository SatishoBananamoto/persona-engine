"""
Enum Coverage Tests — verify all enum members are handled by consumers.

Tests that every Tone, Verbosity, UncertaintyAction, and KnowledgeClaimType
member is properly handled in the generation pipeline (IRPromptBuilder,
StyleModulator).
"""

import pytest

from persona_engine.schema.ir_schema import (
    KnowledgeClaimType,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.style_modulator import StyleModulator


class TestToneCoverage:
    """Every Tone member must be handled by IRPromptBuilder."""

    @pytest.mark.parametrize("tone", list(Tone))
    def test_tone_in_ir_prompt_builder(self, tone: Tone) -> None:
        builder = IRPromptBuilder()
        result = builder._format_tone(tone)
        fallback = tone.value.replace("_", " ")
        assert result != fallback, (
            f"Tone {tone.value} uses fallback in IRPromptBuilder._format_tone"
        )


class TestVerbosityCoverage:
    """Every Verbosity member must be handled by StyleModulator and IRPromptBuilder."""

    @pytest.mark.parametrize("v", list(Verbosity))
    def test_verbosity_in_VERBOSITY_TARGETS(self, v: Verbosity) -> None:
        assert v in StyleModulator.VERBOSITY_TARGETS, (
            f"Verbosity {v.value} missing from VERBOSITY_TARGETS"
        )

    @pytest.mark.parametrize("v", list(Verbosity))
    def test_verbosity_in_ir_prompt_builder(self, v: Verbosity) -> None:
        builder = IRPromptBuilder()
        result = builder._format_verbosity(v)
        assert result != v.value, (
            f"Verbosity {v.value} uses fallback in IRPromptBuilder._format_verbosity"
        )


class TestUncertaintyActionCoverage:
    """Every UncertaintyAction member must be handled by IRPromptBuilder."""

    @pytest.mark.parametrize("ua", list(UncertaintyAction))
    def test_uncertainty_in_ir_prompt_builder(self, ua: UncertaintyAction) -> None:
        builder = IRPromptBuilder()
        result = builder._format_uncertainty(ua)
        assert result != ua.value, (
            f"UncertaintyAction {ua.value} uses fallback in IRPromptBuilder._format_uncertainty"
        )


class TestKnowledgeClaimTypeCoverage:
    """Every KnowledgeClaimType member must be handled by IRPromptBuilder."""

    @pytest.mark.parametrize("ct", list(KnowledgeClaimType))
    def test_claim_type_in_ir_prompt_builder(self, ct: KnowledgeClaimType) -> None:
        builder = IRPromptBuilder()
        result = builder._format_claim_type(ct)
        assert result != ct.value, (
            f"KnowledgeClaimType {ct.value} uses fallback in IRPromptBuilder._format_claim_type"
        )
