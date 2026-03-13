"""
Exhaustive Enum Coverage Tests

Safety net tests that verify every enum member has entries in all consumer
dicts and format methods. Prevents adding enum values without updating
downstream consumers.
"""

import pytest

from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    KnowledgeClaimType,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.style_modulator import StyleModulator
from persona_engine.response.adapters import _OPENERS
from persona_engine.response.prompt_builder import (
    CLAIM_TYPE_PROMPTS,
    TONE_PROMPTS,
    UNCERTAINTY_PROMPTS,
    VERBOSITY_PROMPTS,
)


class TestToneCoverage:
    """Every Tone member must be present in all consumer dicts."""

    @pytest.mark.parametrize("tone", list(Tone))
    def test_tone_in_TONE_PROMPTS(self, tone: Tone) -> None:
        assert tone.value in TONE_PROMPTS, f"Tone {tone.value} missing from TONE_PROMPTS"

    @pytest.mark.parametrize("tone", list(Tone))
    def test_tone_in_OPENERS(self, tone: Tone) -> None:
        assert tone.value in _OPENERS, f"Tone {tone.value} missing from _OPENERS"

    @pytest.mark.parametrize("tone", list(Tone))
    def test_tone_in_ir_prompt_builder(self, tone: Tone) -> None:
        builder = IRPromptBuilder()
        result = builder._format_tone(tone)
        fallback = tone.value.replace("_", " ")
        assert result != fallback, (
            f"Tone {tone.value} uses fallback in IRPromptBuilder._format_tone"
        )


class TestVerbosityCoverage:
    """Every Verbosity member must be present in all consumer dicts."""

    @pytest.mark.parametrize("v", list(Verbosity))
    def test_verbosity_in_VERBOSITY_PROMPTS(self, v: Verbosity) -> None:
        assert v.value in VERBOSITY_PROMPTS, f"Verbosity {v.value} missing from VERBOSITY_PROMPTS"

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
    """Every UncertaintyAction member must be present in all consumer dicts."""

    @pytest.mark.parametrize("ua", list(UncertaintyAction))
    def test_uncertainty_in_UNCERTAINTY_PROMPTS(self, ua: UncertaintyAction) -> None:
        assert ua.value in UNCERTAINTY_PROMPTS, (
            f"UncertaintyAction {ua.value} missing from UNCERTAINTY_PROMPTS"
        )

    @pytest.mark.parametrize("ua", list(UncertaintyAction))
    def test_uncertainty_in_ir_prompt_builder(self, ua: UncertaintyAction) -> None:
        builder = IRPromptBuilder()
        result = builder._format_uncertainty(ua)
        assert result != ua.value, (
            f"UncertaintyAction {ua.value} uses fallback in IRPromptBuilder._format_uncertainty"
        )


class TestKnowledgeClaimTypeCoverage:
    """Every KnowledgeClaimType member must be present in all consumer dicts."""

    @pytest.mark.parametrize("ct", list(KnowledgeClaimType))
    def test_claim_type_in_CLAIM_TYPE_PROMPTS(self, ct: KnowledgeClaimType) -> None:
        assert ct.value in CLAIM_TYPE_PROMPTS, (
            f"KnowledgeClaimType {ct.value} missing from CLAIM_TYPE_PROMPTS"
        )

    @pytest.mark.parametrize("ct", list(KnowledgeClaimType))
    def test_claim_type_in_ir_prompt_builder(self, ct: KnowledgeClaimType) -> None:
        builder = IRPromptBuilder()
        result = builder._format_claim_type(ct)
        assert result != ct.value, (
            f"KnowledgeClaimType {ct.value} uses fallback in IRPromptBuilder._format_claim_type"
        )
