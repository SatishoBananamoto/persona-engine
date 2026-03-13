"""LLM adapters - abstract interface + concrete backends.

Three backends:
  MockAdapter     - deterministic canned responses for unit testing
  TemplateAdapter - rule-based text generation from IR (no API key)
  AnthropicAdapter - real Claude API calls (requires API key)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from persona_engine.response.schema import GeneratedResponse, GenerationBackend

if TYPE_CHECKING:
    from persona_engine.schema.ir_schema import IntermediateRepresentation
    from persona_engine.schema.persona_schema import Persona


# =============================================================================
# Abstract Base
# =============================================================================


class LLMAdapter(ABC):
    """Abstract interface for LLM backends.

    All adapters receive a system prompt and user prompt (both strings)
    and return a GeneratedResponse. Prompt engineering lives in
    prompt_builder.py, not here.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """Generate text from system + user prompts."""
        ...


# =============================================================================
# Mock Adapter (for testing)
# =============================================================================


class MockAdapter(LLMAdapter):
    """Deterministic mock that returns canned responses.

    For testing: captures prompts for assertion, returns fixed text.
    No API key required.
    """

    def __init__(self, response_text: str | None = None):
        self._fixed_response = response_text
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.call_count: int = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1

        text = self._fixed_response or f"[Mock response to: {user_prompt[:50]}]"

        return GeneratedResponse(
            text=text,
            backend=GenerationBackend.MOCK,
            model_id=None,
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            token_usage=None,
            ir_turn_id=None,
        )


# =============================================================================
# Template Adapter (rule-based, no API)
# =============================================================================

# Sentence openers by tone
_OPENERS: dict[str, list[str]] = {
    "warm_enthusiastic": [
        "I'm really excited about this!",
        "Oh, that's a great topic!",
    ],
    "excited_engaged": [
        "This is fascinating!",
        "I love thinking about this!",
    ],
    "thoughtful_engaged": [
        "That's an interesting question.",
        "Let me think about that carefully.",
    ],
    "warm_confident": [
        "I'm glad you asked about this.",
        "Great question!",
    ],
    "friendly_relaxed": [
        "Sure thing!",
        "Yeah, so here's the thing.",
    ],
    "content_calm": [
        "I appreciate you bringing this up.",
        "That's a nice topic to explore.",
    ],
    "satisfied_peaceful": [
        "I'm happy to share my thoughts on this.",
        "That's something I feel good about.",
    ],
    "neutral_calm": [
        "I see.",
        "That's a reasonable question.",
    ],
    "professional_composed": [
        "That's a relevant consideration.",
        "Allow me to address that.",
    ],
    "matter_of_fact": [
        "Here's the thing.",
        "Simply put,",
    ],
    "frustrated_tense": [
        "Look,",
        "Honestly, this is frustrating.",
    ],
    "anxious_stressed": [
        "I'm a bit worried about this, honestly.",
        "Well, that's concerning...",
    ],
    "defensive_agitated": [
        "I don't think that's fair.",
        "That's not quite right, actually.",
    ],
    "concerned_empathetic": [
        "I'm concerned about this.",
        "That does sound difficult.",
    ],
    "disappointed_resigned": [
        "I suppose that's how it is.",
        "Well, that's a bit disappointing.",
    ],
    "sad_subdued": [
        "I suppose...",
        "Well...",
    ],
    "tired_withdrawn": [
        "Mm.",
        "Right, well...",
    ],
}

# Hedging phrases for low-confidence responses
_HEDGES = [
    "I think",
    "From what I understand",
    "If I'm not mistaken",
    "As far as I know",
]

# Experience phrases for moderate-confidence
_EXPERIENCE = [
    "In my experience",
    "From what I've seen",
    "Based on my work",
]


class TemplateAdapter(LLMAdapter):
    """Rule-based text generation using IR fields directly.

    No LLM call needed. Produces readable text that varies based on
    tone, verbosity, confidence, and other IR parameters.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        # Fallback: when called via the standard interface (without IR),
        # return a simple response indicating template mode
        return GeneratedResponse(
            text=f"[Template fallback - no IR provided for: {user_prompt[:50]}]",
            backend=GenerationBackend.TEMPLATE,
            prompt_system=system_prompt,
            prompt_user=user_prompt,
        )

    def generate_from_ir(
        self,
        ir: IntermediateRepresentation,
        user_input: str,
        persona: Persona | None = None,
    ) -> GeneratedResponse:
        """Generate text directly from IR fields (no LLM prompt needed)."""
        tone = ir.communication_style.tone.value
        verbosity = ir.communication_style.verbosity.value
        formality = ir.communication_style.formality
        confidence = ir.response_structure.confidence
        stance = ir.response_structure.stance or ""
        rationale = ir.response_structure.rationale or ""
        uncertainty = ir.knowledge_disclosure.uncertainty_action.value

        parts: list[str] = []

        # 1. Opener based on tone
        openers = _OPENERS.get(tone, _OPENERS["neutral_calm"])
        parts.append(openers[0])

        # 2. Core stance with confidence-appropriate framing
        if stance:
            if confidence < 0.4:
                stance_text = f"{_HEDGES[0]} {stance[0].lower()}{stance[1:]}"
            elif confidence < 0.7:
                stance_text = f"{_EXPERIENCE[0]}, {stance[0].lower()}{stance[1:]}"
            else:
                stance_text = stance
            if not stance_text.endswith("."):
                stance_text += "."
            parts.append(stance_text)

        # 3. Rationale (only for medium/detailed)
        if verbosity != "brief" and rationale:
            parts.append(f"This is because {rationale[0].lower()}{rationale[1:]}.")

        # 4. Uncertainty action
        if uncertainty == "ask_clarifying":
            parts.append(
                "Could you tell me more about what specifically you'd like to know?"
            )
        elif uncertainty == "hedge":
            parts.append(
                "Though I should note I'm not entirely certain about all the details."
            )
        elif uncertainty == "refuse":
            parts.append(
                "I'm not really the right person to speak to that, though."
            )

        # 5. Extra detail for detailed verbosity
        if verbosity == "detailed" and persona and persona.identity.occupation:
            parts.append(
                f"As a {persona.identity.occupation}, "
                f"this is something I've thought about quite a bit."
            )

        # 6. Assemble
        text = " ".join(parts)

        # 7. Formality transform
        if formality > 0.75:
            text = _formalize(text)
        elif formality < 0.25:
            text = _casualize(text)

        # 8. Length enforcement for brief
        if verbosity == "brief":
            sentences = text.split(". ")
            text = ". ".join(sentences[:2])
            if not text.endswith("."):
                text += "."

        return GeneratedResponse(
            text=text,
            backend=GenerationBackend.TEMPLATE,
            model_id=None,
            prompt_system=None,
            prompt_user=user_input,
            ir_turn_id=ir.turn_id,
        )


def _formalize(text: str) -> str:
    """Apply formal language transforms."""
    replacements = {
        "Yeah, ": "Yes, ",
        "yeah, ": "yes, ",
        "Sure thing!": "Certainly.",
        "Here's the thing.": "The matter is as follows.",
        "I'm ": "I am ",
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "that's": "that is",
        "it's": "it is",
        "Mm.": "I see.",
        "Look,": "To be frank,",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _casualize(text: str) -> str:
    """Apply casual language transforms."""
    replacements = {
        "I am ": "I'm ",
        "do not": "don't",
        "cannot": "can't",
        "will not": "won't",
        "That is a relevant consideration.": "So here's the deal.",
        "Allow me to address that.": "Let me jump in here.",
        "Certainly.": "Sure thing!",
        "I see.": "Yeah, got it.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# =============================================================================
# Anthropic Adapter (production, requires API key)
# =============================================================================


class AnthropicAdapter(LLMAdapter):
    """Anthropic Claude API adapter.

    Uses claude-haiku-4-5 by default (cheapest option).
    Tracks token usage in every response for budget monitoring.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "claude-haiku-4-5-20251001",
    ):
        try:
            import anthropic
        except ImportError as e:
            from persona_engine.exceptions import ConfigurationError
            raise ConfigurationError(
                "anthropic package required for AnthropicAdapter. "
                "Install with: pip install anthropic"
            ) from e

        # Uses ANTHROPIC_API_KEY env var if api_key is None
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        message = self._client.messages.create(
            model=self._model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text_block = message.content[0]
        text = text_block.text if hasattr(text_block, "text") else str(text_block)

        return GeneratedResponse(
            text=text,
            backend=GenerationBackend.ANTHROPIC,
            model_id=self._model_id,
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            token_usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            ir_turn_id=None,
        )
