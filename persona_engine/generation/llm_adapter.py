"""
LLM Adapter Abstraction

Provides a unified interface for different LLM providers (Anthropic, OpenAI, Mock).
This module handles all LLM API interactions for the Response Generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional
import os

from persona_engine.exceptions import (
    ConfigurationError,
    LLMAPIKeyError,
)

if TYPE_CHECKING:
    from persona_engine.schema.ir_schema import IntermediateRepresentation
    from persona_engine.schema.persona_schema import Persona


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System/persona identity prompt
            user_prompt: Current turn's generation prompt (constraints + user message)
            max_tokens: Maximum tokens in response
            temperature: Randomness (0=deterministic, 1=creative)
            conversation_history: Prior turns as
                ``[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]``

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used."""
        pass


class AnthropicAdapter(BaseLLMAdapter):
    """
    Anthropic Claude adapter.
    
    Uses Claude 3.5 Sonnet by default for best quality/cost balance.
    Set ANTHROPIC_API_KEY environment variable before use.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMAPIKeyError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ConfigurationError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Generate response using Claude."""
        messages: list[dict[str, str]] = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        text_block = message.content[0]
        return text_block.text if hasattr(text_block, "text") else str(text_block)

    def get_model_name(self) -> str:
        return self.model


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI adapter.
    
    Uses GPT-4o-mini by default for cost efficiency.
    Set OPENAI_API_KEY environment variable before use.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMAPIKeyError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self._client = None
    
    @property
    def client(self) -> Any:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ConfigurationError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Generate response using GPT."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    def get_model_name(self) -> str:
        return self.model


class MockLLMAdapter(BaseLLMAdapter):
    """
    Mock adapter for testing without API calls.
    
    Returns template responses based on constraints in the prompt.
    Useful for unit tests and development without spending API credits.
    """
    
    def __init__(self, response_template: Optional[str] = None):
        """
        Initialize mock adapter.
        
        Args:
            response_template: Custom response template (optional)
        """
        self.response_template = response_template
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.call_count: int = 0
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Generate mock response."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_conversation_history = conversation_history
        self.call_count += 1

        if self.response_template:
            return self.response_template

        # Generate a reasonable mock response based on detected constraints
        return self._generate_contextual_mock(system_prompt, user_prompt)
    
    def _generate_contextual_mock(self, system_prompt: str, user_prompt: str) -> str:
        """Generate context-aware mock response."""
        # Parse verbosity from prompt
        if "brief" in user_prompt.lower() or "1-2 sentences" in user_prompt.lower():
            return "That's an interesting question. I'd be happy to share my perspective on it."
        elif "detailed" in user_prompt.lower() or "thorough" in user_prompt.lower():
            return (
                "That's a great question, and I appreciate you asking. "
                "Based on my experience in this area, I think there are several "
                "important factors to consider. First, we should look at the "
                "practical implications. Second, there's the theoretical framework "
                "to consider. And finally, the real-world applications matter most. "
                "Let me elaborate on each of these points..."
            )
        else:
            # Medium verbosity default
            return (
                "That's an interesting perspective. From my experience, I'd say "
                "there are a few key considerations here. The approach you're "
                "describing has both merits and potential challenges that are "
                "worth exploring further."
            )
    
    def get_model_name(self) -> str:
        return "mock-llm"


class TemplateAdapter(BaseLLMAdapter):
    """
    Rule-based text generation using IR fields directly.

    No LLM call needed. Produces readable text that varies based on
    tone, verbosity, confidence, and other IR parameters.
    Ideal for development, testing, and zero-cost operation.
    """

    _OPENERS: dict[str, list[str]] = {
        "warm_enthusiastic": ["I'm really excited about this!", "Oh, that's a great topic!"],
        "excited_engaged": ["This is fascinating!", "I love thinking about this!"],
        "thoughtful_engaged": ["That's an interesting question.", "Let me think about that carefully."],
        "warm_confident": ["I'm glad you asked about this.", "Great question!"],
        "friendly_relaxed": ["Sure thing!", "Yeah, so here's the thing."],
        "content_calm": ["I appreciate you bringing this up.", "That's a nice topic to explore."],
        "satisfied_peaceful": ["I'm happy to share my thoughts on this.", "That's something I feel good about."],
        "neutral_calm": ["I see.", "That's a reasonable question."],
        "professional_composed": ["That's a relevant consideration.", "Allow me to address that."],
        "matter_of_fact": ["Here's the thing.", "Simply put,"],
        "frustrated_tense": ["Look,", "Honestly, this is frustrating."],
        "anxious_stressed": ["I'm a bit worried about this, honestly.", "Well, that's concerning..."],
        "defensive_agitated": ["I don't think that's fair.", "That's not quite right, actually."],
        "concerned_empathetic": ["I'm concerned about this.", "That does sound difficult."],
        "disappointed_resigned": ["I suppose that's how it is.", "Well, that's a bit disappointing."],
        "sad_subdued": ["I suppose...", "Well..."],
        "tired_withdrawn": ["Mm.", "Right, well..."],
    }
    _HEDGES = ["I think", "From what I understand", "If I'm not mistaken", "As far as I know"]
    _EXPERIENCE = ["In my experience", "From what I've seen", "Based on my work"]

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Fallback when called without IR (standard interface)."""
        return f"[Template fallback - no IR provided for: {user_prompt[:50]}]"

    def get_model_name(self) -> str:
        return "template-rule-based"

    def generate_from_ir(
        self,
        ir: IntermediateRepresentation,
        user_input: str,
        persona: Persona | None = None,
    ) -> str:
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
        openers = self._OPENERS.get(tone, self._OPENERS["neutral_calm"])
        parts.append(openers[0])

        # 2. Core stance with confidence-appropriate framing
        if stance:
            if confidence < 0.4:
                stance_text = f"{self._HEDGES[0]} {stance[0].lower()}{stance[1:]}"
            elif confidence < 0.7:
                stance_text = f"{self._EXPERIENCE[0]}, {stance[0].lower()}{stance[1:]}"
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
            parts.append("Could you tell me more about what specifically you'd like to know?")
        elif uncertainty == "hedge":
            parts.append("Though I should note I'm not entirely certain about all the details.")
        elif uncertainty == "refuse":
            parts.append("I'm not really the right person to speak to that, though.")

        # 5. Extra detail for detailed verbosity
        if verbosity == "detailed" and persona:
            occupation = getattr(getattr(persona, "identity", None), "occupation", None)
            if occupation:
                parts.append(f"As a {occupation}, this is something I've thought about quite a bit.")

        # 6. Assemble
        text = " ".join(parts)

        # 7. Formality transform
        if formality > 0.75:
            text = self._formalize(text)
        elif formality < 0.25:
            text = self._casualize(text)

        # 8. Length enforcement for brief
        if verbosity == "brief":
            sentences = text.split(". ")
            text = ". ".join(sentences[:2])
            if not text.endswith("."):
                text += "."

        return text

    @staticmethod
    def _formalize(text: str) -> str:
        """Apply formal language transforms."""
        replacements = {
            "Yeah, ": "Yes, ", "yeah, ": "yes, ", "Sure thing!": "Certainly.",
            "Here's the thing.": "The matter is as follows.",
            "I'm ": "I am ", "don't": "do not", "can't": "cannot",
            "won't": "will not", "shouldn't": "should not", "wouldn't": "would not",
            "couldn't": "could not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "that's": "that is",
            "it's": "it is", "Mm.": "I see.", "Look,": "To be frank,",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _casualize(text: str) -> str:
        """Apply casual language transforms."""
        replacements = {
            "I am ": "I'm ", "do not": "don't", "cannot": "can't", "will not": "won't",
            "That is a relevant consideration.": "So here's the deal.",
            "Allow me to address that.": "Let me jump in here.",
            "Certainly.": "Sure thing!", "I see.": "Yeah, got it.",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


def create_adapter(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMAdapter:
    """
    Factory function to create an LLM adapter.

    Args:
        provider: "anthropic", "openai", "mock", or "template"
        api_key: Optional API key (defaults to env var)
        model: Optional model override

    Returns:
        Configured LLM adapter
    """
    provider = provider.lower()

    if provider == "anthropic":
        kwargs: dict[str, Any] = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return AnthropicAdapter(**kwargs)

    elif provider == "openai":
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return OpenAIAdapter(**kwargs)

    elif provider == "mock":
        return MockLLMAdapter()

    elif provider == "template":
        return TemplateAdapter()

    else:
        raise ConfigurationError(
            f"Unknown provider: {provider}. Use 'anthropic', 'openai', 'mock', or 'template'"
        )
