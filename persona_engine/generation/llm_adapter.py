"""
LLM Adapter Abstraction

Provides a unified interface for different LLM providers (Anthropic, OpenAI, Mock).
This module handles all LLM API interactions for the Response Generator.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System/persona identity prompt
            user_prompt: User message and generation constraints
            max_tokens: Maximum tokens in response
            temperature: Randomness (0=deterministic, 1=creative)
            
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
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Claude."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.content[0].text
    
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
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate response using GPT."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    
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
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.call_count = 0
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate mock response."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
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


def create_adapter(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> BaseLLMAdapter:
    """
    Factory function to create an LLM adapter.
    
    Args:
        provider: "anthropic", "openai", or "mock"
        api_key: Optional API key (defaults to env var)
        model: Optional model override
        
    Returns:
        Configured LLM adapter
    """
    provider = provider.lower()
    
    if provider == "anthropic":
        kwargs = {"api_key": api_key}
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
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'mock'")
