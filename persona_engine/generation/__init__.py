"""
Response Generation Module

Converts Intermediate Representation (IR) into natural language
responses using LLM adapters.
"""

from persona_engine.generation.llm_adapter import (
    BaseLLMAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    MockLLMAdapter,
    create_adapter,
)
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.style_modulator import StyleModulator, ConstraintViolation
from persona_engine.generation.response_generator import (
    ResponseGenerator,
    GeneratedResponse,
    create_response_generator,
)

__all__ = [
    # Adapters
    "BaseLLMAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "MockLLMAdapter",
    "create_adapter",
    # Prompt building
    "IRPromptBuilder",
    # Style modulation
    "StyleModulator",
    "ConstraintViolation",
    # Response generation
    "ResponseGenerator",
    "GeneratedResponse",
    "create_response_generator",
]

