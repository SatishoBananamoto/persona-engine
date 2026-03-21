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
    TemplateAdapter,
    create_adapter,
)
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.style_modulator import StyleModulator, ConstraintViolation
from persona_engine.generation.response_generator import (
    ResponseGenerator,
    GeneratedResponse,
    GenerationBackend,
    ResponseConfig,
    create_response_generator,
)
from persona_engine.generation.prompt_builder import (
    TONE_PROMPTS,
    VERBOSITY_PROMPTS,
    UNCERTAINTY_PROMPTS,
    CLAIM_TYPE_PROMPTS,
    formality_instruction,
    directness_instruction,
    confidence_instruction,
    elasticity_instruction,
    disclosure_instruction,
    build_ir_prompt,
)

__all__ = [
    # Adapters
    "BaseLLMAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "MockLLMAdapter",
    "TemplateAdapter",
    "create_adapter",
    # Prompt building
    "IRPromptBuilder",
    # Style modulation
    "StyleModulator",
    "ConstraintViolation",
    # Response generation
    "ResponseGenerator",
    "GeneratedResponse",
    "GenerationBackend",
    "ResponseConfig",
    "create_response_generator",
    # Legacy prompt builder utilities
    "TONE_PROMPTS",
    "VERBOSITY_PROMPTS",
    "UNCERTAINTY_PROMPTS",
    "CLAIM_TYPE_PROMPTS",
    "formality_instruction",
    "directness_instruction",
    "confidence_instruction",
    "elasticity_instruction",
    "disclosure_instruction",
    "build_ir_prompt",
]

