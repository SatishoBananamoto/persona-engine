"""Response generation module - converts IR to natural language text."""

from persona_engine.response.adapters import (
    AnthropicAdapter,
    LLMAdapter,
    MockAdapter,
    TemplateAdapter,
)
from persona_engine.response.generator import ResponseGenerator, create_response_generator
from persona_engine.response.prompt_builder import build_system_prompt
from persona_engine.response.schema import GeneratedResponse, GenerationBackend, ResponseConfig

__all__ = [
    "AnthropicAdapter",
    "GeneratedResponse",
    "GenerationBackend",
    "LLMAdapter",
    "MockAdapter",
    "ResponseConfig",
    "ResponseGenerator",
    "TemplateAdapter",
    "build_system_prompt",
    "create_response_generator",
]
