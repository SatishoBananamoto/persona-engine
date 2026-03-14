"""Response generation schema - output models and configuration."""

from enum import StrEnum

from pydantic import BaseModel, Field


class GenerationBackend(StrEnum):
    """Which backend generated the response."""

    MOCK = "mock"
    TEMPLATE = "template"
    ANTHROPIC = "anthropic"


class GeneratedResponse(BaseModel):
    """Output of the response generation layer.

    Contains the generated text plus metadata about how it was produced,
    allowing callers to audit which backend was used and what prompt was sent.
    """

    text: str = Field(description="The generated natural language response")
    backend: GenerationBackend = Field(description="Which backend produced this")
    model_id: str | None = Field(
        default=None,
        description="LLM model used (e.g., 'claude-haiku-4-5'), None for template/mock",
    )
    prompt_system: str | None = Field(
        default=None,
        description="System prompt sent to LLM (for debugging/auditing)",
    )
    prompt_user: str | None = Field(
        default=None,
        description="User prompt sent to LLM",
    )
    token_usage: dict[str, int] | None = Field(
        default=None,
        description="Token counts: input_tokens, output_tokens",
    )
    ir_turn_id: str | None = Field(
        default=None,
        description="Turn ID from the IR that produced this response",
    )


class ResponseConfig(BaseModel):
    """Configuration for response generation."""

    backend: GenerationBackend = GenerationBackend.TEMPLATE
    model_id: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 300
    temperature: float = 0.7
    api_key: str | None = None
    strict_mode: bool = Field(
        default=False,
        description="When True, forces TemplateAdapter for deterministic output regardless of backend setting",
    )
