"""Response generator - orchestrates IR to text conversion.

Sits on top of the existing TurnPlanner pipeline:
  TurnPlanner.generate_ir() → IR → ResponseGenerator.generate() → text

Usage:
    planner = TurnPlanner(persona)
    ir = planner.generate_ir(context)

    generator = ResponseGenerator(persona=persona)
    response = generator.generate(ir, user_input="What do you think about AI?")
    print(response.text)
"""

from __future__ import annotations

from persona_engine.response.adapters import (
    AnthropicAdapter,
    LLMAdapter,
    MockAdapter,
    TemplateAdapter,
)
from persona_engine.response.prompt_builder import build_system_prompt
from persona_engine.response.schema import (
    GeneratedResponse,
    GenerationBackend,
    ResponseConfig,
)
from persona_engine.schema.ir_schema import IntermediateRepresentation
from persona_engine.schema.persona_schema import Persona


class ResponseGenerator:
    """Converts IR to natural language text.

    Supports three backends:
      - template: Rule-based, no API key needed (default)
      - mock: Deterministic canned responses for testing
      - anthropic: Real Claude API calls
    """

    def __init__(
        self,
        config: ResponseConfig | None = None,
        persona: Persona | None = None,
        adapter: LLMAdapter | None = None,
    ):
        self._config = config or ResponseConfig()
        self._persona = persona

        # Allow direct adapter injection (for testing) or auto-create
        if adapter:
            self._adapter = adapter
        else:
            self._adapter = self._create_adapter()

    def _create_adapter(self) -> LLMAdapter:
        """Create adapter from config."""
        if self._config.backend == GenerationBackend.MOCK:
            return MockAdapter()
        if self._config.backend == GenerationBackend.TEMPLATE:
            return TemplateAdapter()
        if self._config.backend == GenerationBackend.ANTHROPIC:
            return AnthropicAdapter(
                api_key=self._config.api_key,
                model_id=self._config.model_id,
            )
        raise ValueError(f"Unknown backend: {self._config.backend}")

    def generate(
        self,
        ir: IntermediateRepresentation,
        user_input: str,
    ) -> GeneratedResponse:
        """Generate natural language response from IR.

        Args:
            ir: Intermediate representation from TurnPlanner
            user_input: Original user input (the message to respond to)

        Returns:
            GeneratedResponse with text and metadata
        """
        # Template backend: bypass prompt building, use IR directly
        if isinstance(self._adapter, TemplateAdapter):
            response = self._adapter.generate_from_ir(
                ir=ir,
                user_input=user_input,
                persona=self._persona,
            )
            response.ir_turn_id = ir.turn_id
            return response

        # LLM backends (mock, anthropic): build prompt from IR
        system_prompt = build_system_prompt(ir, persona=self._persona)

        response = self._adapter.generate(
            system_prompt=system_prompt,
            user_prompt=user_input,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        response.ir_turn_id = ir.turn_id
        return response

    @property
    def adapter(self) -> LLMAdapter:
        """Access the underlying adapter (useful for testing)."""
        return self._adapter


def create_response_generator(
    persona: Persona | None = None,
    backend: str = "template",
    api_key: str | None = None,
    model_id: str = "claude-haiku-4-5-20251001",
) -> ResponseGenerator:
    """Factory function for ResponseGenerator."""
    config = ResponseConfig(
        backend=GenerationBackend(backend),
        api_key=api_key,
        model_id=model_id,
    )
    return ResponseGenerator(config=config, persona=persona)
