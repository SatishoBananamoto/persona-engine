"""
Response Generator

Main orchestrator that combines IR processing, LLM generation,
and post-processing into a complete response generation pipeline.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

from persona_engine.schema.ir_schema import IntermediateRepresentation
from persona_engine.schema.persona_schema import Persona
from persona_engine.generation.llm_adapter import BaseLLMAdapter, TemplateAdapter, create_adapter
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.style_modulator import StyleModulator, ConstraintViolation


@dataclass
class GeneratedResponse:
    """
    Complete response with metadata and validation results.
    """
    # The final generated text
    text: str
    
    # The IR that guided generation
    ir: IntermediateRepresentation
    
    # Raw LLM output (before post-processing)
    raw_text: str
    
    # Model used for generation
    model: str
    
    # Generation timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Constraint violations detected
    violations: list = field(default_factory=list)
    
    # Token usage estimate
    estimated_tokens: int = 0
    
    def is_valid(self) -> bool:
        """Check if response passes all constraints."""
        return not any(v.severity == "error" for v in self.violations)
    
    def has_warnings(self) -> bool:
        """Check if response has warnings."""
        return any(v.severity == "warning" for v in self.violations)
    
    def __repr__(self) -> str:
        status = "VALID" if self.is_valid() else "INVALID"
        warnings = len([v for v in self.violations if v.severity == "warning"])
        return f"GeneratedResponse({status}, {len(self.text)} chars, {warnings} warnings)"


class ResponseGenerator:
    """
    Orchestrates the complete response generation pipeline.
    
    Pipeline:
    1. Build system prompt from persona
    2. Build generation prompt from IR
    3. Call LLM to generate response
    4. Post-process and validate
    5. Return structured response with metadata
    """
    
    def __init__(
        self,
        persona: Persona,
        adapter: Optional[BaseLLMAdapter] = None,
        provider: str = "anthropic",
        strict_mode: bool = False
    ):
        """
        Initialize response generator.
        
        Args:
            persona: The persona to generate responses for
            adapter: Pre-configured LLM adapter (optional)
            provider: LLM provider if adapter not provided ("anthropic", "openai", "mock")
            strict_mode: If True, enforce strict verbosity limits
        """
        self.persona = persona
        self.adapter = adapter or create_adapter(provider)
        self.prompt_builder = IRPromptBuilder()
        self.style_modulator = StyleModulator()
        self.strict_mode = strict_mode
        
        # Pre-build system prompt (constant for persona)
        self._system_prompt = self.prompt_builder.build_system_prompt(persona)
    
    def generate(
        self,
        ir: IntermediateRepresentation,
        user_input: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        memory_context: Optional[dict] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> GeneratedResponse:
        """
        Generate a response based on IR constraints.

        Args:
            ir: Intermediate Representation with behavioral constraints
            user_input: The user's message to respond to
            max_tokens: Maximum tokens in response
            temperature: Override temperature (if None, uses IR-derived value)
            memory_context: Optional memory context from MemoryManager
            conversation_history: Prior turns as
                ``[{"role": "user"|"assistant", "content": ...}, ...]``

        Returns:
            GeneratedResponse with text and metadata
        """
        # Template backend: bypass prompt building, generate from IR directly
        if isinstance(self.adapter, TemplateAdapter):
            raw_text = self.adapter.generate_from_ir(
                ir=ir,
                user_input=user_input,
                persona=self.persona,
            )
            violations = self.style_modulator.validate_constraints(raw_text, ir)
            return GeneratedResponse(
                text=raw_text,
                ir=ir,
                raw_text=raw_text,
                model=self.adapter.get_model_name(),
                violations=violations,
                estimated_tokens=len(raw_text) // 4,
            )

        # LLM backends (anthropic, openai, mock): build prompt from IR
        generation_prompt = self.prompt_builder.build_generation_prompt(
            ir=ir,
            user_input=user_input,
            persona=self.persona,
            memory_context=memory_context,
        )

        # Determine temperature from IR confidence
        if temperature is None:
            # Higher confidence = lower temperature (more deterministic)
            # Lower confidence = higher temperature (more hedging/variation)
            confidence = ir.response_structure.confidence
            temperature = max(0.3, 1.0 - (confidence * 0.5))

        # Generate response
        raw_text = self.adapter.generate(
            system_prompt=self._system_prompt,
            user_prompt=generation_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            conversation_history=conversation_history,
        )

        # Post-process
        processed_text = self.style_modulator.enforce_verbosity(
            raw_text,
            ir.communication_style.verbosity,
            strict=self.strict_mode,
        )

        # Validate constraints
        violations = self.style_modulator.validate_constraints(processed_text, ir)

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(self._system_prompt + generation_prompt + processed_text) // 4

        return GeneratedResponse(
            text=processed_text,
            ir=ir,
            raw_text=raw_text,
            model=self.adapter.get_model_name(),
            violations=violations,
            estimated_tokens=estimated_tokens,
        )
    
    def get_system_prompt(self) -> str:
        """Return the system prompt (for debugging)."""
        return self._system_prompt
    
    def preview_prompt(self, ir: IntermediateRepresentation, user_input: str) -> str:
        """
        Preview the full prompt that would be sent to the LLM.
        
        Useful for debugging without making API calls.
        """
        generation_prompt = self.prompt_builder.build_generation_prompt(
            ir=ir,
            user_input=user_input,
            persona=self.persona
        )
        
        return f"""=== SYSTEM PROMPT ===
{self._system_prompt}

=== USER PROMPT ===
{generation_prompt}
"""


def create_response_generator(
    persona: Persona,
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    strict_mode: bool = False
) -> ResponseGenerator:
    """
    Factory function to create a response generator.
    
    Args:
        persona: The persona to generate responses for
        provider: "anthropic", "openai", or "mock"
        api_key: Optional API key (defaults to env var)
        strict_mode: If True, enforce strict verbosity limits
        
    Returns:
        Configured ResponseGenerator
    """
    adapter = create_adapter(provider=provider, api_key=api_key)
    return ResponseGenerator(
        persona=persona,
        adapter=adapter,
        strict_mode=strict_mode
    )
