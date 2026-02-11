"""
PersonaEngine — Unified SDK entry point.

Wraps the entire pipeline (planning, generation, validation, memory) into
a single class with a simple ``chat()`` / ``plan()`` API.

Usage::

    from persona_engine import PersonaEngine

    engine = PersonaEngine.from_yaml("personas/chef.yaml")

    # Full round-trip (IR → LLM → validation)
    result = engine.chat("What makes a perfect French mother sauce?")
    print(result.text)
    print(result.ir.response_structure.competence)
    print(result.validation.passed)

    # IR-only (no LLM call) — useful for testing
    ir = engine.plan("What makes a perfect French mother sauce?")

    # Multi-turn — state is managed internally
    r1 = engine.chat("Tell me about sauces.")
    r2 = engine.chat("And what about soups?")  # turn 2, memory active
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]

from persona_engine.generation.llm_adapter import BaseLLMAdapter, create_adapter
from persona_engine.generation.response_generator import GeneratedResponse, ResponseGenerator
from persona_engine.memory import MemoryManager
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    IRValidationResult,
    InteractionMode,
    IntermediateRepresentation,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation import PipelineValidator


# ---------------------------------------------------------------------------
# ChatResult — everything from a single turn
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    """Bundle returned by ``PersonaEngine.chat()``.

    Attributes:
        text:        Final LLM-generated response text.
        ir:          The full Intermediate Representation that guided generation.
        validation:  Validation result (coherence + compliance + cross-turn).
        response:    The underlying ``GeneratedResponse`` from the generator.
        turn_number: Which conversation turn this was.
    """

    text: str
    ir: IntermediateRepresentation
    validation: IRValidationResult
    response: GeneratedResponse
    turn_number: int

    @property
    def citations(self) -> list:
        """Shortcut to the IR citation chain."""
        return self.ir.citations

    @property
    def passed(self) -> bool:
        """Whether validation passed."""
        return self.validation.passed

    @property
    def competence(self) -> float:
        return self.ir.response_structure.competence

    @property
    def confidence(self) -> float:
        return self.ir.response_structure.confidence

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ChatResult(turn={self.turn_number}, {status}, "
            f"comp={self.competence:.2f}, conf={self.confidence:.2f}, "
            f"{len(self.text)} chars)"
        )


# ---------------------------------------------------------------------------
# PersonaEngine
# ---------------------------------------------------------------------------

class PersonaEngine:
    """Unified entry point for the persona engine pipeline.

    Manages the full lifecycle: persona loading, IR planning, LLM generation,
    validation, and memory — so callers only need ``chat()`` or ``plan()``.
    """

    def __init__(
        self,
        persona: Persona,
        *,
        llm_provider: str = "anthropic",
        adapter: BaseLLMAdapter | None = None,
        seed: int = 42,
        validate: bool = True,
        strict_mode: bool = False,
        conversation_id: str | None = None,
    ) -> None:
        """
        Args:
            persona:         Loaded Persona model.
            llm_provider:    LLM backend ("anthropic", "openai", "mock", "template").
                             Ignored if *adapter* is provided.
            adapter:         Pre-configured LLM adapter (overrides *llm_provider*).
            seed:            Base seed for deterministic IR generation.
            validate:        Run validation after each ``chat()`` / ``plan()`` call.
            strict_mode:     Pass to ResponseGenerator for strict verbosity enforcement.
            conversation_id: Unique ID for this conversation. Auto-generated if None.
        """
        self._persona = persona
        self._validate = validate
        self._conversation_id = conversation_id or uuid.uuid4().hex[:12]
        self._turn_number = 0

        # Core components
        self._determinism = DeterminismManager(seed=seed)
        self._memory = MemoryManager()
        self._stance_cache = StanceCache()
        self._planner = TurnPlanner(
            persona, self._determinism, memory_manager=self._memory,
        )
        self._validator = PipelineValidator(persona) if validate else None

        # Generation
        resolved_adapter = adapter or create_adapter(llm_provider)
        self._generator = ResponseGenerator(
            persona, adapter=resolved_adapter, strict_mode=strict_mode,
        )

        # History
        self._history: list[ChatResult] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str,
        **kwargs: Any,
    ) -> PersonaEngine:
        """Load a persona from a YAML file and return a ready engine.

        Args:
            path:   Path to persona YAML file.
            **kwargs: Forwarded to ``PersonaEngine.__init__``.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        # Normalise legacy "domains" key
        if "domains" in data and "knowledge_domains" not in data:
            data["knowledge_domains"] = data.pop("domains")
        persona = Persona(**data)
        return cls(persona, **kwargs)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input: str,
        *,
        mode: InteractionMode | None = None,
        goal: ConversationGoal | None = None,
        topic: str | None = None,
    ) -> ChatResult:
        """Process a user message through the full pipeline.

        Steps:
        1. Generate IR via TurnPlanner
        2. Validate IR (if enabled)
        3. Generate LLM response
        4. Write memory intents
        5. Return bundled ChatResult

        Args:
            user_input: The user's message.
            mode:       Interaction mode override (auto-detected if None).
            goal:       Conversation goal override (auto-detected if None).
            topic:      Topic signature override (auto-detected if None).

        Returns:
            ChatResult with text, IR, validation, and metadata.
        """
        self._turn_number += 1
        turn = self._turn_number

        # 1. Plan (generate IR)
        ir = self._generate_ir(user_input, mode=mode, goal=goal, topic=topic)

        # 2. Validate
        validation = self._run_validation(ir, turn, topic or ir.conversation_frame.goal.value)

        # 3. Generate response
        memory_ctx = self._memory.get_context_for_turn(
            topic or "", current_turn=turn,
        )
        response = self._generator.generate(
            ir, user_input, memory_context=memory_ctx,
        )

        # 4. Write memory
        if ir.memory_ops and ir.memory_ops.write_intents:
            self._memory.process_write_intents(
                ir.memory_ops.write_intents,
                turn=turn,
                conversation_id=self._conversation_id,
            )

        # 5. Bundle
        result = ChatResult(
            text=response.text,
            ir=ir,
            validation=validation,
            response=response,
            turn_number=turn,
        )
        self._history.append(result)
        return result

    def plan(
        self,
        user_input: str,
        *,
        mode: InteractionMode | None = None,
        goal: ConversationGoal | None = None,
        topic: str | None = None,
    ) -> IntermediateRepresentation:
        """Generate IR without calling the LLM.

        Useful for testing, debugging, and inspecting the planning layer
        without spending API credits.

        Same turn-counting and memory semantics as ``chat()``.
        """
        self._turn_number += 1
        ir = self._generate_ir(user_input, mode=mode, goal=goal, topic=topic)

        # Write memory (IR may contain write intents even without LLM call)
        if ir.memory_ops and ir.memory_ops.write_intents:
            self._memory.process_write_intents(
                ir.memory_ops.write_intents,
                turn=self._turn_number,
                conversation_id=self._conversation_id,
            )

        return ir

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def persona(self) -> Persona:
        """The loaded persona profile."""
        return self._persona

    @property
    def turn_count(self) -> int:
        """Number of turns processed so far."""
        return self._turn_number

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def history(self) -> list[ChatResult]:
        """Chronological list of ChatResults from ``chat()`` calls."""
        return list(self._history)

    @property
    def memory(self) -> MemoryManager:
        """Direct access to the memory manager."""
        return self._memory

    @property
    def validator(self) -> PipelineValidator | None:
        """The pipeline validator (None if validation is disabled)."""
        return self._validator

    def memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return self._memory.stats()

    def system_prompt(self) -> str:
        """Return the system prompt being sent with every LLM call."""
        return self._generator.get_system_prompt()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset conversation state for a new conversation.

        Clears turn counter, history, memory, stance cache, and
        cross-turn validation state. The persona and components remain.
        """
        self._turn_number = 0
        self._conversation_id = uuid.uuid4().hex[:12]
        self._history.clear()
        self._memory = MemoryManager()
        self._stance_cache = StanceCache()
        self._planner = TurnPlanner(
            self._persona, self._determinism, memory_manager=self._memory,
        )
        if self._validator:
            self._validator.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_ir(
        self,
        user_input: str,
        *,
        mode: InteractionMode | None,
        goal: ConversationGoal | None,
        topic: str | None,
    ) -> IntermediateRepresentation:
        """Build ConversationContext and delegate to TurnPlanner."""
        context = ConversationContext(
            conversation_id=self._conversation_id,
            turn_number=self._turn_number,
            interaction_mode=mode,
            goal=goal,
            topic_signature=topic or "",
            user_input=user_input,
            stance_cache=self._stance_cache,
        )
        return self._planner.generate_ir(context)

    def _run_validation(
        self,
        ir: IntermediateRepresentation,
        turn: int,
        topic: str,
    ) -> IRValidationResult:
        """Validate IR or return a trivial pass result."""
        if self._validator:
            return self._validator.validate(ir, turn_number=turn, topic=topic)
        # Validation disabled — return a synthetic "passed" result
        return IRValidationResult(
            passed=True,
            violations=[],
            checked_invariants=[],
            timestamp="",
        )
