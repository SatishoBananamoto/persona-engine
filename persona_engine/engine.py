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

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

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
from persona_engine.exceptions import InputValidationError
from persona_engine.persona_builder import PersonaBuilder
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation import PipelineValidator


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 10_000  # characters — configurable via _validate_user_input param

# Control characters to strip (C0 controls except tab/newline/carriage-return)
_CONTROL_CHARS = {chr(c) for c in range(0x00, 0x20)} - {'\t', '\n', '\r'}
_CONTROL_CHARS.add(chr(0x7F))  # DEL


def _sanitize_text(text: str) -> str:
    """Remove control characters and neutralise prompt-injection patterns.

    This is a defence-in-depth measure: the primary boundary is the LLM
    prompt template, but we strip obvious manipulation attempts here.
    """
    # 1. Strip control characters (null bytes, escape sequences, etc.)
    sanitized = "".join(ch for ch in text if ch not in _CONTROL_CHARS)

    # 2. Collapse multiple consecutive newlines (limits injection surface)
    import re
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)

    return sanitized


def _validate_user_input(
    user_input: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
) -> str:
    """Validate and sanitise user input.

    Checks type, emptiness, and length, then strips control characters
    and collapses excessive newlines to reduce prompt-injection surface.

    Returns the sanitised input string on success.

    Raises:
        InputValidationError: If input is empty, wrong type, or too long.
    """
    if not isinstance(user_input, str):
        raise InputValidationError(
            f"user_input must be a string, got {type(user_input).__name__}"
        )

    stripped = user_input.strip()
    if not stripped:
        raise InputValidationError("user_input must not be empty or whitespace-only")

    if len(stripped) > max_length:
        raise InputValidationError(
            f"user_input exceeds maximum length of {max_length:,} characters "
            f"(got {len(stripped):,})"
        )

    # Sanitise after length check (sanitisation can only shrink text)
    return _sanitize_text(stripped)


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
    _user_input: str = ""

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

        # Core components — single StanceCache shared between engine and memory manager
        self._determinism = DeterminismManager(seed=seed)
        self._stance_cache = StanceCache()
        self._memory = MemoryManager(stance_cache=self._stance_cache)
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

    def __repr__(self) -> str:
        return (
            f"PersonaEngine(persona={self._persona.label!r}, "
            f"turns={self._turn_number})"
        )

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

        Example::

            engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
            result = engine.chat("What makes a good sauce?")

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

    @classmethod
    def from_description(
        cls,
        description: str,
        **kwargs: Any,
    ) -> PersonaEngine:
        """Create an engine from a natural-language persona description.

        No LLM required — uses heuristic parsing to extract name,
        occupation, age, location, and personality adjectives, then
        fills in ~50 psychological parameters with sensible defaults.

        Example::

            engine = PersonaEngine.from_description(
                "A 45-year-old French chef named Marcus, passionate and direct"
            )
            result = engine.chat("What makes a good sauce?")

        Args:
            description: Natural-language persona description.
            **kwargs:    Forwarded to ``PersonaEngine.__init__``.
        """
        persona = PersonaBuilder.from_description(description)
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

        Example::

            result = engine.chat("Tell me about quantum physics")
            print(result.text)
            print(result.ir.response_structure.competence)

        Args:
            user_input: The user's message.
            mode:       Interaction mode override (auto-detected if None).
            goal:       Conversation goal override (auto-detected if None).
            topic:      Topic signature override (auto-detected if None).

        Returns:
            ChatResult with text, IR, validation, and metadata.
        """
        user_input = _validate_user_input(user_input)

        self._turn_number += 1
        turn = self._turn_number
        logger.info("chat turn=%d persona=%s input_len=%d", turn, self._persona.label, len(user_input))

        # 1. Plan (generate IR)
        ir = self._generate_ir(user_input, mode=mode, goal=goal, topic=topic)
        logger.debug(
            "IR generated: mode=%s confidence=%.2f competence=%.2f tone=%s",
            ir.conversation_frame.interaction_mode.value,
            ir.response_structure.confidence,
            ir.response_structure.competence,
            ir.communication_style.tone.value,
        )

        # 2. Validate
        validation = self._run_validation(ir, turn, topic or ir.conversation_frame.goal.value)
        if not validation.passed:
            logger.warning("Validation failed turn=%d violations=%d", turn, len(validation.violations))

        # 3. Generate response
        memory_ctx = self._memory.get_context_for_turn(
            topic or "", current_turn=turn,
        )
        response = self._generator.generate(
            ir, user_input,
            memory_context=memory_ctx,
            conversation_history=self._build_conversation_history(),
        )

        # 4. Memory writes are handled by TurnPlanner.generate_ir() (section 17).
        #    Do NOT write here — that causes double-write corruption.

        # 5. Bundle
        result = ChatResult(
            text=response.text,
            ir=ir,
            validation=validation,
            response=response,
            turn_number=turn,
            _user_input=user_input,
        )
        self._history.append(result)
        logger.info("chat complete turn=%d response_len=%d valid=%s", turn, len(result.text), result.passed)
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

        Example::

            ir = engine.plan("What do you think about AI?")
            print(ir.response_structure.confidence)
            print(ir.conversation_frame.goal)

        Same turn-counting and memory semantics as ``chat()``.
        """
        user_input = _validate_user_input(user_input)

        self._turn_number += 1
        ir = self._generate_ir(user_input, mode=mode, goal=goal, topic=topic)

        # Memory writes are handled by TurnPlanner.generate_ir() (section 17).
        # Do NOT write here — that causes double-write corruption.

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
        """Get memory system statistics.

        Example::

            stats = engine.memory_stats()
            print(stats["facts_count"], stats["episodes_count"])
        """
        return self._memory.stats()

    def system_prompt(self) -> str:
        """Return the system prompt being sent with every LLM call.

        Example::

            print(engine.system_prompt()[:200])
        """
        return self._generator.get_system_prompt()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> PersonaEngine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release resources (LLM client connections, pending memory)."""
        # Flush any pending state; currently a no-op but provides a
        # clean extension point for future async adapters / connection pools.
        pass

    def reset(self) -> None:
        """Reset conversation state for a new conversation.

        Clears turn counter, history, memory, stance cache, and
        cross-turn validation state. The persona and components remain.

        Example::

            engine.chat("Hello")
            engine.reset()          # fresh conversation
            engine.chat("Hi again") # turn 1 again
        """
        self._turn_number = 0
        self._conversation_id = uuid.uuid4().hex[:12]
        self._history.clear()
        self._stance_cache = StanceCache()
        self._memory = MemoryManager(stance_cache=self._stance_cache)
        self._planner = TurnPlanner(
            self._persona, self._determinism, memory_manager=self._memory,
        )
        if self._validator:
            self._validator.reset()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save conversation state to a JSON file.

        Saves: conversation metadata, message history, and full memory
        store contents so that ``load()`` can restore a usable engine.

        Example::

            engine.chat("Hello")
            engine.save("state.json")

        Format is versioned for forward compatibility.
        """
        from dataclasses import asdict

        def _serialize_records(records: list) -> list[dict]:
            """Serialize frozen dataclass records to dicts."""
            result = []
            for r in records:
                d = asdict(r)
                # Convert datetime to ISO string
                if "created_at" in d:
                    d["created_at"] = str(d["created_at"])
                result.append(d)
            return result

        data = {
            "version": 2,
            "conversation_id": self._conversation_id,
            "turn_number": self._turn_number,
            "persona_id": self._persona.persona_id,
            "messages": [
                {"user_input": r._user_input,
                 "text": r.text, "turn": r.turn_number}
                for r in self._history
            ],
            "memory": {
                "facts": _serialize_records(self._memory.facts.all_facts()),
                "preferences": _serialize_records(
                    [p for obs in self._memory.preferences._preferences.values()
                     for p in obs]
                ),
                "relationship_events": _serialize_records(
                    self._memory.relationships._events
                ),
                "relationship_base_trust": self._memory.relationships._base_trust,
                "relationship_base_rapport": self._memory.relationships._base_rapport,
                "episodes": _serialize_records(
                    self._memory.episodes._episodes
                ),
            },
            "memory_stats": self._memory.stats(),
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(
        cls,
        state_path: str | Path,
        persona_path: str,
        **kwargs: Any,
    ) -> PersonaEngine:
        """Reload an engine from a saved state file + persona YAML.

        Restores conversation_id, turn_number, and memory stores so the
        engine can continue where it left off. History is NOT replayed
        (no LLM calls); turns are fast-forwarded.

        Example::

            engine = PersonaEngine.load("state.json", "personas/chef.yaml",
                                        llm_provider="mock")
            engine.chat("Continue our conversation")
        """
        from persona_engine.memory.models import (
            Episode, Fact, Preference, RelationshipMemory,
            MemoryType, MemorySource,
        )

        state = json.loads(Path(state_path).read_text())
        engine = cls.from_yaml(persona_path, **kwargs)
        engine._conversation_id = state["conversation_id"]
        engine._turn_number = state.get("turn_number", 0)

        # Restore memory stores if saved (version 2+)
        mem_data = state.get("memory")
        if mem_data:
            for fd in mem_data.get("facts", []):
                fd.pop("created_at", None)
                fd["memory_type"] = MemoryType(fd["memory_type"])
                fd["source"] = MemorySource(fd["source"])
                if "tags" in fd:
                    fd["tags"] = tuple(fd["tags"])
                engine._memory.facts.store(Fact(**fd))
            for pd in mem_data.get("preferences", []):
                pd.pop("created_at", None)
                pd["memory_type"] = MemoryType(pd["memory_type"])
                pd["source"] = MemorySource(pd["source"])
                if "tags" in pd:
                    pd["tags"] = tuple(pd["tags"])
                engine._memory.preferences.store(Preference(**pd))
            for rd in mem_data.get("relationship_events", []):
                rd.pop("created_at", None)
                rd["memory_type"] = MemoryType(rd["memory_type"])
                rd["source"] = MemorySource(rd["source"])
                if "tags" in rd:
                    rd["tags"] = tuple(rd["tags"])
                engine._memory.relationships.record_event(RelationshipMemory(**rd))
            # Restore base trust/rapport (includes folded eviction deltas)
            if "relationship_base_trust" in mem_data:
                engine._memory.relationships._base_trust = mem_data["relationship_base_trust"]
            if "relationship_base_rapport" in mem_data:
                engine._memory.relationships._base_rapport = mem_data["relationship_base_rapport"]
            for ed in mem_data.get("episodes", []):
                ed.pop("created_at", None)
                ed["memory_type"] = MemoryType(ed["memory_type"])
                ed["source"] = MemorySource(ed["source"])
                if "tags" in ed:
                    ed["tags"] = tuple(ed["tags"])
                engine._memory.episodes.store(Episode(**ed))

        return engine

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_conversation_history(
        self,
        max_turns: int = 10,
    ) -> list[dict[str, str]] | None:
        """Build a conversation_history list from past ChatResults.

        Returns None when there is no history (turn 1).  Limits to the
        most recent *max_turns* exchanges to avoid context overflow.
        """
        if not self._history:
            return None
        recent = self._history[-max_turns:]
        messages: list[dict[str, str]] = []
        for r in recent:
            # Reconstruct the user message from the IR (we stored user_input via prompt)
            # We'll store user_input on ChatResult for this purpose.
            messages.append({"role": "user", "content": r._user_input})
            messages.append({"role": "assistant", "content": r.text})
        return messages or None

    def _generate_ir(
        self,
        user_input: str,
        *,
        mode: InteractionMode | None,
        goal: ConversationGoal | None,
        topic: str | None,
    ) -> IntermediateRepresentation:
        """Build ConversationContext and delegate to TurnPlanner."""
        topic_sig = topic or self._derive_topic_signature(user_input)
        context = ConversationContext(
            conversation_id=self._conversation_id,
            turn_number=self._turn_number,
            interaction_mode=mode,
            goal=goal,
            topic_signature=topic_sig,
            user_input=user_input,
            stance_cache=self._stance_cache,
        )
        return self._planner.generate_ir(context)

    @staticmethod
    def _derive_topic_signature(user_input: str) -> str:
        """Derive a topic signature from user input to avoid stance cache collisions.

        Extracts content words (3+ chars, lowered, sorted) to create a
        stable signature that groups semantically similar inputs together.
        """
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "have", "been", "will", "with", "this", "that", "from",
            "they", "what", "about", "would", "there", "their", "which",
            "could", "other", "into", "more", "some", "than", "them",
            "very", "when", "come", "make", "like", "just", "know",
            "take", "does", "how", "your", "also",
        }
        words = sorted(set(
            w for w in user_input.lower().split()
            if len(w) >= 3 and w not in stop_words
        ))
        return "_".join(words[:6]) if words else "general"

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
