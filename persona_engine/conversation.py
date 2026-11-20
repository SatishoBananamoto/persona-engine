"""
Conversation — High-level multi-turn dialogue wrapper.

Provides a cleaner interface for multi-turn conversations with
iteration, export, and analysis capabilities.

Usage::

    from persona_engine import PersonaEngine, Conversation

    engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
    convo = Conversation(engine)

    convo.say("What makes a good sauce?")
    convo.say("And what about soups?")

    # Iterate over turns
    for turn in convo:
        print(f"Turn {turn.turn_number}: {turn.text}")

    # Export
    convo.export_json("conversation.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from persona_engine.engine import ChatResult, PersonaEngine
    from persona_engine.schema.ir_schema import (
        ConversationGoal,
        InteractionMode,
    )


class Conversation:
    """Multi-turn conversation wrapper around PersonaEngine.

    Adds iteration, export, and analysis on top of the engine's ``chat()`` API.

    Args:
        engine: A PersonaEngine instance to drive the conversation.
        metadata: Optional metadata dict attached to exports.
    """

    def __init__(
        self,
        engine: PersonaEngine,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._metadata = metadata or {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def say(
        self,
        message: str,
        *,
        mode: InteractionMode | None = None,
        goal: ConversationGoal | None = None,
        topic: str | None = None,
    ) -> ChatResult:
        """Send a message and get the persona's response.

        Example::

            result = convo.say("What do you think about AI?")
            print(result.text)

        Args:
            message: User's message text.
            mode:    Interaction mode override.
            goal:    Conversation goal override.
            topic:   Topic signature override.

        Returns:
            ChatResult with response text, IR, and validation.
        """
        return self._engine.chat(message, mode=mode, goal=goal, topic=topic)

    def say_all(
        self,
        messages: list[str],
        **kwargs: Any,
    ) -> list[ChatResult]:
        """Send multiple messages sequentially.

        Example::

            results = convo.say_all([
                "Hello!",
                "Tell me about yourself",
                "What's your expertise?",
            ])

        Args:
            messages: List of user messages to send in order.
            **kwargs: Forwarded to each ``say()`` call.

        Returns:
            List of ChatResults in order.
        """
        return [self.say(msg, **kwargs) for msg in messages]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def turns(self) -> list[ChatResult]:
        """All turns in this conversation."""
        return self._engine.history

    @property
    def turn_count(self) -> int:
        """Number of turns completed."""
        return self._engine.turn_count

    @property
    def engine(self) -> PersonaEngine:
        """The underlying PersonaEngine."""
        return self._engine

    @property
    def persona_name(self) -> str:
        """Label of the persona in this conversation."""
        return self._engine.persona.label

    def last(self) -> ChatResult | None:
        """Get the most recent turn, or None if no turns yet."""
        turns = self.turns
        return turns[-1] if turns else None

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ChatResult]:
        return iter(self.turns)

    def __len__(self) -> int:
        return self.turn_count

    def __getitem__(self, index: int) -> ChatResult:
        return self.turns[index]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Get a summary of the conversation.

        Returns:
            Dict with turn count, avg confidence/competence, memory stats,
            and persona info.
        """
        turns = self.turns
        if not turns:
            return {
                "persona": self.persona_name,
                "turn_count": 0,
                "conversation_id": self._engine.conversation_id,
            }

        confidences = [t.confidence for t in turns]
        competences = [t.competence for t in turns]

        return {
            "persona": self.persona_name,
            "conversation_id": self._engine.conversation_id,
            "turn_count": len(turns),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_competence": sum(competences) / len(competences),
            "all_passed_validation": all(t.passed for t in turns),
            "memory_stats": self._engine.memory_stats(),
            "metadata": self._metadata,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export conversation as a plain dictionary.

        Returns:
            Dict with persona info, turns, and summary.
        """
        return {
            "persona": self.persona_name,
            "persona_id": self._engine.persona.persona_id,
            "conversation_id": self._engine.conversation_id,
            "metadata": self._metadata,
            "turns": [
                {
                    "turn": t.turn_number,
                    "user": t._user_input,
                    "response": t.text,
                    "confidence": round(t.confidence, 3),
                    "competence": round(t.competence, 3),
                    "tone": t.ir.communication_style.tone.value,
                    "passed_validation": t.passed,
                }
                for t in self.turns
            ],
            "summary": self.summary(),
        }

    def export_json(self, path: str | Path) -> None:
        """Export conversation to a JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str)
        )

    def export_yaml(self, path: str | Path) -> None:
        """Export conversation to a YAML file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(
            yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        )

    def export_transcript(self, path: str | Path | None = None) -> str:
        """Export as a human-readable transcript.

        Args:
            path: Optional output file path. If None, returns string only.

        Returns:
            The transcript as a string.
        """
        lines = [f"# Conversation with {self.persona_name}\n"]

        for t in self.turns:
            lines.append(f"**User (Turn {t.turn_number}):** {t._user_input}")
            lines.append(f"**{self.persona_name}:** {t.text}")
            lines.append(
                f"_[confidence={t.confidence:.2f}, "
                f"competence={t.competence:.2f}, "
                f"tone={t.ir.communication_style.tone.value}]_\n"
            )

        transcript = "\n".join(lines)
        if path:
            Path(path).write_text(transcript)
        return transcript

    def export_markdown(self, path: str | Path | None = None) -> str:
        """Export as a detailed markdown report with IR analysis.

        Includes persona profile, per-turn analysis with IR metrics,
        and a conversation summary. More detailed than export_transcript.

        Args:
            path: Optional output file path. If None, returns string only.

        Returns:
            The markdown report as a string.
        """
        lines = [
            f"# Conversation Report: {self.persona_name}",
            "",
            f"**Persona ID:** {self._engine.persona.persona_id}  ",
            f"**Conversation ID:** {self._engine.conversation_id}  ",
            f"**Turns:** {self.turn_count}",
            "",
        ]

        if self._metadata:
            lines.append("## Metadata")
            for key, value in self._metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("## Conversation")
        lines.append("")

        for t in self.turns:
            ir = t.ir
            status = "PASS" if t.passed else "FAIL"
            lines.extend([
                f"### Turn {t.turn_number}",
                "",
                f"> **User:** {t._user_input}",
                "",
                f"**{self.persona_name}:** {t.text}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Confidence | {t.confidence:.3f} |",
                f"| Competence | {t.competence:.3f} |",
                f"| Tone | {ir.communication_style.tone.value} |",
                f"| Verbosity | {ir.communication_style.verbosity.value} |",
                f"| Formality | {ir.communication_style.formality:.3f} |",
                f"| Directness | {ir.communication_style.directness:.3f} |",
                f"| Disclosure | {ir.knowledge_disclosure.disclosure_level:.3f} |",
                f"| Elasticity | {ir.response_structure.elasticity:.3f} |",
                f"| Validation | {status} |",
                f"| Citations | {len(ir.citations)} |",
                "",
            ])

        # Summary
        if self.turns:
            summary = self.summary()
            lines.extend([
                "## Summary",
                "",
                f"- **Average Confidence:** {summary['avg_confidence']:.3f}",
                f"- **Average Competence:** {summary['avg_competence']:.3f}",
                f"- **All Validation Passed:** {summary['all_passed_validation']}",
                "",
            ])

        result = "\n".join(lines)
        if path:
            Path(path).write_text(result)
        return result

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Conversation(persona={self.persona_name!r}, "
            f"turns={self.turn_count})"
        )
