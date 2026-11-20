"""
Stage 1: Foundation — trace setup, per-turn seed, and memory context.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from persona_engine.planner.trace_context import TraceContext, create_turn_seed
from persona_engine.schema.ir_schema import MemoryOps

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)


class FoundationStage:
    """Initializes trace context, per-turn seed, and loads memory context."""

    def __init__(self, planner: TurnPlanner) -> None:
        self.planner = planner

    def execute(
        self, context: ConversationContext
    ) -> tuple[TraceContext, int, MemoryOps, dict[str, Any]]:
        """Run the foundation stage.

        Returns:
            (trace_context, turn_seed, memory_ops, memory_context)
        """
        p = self.planner
        ctx = TraceContext()
        memory_ops = MemoryOps()
        memory_context: dict[str, Any] = {}

        turn_seed = create_turn_seed(
            base_seed=p.determinism.seed if p.determinism.seed is not None else 0,
            conversation_id=context.conversation_id,
            turn_number=context.turn_number,
        )
        p.determinism.set_seed(turn_seed)

        if p.memory:
            memory_context = p.memory.get_context_for_turn(
                topic=context.topic_signature,
                current_turn=context.turn_number,
            )
            fact_count = len(memory_context.get("known_facts", []))
            logger.debug(
                "Memory context loaded",
                extra={"fact_count": fact_count, "topic": context.topic_signature},
            )
            if memory_context.get("known_facts"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="fact_store",
                    effect=f"Loaded {len(memory_context['known_facts'])} known facts from memory",
                    weight=0.8,
                )
            if memory_context.get("active_preferences"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="preference_store",
                    effect=f"Loaded {len(memory_context['active_preferences'])} active preferences",
                    weight=0.6,
                )
            if memory_context.get("previously_discussed"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="episodic_store",
                    effect=f"Topic '{context.topic_signature}' previously discussed",
                    weight=0.7,
                )

        return ctx, turn_seed, memory_ops, memory_context
