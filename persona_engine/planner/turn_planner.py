"""
Turn Planner - Core Orchestration

The Turn Planner is the heart of the persona engine. It orchestrates all
behavioral interpreters to generate a complete IR following the canonical
modifier composition sequence.

This ensures:
- No double-counting of modifiers
- Clear citation trail
- Deterministic behavior
- Constraint enforcement

Pipeline stages (each in its own module under ``stages/``):
1. Foundation  — trace setup, memory context
2. Interpretation — topic relevance, bias, state, intent, domain, expert eligibility
3. Behavioral metrics — elasticity, stance, confidence, competence, tone, verbosity, comm style
4. Knowledge & safety — disclosure, uncertainty, claim type, patterns, constraints
5. Finalization — memory writes, IR assembly, stance cache, snapshot
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from persona_engine.behavioral import (
    BehavioralRulesEngine,
    BiasModifier,
    BiasSimulator,
    CognitiveStyleInterpreter,
    StateManager,
    TraitInterpreter,
    ValuesInterpreter,
)
from persona_engine.behavioral.trait_interactions import TraitInteractionEngine
from persona_engine.memory import MemoryManager, StanceCache
from persona_engine.planner.engine_config import DEFAULT_CONFIG, EngineConfig
from persona_engine.planner.stages.behavioral import BehavioralMetricsStage
from persona_engine.planner.stages.behavioral_guidance import (
    CognitiveGuidance,
    TraitGuidance,
)
from persona_engine.planner.stages.finalization import FinalizationStage
from persona_engine.planner.stages.foundation import FoundationStage
from persona_engine.planner.stages.interpretation import InterpretationStage
from persona_engine.planner.stages.knowledge import KnowledgeSafetyStage
from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils import DeterminismManager
from persona_engine.validation.cross_turn import TurnSnapshot



# =============================================================================
# ConversationContext (kept here for backward-compatible imports)
# =============================================================================

@dataclass
class ConversationContext:
    """Context for current conversation"""
    conversation_id: str
    turn_number: int
    interaction_mode: InteractionMode | None  # Can be None, inferred by analyze_intent
    goal: ConversationGoal | None  # Can be None, inferred by analyze_intent
    topic_signature: str  # Hash/key for topic
    user_input: str
    stance_cache: StanceCache
    domain: str | None = None


# =============================================================================
# TurnPlanner — thin orchestrator delegating to stage classes
# =============================================================================

class TurnPlanner:
    """
    Orchestrates all behavioral components to generate IR.

    Follows canonical modifier composition sequence:
    base → role → trait → state → constraints

    Each pipeline stage is implemented in a dedicated class under
    ``persona_engine.planner.stages``.
    """

    def __init__(
        self,
        persona: Persona,
        determinism: DeterminismManager | None = None,
        memory_manager: MemoryManager | None = None,
        config: EngineConfig | None = None,
    ):
        self.persona = persona
        self.determinism = determinism or DeterminismManager()
        self.memory = memory_manager
        self.config = config or DEFAULT_CONFIG

        # Initialize all interpreters
        self.traits = TraitInterpreter(persona.psychology.big_five)
        self.trait_interactions = TraitInteractionEngine(persona.psychology.big_five)
        self.values = ValuesInterpreter(persona.psychology.values)
        self.cognitive = CognitiveStyleInterpreter(persona.psychology.cognitive_style)
        self.state = StateManager(
            persona.initial_state,
            persona.psychology.big_five,
            self.determinism
        )
        self.rules = BehavioralRulesEngine(persona)

        # Initialize bias simulator (Phase R6: wire persona-declared biases)
        persona_biases = [
            {"type": b.type, "strength": b.strength}
            for b in persona.biases
        ] if persona.biases else None
        self.bias_simulator = BiasSimulator(
            traits={
                "openness": persona.psychology.big_five.openness,
                "conscientiousness": persona.psychology.big_five.conscientiousness,
                "extraversion": persona.psychology.big_five.extraversion,
                "agreeableness": persona.psychology.big_five.agreeableness,
                "neuroticism": persona.psychology.big_five.neuroticism,
            },
            value_priorities=self.values.get_value_priorities(),
            persona_biases=persona_biases,
            knowledge_boundary_strictness=persona.uncertainty.knowledge_boundary_strictness,
        )

        # Track bias modifiers for current turn (reset per turn)
        self._current_bias_modifiers: list[BiasModifier] = []

        # Cross-turn dynamics: previous turn's IR snapshot for inertia smoothing
        self._prior_snapshot: TurnSnapshot | None = None

        # Initialize pipeline stages
        self._foundation = FoundationStage(self)
        self._interpretation = InterpretationStage(self)
        self._behavioral = BehavioralMetricsStage(self)
        self._knowledge = KnowledgeSafetyStage(self)
        self._finalization = FinalizationStage(self)

    def generate_ir(self, context: ConversationContext) -> IntermediateRepresentation:
        """
        Generate Intermediate Representation for a single turn.

        Orchestrates five pipeline stages:
        1. Foundation  — trace setup, memory context
        2. Interpretation — topic relevance, bias, state, intent, domain, expert eligibility
        3. Behavioral metrics — elasticity, stance, confidence, competence, tone, verbosity, comm style
        4. Knowledge & safety — disclosure, uncertainty, claim type, patterns, constraints
        5. Finalization — memory writes, IR assembly, stance cache, snapshot
        """
        logger.debug(
            "Generating IR",
            extra={"user_input": context.user_input[:50], "turn_number": context.turn_number},
        )
        ctx, turn_seed, memory_ops, memory_context = self._stage_foundation(context)
        foundation = self._stage_interpretation(context, ctx)
        foundation.memory_context = memory_context
        metrics = self._stage_behavioral_metrics(context, ctx, foundation)
        knowledge = self._stage_knowledge_safety(context, ctx, foundation, metrics)
        ir = self._stage_finalization(
            context, ctx, turn_seed, memory_ops, foundation, metrics, knowledge
        )

        # State evolution — applied AFTER IR is finalized (affects next turn, not current)
        self.state.evolve_state_post_turn(
            conversation_length=context.turn_number,
            topic_relevance=foundation.topic_relevance,
        )

        return ir

    # ========================================================================
    # Stage delegation methods (preserve method names for backward compat)
    # ========================================================================

    def _stage_foundation(self, context):
        """Stage 1: TraceContext setup, per-turn seed, and memory context."""
        return self._foundation.execute(context)

    def _stage_interpretation(self, context, ctx):
        """Stage 2: Topic relevance, bias, state evolution, intent, domain, expert eligibility."""
        return self._interpretation.execute(context, ctx)

    def _stage_behavioral_metrics(self, context, ctx, foundation):
        """Stage 3: Elasticity, stance, confidence, competence, tone, verbosity, communication style."""
        return self._behavioral.execute(context, ctx, foundation)

    def _stage_knowledge_safety(self, context, ctx, foundation, metrics):
        """Stage 4: Disclosure, uncertainty action, claim type, patterns, constraints."""
        return self._knowledge.execute(context, ctx, foundation, metrics)

    def _stage_finalization(self, context, ctx, turn_seed, memory_ops, foundation, metrics, knowledge):
        """Stage 5: Memory writes, IR assembly, stance cache, snapshot."""
        return self._finalization.execute(
            context, ctx, turn_seed, memory_ops, foundation, metrics, knowledge
        )



def create_turn_planner(
    persona: Persona,
    determinism: DeterminismManager | None = None,
    memory_manager: MemoryManager | None = None,
    config: EngineConfig | None = None,
) -> TurnPlanner:
    """Factory function to create turn planner"""
    return TurnPlanner(persona, determinism, memory_manager=memory_manager, config=config)
