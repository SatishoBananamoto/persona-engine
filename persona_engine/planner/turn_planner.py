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
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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
from persona_engine.planner.stages.behavioral import (
    BehavioralMetricsStage,
    CognitiveGuidance,
    TraitGuidance,
    _smooth,
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
    UncertaintyAction,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils import DeterminismManager
from persona_engine.validation.cross_turn import TurnSnapshot

# =============================================================================
# CONFIGURATION CONSTANTS (backward-compatible aliases from EngineConfig)
# =============================================================================
DEFAULT_PROFICIENCY = DEFAULT_CONFIG.default_proficiency
EXPERT_THRESHOLD = DEFAULT_CONFIG.expert_threshold
DEFAULT_TOPIC_RELEVANCE = DEFAULT_CONFIG.default_topic_relevance
FORMALITY_ROLE_WEIGHT = DEFAULT_CONFIG.formality_role_weight
FORMALITY_BASE_WEIGHT = DEFAULT_CONFIG.formality_base_weight
DIRECTNESS_IMPATIENCE_BUMP = DEFAULT_CONFIG.directness_impatience_bump
PATIENCE_THRESHOLD = DEFAULT_CONFIG.patience_threshold
ELASTICITY_MIN = DEFAULT_CONFIG.elasticity_min
ELASTICITY_MAX = DEFAULT_CONFIG.elasticity_max
DISCLOSURE_MIN = DEFAULT_CONFIG.disclosure_min
DISCLOSURE_MAX = DEFAULT_CONFIG.disclosure_max
EVIDENCE_STRESS_THRESHOLD = DEFAULT_CONFIG.evidence_stress_threshold
UNKNOWN_DOMAIN_BASE = DEFAULT_CONFIG.unknown_domain_base
OPENNESS_COMPETENCE_WEIGHT = DEFAULT_CONFIG.openness_competence_weight
CROSS_TURN_INERTIA = DEFAULT_CONFIG.cross_turn_inertia
PERSONALITY_FIELD_INERTIA = DEFAULT_CONFIG.personality_field_inertia
FAMILIARITY_BOOST_PER_EPISODE = DEFAULT_CONFIG.familiarity_boost_per_episode
FAMILIARITY_BOOST_CAP = DEFAULT_CONFIG.familiarity_boost_cap
TIME_PRESSURE_TURN_THRESHOLD = DEFAULT_CONFIG.time_pressure_turn_threshold
TIME_PRESSURE_PER_TURN = DEFAULT_CONFIG.time_pressure_per_turn
TIME_PRESSURE_MAX_BUILDUP = DEFAULT_CONFIG.time_pressure_max_buildup


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
        return self._stage_finalization(
            context, ctx, turn_seed, memory_ops, foundation, metrics, knowledge
        )

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

    # ========================================================================
    # Delegating wrappers for test-accessed private methods
    # ========================================================================

    def _detect_domain(self, user_input: str, ctx: TraceContext | None = None) -> str:
        """Detect domain from user input using keyword scoring."""
        return self._interpretation.detect_domain(user_input, ctx=ctx)

    def _get_domain_proficiency(self, domain: str) -> float:
        """Get persona's proficiency in domain."""
        return self._interpretation.get_domain_proficiency(domain)

    def _compute_elasticity(self, proficiency: float, ctx: TraceContext) -> float:
        """Compute elasticity (openness to persuasion)."""
        return self._behavioral.compute_elasticity(proficiency, ctx)

    def _compute_confidence(
        self,
        proficiency: float,
        ctx: TraceContext,
        memory_context: dict[str, Any] | None = None,
    ) -> float:
        """Compute response confidence."""
        return self._behavioral.compute_confidence(proficiency, ctx, memory_context)

    def _compute_competence(
        self,
        domain: str,
        proficiency: float,
        persona_domains: list[dict],
        ctx: TraceContext,
    ) -> float:
        """Compute how equipped the persona is to engage with this topic."""
        return self._behavioral.compute_competence(domain, proficiency, persona_domains, ctx)

    def _select_tone(
        self, ctx: TraceContext, trait_guidance: TraitGuidance | None = None
    ):
        """Select tone from mood + stress + traits + negativity bias + enthusiasm boost."""
        return self._behavioral.select_tone(ctx, trait_guidance=trait_guidance)

    def _compute_verbosity(self, ctx: TraceContext, verbosity_boost: float = 0.0):
        """Compute verbosity level."""
        return self._behavioral.compute_verbosity(ctx, verbosity_boost=verbosity_boost)

    def _compute_communication_style(
        self,
        interaction_mode: InteractionMode,
        ctx: TraceContext,
        trait_guidance: TraitGuidance | None = None,
    ) -> tuple[float, float]:
        """Compute formality and directness."""
        return self._behavioral.compute_communication_style(
            interaction_mode, ctx, trait_guidance=trait_guidance
        )

    def _compute_trait_guidance(self, ctx: TraceContext, user_input: str) -> TraitGuidance:
        """Compute behavioral guidance from trait methods."""
        return self._behavioral.compute_trait_guidance(ctx, user_input)

    def _compute_cognitive_guidance(self, ctx: TraceContext) -> CognitiveGuidance:
        """Compute behavioral guidance from cognitive style methods."""
        return self._behavioral.compute_cognitive_guidance(ctx)

    def _generate_stance(
        self,
        context: ConversationContext,
        proficiency: float,
        expert_allowed: bool,
        evidence_strength: float,
        current_elasticity: float,
        ctx: TraceContext,
    ) -> tuple[str, str]:
        """Generate or retrieve cached stance on topic."""
        return self._behavioral.generate_stance(
            context, proficiency, expert_allowed, evidence_strength, current_elasticity, ctx
        )

    def _compute_disclosure(self, topic_signature: str, ctx: TraceContext) -> float:
        """Compute disclosure level with canonical composition."""
        return self._knowledge.compute_disclosure(topic_signature, ctx)

    def _compute_time_pressure(
        self,
        interaction_mode: InteractionMode,
        turn_number: int,
        ctx: TraceContext,
    ) -> float:
        """Compute context-sensitive time pressure."""
        return self._knowledge.compute_time_pressure(interaction_mode, turn_number, ctx)

    def _infer_claim_type(
        self,
        proficiency: float,
        uncertainty_action: UncertaintyAction,
        domain: str,
        user_input: str = "",
    ) -> str:
        """Infer knowledge claim type."""
        return self._knowledge.infer_claim_type(proficiency, uncertainty_action, domain, user_input)

    def _detect_personal_experience(
        self,
        user_input: str,
        domain: str,
        is_domain_specific: bool,
    ) -> bool:
        """Detect if user is asking about personal experience."""
        return self._knowledge.detect_personal_experience(user_input, domain, is_domain_specific)

    def _generate_intent(
        self,
        user_intent: str,
        conversation_goal: str,
        uncertainty_action: str,
        needs_clarification: bool = False,
        ctx: Optional[TraceContext] = None,
    ) -> str:
        """Generate meaningful response intent using template logic."""
        return self._finalization.generate_intent(
            user_intent, conversation_goal, uncertainty_action, needs_clarification, ctx
        )

    def _apply_patterns_safely(
        self,
        context: ConversationContext,
        disclosure_level: float,
        ctx: TraceContext,
    ) -> None:
        """Apply response patterns with safety checks."""
        return self._knowledge.apply_patterns_safely(context, disclosure_level, ctx)

    def _derive_success_criteria(self) -> list[str] | None:
        """Derive success_criteria from persona's active goals."""
        return self._finalization.derive_success_criteria()


def create_turn_planner(
    persona: Persona,
    determinism: DeterminismManager | None = None,
    memory_manager: MemoryManager | None = None,
    config: EngineConfig | None = None,
) -> TurnPlanner:
    """Factory function to create turn planner"""
    return TurnPlanner(persona, determinism, memory_manager=memory_manager, config=config)
