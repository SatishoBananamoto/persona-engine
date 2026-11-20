"""
Stage 5: Finalization — memory writes, IR assembly, stance cache, snapshot.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from persona_engine.behavioral.linguistic_markers import (
    build_personality_language_directives,
)
from persona_engine.behavioral.social_cognition import (
    AdaptationDirectives,
    SchemaEffect,
)
from persona_engine.planner.domain_detection import generate_intent_string
from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    MemoryOps,
    MemoryReadRequest,
    MemoryWriteIntent,
    ResponseStructure,
)
from persona_engine.planner.stages.behavioral import (
    CognitiveGuidance,
    TraitGuidance,
)
from persona_engine.planner.stages.stage_results import (
    BehavioralMetricsResult,
    InterpretationResult,
    KnowledgeSafetyResult,
)
from persona_engine.validation.cross_turn import TurnSnapshot

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)


class FinalizationStage:
    """Assembles the IR, writes to memory, caches stance, and stores snapshot."""

    def __init__(self, planner: TurnPlanner) -> None:
        self.planner = planner

    def execute(
        self,
        context: ConversationContext,
        ctx: TraceContext,
        turn_seed: int,
        memory_ops: MemoryOps,
        foundation: InterpretationResult,
        metrics: BehavioralMetricsResult,
        knowledge: KnowledgeSafetyResult,
    ) -> IntermediateRepresentation:
        """Run the finalization stage."""
        p = self.planner
        user_intent = foundation.user_intent
        needs_clarification = foundation.needs_clarification
        stance = metrics.stance
        rationale = metrics.rationale
        elasticity = metrics.elasticity
        confidence = metrics.confidence
        competence = metrics.competence
        tone = metrics.tone
        verbosity = metrics.verbosity
        formality = metrics.formality
        directness = metrics.directness
        disclosure_level = knowledge.disclosure_level
        uncertainty_action = knowledge.uncertainty_action
        claim_enum = knowledge.claim_enum

        # Memory read requests
        read_requests: list[MemoryReadRequest] = []
        memory_context = foundation.memory_context
        if p.memory and memory_context:
            if memory_context.get("known_facts"):
                read_requests.append(MemoryReadRequest(
                    query_type="fact",
                    query=f"Facts relevant to {context.topic_signature}",
                ))
            if memory_context.get("active_preferences"):
                read_requests.append(MemoryReadRequest(
                    query_type="preference",
                    query="Active user preferences",
                ))
            if memory_context.get("topic_episodes"):
                read_requests.append(MemoryReadRequest(
                    query_type="episode",
                    query=f"Episodes about {context.topic_signature}",
                ))
            if memory_context.get("relationship"):
                read_requests.append(MemoryReadRequest(
                    query_type="relationship",
                    query="Current relationship state",
                ))

        # Memory write intents
        write_intents: list[MemoryWriteIntent] = []
        if p.memory:
            write_intents.append(MemoryWriteIntent(
                content_type="episode",
                content=f"Discussed {context.topic_signature}: persona {uncertainty_action.value}ed with {claim_enum.value} claim",
                confidence=0.9,
                privacy_level=0.2,
                source="observed_behavior",
            ))
            if stance and confidence > 0.5:
                rel_content = f"Engaged on {context.topic_signature}"
                if confidence > 0.7:
                    rel_content += " — validated shared expertise"
                if claim_enum.value == "personal_experience":
                    rel_content += " — shared personal perspective"
                if user_intent == "challenge":
                    rel_content += " — challenged viewpoint"
                elif user_intent == "share":
                    rel_content += " — friendly exchange"
                write_intents.append(MemoryWriteIntent(
                    content_type="relationship",
                    content=rel_content,
                    confidence=0.7,
                    privacy_level=0.1,
                    source="observed_behavior",
                ))
            memory_ops = MemoryOps(
                read_requests=read_requests,
                write_intents=write_intents,
            )

        # Propagate invariants into safety plan
        ctx.safety_plan.cannot_claim = list(p.persona.invariants.cannot_claim)
        ctx.safety_plan.must_avoid = list(p.persona.invariants.must_avoid)

        # Collect behavioral directives
        trait_guidance: TraitGuidance | None = metrics.trait_guidance
        cognitive_guidance: CognitiveGuidance | None = metrics.cognitive_guidance
        adaptation: AdaptationDirectives | None = metrics.adaptation
        schema_effect: SchemaEffect | None = metrics.schema_effect
        behavioral_directives: list[str] = []
        if trait_guidance:
            behavioral_directives.extend(trait_guidance.prompt_directives)
        if cognitive_guidance:
            behavioral_directives.extend(cognitive_guidance.prompt_directives)

        if adaptation and adaptation.prompt_directives:
            behavioral_directives.extend(adaptation.prompt_directives)
        if schema_effect and schema_effect.prompt_directive:
            behavioral_directives.append(schema_effect.prompt_directive)

        # Personality-specific language directives
        current_mood = p.state.get_mood()
        linguistic_profile = build_personality_language_directives(
            traits=p.persona.psychology.big_five,
            determinism=p.determinism,
            mood_valence=current_mood[0],
            mood_arousal=current_mood[1],
            interaction_formality=formality,
        )
        personality_language: list[str] = []
        personality_language.extend(linguistic_profile.personality_directives)
        personality_language.extend(linguistic_profile.marker_directives)
        if linguistic_profile.emotional_coloring:
            personality_language.append(linguistic_profile.emotional_coloring)

        if personality_language:
            ctx.add_basic_citation(
                source_type="trait",
                source_id="linguistic_markers",
                effect=f"Phase R5: {len(personality_language)} personality language directives",
                weight=0.6,
            )

        # Assemble IR
        ir = IntermediateRepresentation(
            conversation_frame=ConversationFrame(
                interaction_mode=context.interaction_mode,
                goal=context.goal,
                success_criteria=self.derive_success_criteria(),
            ),
            response_structure=ResponseStructure(
                intent=self.generate_intent(
                    user_intent=user_intent,
                    conversation_goal=str(context.goal.value) if context.goal else "inform",
                    uncertainty_action=str(uncertainty_action.value),
                    needs_clarification=needs_clarification,
                    ctx=ctx,
                ),
                stance=stance,
                rationale=rationale,
                elasticity=elasticity,
                confidence=confidence,
                competence=competence,
            ),
            communication_style=CommunicationStyle(
                tone=tone,
                verbosity=verbosity,
                formality=formality,
                directness=directness,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=disclosure_level,
                uncertainty_action=uncertainty_action,
                knowledge_claim_type=claim_enum,
            ),
            citations=ctx.citations,
            safety_plan=ctx.safety_plan,
            memory_ops=memory_ops,
            behavioral_directives=behavioral_directives,
            personality_language=personality_language,
            turn_id=f"{context.conversation_id}_turn_{context.turn_number}",
            seed=turn_seed,
        )

        logger.info(
            "IR complete",
            extra={
                "turn_id": ir.turn_id,
                "confidence": confidence,
                "tone": str(tone),
                "verbosity": str(verbosity),
            },
        )

        # Cache stance
        if stance:
            context.stance_cache.store_stance(
                topic_signature=context.topic_signature,
                interaction_mode=str(context.interaction_mode.value),
                stance=stance,
                rationale=rationale,
                elasticity=elasticity,
                confidence=confidence,
                turn_number=context.turn_number,
            )

        # Process memory writes
        if p.memory and ir.memory_ops.write_intents:
            p.memory.process_write_intents(
                intents=ir.memory_ops.write_intents,
                turn=context.turn_number,
                conversation_id=context.conversation_id,
                write_policy=ir.memory_ops.write_policy,
            )

        # Store snapshot for cross-turn dynamics
        p._prior_snapshot = TurnSnapshot.from_ir(
            ir, context.turn_number, context.topic_signature
        )

        return ir

    # ========================================================================
    # Helpers
    # ========================================================================

    def derive_success_criteria(self) -> list[str] | None:
        """Derive success_criteria from persona's active goals."""
        p = self.planner
        all_goals = []
        for g in p.persona.primary_goals:
            all_goals.append((g.goal, g.weight * 1.0))
        for g in p.persona.secondary_goals:
            all_goals.append((g.goal, g.weight * 0.5))

        if not all_goals:
            return None

        all_goals.sort(key=lambda x: x[1], reverse=True)
        criteria = [g[0] for g in all_goals[:3]]
        return criteria

    def generate_intent(
        self,
        user_intent: str,
        conversation_goal: str,
        uncertainty_action: str,
        needs_clarification: bool = False,
        ctx: Optional[TraceContext] = None,
    ) -> str:
        """Generate meaningful response intent using template logic."""
        return generate_intent_string(
            user_intent=user_intent,
            conversation_goal=conversation_goal,
            uncertainty_action=uncertainty_action,
            needs_clarification=needs_clarification,
            ctx=ctx,
        )
