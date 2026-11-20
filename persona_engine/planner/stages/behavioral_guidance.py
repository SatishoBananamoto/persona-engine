"""
Behavioral Guidance — trait guidance, cognitive guidance, stance generation.

Extracted from behavioral.py to keep files under ~400 lines.
Used as a mixin by BehavioralMetricsStage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

from persona_engine.planner.stance_generator import generate_stance_safe
from persona_engine.planner.trace_context import TraceContext

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner


# ============================================================================
# Behavioral Guidance Dataclasses
# ============================================================================

@dataclass
class TraitGuidance:
    """Behavioral guidance derived from formerly-orphaned trait methods.

    These modifiers flow into the IR and prompt builder as personality-driven
    behavioral directives that were previously computed but never used.
    """
    # Agreeableness-driven
    should_validate_first: bool = False
    hedging_level: float = 0.0
    conflict_avoidance_boost: float = 0.0

    # Extraversion-driven
    enthusiasm_boost: float = 0.0
    proactive_followup: bool = False

    # Openness-driven
    prefer_abstract_language: bool = False
    prefer_novelty: bool = False

    # Neuroticism-driven
    negative_tone_weight: float = 0.0

    # Prompt guidance strings accumulated from trait effects
    prompt_directives: list[str] = field(default_factory=list)


@dataclass
class CognitiveGuidance:
    """Behavioral guidance derived from formerly-orphaned cognitive style methods.

    These influence reasoning presentation, stance complexity, and how
    the persona structures its arguments.
    """
    reasoning_style: Literal["analytical", "intuitive", "mixed"] = "mixed"
    rationale_depth: int = 2
    acknowledge_tradeoffs: bool = False
    stance_dimensions: int = 1
    nuance_level: Literal["low", "moderate", "high"] = "moderate"

    # Prompt guidance strings
    prompt_directives: list[str] = field(default_factory=list)


class GuidanceMixin:
    """Mixin providing trait guidance, cognitive guidance, and stance generation."""

    planner: TurnPlanner  # provided by BehavioralMetricsStage

    def compute_trait_guidance(
        self,
        ctx: TraceContext,
        user_input: str,
    ) -> TraitGuidance:
        """Compute behavioral guidance from trait methods."""
        p = self.planner
        guidance = TraitGuidance()

        # Agreeableness
        validation_tendency = p.traits.get_validation_tendency()
        guidance.should_validate_first = validation_tendency > 0.7
        guidance.hedging_level = p.traits.influences_hedging_frequency()

        contentious_markers = [
            "wrong", "disagree", "terrible", "stupid", "hate",
            "ridiculous", "nonsense", "bad idea", "completely",
            "absolutely not", "no way",
        ]
        is_contentious = any(m in user_input.lower() for m in contentious_markers)
        if is_contentious:
            guidance.conflict_avoidance_boost = p.traits.get_conflict_avoidance() * 0.15
        else:
            guidance.conflict_avoidance_boost = 0.0

        if guidance.should_validate_first:
            guidance.prompt_directives.append(
                "Acknowledge the other person's point before expressing your own view. "
                "Use phrases like 'I see what you mean' or 'That's a fair point'."
            )
        if guidance.hedging_level > 0.4:
            guidance.prompt_directives.append(
                "Use hedging language: 'I think', 'it seems to me', 'perhaps', "
                "'in my experience'. Soften strong claims."
            )

        # Extraversion
        enthusiasm = p.traits.get_enthusiasm_baseline()
        guidance.enthusiasm_boost = enthusiasm * 0.2
        guidance.proactive_followup = p.traits.influences_proactivity() > 0.7

        if guidance.proactive_followup:
            guidance.prompt_directives.append(
                "End with a follow-up question or enthusiastic invitation to continue. "
                "Show active interest in the conversation."
            )

        # Openness
        guidance.prefer_abstract_language = p.traits.influences_abstract_reasoning()
        guidance.prefer_novelty = p.traits.get_novelty_seeking() > 0.7

        if guidance.prefer_abstract_language:
            guidance.prompt_directives.append(
                "Use metaphors, analogies, and abstract reasoning. "
                "Connect ideas across different domains."
            )
        if guidance.prefer_novelty:
            guidance.prompt_directives.append(
                "Favor creative, unconventional, and exploratory perspectives "
                "over tried-and-true conventional ones."
            )

        # Neuroticism
        guidance.negative_tone_weight = p.traits.get_negative_tone_bias()

        ctx.add_basic_citation(
            source_type="trait",
            source_id="trait_guidance",
            effect=(
                f"TraitGuidance: validate={guidance.should_validate_first}, "
                f"hedging={guidance.hedging_level:.2f}, "
                f"enthusiasm_boost={guidance.enthusiasm_boost:.2f}, "
                f"proactive={guidance.proactive_followup}, "
                f"abstract={guidance.prefer_abstract_language}, "
                f"novelty={guidance.prefer_novelty}, "
                f"neg_tone={guidance.negative_tone_weight:.2f}"
            ),
            weight=0.8,
        )

        return guidance

    def compute_cognitive_guidance(self, ctx: TraceContext) -> CognitiveGuidance:
        """Compute behavioral guidance from cognitive style methods."""
        p = self.planner
        guidance = CognitiveGuidance()

        guidance.reasoning_style = p.cognitive.get_reasoning_approach()
        guidance.rationale_depth = p.cognitive.get_rationale_depth()
        guidance.acknowledge_tradeoffs = p.cognitive.should_acknowledge_tradeoffs()
        guidance.stance_dimensions = p.cognitive.get_stance_complexity_level()
        guidance.nuance_level = p.cognitive.get_nuance_capacity()

        if guidance.reasoning_style == "analytical":
            guidance.prompt_directives.append(
                "Present your reasoning step by step. Use logical structure, "
                "evidence, and clear cause-and-effect chains."
            )
        elif guidance.reasoning_style == "intuitive":
            guidance.prompt_directives.append(
                "Share your gut feeling first, then briefly explain why. "
                "Trust your instincts and use pattern-based reasoning."
            )

        if guidance.acknowledge_tradeoffs:
            guidance.prompt_directives.append(
                "Acknowledge tradeoffs and counterarguments. Show that you can "
                "see multiple sides even while holding your position."
            )

        if guidance.nuance_level == "low":
            guidance.prompt_directives.append(
                "Take a clear, decisive position. Avoid excessive qualification. "
                "Black-and-white thinking is natural for you."
            )
        elif guidance.nuance_level == "high":
            guidance.prompt_directives.append(
                "Express nuanced, multifaceted views. Qualify your statements "
                "and consider edge cases and exceptions."
            )

        ctx.add_basic_citation(
            source_type="trait",
            source_id="cognitive_guidance",
            effect=(
                f"CognitiveGuidance: reasoning={guidance.reasoning_style}, "
                f"depth={guidance.rationale_depth}, "
                f"tradeoffs={guidance.acknowledge_tradeoffs}, "
                f"dimensions={guidance.stance_dimensions}, "
                f"nuance={guidance.nuance_level}"
            ),
            weight=0.7,
        )

        return guidance

    def generate_stance(
        self,
        context: ConversationContext,
        proficiency: float,
        expert_allowed: bool,
        evidence_strength: float,
        current_elasticity: float,
        ctx: TraceContext,
    ) -> tuple[str, str]:
        """Generate or retrieve cached stance on topic.

        Returns: (stance, rationale)
        """
        p = self.planner

        cached = context.stance_cache.get_stance(
            context.topic_signature,
            str(context.interaction_mode.value) if context.interaction_mode else "casual_chat",
            context.turn_number
        )

        if cached:
            should_reconsider = context.stance_cache.should_reconsider(
                cached=cached,
                new_evidence_strength=evidence_strength,
                current_elasticity=current_elasticity,
                current_turn=context.turn_number
            )

            if not should_reconsider:
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="stance_cache",
                    effect=f"Using cached stance (age: {context.turn_number - cached.created_turn} turns)",
                    weight=1.0
                )
                return cached.stance, cached.rationale_seeds

            else:
                ctx.add_basic_citation(
                    source_type="rule",
                    source_id="stance_reconsideration",
                    effect=f"Reconsidering cached stance (evidence={evidence_strength:.2f}, elasticity={current_elasticity:.2f})",
                    weight=1.0
                )

        stance, rationale = generate_stance_safe(
            persona=p.persona,
            values=p.values,
            cognitive=p.cognitive,
            user_input=context.user_input,
            topic_signature=context.topic_signature,
            proficiency=proficiency,
            expert_allowed=expert_allowed,
            ctx=ctx
        )

        p.bias_simulator.set_anchor(stance)

        return stance, rationale
