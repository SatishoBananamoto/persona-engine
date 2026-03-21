"""
Behavioral Style — tone, verbosity, communication style (formality + directness).

Extracted from behavioral.py to keep files under ~400 lines.
Used as a mixin by BehavioralMetricsStage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from persona_engine.behavioral import MAX_BIAS_IMPACT
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.trace_context import TraceContext, clamp01
from persona_engine.schema.ir_schema import InteractionMode, Tone, Verbosity

if TYPE_CHECKING:
    from persona_engine.planner.stages.behavioral_guidance import TraitGuidance
    from persona_engine.planner.turn_planner import TurnPlanner

# Config aliases
FORMALITY_ROLE_WEIGHT = DEFAULT_CONFIG.formality_role_weight
FORMALITY_BASE_WEIGHT = DEFAULT_CONFIG.formality_base_weight
DIRECTNESS_IMPATIENCE_BUMP = DEFAULT_CONFIG.directness_impatience_bump
PATIENCE_THRESHOLD = DEFAULT_CONFIG.patience_threshold


class StyleMixin:
    """Mixin providing tone, verbosity, and communication style computation."""

    planner: TurnPlanner  # provided by BehavioralMetricsStage

    def select_tone(
        self, ctx: TraceContext, trait_guidance: TraitGuidance | None = None
    ) -> Tone:
        """Select tone from mood + stress + traits + negativity bias + enthusiasm boost."""
        p = self.planner
        valence, arousal = p.state.get_mood()
        stress = p.state.get_stress()

        if trait_guidance and trait_guidance.enthusiasm_boost > 0.1:
            arousal_before = arousal
            arousal = min(1.0, arousal + trait_guidance.enthusiasm_boost)
            if abs(arousal - arousal_before) > 0.001:
                ctx.num(
                    source_type="trait",
                    source_id="extraversion_enthusiasm",
                    target_field="communication_style.arousal",
                    operation="add",
                    before=arousal_before,
                    after=arousal,
                    effect=f"Extraversion enthusiasm boost: +{trait_guidance.enthusiasm_boost:.3f}",
                    weight=0.6,
                    reason=f"E={p.persona.psychology.big_five.extraversion:.2f}"
                )

        arousal_bias = p.bias_simulator.get_total_modifier_for_field(
            p._current_bias_modifiers,
            "communication_style.arousal"
        )
        if abs(arousal_bias) > 0.001:
            arousal_before = arousal
            arousal = min(1.0, max(0.0, arousal + arousal_bias))
            bias_names = [
                m.bias_type.value for m in p._current_bias_modifiers
                if m.target_field == "communication_style.arousal"
            ]
            ctx.num(
                source_type="trait",
                source_id="arousal_biases",
                target_field="communication_style.arousal",
                operation="add",
                before=arousal_before,
                after=arousal,
                effect=f"Bias ({', '.join(bias_names)}): arousal {arousal_bias:+.3f}",
                weight=min(abs(arousal_bias) / MAX_BIAS_IMPACT, 1.0),
                reason=f"combined_bias={arousal_bias:+.3f}"
            )

        tone = p.traits.get_tone_from_mood(valence, arousal, stress)

        ctx.base_enum(
            field_name="tone.mood_stress_traits",
            target_field="communication_style.tone",
            value=tone.value,
            effect=f"Mood (v={valence:.2f}, a={arousal:.2f}) + stress ({stress:.2f}) → {tone.value}"
        )

        return tone

    def compute_verbosity(
        self,
        ctx: TraceContext,
        verbosity_boost: float = 0.0,
    ) -> Verbosity:
        """Compute verbosity level."""
        p = self.planner
        base_verbosity = p.persona.psychology.communication.verbosity + verbosity_boost
        base_verbosity = max(0.0, min(1.0, base_verbosity))
        verbosity_enum = p.traits.influences_verbosity(base_verbosity)

        current = ctx.base_enum(
            field_name="verbosity.trait_derived",
            target_field="communication_style.verbosity",
            value=verbosity_enum.value,
            effect=f"Trait-derived verbosity ({verbosity_enum.value})"
        )

        modifier = p.state.get_verbosity_modifier()

        if modifier == -1 and verbosity_enum != Verbosity.BRIEF:
            ctx.enum(
                source_type="state",
                source_id="fatigue",
                target_field="communication_style.verbosity",
                operation="override",
                before=current,
                after=Verbosity.BRIEF.value,
                effect="High fatigue reduces verbosity",
                weight=0.6
            )
            return Verbosity.BRIEF

        if modifier == 1 and verbosity_enum != Verbosity.DETAILED:
            ctx.enum(
                source_type="state",
                source_id="engagement",
                target_field="communication_style.verbosity",
                operation="override",
                before=current,
                after=Verbosity.DETAILED.value,
                effect="High engagement increases verbosity",
                weight=0.6
            )
            return Verbosity.DETAILED

        return verbosity_enum

    def compute_communication_style(
        self,
        interaction_mode: InteractionMode,
        ctx: TraceContext,
        trait_guidance: TraitGuidance | None = None,
    ) -> tuple[float, float]:
        """Compute formality and directness using canonical sequence.

        Sequence: base → role → trait → conflict_avoidance → state → constraints

        Returns: (formality, directness)
        """
        p = self.planner

        formality = ctx.base(
            field_name="communication.formality",
            target_field="communication_style.formality",
            value=p.persona.psychology.communication.formality,
            effect=f"Base formality from persona ({p.persona.psychology.communication.formality:.2f})"
        )

        directness = ctx.base(
            field_name="communication.directness",
            target_field="communication_style.directness",
            value=p.persona.psychology.communication.directness,
            effect=f"Base directness from persona ({p.persona.psychology.communication.directness:.2f})"
        )

        # Role blend
        role_mode = p.rules.get_social_role_mode(interaction_mode)
        role_adjustments = p.rules.apply_social_role_adjustments(
            interaction_mode,
            formality,
            directness,
            p.persona.psychology.communication.emotional_expressiveness
        )

        formality = ctx.num(
            source_type="rule",
            source_id=f"social_role_{role_mode}",
            target_field="communication_style.formality",
            operation="blend",
            before=formality,
            after=role_adjustments["formality"],
            effect=f"Role blend applied (70/30) for {role_mode}",
            weight=1.0,
            reason=f"interaction_mode={interaction_mode.value}"
        )

        directness = ctx.num(
            source_type="rule",
            source_id=f"social_role_{role_mode}",
            target_field="communication_style.directness",
            operation="blend",
            before=directness,
            after=role_adjustments["directness"],
            effect=f"Role blend applied (70/30) for {role_mode}",
            weight=1.0,
            reason=f"interaction_mode={interaction_mode.value}"
        )

        # Mode-specific overlay
        if interaction_mode == InteractionMode.DEBATE:
            before_d_debate = directness
            directness = min(1.0, directness + 0.15)
            if abs(directness - before_d_debate) > 0.001:
                ctx.num(
                    source_type="rule",
                    source_id="debate_mode_overlay",
                    target_field="communication_style.directness",
                    operation="add",
                    before=before_d_debate,
                    after=directness,
                    effect="Debate mode: +0.15 directness for argumentative stance",
                    weight=0.9,
                    reason="interaction_mode=debate"
                )

        # Trait modifier
        before_directness = directness
        directness_after_trait = p.traits.influences_directness(directness)

        directness = ctx.num(
            source_type="trait",
            source_id="agreeableness",
            target_field="communication_style.directness",
            operation="add",
            before=before_directness,
            after=directness_after_trait,
            effect=f"Agreeableness ({p.persona.psychology.big_five.agreeableness:.2f}) reduces directness",
            weight=0.8,
            reason=f"A={p.persona.psychology.big_five.agreeableness:.2f} → modifier={directness_after_trait - before_directness:+.3f}"
        )

        # Conflict avoidance
        if trait_guidance and trait_guidance.conflict_avoidance_boost > 0.01:
            before_ca = directness
            directness = max(0.0, directness - trait_guidance.conflict_avoidance_boost)
            if abs(directness - before_ca) > 0.001:
                ctx.num(
                    source_type="trait",
                    source_id="conflict_avoidance",
                    target_field="communication_style.directness",
                    operation="add",
                    before=before_ca,
                    after=directness,
                    effect=f"Conflict avoidance: -{trait_guidance.conflict_avoidance_boost:.3f} (contentious input)",
                    weight=0.7,
                    reason=f"A={p.persona.psychology.big_five.agreeableness:.2f}, contentious input detected"
                )

        # State modifier
        patience = p.state.get_patience_level()
        if patience < PATIENCE_THRESHOLD:
            before_patience = directness
            directness = min(1.0, directness + DIRECTNESS_IMPATIENCE_BUMP)

            directness = ctx.num(
                source_type="state",
                source_id="patience",
                target_field="communication_style.directness",
                operation="add",
                before=before_patience,
                after=directness,
                effect=f"Low patience ({patience:.2f}) increases directness",
                weight=0.5,
                reason=f"patience={patience:.2f} < {PATIENCE_THRESHOLD} → +{DIRECTNESS_IMPATIENCE_BUMP} directness"
            )

        # Clamp
        formality = clamp01(ctx, field_name="formality", target_field="communication_style.formality", value=formality)
        directness = clamp01(ctx, field_name="directness", target_field="communication_style.directness", value=directness)

        return formality, directness
