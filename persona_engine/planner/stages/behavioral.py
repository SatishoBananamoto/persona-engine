"""
Stage 3: Behavioral Metrics — elasticity, stance, confidence, competence,
tone, verbosity, communication style, trait/cognitive guidance.

This module is the orchestrator. Computation methods live in:
- behavioral_metrics.py  (elasticity, confidence, competence)
- behavioral_style.py    (tone, verbosity, formality, directness)
- behavioral_guidance.py (trait guidance, cognitive guidance, stance)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from persona_engine.behavioral.emotional_appraisal import (
    appraise_event,
    detect_user_emotion,
)
from persona_engine.behavioral.social_cognition import (
    compute_adaptation,
    compute_schema_effect,
    detect_schema_relevance,
    infer_user_model,
)
from persona_engine.planner.domain_detection import detect_evidence_strength
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.stages.behavioral_guidance import (
    CognitiveGuidance,
    GuidanceMixin,
    TraitGuidance,
)
from persona_engine.planner.stages.behavioral_metrics import MetricsMixin
from persona_engine.planner.stages.behavioral_style import StyleMixin
from persona_engine.planner.stages.stage_results import (
    BehavioralMetricsResult,
    InterpretationResult,
)
from persona_engine.planner.trace_context import TraceContext

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)

# Config aliases
EVIDENCE_STRESS_THRESHOLD = DEFAULT_CONFIG.evidence_stress_threshold
CROSS_TURN_INERTIA = DEFAULT_CONFIG.cross_turn_inertia
PERSONALITY_FIELD_INERTIA = DEFAULT_CONFIG.personality_field_inertia


def _smooth(prev: float, new: float, alpha: float) -> float:
    """Cross-turn inertia smoothing: blend previous and current values."""
    return prev * alpha + new * (1 - alpha)


class BehavioralMetricsStage(MetricsMixin, StyleMixin, GuidanceMixin):
    """Computes all behavioral metrics: elasticity, stance, confidence,
    competence, tone, verbosity, formality, directness, plus trait and
    cognitive guidance.

    Computation methods are provided by mixin classes:
    - MetricsMixin:  compute_elasticity, compute_confidence, compute_competence
    - StyleMixin:    select_tone, compute_verbosity, compute_communication_style
    - GuidanceMixin: compute_trait_guidance, compute_cognitive_guidance, generate_stance
    """

    def __init__(self, planner: TurnPlanner) -> None:
        self.planner = planner

    def execute(
        self,
        context: ConversationContext,
        ctx: TraceContext,
        foundation: InterpretationResult,
    ) -> BehavioralMetricsResult:
        """Run the behavioral metrics stage."""
        p = self.planner
        domain = foundation.domain
        proficiency = foundation.proficiency
        persona_domains = foundation.persona_domains

        # Elasticity (early for stance cache logic)
        elasticity = self.compute_elasticity(proficiency, ctx)

        # Cross-turn inertia smoothing — elasticity
        if p._prior_snapshot:
            before_e = elasticity
            elasticity = _smooth(p._prior_snapshot.elasticity, elasticity, PERSONALITY_FIELD_INERTIA)
            if abs(elasticity - before_e) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="response_structure.elasticity",
                    operation="blend",
                    before=before_e,
                    after=elasticity,
                    effect=f"Cross-turn inertia: {before_e:.3f} → {elasticity:.3f} (prev={p._prior_snapshot.elasticity:.3f})",
                    weight=0.7,
                    reason=f"inertia={PERSONALITY_FIELD_INERTIA}",
                )

        # Stance
        evidence_strength = detect_evidence_strength(context.user_input, ctx=ctx)
        if evidence_strength > EVIDENCE_STRESS_THRESHOLD:
            p.state.apply_stress_trigger("conflict", intensity=evidence_strength)
            ctx.add_basic_citation(
                source_type="state",
                source_id="stress_trigger",
                effect=f"Evidence {evidence_strength:.2f} > threshold {EVIDENCE_STRESS_THRESHOLD} → stress trigger",
                weight=1.0,
            )

        stance, rationale = self.generate_stance(
            context=context,
            proficiency=proficiency,
            expert_allowed=foundation.expert_allowed,
            evidence_strength=evidence_strength,
            current_elasticity=elasticity,
            ctx=ctx,
        )

        # Confidence (+ cross-turn smoothing)
        confidence = self.compute_confidence(proficiency, ctx, memory_context=foundation.memory_context)
        if p._prior_snapshot:
            before_smooth = confidence
            confidence = _smooth(p._prior_snapshot.confidence, confidence, CROSS_TURN_INERTIA)
            if abs(confidence - before_smooth) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="response_structure.confidence",
                    operation="blend",
                    before=before_smooth,
                    after=confidence,
                    effect=f"Cross-turn inertia: {before_smooth:.3f} → {confidence:.3f} (prev={p._prior_snapshot.confidence:.3f})",
                    weight=0.7,
                    reason=f"inertia={CROSS_TURN_INERTIA}",
                )

        competence = self.compute_competence(domain, proficiency, persona_domains, ctx)

        # Cross-turn inertia smoothing — competence
        if p._prior_snapshot:
            before_comp = competence
            competence = _smooth(p._prior_snapshot.competence, competence, CROSS_TURN_INERTIA)
            if abs(competence - before_comp) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="response_structure.competence",
                    operation="blend",
                    before=before_comp,
                    after=competence,
                    effect=f"Cross-turn inertia: {before_comp:.3f} → {competence:.3f} (prev={p._prior_snapshot.competence:.3f})",
                    weight=0.7,
                    reason=f"inertia={CROSS_TURN_INERTIA}",
                )

        # Phase R4: Emotional appraisal
        user_emotion = detect_user_emotion(context.user_input)
        if sum(user_emotion.values()) > 0.25:
            appraisal = appraise_event(
                user_emotion, p.persona.psychology.big_five, p.state.get_stress()
            )
            if abs(appraisal.valence_delta) > 0.01 or abs(appraisal.arousal_delta) > 0.01:
                p.state.update_mood_from_event(appraisal.valence_delta, appraisal.arousal_delta)
                ctx.add_basic_citation(
                    source_type="state",
                    source_id="emotional_appraisal",
                    effect=(
                        f"Appraisal ({appraisal.dominant_emotion}): "
                        f"valence {appraisal.valence_delta:+.3f}, "
                        f"arousal {appraisal.arousal_delta:+.3f}"
                    ),
                    weight=0.7,
                )

        # Phase R6: Social cognition
        user_model = infer_user_model(context.user_input)
        adaptation = compute_adaptation(
            user_model, p.persona.psychology.big_five,
            base_disclosure=p.persona.disclosure_policy.base_openness,
        )

        # Phase R6: Self-schema protection
        self_schemas = getattr(p.persona, 'self_schemas', []) or []
        schema_match, schema_challenge = detect_schema_relevance(
            context.user_input, self_schemas
        )
        schema_effect = compute_schema_effect(schema_match, schema_challenge)
        if schema_match:
            confidence += schema_effect.confidence_modifier
            elasticity += schema_effect.elasticity_modifier
            if abs(schema_effect.confidence_modifier) > 0.001 or abs(schema_effect.elasticity_modifier) > 0.001:
                ctx.add_basic_citation(
                    source_type="rule",
                    source_id="self_schema",
                    effect=(
                        f"Self-schema '{schema_match}' {'challenged' if schema_challenge else 'validated'}: "
                        f"conf {schema_effect.confidence_modifier:+.2f}, "
                        f"elast {schema_effect.elasticity_modifier:+.2f}"
                    ),
                    weight=0.7,
                )

        # Trait & cognitive guidance
        trait_guidance = self.compute_trait_guidance(ctx, context.user_input)
        cognitive_guidance = self.compute_cognitive_guidance(ctx)

        # Phase R3: Trait interaction patterns
        interaction_effects = p.trait_interactions.detect_active_patterns(threshold=0.1)
        interaction_modifiers = p.trait_interactions.get_aggregate_modifiers(threshold=0.1)

        interaction_names = ", ".join(e.pattern_name for e in interaction_effects) if interaction_effects else ""
        if interaction_modifiers:
            for field_name in ("confidence", "elasticity"):
                if field_name in interaction_modifiers:
                    before_val = confidence if field_name == "confidence" else elasticity
                    # Confidence floor 0.12: even extreme-N personas retain
                    # some confidence (TF-002 fix). Elasticity floor stays 0.1.
                    lo, hi = (0.12, 0.95) if field_name == "confidence" else (0.1, 0.9)
                    new_val = max(lo, min(hi, before_val + interaction_modifiers[field_name]))
                    if field_name == "confidence":
                        confidence = new_val
                    else:
                        elasticity = new_val
                    if abs(new_val - before_val) > 0.001:
                        ctx.num(
                            source_type="trait_interaction",
                            source_id=f"{field_name}_modifier",
                            target_field=f"response_structure.{field_name}",
                            operation="add",
                            before=before_val,
                            after=new_val,
                            effect=f"Trait interaction: {field_name} {interaction_modifiers[field_name]:+.3f}",
                            weight=0.8,
                            reason=interaction_names,
                        )

            if "hedging_level" in interaction_modifiers:
                trait_guidance.hedging_level = min(1.0, trait_guidance.hedging_level + interaction_modifiers["hedging_level"])
            if "enthusiasm_boost" in interaction_modifiers:
                trait_guidance.enthusiasm_boost = max(0.0, trait_guidance.enthusiasm_boost + interaction_modifiers["enthusiasm_boost"])
            if "negative_tone_bias" in interaction_modifiers:
                trait_guidance.negative_tone_weight = min(1.0, trait_guidance.negative_tone_weight + interaction_modifiers["negative_tone_bias"])

            for effect in interaction_effects:
                if effect.activation_strength > 0.3:
                    trait_guidance.prompt_directives.append(
                        f"[{effect.pattern_name.replace('_', ' ').title()}] {effect.prompt_guidance}"
                    )

        tone = self.select_tone(ctx, trait_guidance=trait_guidance)
        verbosity = self.compute_verbosity(
            ctx, verbosity_boost=interaction_modifiers.get("verbosity_boost", 0.0)
        )

        # Communication style (+ cross-turn smoothing)
        formality, directness = self.compute_communication_style(
            context.interaction_mode, ctx, trait_guidance=trait_guidance
        )

        # Phase R3: Apply interaction directness modifier
        if "directness" in interaction_modifiers:
            before_id = directness
            directness = max(0.0, min(1.0, directness + interaction_modifiers["directness"]))
            if abs(directness - before_id) > 0.001:
                ctx.num(
                    source_type="trait_interaction",
                    source_id="directness_modifier",
                    target_field="communication_style.directness",
                    operation="add",
                    before=before_id,
                    after=directness,
                    effect=f"Trait interaction: directness {interaction_modifiers['directness']:+.3f}",
                    weight=0.8,
                    reason=interaction_names,
                )
        if p._prior_snapshot:
            before_f = formality
            formality = _smooth(p._prior_snapshot.formality, formality, PERSONALITY_FIELD_INERTIA)
            if abs(formality - before_f) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="communication_style.formality",
                    operation="blend",
                    before=before_f,
                    after=formality,
                    effect=f"Cross-turn inertia: {before_f:.3f} → {formality:.3f}",
                    weight=0.7,
                    reason=f"inertia={PERSONALITY_FIELD_INERTIA}",
                )

            before_d = directness
            directness = _smooth(p._prior_snapshot.directness, directness, PERSONALITY_FIELD_INERTIA)
            if abs(directness - before_d) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="communication_style.directness",
                    operation="blend",
                    before=before_d,
                    after=directness,
                    effect=f"Cross-turn inertia: {before_d:.3f} → {directness:.3f}",
                    weight=0.7,
                    reason=f"inertia={PERSONALITY_FIELD_INERTIA}",
                )

        # Phase R6: Apply adaptation modifiers
        if adaptation:
            if abs(adaptation.formality_shift) > 0.01:
                before_af = formality
                formality = max(0.0, min(1.0, formality + adaptation.formality_shift))
                if abs(formality - before_af) > 0.001:
                    ctx.num(
                        source_type="rule",
                        source_id="social_adaptation_formality",
                        target_field="communication_style.formality",
                        operation="add",
                        before=before_af,
                        after=formality,
                        effect=f"Social adaptation: formality {adaptation.formality_shift:+.3f}",
                        weight=0.6,
                    )
            if abs(adaptation.depth_shift) > 0.01:
                from persona_engine.schema.ir_schema import Verbosity as _Verb
                _verb_order = [_Verb.BRIEF, _Verb.MEDIUM, _Verb.DETAILED]
                _cur_idx = _verb_order.index(verbosity) if verbosity in _verb_order else 1
                if adaptation.depth_shift > 0 and _cur_idx < 2:
                    before_av = verbosity
                    verbosity = _verb_order[_cur_idx + 1]
                    ctx.enum(
                        source_type="rule",
                        source_id="social_adaptation_depth",
                        target_field="communication_style.verbosity",
                        operation="override",
                        before=before_av.value,
                        after=verbosity.value,
                        effect="Social adaptation: depth up for expert user",
                        weight=0.6,
                    )
                elif adaptation.depth_shift < 0 and _cur_idx > 0:
                    before_av = verbosity
                    verbosity = _verb_order[_cur_idx - 1]
                    ctx.enum(
                        source_type="rule",
                        source_id="social_adaptation_depth",
                        target_field="communication_style.verbosity",
                        operation="override",
                        before=before_av.value,
                        after=verbosity.value,
                        effect="Social adaptation: depth down for novice user",
                        weight=0.6,
                    )

        logger.debug(
            "Behavioral metrics computed",
            extra={"confidence": confidence, "elasticity": elasticity, "tone": str(tone)},
        )
        return BehavioralMetricsResult(
            elasticity=elasticity,
            stance=stance,
            rationale=rationale,
            confidence=confidence,
            competence=competence,
            tone=tone,
            verbosity=verbosity,
            formality=formality,
            directness=directness,
            trait_guidance=trait_guidance,
            cognitive_guidance=cognitive_guidance,
            adaptation=adaptation,
            schema_effect=schema_effect,
        )
