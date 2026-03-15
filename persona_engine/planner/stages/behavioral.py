"""
Stage 3: Behavioral Metrics — elasticity, stance, confidence, competence,
tone, verbosity, communication style, trait/cognitive guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from persona_engine.behavioral import (
    MAX_BIAS_IMPACT,
    apply_response_pattern_safely,
)
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
from persona_engine.planner.domain_detection import (
    compute_domain_adjacency,
    detect_evidence_strength,
)
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.stance_generator import generate_stance_safe
from persona_engine.planner.stages.stage_results import (
    BehavioralMetricsResult,
    InterpretationResult,
    KnowledgeSafetyResult,
)
from persona_engine.planner.trace_context import TraceContext, clamp01
from persona_engine.schema.ir_schema import (
    Tone,
    Verbosity,
)

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)

# Config aliases
ELASTICITY_MIN = DEFAULT_CONFIG.elasticity_min
ELASTICITY_MAX = DEFAULT_CONFIG.elasticity_max
EVIDENCE_STRESS_THRESHOLD = DEFAULT_CONFIG.evidence_stress_threshold
UNKNOWN_DOMAIN_BASE = DEFAULT_CONFIG.unknown_domain_base
OPENNESS_COMPETENCE_WEIGHT = DEFAULT_CONFIG.openness_competence_weight
CROSS_TURN_INERTIA = DEFAULT_CONFIG.cross_turn_inertia
PERSONALITY_FIELD_INERTIA = DEFAULT_CONFIG.personality_field_inertia
FAMILIARITY_BOOST_PER_EPISODE = DEFAULT_CONFIG.familiarity_boost_per_episode
FAMILIARITY_BOOST_CAP = DEFAULT_CONFIG.familiarity_boost_cap
FORMALITY_ROLE_WEIGHT = DEFAULT_CONFIG.formality_role_weight
FORMALITY_BASE_WEIGHT = DEFAULT_CONFIG.formality_base_weight
DIRECTNESS_IMPATIENCE_BUMP = DEFAULT_CONFIG.directness_impatience_bump
PATIENCE_THRESHOLD = DEFAULT_CONFIG.patience_threshold
DEFAULT_PROFICIENCY = DEFAULT_CONFIG.default_proficiency

from persona_engine.schema.ir_schema import InteractionMode


def _smooth(prev: float, new: float, alpha: float) -> float:
    """Cross-turn inertia smoothing: blend previous and current values."""
    return prev * alpha + new * (1 - alpha)


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


class BehavioralMetricsStage:
    """Computes all behavioral metrics: elasticity, stance, confidence,
    competence, tone, verbosity, formality, directness, plus trait and
    cognitive guidance."""

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
                    lo, hi = (0.1, 0.95) if field_name == "confidence" else (0.1, 0.9)
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

    # ========================================================================
    # Elasticity
    # ========================================================================

    def compute_elasticity(self, proficiency: float, ctx: TraceContext) -> float:
        """Compute elasticity (openness to persuasion).

        Sequence: base (trait) → cognitive blend → confirmation bias → bounds [0.1, 0.9]
        """
        p = self.planner

        trait_elasticity = p.traits.get_elasticity(proficiency)
        elasticity = ctx.base(
            field_name="elasticity.trait_base",
            target_field="response_structure.elasticity",
            value=trait_elasticity,
            effect=f"Trait-based elasticity ({trait_elasticity:.2f})"
        )

        cognitive_elasticity = p.cognitive.get_elasticity_from_cognitive_style()
        blended = (elasticity + cognitive_elasticity) / 2.0

        elasticity = ctx.num(
            source_type="trait",
            source_id="cognitive_complexity",
            target_field="response_structure.elasticity",
            operation="blend",
            before=elasticity,
            after=blended,
            effect="Blend trait elasticity with cognitive elasticity",
            weight=0.7,
            reason=f"trait={trait_elasticity:.2f}, cognitive={cognitive_elasticity:.2f}"
        )

        # Apply elasticity bias modifiers
        elasticity_bias = p.bias_simulator.get_total_modifier_for_field(
            p._current_bias_modifiers,
            "response_structure.elasticity"
        )
        if abs(elasticity_bias) > 0.001:
            biased = elasticity + elasticity_bias
            bias_names = [
                m.bias_type.value for m in p._current_bias_modifiers
                if m.target_field == "response_structure.elasticity"
            ]
            elasticity = ctx.num(
                source_type="rule",
                source_id="elasticity_biases",
                target_field="response_structure.elasticity",
                operation="add",
                before=elasticity,
                after=biased,
                effect=f"Bias ({', '.join(bias_names)}): {elasticity_bias:+.3f}",
                weight=min(abs(elasticity_bias) / MAX_BIAS_IMPACT, 1.0),
                reason=f"combined_bias={elasticity_bias:+.3f}"
            )

        elasticity = ctx.clamp(
            field_name="elasticity",
            target_field="response_structure.elasticity",
            proposed=elasticity,
            minimum=ELASTICITY_MIN,
            maximum=ELASTICITY_MAX,
            constraint_name="bounds_check",
            reason=f"Elasticity bounds [{ELASTICITY_MIN}, {ELASTICITY_MAX}]"
        )

        return elasticity

    # ========================================================================
    # Confidence
    # ========================================================================

    def compute_confidence(
        self,
        proficiency: float,
        ctx: TraceContext,
        memory_context: dict[str, Any] | None = None,
    ) -> float:
        """Compute response confidence.

        Sequence: base (proficiency) → trait → cognitive → authority bias → memory → bounds
        """
        p = self.planner

        confidence = ctx.base(
            field_name="confidence.proficiency_base",
            target_field="response_structure.confidence",
            value=proficiency,
            effect=f"Base confidence from domain proficiency ({proficiency:.2f})"
        )

        adjusted = p.traits.get_confidence_modifier(confidence)
        confidence = ctx.num(
            source_type="trait",
            source_id="confidence_traits",
            target_field="response_structure.confidence",
            operation="add",
            before=confidence,
            after=adjusted,
            effect="Traits adjust confidence (conscientiousness/neuroticism)",
            weight=0.8,
            reason=f"C={p.persona.psychology.big_five.conscientiousness:.2f}, N={p.persona.psychology.big_five.neuroticism:.2f}"
        )

        adjusted2 = p.cognitive.get_confidence_adjustment(confidence)
        confidence = ctx.num(
            source_type="trait",
            source_id="cognitive_style",
            target_field="response_structure.confidence",
            operation="add",
            before=confidence,
            after=adjusted2,
            effect="Cognitive style adjusts confidence",
            weight=0.6,
            reason="Risk tolerance/need for closure influence"
        )

        # Apply confidence bias modifiers
        confidence_bias = p.bias_simulator.get_total_modifier_for_field(
            p._current_bias_modifiers,
            "response_structure.confidence"
        )
        if abs(confidence_bias) > 0.001:
            biased = confidence + confidence_bias
            bias_names = [
                m.bias_type.value for m in p._current_bias_modifiers
                if m.target_field == "response_structure.confidence"
            ]
            confidence = ctx.num(
                source_type="rule",
                source_id="confidence_biases",
                target_field="response_structure.confidence",
                operation="add",
                before=confidence,
                after=biased,
                effect=f"Bias ({', '.join(bias_names)}): {confidence_bias:+.3f}",
                weight=min(abs(confidence_bias) / MAX_BIAS_IMPACT, 1.0),
                reason=f"combined_bias={confidence_bias:+.3f}"
            )

        # Memory knowledge boost
        if memory_context and memory_context.get("known_facts"):
            fact_count = len(memory_context["known_facts"])
            memory_boost = min(0.15, 0.05 + 0.03 * (fact_count - 1))
            before_mem = confidence
            confidence = confidence + memory_boost
            ctx.num(
                source_type="memory",
                source_id="known_facts_boost",
                target_field="response_structure.confidence",
                operation="add",
                before=before_mem,
                after=confidence,
                effect=f"Memory boost: +{memory_boost:.3f} from {fact_count} known fact(s)",
                weight=0.7,
                reason=f"{fact_count} facts retrieved from memory",
            )
        if memory_context and memory_context.get("previously_discussed"):
            before_disc = confidence
            familiarity_boost = 0.03
            confidence = confidence + familiarity_boost
            ctx.num(
                source_type="memory",
                source_id="topic_familiarity",
                target_field="response_structure.confidence",
                operation="add",
                before=before_disc,
                after=confidence,
                effect=f"Topic familiarity boost: +{familiarity_boost:.3f}",
                weight=0.5,
                reason="Topic previously discussed",
            )

        confidence = clamp01(
            ctx,
            field_name="confidence",
            target_field="response_structure.confidence",
            value=confidence
        )

        return confidence

    # ========================================================================
    # Competence
    # ========================================================================

    def compute_competence(
        self,
        domain: str,
        proficiency: float,
        persona_domains: list[dict],
        ctx: TraceContext,
    ) -> float:
        """Compute how equipped the persona is to engage with this topic.

        Sequence: direct match | adjacency fallback → openness modifier → clamp
        """
        p = self.planner

        is_direct_match = any(
            kd.domain.lower() == domain.lower()
            for kd in p.persona.knowledge_domains
        )

        if is_direct_match:
            competence = ctx.base(
                field_name="competence.domain_match",
                target_field="response_structure.competence",
                value=proficiency,
                effect=f"Direct domain match '{domain}' — competence = proficiency ({proficiency:.2f})",
            )
        else:
            adjacency_score, nearest = compute_domain_adjacency(
                detected_domain=domain,
                persona_domains=persona_domains,
                ctx=ctx,
            )

            if adjacency_score > 0 and nearest:
                competence = ctx.base(
                    field_name="competence.adjacency",
                    target_field="response_structure.competence",
                    value=adjacency_score,
                    effect=(
                        f"No direct match for '{domain}' — adjacent to "
                        f"'{nearest}' (adjacency={adjacency_score:.3f})"
                    ),
                )
            else:
                competence = ctx.base(
                    field_name="competence.unknown",
                    target_field="response_structure.competence",
                    value=UNKNOWN_DOMAIN_BASE,
                    effect=f"No domain match or adjacency for '{domain}' — base floor ({UNKNOWN_DOMAIN_BASE})",
                )

        # Openness modifier
        openness = p.persona.psychology.big_five.openness
        openness_boost = openness * OPENNESS_COMPETENCE_WEIGHT
        before = competence
        competence = competence + openness_boost

        competence = ctx.num(
            source_type="trait",
            source_id="openness",
            target_field="response_structure.competence",
            operation="add",
            before=before,
            after=competence,
            effect=f"Openness ({openness:.2f}) adds novelty comfort ({openness_boost:.3f})",
            weight=0.5,
            reason=f"O={openness:.2f} * weight={OPENNESS_COMPETENCE_WEIGHT}",
        )

        # Familiarity boost
        if p.memory:
            previously_discussed = p.memory.episodes.has_discussed(domain)
            topic_episodes = p.memory.episodes.get_by_topic(domain)
            if previously_discussed and topic_episodes:
                boost = min(
                    len(topic_episodes) * FAMILIARITY_BOOST_PER_EPISODE,
                    FAMILIARITY_BOOST_CAP,
                )
                before_fam = competence
                competence = competence + boost
                ctx.num(
                    source_type="memory",
                    source_id="topic_familiarity",
                    target_field="response_structure.competence",
                    operation="add",
                    before=before_fam,
                    after=competence,
                    effect=f"Familiarity boost: +{boost:.3f} ({len(topic_episodes)} prior episodes on '{domain}')",
                    weight=0.6,
                    reason=f"episodes={len(topic_episodes)}, boost_per={FAMILIARITY_BOOST_PER_EPISODE}, cap={FAMILIARITY_BOOST_CAP}",
                )

        # Known fact boost
        if p.memory:
            relevant_facts = p.memory.facts.search(
                domain, current_turn=0, min_confidence=0.5,
            )
            if relevant_facts:
                fact_boost = min(len(relevant_facts) * 0.03, 0.10)
                before_fact = competence
                competence = competence + fact_boost
                ctx.num(
                    source_type="memory",
                    source_id="known_facts",
                    target_field="response_structure.competence",
                    operation="add",
                    before=before_fact,
                    after=competence,
                    effect=f"Known facts boost: +{fact_boost:.3f} ({len(relevant_facts)} facts about '{domain}')",
                    weight=0.5,
                    reason=f"facts={len(relevant_facts)}, boost_per=0.03, cap=0.10",
                )

        competence = clamp01(
            ctx,
            field_name="competence",
            target_field="response_structure.competence",
            value=competence,
        )

        return competence

    # ========================================================================
    # Tone
    # ========================================================================

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

    # ========================================================================
    # Verbosity
    # ========================================================================

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

    # ========================================================================
    # Communication Style
    # ========================================================================

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

    # ========================================================================
    # Trait Guidance
    # ========================================================================

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

    # ========================================================================
    # Cognitive Guidance
    # ========================================================================

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

    # ========================================================================
    # Stance Generation
    # ========================================================================

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
