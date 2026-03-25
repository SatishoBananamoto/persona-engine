"""
Behavioral Metrics — elasticity, confidence, competence computation.

Extracted from behavioral.py to keep files under ~400 lines.
Used as a mixin by BehavioralMetricsStage.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from persona_engine.behavioral import MAX_BIAS_IMPACT
from persona_engine.planner.context_classifier import (
    ADVERSARIAL, EMOTIONAL, KNOWLEDGE, OPINION, PERSONAL, SOCIAL,
)
from persona_engine.planner.domain_detection import compute_domain_adjacency
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.trace_context import TraceContext, clamp01

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import TurnPlanner

# Config aliases
ELASTICITY_MIN = DEFAULT_CONFIG.elasticity_min
ELASTICITY_MAX = DEFAULT_CONFIG.elasticity_max
UNKNOWN_DOMAIN_BASE = DEFAULT_CONFIG.unknown_domain_base
OPENNESS_COMPETENCE_WEIGHT = DEFAULT_CONFIG.openness_competence_weight
FAMILIARITY_BOOST_PER_EPISODE = DEFAULT_CONFIG.familiarity_boost_per_episode
FAMILIARITY_BOOST_CAP = DEFAULT_CONFIG.familiarity_boost_cap


class MetricsMixin:
    """Mixin providing elasticity, confidence, and competence computation."""

    planner: TurnPlanner  # provided by BehavioralMetricsStage

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

    def compute_confidence(
        self,
        proficiency: float,
        ctx: TraceContext,
        memory_context: dict[str, Any] | None = None,
        context_type: str = KNOWLEDGE,
    ) -> float:
        """Compute response confidence.

        CC-2: For knowledge contexts, confidence is proficiency-based (current behavior).
        For non-knowledge contexts, confidence is personality-based — extroverts are
        confident in opinions and social settings, low-N people are emotionally steady.

        Sequence: base (source depends on context) → trait → cognitive → authority bias → memory → bounds
        """
        p = self.planner
        bf = p.persona.psychology.big_five

        # CC-2: Base confidence depends on context type
        if context_type == KNOWLEDGE:
            base = proficiency
            base_label = f"Base confidence from domain proficiency ({proficiency:.2f})"
        elif context_type == OPINION:
            base = 0.35 + bf.extraversion * 0.15 + bf.conscientiousness * 0.08 - bf.neuroticism * 0.08
            base_label = f"Opinion confidence from personality (E={bf.extraversion:.2f}, C={bf.conscientiousness:.2f}, N={bf.neuroticism:.2f})"
        elif context_type == SOCIAL:
            base = 0.30 + bf.extraversion * 0.20 - bf.neuroticism * 0.10
            base_label = f"Social confidence from E/N (E={bf.extraversion:.2f}, N={bf.neuroticism:.2f})"
        elif context_type == EMOTIONAL:
            base = 0.40 - bf.neuroticism * 0.15 + bf.extraversion * 0.05
            base_label = f"Emotional confidence from N/E (N={bf.neuroticism:.2f}, E={bf.extraversion:.2f})"
        else:
            # personal, adversarial — neutral baseline
            base = 0.45
            base_label = f"Neutral confidence baseline for '{context_type}' context"

        confidence = ctx.base(
            field_name="confidence.context_base",
            target_field="response_structure.confidence",
            value=base,
            effect=base_label,
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

    def compute_competence(
        self,
        domain: str,
        proficiency: float,
        persona_domains: list[dict],
        ctx: TraceContext,
        context_type: str = KNOWLEDGE,
    ) -> float:
        """Compute how equipped the persona is to engage with this topic.

        CC-2: For non-knowledge contexts, competence is neutral (0.5). Competence
        is a domain expertise concept — it doesn't apply to "how would you handle
        a party?" or "what's your opinion on remote work?"

        Sequence: direct match | adjacency fallback → openness modifier → clamp
        """
        p = self.planner

        # CC-2: Non-knowledge contexts get neutral competence
        if context_type != KNOWLEDGE:
            competence = ctx.base(
                field_name="competence.context_neutral",
                target_field="response_structure.competence",
                value=0.5,
                effect=f"Neutral competence for '{context_type}' context (not a domain question)",
            )
            return clamp01(
                ctx,
                field_name="competence",
                target_field="response_structure.competence",
                value=competence,
            )

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
