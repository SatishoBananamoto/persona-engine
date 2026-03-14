"""
Persona Analysis & Comparison Utilities.

Provides two analyzer classes for inspecting persona behavior:

- **PersonaAnalyzer**: Static persona profiling, pairwise comparison,
  benchmark execution, and trait-influence reporting.
- **ConversationAnalyzer**: Post-hoc analysis of multi-turn conversations
  including drift detection and summary statistics.

Usage::

    from persona_engine import PersonaEngine
    from persona_engine.analysis import PersonaAnalyzer, ConversationAnalyzer

    engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
    results = [engine.chat(p) for p in ["Hello", "Tell me about sauces"]]

    # Persona-level analysis
    summary = PersonaAnalyzer.profile_summary(engine.persona)
    report  = PersonaAnalyzer.trait_influence_report(engine, ["What is a roux?"])

    # Conversation-level analysis
    conv_summary = ConversationAnalyzer.summarize_conversation(results)
    drift        = ConversationAnalyzer.detect_drift(results)
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_engine.engine import ChatResult, PersonaEngine
    from persona_engine.schema.persona_schema import Persona


# ============================================================================
# PersonaAnalyzer
# ============================================================================

class PersonaAnalyzer:
    """Static and dynamic analysis of persona profiles and engine output."""

    # ------------------------------------------------------------------
    # profile_summary
    # ------------------------------------------------------------------

    @staticmethod
    def profile_summary(persona: Persona) -> dict:
        """Return a structured summary of a persona's traits, values, and knowledge.

        Extracts the key psychological dimensions, top values, knowledge
        domains, and communication defaults into a flat, inspectable dict.

        Args:
            persona: A loaded Persona model.

        Returns:
            Dict with keys: persona_id, label, identity, big_five, top_values,
            cognitive_style, communication, knowledge_domains, primary_goals.
        """
        big_five = persona.psychology.big_five
        values = persona.psychology.values

        # Rank Schwartz values by weight (descending)
        value_dict = values.model_dump()
        sorted_values = sorted(value_dict.items(), key=lambda kv: kv[1], reverse=True)

        # Knowledge domains sorted by proficiency
        domains = sorted(
            persona.knowledge_domains,
            key=lambda d: d.proficiency,
            reverse=True,
        )

        return {
            "persona_id": persona.persona_id,
            "label": persona.label,
            "identity": {
                "age": persona.identity.age,
                "occupation": persona.identity.occupation,
                "location": persona.identity.location,
                "education": persona.identity.education,
            },
            "big_five": big_five.model_dump(),
            "top_values": sorted_values[:5],
            "cognitive_style": persona.psychology.cognitive_style.model_dump(),
            "communication": persona.psychology.communication.model_dump(),
            "knowledge_domains": [
                {"domain": d.domain, "proficiency": d.proficiency}
                for d in domains
            ],
            "primary_goals": [
                {"goal": g.goal, "weight": g.weight}
                for g in persona.primary_goals
            ],
        }

    # ------------------------------------------------------------------
    # compare_personas
    # ------------------------------------------------------------------

    @staticmethod
    def compare_personas(p1: Persona, p2: Persona) -> dict:
        """Compare two personas and highlight trait differences.

        Computes per-trait deltas for Big Five, Schwartz values, and
        communication preferences.  Also reports the single most
        divergent trait across all dimensions.

        Args:
            p1: First persona.
            p2: Second persona.

        Returns:
            Dict with keys: personas, big_five_delta, values_delta,
            communication_delta, most_divergent_trait.
        """
        b1 = p1.psychology.big_five.model_dump()
        b2 = p2.psychology.big_five.model_dump()
        big_five_delta = {
            k: round(b2[k] - b1[k], 4) for k in b1
        }

        v1 = p1.psychology.values.model_dump()
        v2 = p2.psychology.values.model_dump()
        values_delta = {
            k: round(v2[k] - v1[k], 4) for k in v1
        }

        c1 = p1.psychology.communication.model_dump()
        c2 = p2.psychology.communication.model_dump()
        communication_delta = {
            k: round(c2[k] - c1[k], 4) for k in c1
        }

        # Find the single most divergent trait across all dimensions
        all_deltas: dict[str, float] = {}
        all_deltas.update({f"big_five.{k}": v for k, v in big_five_delta.items()})
        all_deltas.update({f"values.{k}": v for k, v in values_delta.items()})
        all_deltas.update({f"communication.{k}": v for k, v in communication_delta.items()})

        most_divergent_key = max(all_deltas, key=lambda k: abs(all_deltas[k]))

        return {
            "personas": (p1.label, p2.label),
            "big_five_delta": big_five_delta,
            "values_delta": values_delta,
            "communication_delta": communication_delta,
            "most_divergent_trait": {
                "trait": most_divergent_key,
                "delta": all_deltas[most_divergent_key],
            },
        }

    # ------------------------------------------------------------------
    # run_benchmark
    # ------------------------------------------------------------------

    @staticmethod
    def run_benchmark(engine: PersonaEngine, prompts: list[str]) -> dict:
        """Run a set of prompts through the engine and collect IR statistics.

        Uses ``engine.plan()`` (no LLM call) to generate IRs, then
        aggregates confidence, competence, elasticity, disclosure, and
        tone distributions.

        The engine is ``reset()`` before and after the benchmark so that
        the caller's conversation state is not affected.

        Args:
            engine:  A PersonaEngine instance.
            prompts: List of user-input strings.

        Returns:
            Dict with keys: prompt_count, confidence (mean/min/max),
            competence (mean/min/max), elasticity (mean/min/max),
            disclosure (mean/min/max), tone_distribution, verbosity_distribution.
        """
        engine.reset()

        confidences: list[float] = []
        competences: list[float] = []
        elasticities: list[float] = []
        disclosures: list[float] = []
        tones: list[str] = []
        verbosities: list[str] = []

        for prompt in prompts:
            ir = engine.plan(prompt)
            rs = ir.response_structure
            confidences.append(rs.confidence)
            competences.append(rs.competence)
            if rs.elasticity is not None:
                elasticities.append(rs.elasticity)
            disclosures.append(ir.knowledge_disclosure.disclosure_level)
            tones.append(ir.communication_style.tone.value)
            verbosities.append(ir.communication_style.verbosity.value)

        engine.reset()

        def _stats(values: list[float]) -> dict:
            if not values:
                return {"mean": None, "min": None, "max": None}
            return {
                "mean": round(statistics.mean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        return {
            "prompt_count": len(prompts),
            "confidence": _stats(confidences),
            "competence": _stats(competences),
            "elasticity": _stats(elasticities),
            "disclosure": _stats(disclosures),
            "tone_distribution": dict(Counter(tones)),
            "verbosity_distribution": dict(Counter(verbosities)),
        }

    # ------------------------------------------------------------------
    # trait_influence_report
    # ------------------------------------------------------------------

    @staticmethod
    def trait_influence_report(engine: PersonaEngine, prompts: list[str]) -> dict:
        """Show which persona traits most influenced the IR across prompts.

        Examines the citation chain of each IR to tally how often each
        source (trait, value, rule, state, etc.) appears and the total
        absolute weight contributed.

        The engine is ``reset()`` before and after.

        Args:
            engine:  A PersonaEngine instance.
            prompts: List of user-input strings.

        Returns:
            Dict with keys: prompt_count, by_source_type (count + total_weight
            per source_type), top_sources (source_id ranked by total weight),
            most_influential (single most influential source_id).
        """
        engine.reset()

        source_type_stats: dict[str, dict] = {}  # source_type -> {count, total_weight}
        source_id_weight: Counter[str] = Counter()

        for prompt in prompts:
            ir = engine.plan(prompt)
            for cit in ir.citations:
                st = cit.source_type
                if st not in source_type_stats:
                    source_type_stats[st] = {"count": 0, "total_weight": 0.0}
                source_type_stats[st]["count"] += 1
                source_type_stats[st]["total_weight"] += cit.weight
                source_type_stats[st]["total_weight"] = round(
                    source_type_stats[st]["total_weight"], 4,
                )

                source_id_weight[cit.source_id] += cit.weight

        engine.reset()

        # Sort source_ids by cumulative weight descending
        top_sources = [
            {"source_id": sid, "total_weight": round(w, 4)}
            for sid, w in source_id_weight.most_common()
        ]

        most_influential = top_sources[0]["source_id"] if top_sources else None

        return {
            "prompt_count": len(prompts),
            "by_source_type": source_type_stats,
            "top_sources": top_sources,
            "most_influential": most_influential,
        }


# ============================================================================
# ConversationAnalyzer
# ============================================================================

class ConversationAnalyzer:
    """Post-hoc analysis of multi-turn conversation results."""

    # ------------------------------------------------------------------
    # summarize_conversation
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_conversation(results: list[ChatResult]) -> dict:
        """Summarize a multi-turn conversation.

        Computes average confidence, competence, tone distribution,
        disclosure trend (first-half vs second-half average), and
        validation pass rate.

        Args:
            results: List of ChatResult objects (from ``engine.chat()``
                     or ``Conversation.turns``).

        Returns:
            Dict with keys: turn_count, avg_confidence, avg_competence,
            tone_distribution, disclosure_trend, validation_pass_rate.
        """
        if not results:
            return {
                "turn_count": 0,
                "avg_confidence": None,
                "avg_competence": None,
                "tone_distribution": {},
                "disclosure_trend": {},
                "validation_pass_rate": None,
            }

        confidences = [r.confidence for r in results]
        competences = [r.competence for r in results]
        tones = [r.ir.communication_style.tone.value for r in results]
        disclosures = [r.ir.knowledge_disclosure.disclosure_level for r in results]
        passed = [r.passed for r in results]

        n = len(results)
        mid = max(1, n // 2)
        first_half_disc = disclosures[:mid]
        second_half_disc = disclosures[mid:]

        disclosure_trend = {
            "first_half_avg": round(statistics.mean(first_half_disc), 4),
        }
        if second_half_disc:
            disclosure_trend["second_half_avg"] = round(
                statistics.mean(second_half_disc), 4,
            )
            disclosure_trend["direction"] = (
                "increasing"
                if disclosure_trend["second_half_avg"] > disclosure_trend["first_half_avg"]
                else "decreasing"
                if disclosure_trend["second_half_avg"] < disclosure_trend["first_half_avg"]
                else "stable"
            )
        else:
            disclosure_trend["second_half_avg"] = disclosure_trend["first_half_avg"]
            disclosure_trend["direction"] = "stable"

        return {
            "turn_count": n,
            "avg_confidence": round(statistics.mean(confidences), 4),
            "avg_competence": round(statistics.mean(competences), 4),
            "tone_distribution": dict(Counter(tones)),
            "disclosure_trend": disclosure_trend,
            "validation_pass_rate": round(sum(passed) / n, 4),
        }

    # ------------------------------------------------------------------
    # detect_drift
    # ------------------------------------------------------------------

    @staticmethod
    def detect_drift(results: list[ChatResult]) -> dict:
        """Detect personality drift across conversation turns.

        Measures how confidence, competence, formality, and directness
        change over the conversation.  Reports per-metric trend
        (increasing / decreasing / stable) and the maximum absolute
        shift observed between any two consecutive turns.

        Args:
            results: List of ChatResult objects in chronological order.

        Returns:
            Dict with keys: turn_count, metrics (dict of metric name to
            {values, trend, max_shift}), drifted (bool -- True if any
            metric shifted by more than 0.15 in a single step).
        """
        if len(results) < 2:
            return {
                "turn_count": len(results),
                "metrics": {},
                "drifted": False,
            }

        # Extract per-turn series
        series: dict[str, list[float]] = {
            "confidence": [r.confidence for r in results],
            "competence": [r.competence for r in results],
            "formality": [r.ir.communication_style.formality for r in results],
            "directness": [r.ir.communication_style.directness for r in results],
        }

        DRIFT_THRESHOLD = 0.15
        any_drifted = False
        metrics: dict[str, dict] = {}

        for name, values in series.items():
            deltas = [
                round(values[i + 1] - values[i], 4)
                for i in range(len(values) - 1)
            ]
            max_shift = max(abs(d) for d in deltas)
            net_change = values[-1] - values[0]

            if abs(net_change) < 0.02:
                trend = "stable"
            elif net_change > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            if max_shift > DRIFT_THRESHOLD:
                any_drifted = True

            metrics[name] = {
                "values": [round(v, 4) for v in values],
                "trend": trend,
                "max_shift": round(max_shift, 4),
            }

        return {
            "turn_count": len(results),
            "metrics": metrics,
            "drifted": any_drifted,
        }
