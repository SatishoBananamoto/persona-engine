"""
Comprehensive tests for persona_engine.behavioral.uncertainty_resolver

Targets 95%+ coverage of resolve_uncertainty_action and infer_knowledge_claim_type.
"""

import pytest

from persona_engine.behavioral.uncertainty_resolver import (
    infer_knowledge_claim_type,
    resolve_uncertainty_action,
)
from persona_engine.schema.ir_schema import Citation, UncertaintyAction


# ============================================================================
# Helpers
# ============================================================================

def _fresh_citations() -> list[Citation]:
    """Return an empty mutable citations list for each test."""
    return []


# ============================================================================
# resolve_uncertainty_action — Hard constraint (proficiency < 0.3)
# ============================================================================


class TestHardConstraintRefuse:
    """Proficiency < 0.3 + claim_policy 'refuse' -> REFUSE."""

    def test_refuse_policy_returns_refuse(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.9,       # irrelevant when hard constraint fires
            risk_tolerance=0.9,
            need_for_closure=0.9,
            time_pressure=0.9,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.REFUSE

    def test_refuse_policy_appends_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.2,
            confidence=0.8,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "0.20" in c.effect
        assert "refuse" in c.effect.lower()
        assert c.weight == 1.0

    def test_refuse_at_boundary_proficiency_0_29(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.29,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.REFUSE

    def test_refuse_proficiency_zero(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.0,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.REFUSE


class TestHardConstraintHedge:
    """Proficiency < 0.3 + claim_policy 'hedge' -> HEDGE."""

    def test_hedge_policy_returns_hedge(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.15,
            confidence=0.9,
            risk_tolerance=0.9,
            need_for_closure=0.9,
            time_pressure=0.9,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_hedge_policy_citation_content(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.15,
            confidence=0.9,
            risk_tolerance=0.9,
            need_for_closure=0.9,
            time_pressure=0.9,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "0.15" in c.effect
        assert "hedge" in c.effect.lower()
        assert c.weight == 1.0


class TestHardConstraintAsk:
    """Proficiency < 0.3 + claim_policy 'ask' -> ASK_CLARIFYING."""

    def test_ask_policy_returns_ask_clarifying(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.25,
            confidence=0.9,
            risk_tolerance=0.9,
            need_for_closure=0.9,
            time_pressure=0.9,
            claim_policy_lookup_behavior="ask",
            citations=citations,
        )
        assert result == UncertaintyAction.ASK_CLARIFYING

    def test_ask_policy_citation_content(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="ask",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "ask" in c.effect.lower()
        assert c.weight == 1.0


class TestHardConstraintSpeculateFallthrough:
    """Proficiency < 0.3 + claim_policy 'speculate' does NOT match any
    hard-constraint branch and falls through to time pressure / cognitive style."""

    def test_speculate_with_high_time_pressure_falls_to_answer(self):
        """Falls through hard constraint, caught by time-pressure override."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.5,      # > 0.4
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.8,   # > 0.7
            claim_policy_lookup_behavior="speculate",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER

    def test_speculate_falls_to_cognitive_hedge(self):
        """Falls through hard constraint and time pressure, lands on moderate
        confidence + low risk -> HEDGE."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.5,      # moderate
            risk_tolerance=0.4,  # <= 0.6
            need_for_closure=0.5,
            time_pressure=0.3,   # low
            claim_policy_lookup_behavior="speculate",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_unknown_policy_also_falls_through(self):
        """Any unrecognized policy string falls through the hard constraint."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.8,      # high -> ANSWER via cognitive style
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.3,
            claim_policy_lookup_behavior="something_else",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER


class TestProficiencyBypassesHardConstraint:
    """When proficiency >= 0.3 the hard constraint block is skipped entirely."""

    def test_proficiency_exactly_0_3_bypasses(self):
        """0.3 is NOT < 0.3, so hard constraint does not fire."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.3,
            confidence=0.8,      # high -> ANSWER
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.3,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER
        # No claim_policy citation should be appended
        assert all(c.source_id != "claim_policy" for c in citations)

    def test_high_proficiency_ignores_refuse_policy(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.9,
            confidence=0.8,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.3,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER


# ============================================================================
# resolve_uncertainty_action — Time pressure override
# ============================================================================


class TestTimePressureOverride:
    """time_pressure > 0.7 AND confidence > 0.4 -> ANSWER."""

    def test_high_time_moderate_confidence_answers(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,      # > 0.4
            risk_tolerance=0.1,  # would cause HEDGE in cognitive style
            need_for_closure=0.1,
            time_pressure=0.8,   # > 0.7
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER

    def test_time_pressure_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.8,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "state"
        assert c.source_id == "time_scarcity"
        assert "0.80" in c.effect
        assert c.weight == 0.8

    def test_time_pressure_exactly_0_7_not_triggered(self):
        """0.7 is NOT > 0.7, so time pressure override does NOT fire."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.7,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        # Falls to cognitive style: moderate confidence + low risk -> HEDGE
        assert result == UncertaintyAction.HEDGE

    def test_low_confidence_prevents_time_pressure(self):
        """Even high time pressure won't fire if confidence <= 0.4."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.4,      # NOT > 0.4
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.9,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        # Falls to cognitive style: low confidence branch
        assert result != UncertaintyAction.ANSWER or len(citations) == 0
        # More precisely: confidence 0.4 is NOT > 0.4, so goes to low confidence
        # need_for_closure 0.5 <= 0.6, risk_tolerance 0.5 >= 0.3 -> HEDGE
        assert result == UncertaintyAction.HEDGE

    def test_low_time_high_confidence_no_override(self):
        """Low time pressure shouldn't trigger the override."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.3,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        # moderate confidence + low risk_tolerance -> HEDGE
        assert result == UncertaintyAction.HEDGE
        assert all(c.source_id != "time_scarcity" for c in citations)


# ============================================================================
# resolve_uncertainty_action — Cognitive style: High confidence
# ============================================================================


class TestCognitiveHighConfidence:
    """confidence > 0.7 -> ANSWER (no citation appended)."""

    def test_high_confidence_returns_answer(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.8,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER

    def test_high_confidence_no_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.8,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 0

    def test_confidence_exactly_0_7_not_high(self):
        """0.7 is NOT > 0.7, so this falls to moderate confidence."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.7,
            risk_tolerance=0.1,  # <= 0.6 -> HEDGE
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_confidence_1_0_answers(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=1.0,
            risk_tolerance=0.0,
            need_for_closure=0.0,
            time_pressure=0.0,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER


# ============================================================================
# resolve_uncertainty_action — Cognitive style: Moderate confidence
# ============================================================================


class TestCognitiveModerateConfidence:
    """confidence > 0.4 and <= 0.7 -> depends on risk_tolerance."""

    def test_moderate_confidence_high_risk_returns_answer(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.7,   # > 0.6
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER

    def test_moderate_confidence_high_risk_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.8,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "risk_tolerance"
        assert "0.80" in c.effect
        assert c.weight == 0.7

    def test_moderate_confidence_risk_exactly_0_6_hedges(self):
        """0.6 is NOT > 0.6, so falls to else -> HEDGE."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.6,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_moderate_confidence_low_risk_hedges(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.3,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_moderate_confidence_hedge_no_citation(self):
        """The HEDGE branch in moderate confidence adds no citation."""
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.3,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 0

    def test_confidence_at_0_41_is_moderate(self):
        """Just above the moderate threshold."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.41,
            risk_tolerance=0.9,   # > 0.6 -> ANSWER
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER


# ============================================================================
# resolve_uncertainty_action — Cognitive style: Low confidence
# ============================================================================


class TestCognitiveLowConfidence:
    """confidence <= 0.4 -> depends on need_for_closure and risk_tolerance."""

    def test_low_confidence_high_closure_asks(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,
            need_for_closure=0.7,   # > 0.6
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ASK_CLARIFYING

    def test_low_confidence_high_closure_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.3,
            risk_tolerance=0.5,
            need_for_closure=0.8,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "need_for_closure"
        assert "0.80" in c.effect
        assert c.weight == 0.8

    def test_low_confidence_closure_exactly_0_6_not_ask(self):
        """0.6 is NOT > 0.6, so does not trigger ASK_CLARIFYING."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,  # >= 0.3 -> default HEDGE
            need_for_closure=0.6,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_low_confidence_low_risk_refuses(self):
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.2,   # < 0.3
            need_for_closure=0.3, # <= 0.6
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.REFUSE

    def test_low_confidence_low_risk_citation(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.1,
            risk_tolerance=0.1,
            need_for_closure=0.3,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 1
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "risk_tolerance"
        assert "0.10" in c.effect
        assert "refuse" in c.effect.lower()
        assert c.weight == 0.9

    def test_low_confidence_risk_exactly_0_3_not_refuse(self):
        """0.3 is NOT < 0.3, so falls to default HEDGE."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.3,
            need_for_closure=0.3,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_low_confidence_default_hedge(self):
        """Moderate need_for_closure + moderate risk -> default HEDGE."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,   # >= 0.3
            need_for_closure=0.4, # <= 0.6
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE

    def test_low_confidence_default_hedge_no_citation(self):
        """Default HEDGE in low confidence adds no citation."""
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,
            need_for_closure=0.4,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert len(citations) == 0

    def test_confidence_exactly_0_4_is_low(self):
        """0.4 is NOT > 0.4, so it lands in the low confidence branch."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.4,
            risk_tolerance=0.5,
            need_for_closure=0.9,   # > 0.6 -> ASK_CLARIFYING
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ASK_CLARIFYING

    def test_confidence_zero(self):
        """Edge: confidence=0 is low; moderate risk + low closure -> HEDGE."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.0,
            risk_tolerance=0.5,
            need_for_closure=0.3,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.HEDGE


# ============================================================================
# resolve_uncertainty_action — Precedence & interaction checks
# ============================================================================


class TestPrecedence:
    """Verify correct precedence: hard constraint > time pressure > cognitive."""

    def test_hard_constraint_beats_time_pressure(self):
        """Even with high time pressure, if proficiency < 0.3 and policy
        is 'refuse', hard constraint wins."""
        citations: list[Citation] = _fresh_citations()
        result = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.9,
            risk_tolerance=0.9,
            need_for_closure=0.9,
            time_pressure=0.9,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert result == UncertaintyAction.REFUSE
        assert citations[0].source_id == "claim_policy"

    def test_time_pressure_beats_cognitive_style(self):
        """High time pressure overrides what cognitive style would choose."""
        citations: list[Citation] = _fresh_citations()
        # Without time pressure: moderate confidence + low risk -> HEDGE
        result = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.1,
            need_for_closure=0.1,
            time_pressure=0.9,   # override
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        assert result == UncertaintyAction.ANSWER
        assert citations[0].source_id == "time_scarcity"

    def test_citations_accumulate_only_from_matching_branch(self):
        """Only the branch that fires should add citations."""
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.8,   # high confidence -> ANSWER (no citation)
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert len(citations) == 0


# ============================================================================
# resolve_uncertainty_action — Citation format verification
# ============================================================================


class TestCitationFormat:
    """Verify citation fields are correctly populated for every branch that
    produces a citation."""

    def test_claim_policy_refuse_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.05,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "Proficiency 0.05 < 0.3" in c.effect
        assert "refuse per claim policy" in c.effect
        assert c.weight == 1.0

    def test_claim_policy_hedge_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.22,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "0.22" in c.effect
        assert "hedge per claim policy" in c.effect
        assert c.weight == 1.0

    def test_claim_policy_ask_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.10,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="ask",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "rule"
        assert c.source_id == "claim_policy"
        assert "ask per claim policy" in c.effect.lower()
        assert c.weight == 1.0

    def test_time_pressure_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.85,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "state"
        assert c.source_id == "time_scarcity"
        assert "0.85" in c.effect
        assert "answer quickly" in c.effect.lower()
        assert c.weight == 0.8

    def test_risk_tolerance_answer_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.75,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "risk_tolerance"
        assert "0.75" in c.effect
        assert "answer" in c.effect.lower()
        assert c.weight == 0.7

    def test_need_for_closure_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,
            need_for_closure=0.9,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "need_for_closure"
        assert "0.90" in c.effect
        assert "ask for clarity" in c.effect.lower()
        assert c.weight == 0.8

    def test_low_risk_refuse_citation_format(self):
        citations: list[Citation] = _fresh_citations()
        resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.15,
            need_for_closure=0.3,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        c = citations[0]
        assert c.source_type == "trait"
        assert c.source_id == "risk_tolerance"
        assert "0.15" in c.effect
        assert "refuse" in c.effect.lower()
        assert c.weight == 0.9


# ============================================================================
# resolve_uncertainty_action — Citations list is mutated in-place
# ============================================================================


class TestCitationsMutation:
    """Verify citations list is mutated, not replaced."""

    def test_existing_citations_preserved(self):
        pre_existing = Citation(
            source_type="trait",
            source_id="openness",
            effect="Pre-existing citation",
            weight=0.5,
        )
        citations: list[Citation] = [pre_existing]
        resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        assert len(citations) == 2
        assert citations[0] is pre_existing
        assert citations[1].source_id == "claim_policy"


# ============================================================================
# infer_knowledge_claim_type
# ============================================================================


class TestInferKnowledgeClaimRefuse:
    """uncertainty_action == REFUSE -> 'none', regardless of other args."""

    def test_refuse_returns_none(self):
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.REFUSE,
        )
        assert result == "none"

    def test_refuse_beats_personal_experience(self):
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.REFUSE,
            is_personal_experience=True,
        )
        assert result == "none"

    def test_refuse_beats_domain_specific(self):
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.REFUSE,
            is_domain_specific=True,
        )
        assert result == "none"

    def test_refuse_beats_everything_combined(self):
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.REFUSE,
            is_personal_experience=True,
            is_domain_specific=True,
        )
        assert result == "none"


class TestInferKnowledgeClaimPersonalExperience:
    """is_personal_experience=True -> 'personal_experience' (unless REFUSE)."""

    def test_personal_experience_with_answer(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_personal_experience=True,
        )
        assert result == "personal_experience"

    def test_personal_experience_with_hedge(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
            is_personal_experience=True,
        )
        assert result == "personal_experience"

    def test_personal_experience_beats_domain_specific(self):
        """personal_experience is checked before domain_specific."""
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_personal_experience=True,
            is_domain_specific=True,
        )
        assert result == "personal_experience"


class TestInferKnowledgeClaimDomainExpert:
    """is_domain_specific=True AND proficiency > 0.7 -> 'domain_expert'."""

    def test_domain_expert(self):
        result = infer_knowledge_claim_type(
            proficiency=0.8,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_domain_specific=True,
        )
        assert result == "domain_expert"

    def test_domain_specific_low_proficiency_not_expert(self):
        """Domain specific but proficiency <= 0.7 -> falls through."""
        result = infer_knowledge_claim_type(
            proficiency=0.7,   # NOT > 0.7
            uncertainty_action=UncertaintyAction.ANSWER,
            is_domain_specific=True,
        )
        # Falls through to default
        assert result == "general_common_knowledge"

    def test_domain_specific_proficiency_0_5(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_domain_specific=True,
        )
        assert result == "general_common_knowledge"

    def test_domain_expert_requires_domain_flag(self):
        """High proficiency alone does not produce domain_expert."""
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_domain_specific=False,
        )
        assert result == "general_common_knowledge"


class TestInferKnowledgeClaimSpeculative:
    """uncertainty_action in [HEDGE, ASK_CLARIFYING] -> 'speculative'."""

    def test_hedge_returns_speculative(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
        )
        assert result == "speculative"

    def test_ask_clarifying_returns_speculative(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.ASK_CLARIFYING,
        )
        assert result == "speculative"

    def test_hedge_with_domain_specific_low_proficiency(self):
        """Domain-specific but low proficiency + HEDGE -> speculative (not domain_expert)."""
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
            is_domain_specific=True,
        )
        assert result == "speculative"

    def test_ask_clarifying_with_high_proficiency_domain(self):
        """High proficiency + domain + ASK_CLARIFYING: domain_expert check
        happens first (proficiency > 0.7 and domain), so it should return
        domain_expert, not speculative."""
        result = infer_knowledge_claim_type(
            proficiency=0.8,
            uncertainty_action=UncertaintyAction.ASK_CLARIFYING,
            is_domain_specific=True,
        )
        assert result == "domain_expert"


class TestInferKnowledgeClaimDefault:
    """Default path -> 'general_common_knowledge'."""

    def test_answer_no_flags(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.ANSWER,
        )
        assert result == "general_common_knowledge"

    def test_answer_all_flags_false(self):
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_personal_experience=False,
            is_domain_specific=False,
        )
        assert result == "general_common_knowledge"

    def test_answer_domain_specific_but_low_proficiency(self):
        """Domain-specific + ANSWER + low proficiency -> general_common_knowledge."""
        result = infer_knowledge_claim_type(
            proficiency=0.3,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_domain_specific=True,
        )
        assert result == "general_common_knowledge"


# ============================================================================
# infer_knowledge_claim_type — Priority of checks
# ============================================================================


class TestInferPriority:
    """Verify the ordering of checks inside infer_knowledge_claim_type."""

    def test_refuse_highest_priority(self):
        """REFUSE trumps everything."""
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.REFUSE,
            is_personal_experience=True,
            is_domain_specific=True,
        )
        assert result == "none"

    def test_personal_experience_over_domain_expert(self):
        result = infer_knowledge_claim_type(
            proficiency=0.9,
            uncertainty_action=UncertaintyAction.ANSWER,
            is_personal_experience=True,
            is_domain_specific=True,
        )
        assert result == "personal_experience"

    def test_domain_expert_over_speculative(self):
        """If domain-specific + high proficiency, domain_expert wins even
        with HEDGE action."""
        result = infer_knowledge_claim_type(
            proficiency=0.8,
            uncertainty_action=UncertaintyAction.HEDGE,
            is_domain_specific=True,
        )
        assert result == "domain_expert"

    def test_speculative_over_default(self):
        """HEDGE without domain-expert conditions -> speculative, not default."""
        result = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
            is_domain_specific=False,
        )
        assert result == "speculative"


# ============================================================================
# End-to-end integration: resolve then infer
# ============================================================================


class TestEndToEnd:
    """Combine both functions as they would be used together."""

    def test_low_proficiency_refuse_then_infer_none(self):
        citations: list[Citation] = _fresh_citations()
        action = resolve_uncertainty_action(
            proficiency=0.1,
            confidence=0.5,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.5,
            claim_policy_lookup_behavior="refuse",
            citations=citations,
        )
        claim = infer_knowledge_claim_type(
            proficiency=0.1,
            uncertainty_action=action,
        )
        assert action == UncertaintyAction.REFUSE
        assert claim == "none"

    def test_high_confidence_answer_then_general_knowledge(self):
        citations: list[Citation] = _fresh_citations()
        action = resolve_uncertainty_action(
            proficiency=0.8,
            confidence=0.9,
            risk_tolerance=0.5,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        claim = infer_knowledge_claim_type(
            proficiency=0.8,
            uncertainty_action=action,
            is_domain_specific=True,
        )
        assert action == UncertaintyAction.ANSWER
        assert claim == "domain_expert"

    def test_moderate_confidence_hedge_then_speculative(self):
        citations: list[Citation] = _fresh_citations()
        action = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.5,
            risk_tolerance=0.3,
            need_for_closure=0.5,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        claim = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=action,
        )
        assert action == UncertaintyAction.HEDGE
        assert claim == "speculative"

    def test_low_confidence_ask_then_speculative(self):
        citations: list[Citation] = _fresh_citations()
        action = resolve_uncertainty_action(
            proficiency=0.5,
            confidence=0.2,
            risk_tolerance=0.5,
            need_for_closure=0.8,
            time_pressure=0.1,
            claim_policy_lookup_behavior="hedge",
            citations=citations,
        )
        claim = infer_knowledge_claim_type(
            proficiency=0.5,
            uncertainty_action=action,
            is_personal_experience=True,
        )
        assert action == UncertaintyAction.ASK_CLARIFYING
        assert claim == "personal_experience"
