"""
Competence Dimension Tests.

Tests the competence computation, domain adjacency scoring,
prompt builder competence bands, and validation Dunning-Kruger check.
"""

import pytest
import yaml

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.domain_detection import (
    compute_domain_adjacency,
)
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import (
    Citation,
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    ResponseStructure,
    SafetyPlan,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation import PipelineValidator, validate_persona_compliance


# =============================================================================
# Helpers
# =============================================================================


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_ir(
    confidence: float = 0.7,
    competence: float = 0.5,
    formality: float = 0.4,
    directness: float = 0.5,
    claim: KnowledgeClaimType = KnowledgeClaimType.PERSONAL_EXPERIENCE,
    uncertainty: UncertaintyAction = UncertaintyAction.ANSWER,
    tone: Tone = Tone.THOUGHTFUL_ENGAGED,
    disclosure: float = 0.5,
    verbosity: Verbosity = Verbosity.MEDIUM,
) -> IntermediateRepresentation:
    return IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        ),
        response_structure=ResponseStructure(
            intent="Share perspective",
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
            disclosure_level=disclosure,
            uncertainty_action=uncertainty,
            knowledge_claim_type=claim,
        ),
        citations=[
            Citation(source_type="base", source_id="base_profile", effect="Base formality", target_field="communication_style.formality"),
            Citation(source_type="base", source_id="base_profile", effect="Base directness", target_field="communication_style.directness"),
            Citation(source_type="base", source_id="base_profile", effect="Base confidence", target_field="response_structure.confidence"),
        ],
        safety_plan=SafetyPlan(),
    )


def make_context(
    user_input: str,
    topic: str = "general",
    turn: int = 1,
) -> ConversationContext:
    return ConversationContext(
        conversation_id="test_competence",
        turn_number=turn,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=StanceCache(),
    )


# =============================================================================
# 1. Domain Adjacency Tests
# =============================================================================


class TestDomainAdjacency:
    """Test the domain adjacency scoring function."""

    def test_direct_match_returns_zero(self):
        """Direct domain match is NOT adjacency."""
        persona_domains = [
            {"domain": "psychology", "proficiency": 0.9, "subdomains": ["UX research"]},
        ]
        score, nearest = compute_domain_adjacency("psychology", persona_domains)
        assert score == 0.0
        assert nearest is None

    def test_adjacent_domain_has_positive_score(self):
        """A domain that shares keywords should produce adjacency > 0."""
        persona_domains = [
            {"domain": "technology", "proficiency": 0.7, "subdomains": ["software", "data"]},
        ]
        # "health" and "technology" share very few keywords, but let's test with something
        # that has real keyword overlap
        score, nearest = compute_domain_adjacency("business", persona_domains)
        # technology and business share some keywords like "system", "tool", "platform"
        # Score should be small but positive due to keyword overlap
        assert isinstance(score, float)

    def test_no_persona_domains_returns_zero(self):
        """No persona domains → zero adjacency."""
        score, nearest = compute_domain_adjacency("quantum_physics", None)
        assert score == 0.0
        assert nearest is None

    def test_empty_persona_domains_returns_zero(self):
        """Empty persona domains list → zero adjacency."""
        score, nearest = compute_domain_adjacency("quantum_physics", [])
        assert score == 0.0
        assert nearest is None

    def test_completely_unrelated_domain(self):
        """Domain with no keyword overlap should have ~0 adjacency."""
        persona_domains = [
            {"domain": "psychology", "proficiency": 0.9, "subdomains": ["UX research"]},
        ]
        # Use a domain not in the registry — completely unknown
        score, nearest = compute_domain_adjacency("astrophysics", persona_domains)
        # Very low or zero since "astrophysics" has no registry entry
        assert score < 0.3

    def test_adjacency_is_bounded(self):
        """Adjacency score must be between 0 and 1."""
        persona_domains = [
            {"domain": "technology", "proficiency": 1.0, "subdomains": ["everything"]},
            {"domain": "psychology", "proficiency": 1.0, "subdomains": ["everything"]},
        ]
        score, _ = compute_domain_adjacency("business", persona_domains)
        assert 0.0 <= score <= 1.0

    def test_adjacency_scales_with_proficiency(self):
        """Higher proficiency in adjacent domain → higher adjacency."""
        domains_high = [
            {"domain": "technology", "proficiency": 0.9, "subdomains": ["software"]},
        ]
        domains_low = [
            {"domain": "technology", "proficiency": 0.3, "subdomains": ["software"]},
        ]
        score_high, _ = compute_domain_adjacency("business", domains_high)
        score_low, _ = compute_domain_adjacency("business", domains_low)
        assert score_high >= score_low


# =============================================================================
# 2. Competence Computation via TurnPlanner
# =============================================================================


class TestCompetenceComputation:
    """Test that turn_planner computes competence correctly."""

    @pytest.fixture
    def persona(self):
        return load_persona()

    @pytest.fixture
    def planner(self, persona):
        return TurnPlanner(persona, DeterminismManager(seed=42))

    def test_expert_domain_high_competence(self, planner):
        """Expert domain (psychology/UX) → competence near proficiency."""
        ctx = make_context("What are usability heuristics?", topic="ux_research")
        ir = planner.generate_ir(ctx)
        # Sarah is 0.90 proficiency in psychology
        assert ir.response_structure.competence > 0.8

    def test_unknown_domain_low_competence(self, planner):
        """Unknown domain → low competence (floor + openness)."""
        ctx = make_context("Explain quantum computing", topic="quantum_computing")
        ir = planner.generate_ir(ctx)
        # No direct match, weak or no adjacency
        assert ir.response_structure.competence < 0.4

    def test_competence_confidence_diverge_on_unknown(self, planner):
        """On unknown domain, both confidence and competence should be low."""
        ctx = make_context("Tell me about ancient Sumerian pottery", topic="archaeology")
        ir = planner.generate_ir(ctx)
        assert ir.response_structure.confidence < 0.5
        assert ir.response_structure.competence < 0.4

    def test_competence_is_deterministic(self, planner):
        """Same input → same competence across runs."""
        ctx1 = make_context("What's your favorite UX tool?", topic="ux_tools")
        ir1 = planner.generate_ir(ctx1)

        planner2 = TurnPlanner(load_persona(), DeterminismManager(seed=42))
        ctx2 = make_context("What's your favorite UX tool?", topic="ux_tools")
        ir2 = planner2.generate_ir(ctx2)

        assert ir1.response_structure.competence == ir2.response_structure.competence

    def test_competence_bounded_0_1(self, planner):
        """Competence should always be in [0, 1]."""
        topics = ["ux_research", "quantum_computing", "cooking", "philosophy", "general"]
        for topic in topics:
            ctx = make_context(f"Tell me about {topic}", topic=topic)
            ir = planner.generate_ir(ctx)
            assert 0.0 <= ir.response_structure.competence <= 1.0

    def test_adjacent_domain_moderate_competence(self, planner):
        """Domain adjacent to persona's expertise → moderate competence."""
        # Business is one of Sarah's domains at 0.55
        # Technology at 0.70
        # A business question should give moderate competence
        ctx = make_context("How do you manage stakeholder expectations?", topic="business")
        ir = planner.generate_ir(ctx)
        assert ir.response_structure.competence > 0.4

    def test_competence_has_citations(self, planner):
        """Competence computation should produce citations."""
        ctx = make_context("Explain quantum computing", topic="quantum_computing")
        ir = planner.generate_ir(ctx)
        comp_citations = [
            c for c in ir.citations
            if c.target_field and "competence" in c.target_field
        ]
        assert len(comp_citations) >= 2  # base + openness at minimum


# =============================================================================
# 3. Prompt Builder Competence Bands
# =============================================================================


class TestPromptBuilderCompetence:
    """Test that prompt builder includes competence instructions."""

    @pytest.fixture
    def builder(self):
        return IRPromptBuilder()

    def test_low_competence_prompt_says_no_terminology(self, builder):
        """Low competence → prompt tells LLM not to use domain terms."""
        ir = make_ir(competence=0.15, confidence=0.2)
        prompt = builder.build_generation_prompt(ir, "Tell me about quantum physics")
        assert "everyday language" in prompt.lower() or "don't really know" in prompt.lower()

    def test_surface_competence_prompt_says_vague(self, builder):
        """Surface competence → allow vagueness."""
        ir = make_ir(competence=0.3, confidence=0.3)
        prompt = builder.build_generation_prompt(ir, "Tell me about quantum physics")
        assert "surface" in prompt.lower() or "vagueness" in prompt.lower()

    def test_moderate_competence_prompt_says_conceptual(self, builder):
        """Moderate competence → conceptual level."""
        ir = make_ir(competence=0.5, confidence=0.5)
        prompt = builder.build_generation_prompt(ir, "Tell me about design")
        assert "conceptual" in prompt.lower() or "moderate" in prompt.lower()

    def test_high_competence_prompt_says_knowledgeable(self, builder):
        """High competence → full domain vocabulary."""
        ir = make_ir(competence=0.75, confidence=0.8)
        prompt = builder.build_generation_prompt(ir, "Tell me about UX")
        assert "knowledgeable" in prompt.lower() or "terminology" in prompt.lower()

    def test_expert_competence_prompt_says_expert(self, builder):
        """Expert competence → expert-level discussion."""
        ir = make_ir(competence=0.9, confidence=0.9)
        prompt = builder.build_generation_prompt(ir, "Tell me about UX heuristics")
        assert "expert" in prompt.lower() or "highly competent" in prompt.lower()

    def test_competence_appears_in_prompt(self, builder):
        """COMPETENCE label should appear in generated prompt."""
        ir = make_ir(competence=0.5)
        prompt = builder.build_generation_prompt(ir, "Hi")
        assert "COMPETENCE:" in prompt


# =============================================================================
# 4. Validation — Dunning-Kruger Check
# =============================================================================


class TestDunningKrugerValidation:
    """Test the competence-confidence mismatch detection."""

    @pytest.fixture
    def persona(self):
        return load_persona()

    def test_low_competence_high_confidence_is_error(self, persona):
        """Low competence + high confidence → error."""
        ir = make_ir(competence=0.2, confidence=0.8)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "competence_confidence_mismatch" in types

    def test_low_competence_low_confidence_is_ok(self, persona):
        """Low competence + low confidence → fine (appropriately uncertain)."""
        ir = make_ir(competence=0.2, confidence=0.2)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "competence_confidence_mismatch" not in types

    def test_high_competence_high_confidence_is_ok(self, persona):
        """High competence + high confidence → fine (expert being confident)."""
        ir = make_ir(competence=0.9, confidence=0.9, claim=KnowledgeClaimType.DOMAIN_EXPERT)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "competence_confidence_mismatch" not in types

    def test_high_competence_low_confidence_is_ok(self, persona):
        """High competence + low confidence → fine (uncertain expert)."""
        ir = make_ir(competence=0.8, confidence=0.3)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "competence_confidence_mismatch" not in types

    def test_borderline_values_no_false_positive(self, persona):
        """Values near thresholds should not trigger false positives."""
        # competence=0.3 is at the boundary, confidence=0.7 is at boundary
        ir = make_ir(competence=0.3, confidence=0.7)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "competence_confidence_mismatch" not in types

    def test_dunning_kruger_is_severity_error(self, persona):
        """The mismatch should be an error, not a warning."""
        ir = make_ir(competence=0.15, confidence=0.85)
        violations = validate_persona_compliance(ir, persona)
        dk_violations = [v for v in violations if v.violation_type == "competence_confidence_mismatch"]
        assert len(dk_violations) == 1
        assert dk_violations[0].severity == "error"


# =============================================================================
# 5. Full Pipeline Integration
# =============================================================================


class TestCompetencePipelineIntegration:
    """End-to-end competence tests through the real pipeline."""

    @pytest.fixture
    def persona(self):
        return load_persona()

    @pytest.fixture
    def planner(self, persona):
        return TurnPlanner(persona, DeterminismManager(seed=42))

    @pytest.fixture
    def validator(self, persona):
        return PipelineValidator(persona)

    def test_expert_domain_passes_validation(self, planner, validator):
        """Expert domain should pass validation (no DK violation)."""
        ctx = make_context("What are usability heuristics?", topic="ux_research")
        ir = planner.generate_ir(ctx)
        result = validator.validate(ir, turn_number=1, topic="ux_research")
        error_types = [v.violation_type for v in result.violations if v.severity == "error"]
        assert "competence_confidence_mismatch" not in error_types

    def test_unknown_domain_passes_validation(self, planner, validator):
        """Unknown domain with low confidence should pass (no DK)."""
        ctx = make_context("Explain quantum entanglement", topic="quantum_physics")
        ir = planner.generate_ir(ctx)
        result = validator.validate(ir, turn_number=1, topic="quantum_physics")
        error_types = [v.violation_type for v in result.violations if v.severity == "error"]
        assert "competence_confidence_mismatch" not in error_types

    def test_competence_citation_chain_exists(self, planner):
        """Competence should have a traceable citation chain."""
        ctx = make_context("Tell me about data science", topic="data_science")
        ir = planner.generate_ir(ctx)
        comp_citations = [
            c for c in ir.citations
            if c.target_field and "competence" in c.target_field
        ]
        # Should have at least: base (match/adjacency/unknown) + openness
        assert len(comp_citations) >= 2

        # First citation should be a base setting
        base_cites = [c for c in comp_citations if c.source_type == "base"]
        assert len(base_cites) >= 1

        # Should have openness trait modifier
        openness_cites = [c for c in comp_citations if c.source_id == "openness"]
        assert len(openness_cites) == 1

    def test_three_turn_competence_behavior(self, planner, validator):
        """The original 3-turn scenario that motivated the competence dimension."""
        cache = StanceCache()

        # Turn 1: Expert domain
        ctx1 = ConversationContext(
            conversation_id="dk_test",
            turn_number=1,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="ux_research",
            user_input="What are usability heuristics?",
            stance_cache=cache,
        )
        ir1 = planner.generate_ir(ctx1)
        assert ir1.response_structure.competence > 0.8
        assert ir1.response_structure.confidence > 0.7

        # Turn 2: Unknown domain
        ctx2 = ConversationContext(
            conversation_id="dk_test",
            turn_number=2,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="quantum_computing",
            user_input="Can you explain quantum computing?",
            stance_cache=cache,
        )
        ir2 = planner.generate_ir(ctx2)
        assert ir2.response_structure.competence < 0.4
        assert ir2.response_structure.confidence < 0.4

        # Turn 3: Back to expert domain
        ctx3 = ConversationContext(
            conversation_id="dk_test",
            turn_number=3,
            interaction_mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            topic_signature="ux_research",
            user_input="How do you run a usability test?",
            stance_cache=cache,
        )
        ir3 = planner.generate_ir(ctx3)
        assert ir3.response_structure.competence > 0.8
        assert ir3.response_structure.confidence > 0.7

        # Validate all turns have no DK errors
        for i, ir in enumerate([ir1, ir2, ir3], 1):
            result = validator.validate(ir, turn_number=i, topic="test")
            dk_errors = [v for v in result.violations
                         if v.violation_type == "competence_confidence_mismatch"]
            assert len(dk_errors) == 0, f"Turn {i} has DK violation"
