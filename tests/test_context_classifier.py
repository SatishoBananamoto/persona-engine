"""
Tests for the context classifier (CC-1) and pipeline routing (CC-2).

Verifies:
1. Keyword classification accuracy
2. Pipeline routing produces correct IR values per context type
3. Context type is stored in the IR
"""

import pytest

from persona_engine import PersonaEngine
from persona_engine.planner.context_classifier import (
    ADVERSARIAL, EMOTIONAL, KNOWLEDGE, OPINION, PERSONAL, SOCIAL,
    classify_context,
)


# =============================================================================
# 1. Classifier Unit Tests
# =============================================================================

class TestClassifyContext:
    """Test keyword-based context classification."""

    def test_knowledge_default(self):
        assert classify_context("Explain quantum entanglement") == KNOWLEDGE

    def test_knowledge_technical(self):
        assert classify_context("How does photosynthesis work?") == KNOWLEDGE

    def test_opinion_think(self):
        assert classify_context("What do you think about remote work?") == OPINION

    def test_opinion_view(self):
        assert classify_context("What's your view on AI regulation?") == OPINION

    def test_opinion_prefer(self):
        assert classify_context("Do you prefer cats or dogs?") == OPINION

    def test_social_party(self):
        assert classify_context("You're at a party where you know nobody") == SOCIAL

    def test_social_introduce(self):
        assert classify_context("How would you introduce yourself to a stranger?") == SOCIAL

    def test_social_networking(self):
        assert classify_context("Imagine you're at a networking event") == SOCIAL

    def test_emotional_feeling(self):
        assert classify_context("How are you feeling today?") == EMOTIONAL

    def test_emotional_worried(self):
        assert classify_context("Are you worried about the deadline?") == EMOTIONAL

    def test_emotional_mood(self):
        assert classify_context("How's your mood right now?") == EMOTIONAL

    def test_personal_routine(self):
        assert classify_context("Tell me about your morning routine") == PERSONAL

    def test_personal_yourself(self):
        assert classify_context("Tell me about yourself") == PERSONAL

    def test_personal_hobbies(self):
        assert classify_context("What do you enjoy doing for fun?") == PERSONAL

    def test_adversarial_wrong(self):
        assert classify_context("You're wrong about that") == ADVERSARIAL

    def test_adversarial_defend(self):
        assert classify_context("Defend your position on this") == ADVERSARIAL

    def test_adversarial_disagree(self):
        assert classify_context("I disagree with everything you said") == ADVERSARIAL

    def test_priority_adversarial_over_social(self):
        """Adversarial takes priority — 'you're wrong about parties' is adversarial."""
        assert classify_context("You're wrong about how parties work") == ADVERSARIAL

    def test_priority_emotional_over_personal(self):
        """Emotional takes priority — 'how are you feeling about your routine' is emotional."""
        assert classify_context("How are you feeling about your routine?") == EMOTIONAL

    def test_ambiguous_defaults_to_knowledge(self):
        """Unmatched input falls through to knowledge."""
        assert classify_context("The mitochondria is the powerhouse of the cell") == KNOWLEDGE


# =============================================================================
# 2. Pipeline Routing Integration Tests
# =============================================================================

class TestContextRouting:
    """Test that the pipeline routes correctly by context type."""

    @pytest.fixture
    def chef_engine(self):
        return PersonaEngine.from_yaml(
            "personas/chef.yaml", llm_provider="mock", seed=42,
        )

    def test_knowledge_context_uses_proficiency(self, chef_engine):
        """Knowledge questions should use domain proficiency for confidence."""
        ir = chef_engine.plan("What makes a perfect French mother sauce?")
        assert ir.context_type == KNOWLEDGE
        # Chef asking about cooking — should have domain-based confidence
        assert ir.response_structure.competence > 0.3  # domain match

    def test_opinion_context_personality_confidence(self, chef_engine):
        """Opinion questions should derive confidence from personality, not proficiency."""
        ir = chef_engine.plan("What do you think about remote work policies?")
        assert ir.context_type == OPINION
        # Opinion context: competence is neutral (0.5)
        assert abs(ir.response_structure.competence - 0.5) < 0.01
        # Claim type should be personal_experience, not speculative
        from persona_engine.schema.ir_schema import KnowledgeClaimType
        assert ir.knowledge_disclosure.knowledge_claim_type == KnowledgeClaimType.PERSONAL_EXPERIENCE

    def test_social_context_neutral_competence(self, chef_engine):
        """Social scenarios should not assess domain competence."""
        ir = chef_engine.plan("You're at a dinner party with strangers. What do you do?")
        assert ir.context_type == SOCIAL
        assert abs(ir.response_structure.competence - 0.5) < 0.01

    def test_emotional_context_answer_action(self, chef_engine):
        """Emotional check-ins should use 'answer' uncertainty, not 'hedge'."""
        ir = chef_engine.plan("How are you feeling about the new menu?")
        assert ir.context_type == EMOTIONAL
        from persona_engine.schema.ir_schema import UncertaintyAction
        assert ir.knowledge_disclosure.uncertainty_action == UncertaintyAction.ANSWER

    def test_adversarial_context_type(self, chef_engine):
        """Adversarial inputs should be classified correctly."""
        ir = chef_engine.plan("You're wrong about butter being essential in sauces")
        assert ir.context_type == ADVERSARIAL

    def test_personal_context_type(self, chef_engine):
        """Personal questions should be classified correctly."""
        ir = chef_engine.plan("Tell me about your childhood and how you got into cooking")
        assert ir.context_type == PERSONAL

    def test_context_type_stored_in_ir(self, chef_engine):
        """context_type should be stored in the IR for downstream use."""
        ir = chef_engine.plan("What do you think about fusion cuisine?")
        assert hasattr(ir, "context_type")
        assert ir.context_type in (KNOWLEDGE, OPINION, SOCIAL, EMOTIONAL, PERSONAL, ADVERSARIAL)

    def test_knowledge_still_works_normally(self, chef_engine):
        """Knowledge questions should still use the full pipeline (no regression)."""
        ir = chef_engine.plan("What temperature should I cook a steak to?")
        assert ir.context_type == KNOWLEDGE
        # Should have domain-based claim type assessment
        from persona_engine.schema.ir_schema import KnowledgeClaimType
        assert ir.knowledge_disclosure.knowledge_claim_type != KnowledgeClaimType.NONE
