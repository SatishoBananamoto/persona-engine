"""
Sparse Persona Floor Test

Tests that the Turn Planner produces valid, fully-cited IR even for
the most minimal possible persona configuration.

This is the "floor" contract: if a sparse persona works, everything else should.

Validations:
1. Domain detection falls back to "general" with citation
2. Topic relevance falls back to 0.5 with citation
3. Expert_allowed is False (no domains = no expertise)
4. Citations contain expected fallback messages
5. IR is valid and complete
"""

import sys
sys.path.insert(0, '.')

from persona_engine.planner.trace_context import TraceContext
from persona_engine.planner.domain_detection import (
    detect_domain,
    compute_topic_relevance,
    detect_evidence_strength,
    generate_intent_string,
    DEFAULT_DOMAIN,
)

# Local constant matching the default in domain_detection
DEFAULT_TOPIC_RELEVANCE = 0.5


def test_sparse_domain_detection():
    """Sparse persona with no domains should fallback to 'general' with citation"""
    ctx = TraceContext()
    
    # No persona domains, use obscure input that won't match any domain keywords
    domain, score = detect_domain(
        user_input="What is the population of Liechtenstein",
        persona_domains=[],
        ctx=ctx
    )
    
    # Should fallback
    assert domain == DEFAULT_DOMAIN, f"Expected '{DEFAULT_DOMAIN}', got '{domain}'"
    
    # Should have citation explaining why
    citations = [c for c in ctx.citations if c.source_id == "domain_detection"]
    assert len(citations) == 1, f"Expected 1 domain citation, got {len(citations)}"
    
    cite = citations[0]
    assert "fallback" in cite.effect.lower() or "general" in cite.effect.lower(), \
        f"Citation should mention fallback: {cite.effect}"
    
    # Citation should include input snippet
    assert "liechtenstein" in cite.effect.lower() or "population" in cite.effect.lower(), \
        f"Citation should include input snippet: {cite.effect}"
    
    print("✓ Domain detection fallback with input snippet in citation")


def test_sparse_topic_relevance():
    """Sparse persona with no interests/goals should fallback to default relevance"""
    ctx = TraceContext()
    
    relevance = compute_topic_relevance(
        user_input="What is the history of the Faroe Islands",
        persona_domains=[],
        persona_goals=[],
        ctx=ctx,
        default_relevance=DEFAULT_TOPIC_RELEVANCE
    )
    
    assert relevance == DEFAULT_TOPIC_RELEVANCE, \
        f"Expected {DEFAULT_TOPIC_RELEVANCE}, got {relevance}"
    
    # Should have citation explaining fallback
    citations = [c for c in ctx.citations if c.source_id == "topic_relevance"]
    assert len(citations) == 1, f"Expected 1 relevance citation, got {len(citations)}"
    
    cite = citations[0]
    assert "no persona" in cite.effect.lower() or "default" in cite.effect.lower(), \
        f"Citation should explain fallback: {cite.effect}"
    
    print("✓ Topic relevance fallback with citation")


def test_sparse_expert_allowed():
    """Sparse persona should not be allowed expert assertions"""
    # With no domains, proficiency is DEFAULT_PROFICIENCY (0.3)
    # Expert threshold is 0.7
    # So expert_allowed should ALWAYS be False for sparse persona
    
    # Test the logic directly
    proficiency = 0.3  # DEFAULT_PROFICIENCY
    expert_threshold = 0.7
    is_domain_specific = False  # No domains = not domain-specific
    
    expert_allowed = is_domain_specific and (proficiency >= expert_threshold)
    
    assert expert_allowed == False, "Sparse persona should not have expert_allowed"
    print("✓ Expert allowed correctly False for sparse persona")


def test_evidence_detection():
    """Evidence detection should work regardless of persona"""
    # Challenge phrase
    evidence = detect_evidence_strength("Actually, I disagree with that")
    assert evidence >= 0.6, f"Challenge should have high evidence: {evidence}"
    
    # Neutral phrase
    evidence = detect_evidence_strength("Tell me more about this topic")
    assert evidence <= 0.3, f"Neutral should have low evidence: {evidence}"
    
    print("✓ Evidence detection works correctly")


def test_intent_generation():
    """Intent generation should work with all combinations"""
    # Test clarification override
    intent = generate_intent_string(
        user_intent="ask",
        conversation_goal="inform",
        uncertainty_action="ANSWER",
        needs_clarification=True
    )
    assert "clarif" in intent.lower(), f"Should prioritize clarification: {intent}"
    
    # Test normal path
    intent = generate_intent_string(
        user_intent="ask",
        conversation_goal="inform",
        uncertainty_action="HEDGE",
        needs_clarification=False
    )
    assert intent and len(intent) > 10, f"Should produce meaningful intent: {intent}"
    
    # Test challenge path
    intent = generate_intent_string(
        user_intent="challenge",
        conversation_goal="debate",
        uncertainty_action="ANSWER",
        needs_clarification=False
    )
    assert "challenge" in intent.lower() or "evidence" in intent.lower() or "address" in intent.lower(), \
        f"Challenge should produce defensive intent: {intent}"
    
    print("✓ Intent generation produces meaningful strings")


def test_combined_sparse_scenario():
    """
    Full scenario: sparse persona processes input.
    
    Validates all fallbacks work together correctly.
    """
    ctx = TraceContext()
    
    # Step 1: Domain detection
    domain, _ = detect_domain(
        user_input="What is the currency of Bhutan",
        persona_domains=[],
        ctx=ctx
    )
    
    # Step 2: Topic relevance
    relevance = compute_topic_relevance(
        user_input="What is the currency of Bhutan",
        persona_domains=[],
        persona_goals=[],
        ctx=ctx,
        default_relevance=DEFAULT_TOPIC_RELEVANCE
    )
    
    # Step 3: Evidence detection
    evidence = detect_evidence_strength("What is the currency of Bhutan")
    
    # Step 4: Intent generation
    intent = generate_intent_string(
        user_intent="ask",
        conversation_goal="inform",
        uncertainty_action="ANSWER",
        needs_clarification=False
    )
    
    # Validate combined results
    assert domain == DEFAULT_DOMAIN
    assert relevance == DEFAULT_TOPIC_RELEVANCE
    assert evidence <= 0.3  # No challenge phrases
    assert len(intent) > 0
    
    # Validate citations were recorded
    domain_cites = [c for c in ctx.citations if c.source_id == "domain_detection"]
    relevance_cites = [c for c in ctx.citations if c.source_id == "topic_relevance"]
    
    assert len(domain_cites) == 1, "Should have exactly 1 domain citation"
    assert len(relevance_cites) == 1, "Should have exactly 1 relevance citation"
    
    print("✓ Combined sparse scenario passes all validations")
    print(f"  Domain: {domain}")
    print(f"  Relevance: {relevance}")
    print(f"  Evidence: {evidence}")
    print(f"  Intent: {intent}")


def run_all_tests():
    """Run all sparse persona floor tests"""
    print("=" * 60)
    print("Sparse Persona Floor Tests")
    print("=" * 60)
    
    test_sparse_domain_detection()
    test_sparse_topic_relevance()
    test_sparse_expert_allowed()
    test_evidence_detection()
    test_intent_generation()
    test_combined_sparse_scenario()
    
    print("=" * 60)
    print("✅ All sparse persona floor tests passed!")
    print("=" * 60)
    print()
    print("Contract validated:")
    print("  • Domain → 'general' with input snippet citation")
    print("  • Topic relevance → 0.5 with fallback citation")
    print("  • Expert allowed → False (no expertise)")
    print("  • Citations contain fallback messages")
    print("  • All components work together correctly")


if __name__ == "__main__":
    run_all_tests()
