"""
Tests for Domain Detection Refinements (Phase 3 Fixes)

Verifies:
1. Stopwords/Short token whitelist behavior
2. Dictionary/Object hybrid access support
3. Token coverage logic (no double coverage in topic relevance)
4. Citation accuracy
"""

import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, '.')

from persona_engine.planner.trace_context import TraceContext
from persona_engine.planner.domain_detection import (
    detect_domain,
    compute_topic_relevance,
    generate_intent_string,
    TECHNICAL_TERMS_WHITELIST
)

# =============================================================================
# MOCKS
# =============================================================================

@dataclass
class MockDomain:
    domain: str
    proficiency: float
    subdomains: List[str]

@dataclass
class MockGoal:
    goal: str
    weight: float

# =============================================================================
# TESTS
# =============================================================================

def test_short_token_whitelist():
    """Verify that short tokens are ignored unless whitelisted."""
    print("\n--- Testing Short Token Whitelist ---")
    ctx = TraceContext()
    
    # "is", "a" are stopwords < 3 chars. "api" is whitelisted.
    # Note: `tokenize_with_ngrams` returns ALL tokens. Filtering happens in logic.
    
    # 1. Non-whitelisted short token "to" should contribute 0 relevance
    goal_obj = MockGoal(goal="go to", weight=1.0)
    # Both "go" and "to" < 3 chars and not whitelisted -> empty keywords set.
    
    relevance = compute_topic_relevance(
        user_input="go to",
        persona_goals=[goal_obj],
        ctx=ctx
    )
    
    # If interest_keywords empty -> fallback default (0.5).
    assert relevance == 0.5, f"Expected default relevance for stopwords goal, got {relevance}"
    
    # 2. Whitelisted token "ux"
    domain_obj = MockDomain(domain="Design", proficiency=1.0, subdomains=["UI/UX"])
    
    # Input: "do ux work"
    # "do" (2, skip), "ux" (2, whitelist), "work" (4, keep)
    # Interest keywords should contain "ui", "ux", "design".
    # Token "do" is filtered out from relevance check (length < 3).
    # Token "ux" matches.
    # Token "work" doesn't match.
    # Numerator=1 (ux). Denominator=3 ("do", "ux", "work").
    # Relevance = 1/3 = 0.33.
    
    relevance = compute_topic_relevance(
        user_input="do ux work",
        persona_domains=[domain_obj],
        ctx=ctx
    )
    
    assert round(relevance, 4) == round(1/3, 4), f"Expected match for 'ux' (1/3), got {relevance}"
    print("✓ Short token 'ux' matched due to whitelist")


def test_object_access_support():
    """Verify that the module works with Objects, not just Dicts."""
    print("\n--- Testing Object Access ---")
    ctx = TraceContext()
    
    domain_obj = MockDomain(domain="Cooking", proficiency=0.9, subdomains=["Baking"])
    goal_obj = MockGoal(goal="Master Chef", weight=1.0)
    
    # Detect Domain
    dom, score = detect_domain(
        user_input="cooking baking",
        persona_domains=[domain_obj],
        ctx=ctx
    )
    
    assert dom == "cooking", f"Expected domain 'cooking', got '{dom}'"
    print("✓ detect_domain works with Objects")
    
    # Topic Relevance
    relevance = compute_topic_relevance(
        user_input="cooking baking",
        persona_domains=[domain_obj],
        persona_goals=[goal_obj],
        ctx=ctx
    )
    
    assert relevance > 0.5, f"Expected high relevance with Objects, got {relevance}"
    print("✓ compute_topic_relevance works with Objects")


def test_double_counting_prevention():
    """
    Verify that phrases consume tokens preventing double-counting.
    Case where double-counting is distinguishable from clamping.
    """
    print("\n--- Testing Double Counting Prevention ---")
    ctx = TraceContext()
    
    # Setup: "Mental Health" as a subdomain (Phrase).
    domain_obj = MockDomain(domain="Health", proficiency=1.0, subdomains=["Mental Health"])
    
    # Input: "Mental health is important for wellness"
    # Tokens: [mental, health, is, important, for, wellness] (6 tokens)
    # Interest: {mental health, mental, health, wellness}
    # Correct: Phrase "mental health" (2) + Unigram "wellness" (1) = 3/6 = 0.5
    
    goal_obj = MockGoal(goal="wellness", weight=1.0)
    
    relevance = compute_topic_relevance(
        user_input="mental health is important for wellness",
        persona_domains=[domain_obj],
        persona_goals=[goal_obj],
        ctx=ctx
    )
    
    print(f"Relevance: {relevance}")
    assert abs(relevance - 0.5) < 0.01, f"Expected 0.5 (3/6 tokens), got {relevance}"
    print("✓ Double counting properly prevented (3/6 matches)")


def test_citation_accuracy():
    """Verify fallback and evidence strength citations."""
    print("\n--- Testing Citation Accuracy ---")
    ctx = TraceContext()
    
    # 1. Fallback citation
    input_str = "xyzabc123" 
    detect_domain(input_str, persona_domains=[], ctx=ctx)
    
    cites = [c for c in ctx.citations if c.source_id == "domain_detection"]
    assert len(cites) > 0
    msg = cites[0].effect
    
    assert "No domain exceeded min_score" in msg or "No candidates" in msg
    assert input_str in msg, "Expected input snippet in message"
    print("✓ Domain fallback citation correct")
    
    # 2. Evidence strength citation
    from persona_engine.planner.domain_detection import detect_evidence_strength
    
    ctx = TraceContext()
    strength = detect_evidence_strength("I disagree with that", ctx=ctx)
    assert strength == 0.8, f"Expected strong challenge 0.8, got {strength}"
    
    cites = [c for c in ctx.citations if c.source_id == "evidence_strength"]
    assert len(cites) == 1
    assert "strong challenge" in cites[0].effect.lower()
    assert "i disagree" in cites[0].effect.lower()
    print("✓ Evidence strength citation correct")


if __name__ == "__main__":
    test_short_token_whitelist()
    test_object_access_support()
    test_double_counting_prevention()
    test_citation_accuracy()
    print("\nAll Fix Verification Tests Passed!")
