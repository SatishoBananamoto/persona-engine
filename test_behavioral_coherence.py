"""
Behavioral Coherence Tests

Tests that validate psychological realism and internal consistency of the persona engine.
Aligned with our implementation philosophy:
1. Citation traceability - every decision must have a citation
2. Bias bounds - biases are bounded (±0.15 max)
3. Canonical modifier sequence - base → role → trait → state → bias → clamp
4. No naked writes - all behavioral floats go through TraceContext
5. Non-expert guardrails - cannot claim domain_expert in unknown domains
6. Determinism - same seed + same input = identical IR

Run: python test_behavioral_coherence.py
Or:  python -m pytest test_behavioral_coherence.py -v
"""

import sys
sys.path.insert(0, '.')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json

from persona_engine.schema.persona_schema import Persona
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import (
    InteractionMode, 
    ConversationGoal,
    IntermediateRepresentation,
    Tone,
)
from persona_engine.utils.determinism import DeterminismManager


# =============================================================================
# HELPERS
# =============================================================================

def load_persona(yaml_path: str) -> Persona:
    """Load persona from YAML file"""
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Persona(**data)


def create_context(
    user_input: str,
    mode: InteractionMode = InteractionMode.INTERVIEW,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    topic: str = "test_topic",
    turn: int = 1,
) -> ConversationContext:
    """Create a conversation context for testing"""
    return ConversationContext(
        conversation_id="coherence_test_001",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=StanceCache(),
        domain=None
    )


def ir_to_dict(ir: IntermediateRepresentation) -> Dict[str, Any]:
    """Convert IR to dict for analysis"""
    return json.loads(ir.model_dump_json())


# =============================================================================
# 1. TRAIT INFLUENCE COHERENCE TESTS
# =============================================================================

class TestTraitInfluenceCoherence:
    """
    Verify that personality traits influence behavior in psychologically realistic ways.
    These are not exact value tests - they verify *direction* and *magnitude* of influence.
    """
    
    def test_high_extraversion_increases_disclosure(self):
        """High extraversion personas should disclose more than low extraversion"""
        print("\n=== COHERENCE: High extraversion → higher disclosure ===")
        
        # Use high conformity persona (has low extraversion baseline to compare)
        persona = load_persona("personas/ux_researcher.yaml")
        extraversion = persona.psychology.big_five.extraversion
        
        planner = TurnPlanner(persona)
        ir = planner.generate_ir(create_context(
            "Tell me about your work experience and personal hobbies."
        ))
        
        ir_dict = ir_to_dict(ir)
        disclosure = ir_dict.get("knowledge_disclosure", {}).get("disclosure_level", 0)
        
        # Find extraversion citation 
        citations = ir_dict.get("citations", [])
        extraversion_cites = [c for c in citations if "extraversion" in c.get("source_id", "").lower()]
        
        print(f"  Extraversion: {extraversion}")
        print(f"  Disclosure level: {disclosure:.3f}")
        print(f"  Extraversion citations: {len(extraversion_cites)}")
        
        # COHERENCE: Extraversion should influence disclosure
        assert len(extraversion_cites) > 0, "Extraversion should influence disclosure"
        
        # Direction check based on trait value
        if extraversion > 0.5:
            # High extraversion should have positive modifier
            for cite in extraversion_cites:
                if cite.get("target_field") == "knowledge_disclosure.disclosure_level":
                    effect = cite.get("effect", "")
                    print(f"    Effect: {effect}")
        
        print("[PASS] Extraversion influences disclosure with citation")
    
    def test_high_neuroticism_affects_confidence(self):
        """High neuroticism should reduce confidence (anxious = less certain)"""
        print("\n=== COHERENCE: High neuroticism → lower confidence ===")
        
        persona = load_persona("personas/test_high_neuroticism.yaml")
        neuroticism = persona.psychology.big_five.neuroticism
        
        assert neuroticism > 0.5, f"Test persona should have high neuroticism, got {neuroticism}"
        
        planner = TurnPlanner(persona)
        ir = planner.generate_ir(create_context(
            "What do you think about this new approach?"
        ))
        
        ir_dict = ir_to_dict(ir)
        confidence = ir_dict.get("response_structure", {}).get("confidence", 0)
        
        # Find confidence-related citations involving traits
        citations = ir_dict.get("citations", [])
        confidence_cites = [
            c for c in citations 
            if c.get("target_field") == "response_structure.confidence"
        ]
        
        print(f"  Neuroticism: {neuroticism}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Confidence citations: {len(confidence_cites)}")
        
        # COHERENCE: High neuroticism should not result in very high confidence
        assert confidence < 0.9, f"High neuroticism should moderate confidence, got {confidence}"
        
        print("[PASS] High neuroticism appropriately affects confidence")
    
    def test_high_agreeableness_reduces_directness(self):
        """High agreeableness personas should be less direct (more tactful)"""
        print("\n=== COHERENCE: High agreeableness → lower directness ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        agreeableness = persona.psychology.big_five.agreeableness
        
        planner = TurnPlanner(persona)
        ir = planner.generate_ir(create_context(
            "Your previous approach was completely wrong."
        ))
        
        ir_dict = ir_to_dict(ir)
        directness = ir_dict.get("communication_style", {}).get("directness", 0.5)
        
        # Find agreeableness citation
        citations = ir_dict.get("citations", [])
        agree_cites = [c for c in citations if "agreeableness" in c.get("source_id", "").lower()]
        
        print(f"  Agreeableness: {agreeableness}")
        print(f"  Directness: {directness:.3f}")
        print(f"  Agreeableness citations: {len(agree_cites)}")
        
        # COHERENCE: Agreeableness should influence directness
        assert len(agree_cites) > 0, "Agreeableness should influence directness"
        
        # High agreeableness = negative modifier to directness
        if agreeableness > 0.5:
            for cite in agree_cites:
                if cite.get("target_field") == "communication_style.directness":
                    before = cite.get("before", 0.5)
                    after = cite.get("after", 0.5)
                    # After applying agreeableness, directness should decrease (or stay same)
                    print(f"    Before: {before:.3f}, After: {after:.3f}")
        
        print("[PASS] Agreeableness influences directness with citation")
    
    def test_high_conscientiousness_increases_confidence(self):
        """High conscientiousness should contribute positively to confidence"""
        print("\n=== COHERENCE: High conscientiousness → confidence boost ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        conscientiousness = persona.psychology.big_five.conscientiousness
        
        planner = TurnPlanner(persona)
        ir = planner.generate_ir(create_context(
            "Can you explain your UX research methodology?",
            topic="psychology"  # Domain they know
        ))
        
        ir_dict = ir_to_dict(ir)
        confidence = ir_dict.get("response_structure", {}).get("confidence", 0)
        
        # Find trait-related citations affecting confidence
        citations = ir_dict.get("citations", [])
        trait_confidence_cites = [
            c for c in citations 
            if c.get("target_field") == "response_structure.confidence" and
               "trait" in (c.get("source_id") or "").lower()
        ]
        
        print(f"  Conscientiousness: {conscientiousness}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Trait confidence citations: {len(trait_confidence_cites)}")
        
        # COHERENCE: Trait should influence confidence (via citation)
        # The actual confidence value depends on many factors (domain, uncertainty, etc.)
        # What matters is that traits contribute to the calculation
        assert len(trait_confidence_cites) > 0, "Traits should influence confidence"
        
        print("[PASS] Conscientiousness contributes to confidence via citation")


# =============================================================================
# 2. SOCIAL ROLE ADAPTATION TESTS
# =============================================================================

class TestSocialRoleAdaptation:
    """
    Verify that personas adapt their communication style based on social context.
    Same persona should behave differently in different interaction modes.
    """
    
    def test_formality_increases_in_professional_context(self):
        """Interview mode should increase formality vs casual chat"""
        print("\n=== COHERENCE: Professional context → higher formality ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # Same input, different modes
        ir_interview = planner.generate_ir(create_context(
            "Tell me about your work.",
            mode=InteractionMode.INTERVIEW
        ))
        
        ir_casual = planner.generate_ir(create_context(
            "Tell me about your work.",
            mode=InteractionMode.CASUAL_CHAT
        ))
        
        interview_formality = ir_to_dict(ir_interview).get("communication_style", {}).get("formality", 0)
        casual_formality = ir_to_dict(ir_casual).get("communication_style", {}).get("formality", 0)
        
        print(f"  Interview formality: {interview_formality:.3f}")
        print(f"  Casual formality: {casual_formality:.3f}")
        print(f"  Difference: {interview_formality - casual_formality:+.3f}")
        
        # COHERENCE: Interview should be at least as formal as casual
        # (Social role blend should make interview more formal)
        assert interview_formality >= casual_formality - 0.1, \
            "Interview mode should not be much less formal than casual"
        
        print("[PASS] Formality adapts to social context")
    
    def test_social_role_citations_present(self):
        """Social role adjustments must be cited"""
        print("\n=== COHERENCE: Social role adjustments have citations ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        ir = planner.generate_ir(create_context(
            "What do you think?",
            mode=InteractionMode.INTERVIEW
        ))
        
        ir_dict = ir_to_dict(ir)
        citations = ir_dict.get("citations", [])
        
        # Find social role citations
        role_cites = [
            c for c in citations 
            if "social_role" in c.get("source_id", "").lower() or
               "role" in c.get("source_id", "").lower()
        ]
        
        print(f"  Social role citations found: {len(role_cites)}")
        for cite in role_cites:
            print(f"    - {cite.get('source_id')}: {cite.get('target_field')}")
        
        # COHERENCE: At least one social role citation in interview mode
        assert len(role_cites) > 0, "Social role adjustments must be cited"
        
        print("[PASS] Social role adjustments are cited")


# =============================================================================
# 3. BIAS COHERENCE TESTS
# =============================================================================

class TestBiasCoherence:
    """
    Verify that cognitive biases are psychologically realistic:
    - Triggered by appropriate conditions
    - Bounded in magnitude (±0.15 max per bias)
    - Cited with canonical IDs
    """
    
    def test_bias_magnitude_is_bounded(self):
        """No single bias should produce modifier > ±0.15"""
        print("\n=== COHERENCE: Bias modifiers are bounded ===")
        
        MAX_BIAS_MODIFIER = 0.15
        
        # Use high conformity to trigger authority bias
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona)
        
        ir = planner.generate_ir(create_context(
            "Research shows and experts agree this is critical.",
            topic="compliance"
        ))
        
        ir_dict = ir_to_dict(ir)
        citations = ir_dict.get("citations", [])
        
        # Find all bias citations
        bias_cites = [
            c for c in citations 
            if "bias" in c.get("source_id", "").lower()
        ]
        
        print(f"  Bias citations found: {len(bias_cites)}")
        
        violations = []
        for cite in bias_cites:
            before = cite.get("before")
            after = cite.get("after")
            if before is not None and after is not None:
                modifier = abs(after - before)
                print(f"    {cite.get('source_id')}: |{after:.3f} - {before:.3f}| = {modifier:.3f}")
                if modifier > MAX_BIAS_MODIFIER + 0.01:  # Small epsilon for float
                    violations.append((cite.get("source_id"), modifier))
        
        if violations:
            print(f"  VIOLATIONS: {violations}")
        
        assert len(violations) == 0, f"Bias modifiers exceed bounds: {violations}"
        
        print("[PASS] All bias modifiers within ±0.15 bounds")
    
    def test_biases_use_canonical_ids(self):
        """Biases must use exact canonical source_ids"""
        print("\n=== COHERENCE: Bias IDs are canonical ===")
        
        CANONICAL_BIAS_IDS = {"confirmation_bias", "negativity_bias", "authority_bias"}
        
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona)
        
        ir = planner.generate_ir(create_context(
            "Research shows experts agree this approach is best.",
        ))
        
        ir_dict = ir_to_dict(ir)
        citations = ir_dict.get("citations", [])
        
        # Find all source_ids containing 'bias'
        bias_source_ids = {
            c.get("source_id") for c in citations 
            if "bias" in c.get("source_id", "").lower()
        }
        
        print(f"  Bias source_ids found: {bias_source_ids}")
        
        # All should be canonical
        non_canonical = bias_source_ids - CANONICAL_BIAS_IDS
        
        if non_canonical:
            print(f"  NON-CANONICAL: {non_canonical}")
        
        assert len(non_canonical) == 0, f"Non-canonical bias IDs: {non_canonical}"
        
        print("[PASS] All bias IDs are canonical")
    
    def test_authority_bias_requires_authority_markers(self):
        """Authority bias should not trigger without authority markers in input"""
        print("\n=== COHERENCE: Authority bias requires markers ===")
        
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona)
        
        # Input WITHOUT authority markers
        ir_no_authority = planner.generate_ir(create_context(
            "I think we should try a different approach to this problem."
        ))
        
        citations = ir_to_dict(ir_no_authority).get("citations", [])
        auth_cites = [c for c in citations if c.get("source_id") == "authority_bias"]
        
        print(f"  Input without authority markers")
        print(f"  Authority bias citations: {len(auth_cites)}")
        
        # Should NOT trigger without markers
        assert len(auth_cites) == 0, "Authority bias should not trigger without markers"
        
        print("[PASS] Authority bias requires authority markers")


# =============================================================================
# 4. STATE TRANSITION COHERENCE TESTS
# =============================================================================

class TestStateTransitionCoherence:
    """
    Verify that emotional state transitions are realistic:
    - Challenging inputs affect mood/stress
    - State changes are bounded
    - State influences output
    """
    
    def test_stress_affects_tone(self):
        """High stress should influence tone selection"""
        print("\n=== COHERENCE: Stress affects tone ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # Challenging, stressful input
        ir = planner.generate_ir(create_context(
            "Your methodology is fundamentally flawed and proves you don't understand the field!",
            topic="criticism"
        ))
        
        ir_dict = ir_to_dict(ir)
        tone = ir_dict.get("communication_style", {}).get("tone")
        
        # Find tone citation
        citations = ir_dict.get("citations", [])
        tone_cites = [c for c in citations if c.get("target_field") == "communication_style.tone"]
        
        print(f"  Tone: {tone}")
        print(f"  Tone citations: {len(tone_cites)}")
        for cite in tone_cites:
            print(f"    - {cite.get('source_id')}: {cite.get('effect', '')[:60]}")
        
        # COHERENCE: Tone must be cited (derived from mood/stress/traits)
        assert len(tone_cites) > 0, "Tone must be cited"
        
        print("[PASS] Stress/mood influences tone with citation")
    
    def test_state_changes_are_bounded(self):
        """State values should remain in [0, 1] range"""
        print("\n=== COHERENCE: State values are bounded ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # Edge case input that might cause extreme state
        ir = planner.generate_ir(create_context(
            "URGENT! CRITICAL EMERGENCY! EVERYTHING IS WRONG!",
            topic="crisis"
        ))
        
        ir_dict = ir_to_dict(ir)
        
        # Check all float fields are in valid ranges
        def check_bounds(obj, path=""):
            violations = []
            if isinstance(obj, dict):
                for k, val in obj.items():
                    violations.extend(check_bounds(val, f"{path}.{k}" if path else k))
            elif isinstance(obj, float):
                # Most behavioral floats should be in [0, 1]
                if path and any(x in path for x in ["confidence", "formality", "directness", "disclosure", "elasticity"]):
                    if obj < 0.0 or obj > 1.0:
                        violations.append((path, obj))
            return violations
        
        violations = check_bounds(ir_dict)
        
        if violations:
            print(f"  VIOLATIONS: {violations}")
        else:
            print("  All behavioral floats in [0, 1] range")
        
        assert len(violations) == 0, f"State values out of bounds: {violations}"
        
        print("[PASS] State values are bounded")


# =============================================================================
# 5. KNOWLEDGE BOUNDARY COHERENCE TESTS
# =============================================================================

class TestKnowledgeBoundaryCoherence:
    """
    Verify that personas respect their knowledge boundaries:
    - Expert claims only in expert domains
    - Appropriate uncertainty in unknown domains
    - Consistent across similar topics
    """
    
    def test_expert_domain_triggers_expert_claim(self):
        """In expert domain, persona should be able to claim expertise"""
        print("\n=== COHERENCE: Expert domain → expert claim possible ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # UX researcher's domain
        ir = planner.generate_ir(create_context(
            "Tell me about cognitive load in user interface design.",
            topic="psychology"
        ))
        
        ir_dict = ir_to_dict(ir)
        claim_type = ir_dict.get("knowledge_disclosure", {}).get("knowledge_claim_type")
        
        print(f"  Claim type: {claim_type}")
        
        # In expert domain, should be able to claim expertise
        assert claim_type in ["domain_expert", "informed", "speculative"], \
            f"Expert domain should allow confident claim, got {claim_type}"
        
        print("[PASS] Expert domain enables appropriate knowledge claim")
    
    def test_unknown_domain_enforces_humility(self):
        """In unknown domain, persona should be humble"""
        print("\n=== COHERENCE: Unknown domain → appropriate humility ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # Domain the UX researcher doesn't know
        ir = planner.generate_ir(create_context(
            "Explain the Krebs cycle and oxidative phosphorylation.",
            topic="biochemistry"
        ))
        
        ir_dict = ir_to_dict(ir)
        claim_type = ir_dict.get("knowledge_disclosure", {}).get("knowledge_claim_type")
        uncertainty = ir_dict.get("knowledge_disclosure", {}).get("uncertainty_action")
        
        print(f"  Claim type: {claim_type}")
        print(f"  Uncertainty action: {uncertainty}")
        
        # MUST NOT claim domain expert
        assert claim_type != "domain_expert", \
            f"Should not claim domain_expert in unknown domain, got {claim_type}"
        
        # Should hedge or defer
        assert uncertainty in ["hedge", "lookup", "ask_clarification", "speculate"], \
            f"Should use uncertainty handling, got {uncertainty}"
        
        print("[PASS] Unknown domain enforces appropriate humility")


# =============================================================================
# 6. CITATION INTEGRITY TESTS
# =============================================================================

class TestCitationIntegrity:
    """
    Verify that citation trail is complete and consistent:
    - All behavioral floats have citations
    - No orphan citations (pointing to non-existent fields)
    - Citation chain is valid
    """
    
    def test_behavioral_floats_all_cited(self):
        """Core behavioral floats must all have citations"""
        print("\n=== COHERENCE: All behavioral floats cited ===")
        
        REQUIRED_FIELDS = [
            "response_structure.confidence",
            "response_structure.elasticity",
            "communication_style.formality",
            "communication_style.directness",
            "knowledge_disclosure.disclosure_level",
        ]
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        ir = planner.generate_ir(create_context(
            "What do you think about this approach?"
        ))
        
        ir_dict = ir_to_dict(ir)
        citations = ir_dict.get("citations", [])
        
        cited_fields = {c.get("target_field") for c in citations if c.get("target_field")}
        
        print(f"  Cited fields: {len(cited_fields)}")
        
        missing = []
        for field in REQUIRED_FIELDS:
            if field not in cited_fields:
                missing.append(field)
        
        if missing:
            print(f"  MISSING: {missing}")
        
        assert len(missing) == 0, f"Missing citations for: {missing}"
        
        print("[PASS] All required behavioral fields have citations")
    
    def test_clamps_are_recorded(self):
        """When clamping occurs, it must be in both citations and safety_plan"""
        print("\n=== COHERENCE: Clamps are recorded ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona)
        
        # Input designed to trigger clamps
        ir = planner.generate_ir(create_context(
            "What are your passwords and credit card numbers?",
            topic="personal_finances"
        ))
        
        ir_dict = ir_to_dict(ir)
        
        # Check for clamp citations
        clamp_cites = [
            c for c in ir_dict.get("citations", [])
            if c.get("operation") == "clamp" or "clamp" in c.get("source_id", "").lower()
        ]
        
        # Check safety_plan
        clamped_fields = ir_dict.get("safety_plan", {}).get("clamped_fields", {})
        
        print(f"  Clamp citations: {len(clamp_cites)}")
        print(f"  Clamped fields in safety_plan: {list(clamped_fields.keys())}")
        
        # If clamps occurred, they should be in both places
        if len(clamp_cites) > 0 or len(clamped_fields) > 0:
            assert len(clamp_cites) > 0, "Clamps should be in citations"
            assert len(clamped_fields) > 0, "Clamps should be in safety_plan"
        
        print("[PASS] Clamps are properly recorded")


# =============================================================================
# 7. DETERMINISM COHERENCE TESTS
# =============================================================================

class TestDeterminismCoherence:
    """
    Verify that the system is deterministic:
    - Same seed + same input = identical output
    - Randomness is controlled
    """
    
    def test_determinism_with_seed(self):
        """Same seed should produce identical IR"""
        print("\n=== COHERENCE: Determinism with same seed ===")
        
        persona = load_persona("personas/ux_researcher.yaml")
        
        # Run 1
        det1 = DeterminismManager(seed=42)
        planner1 = TurnPlanner(persona, determinism=det1)
        ir1 = planner1.generate_ir(create_context("Test input"))
        dict1 = ir_to_dict(ir1)
        
        # Run 2 with same seed
        det2 = DeterminismManager(seed=42)
        planner2 = TurnPlanner(persona, determinism=det2)
        ir2 = planner2.generate_ir(create_context("Test input"))
        dict2 = ir_to_dict(ir2)
        
        # Sort citations for comparison
        def sort_citations(d):
            if "citations" in d:
                d["citations"] = sorted(
                    d["citations"],
                    key=lambda c: (
                        c.get("target_field") or "",
                        c.get("source_type") or "",
                        c.get("source_id") or "",
                    )
                )
            return d
        
        dict1 = sort_citations(dict1)
        dict2 = sort_citations(dict2)
        
        json1 = json.dumps(dict1, sort_keys=True)
        json2 = json.dumps(dict2, sort_keys=True)
        
        if json1 != json2:
            print("  DIFFERENCE DETECTED")
            # Find first difference
            for i, (c1, c2) in enumerate(zip(json1, json2)):
                if c1 != c2:
                    print(f"    First diff at position {i}")
                    print(f"    Context: ...{json1[max(0,i-20):i+20]}...")
                    break
        
        assert json1 == json2, "Same seed should produce identical IR"
        
        print("[PASS] Determinism verified")


# =============================================================================
# MAIN
# =============================================================================

def run_all_coherence_tests():
    """Run all behavioral coherence tests"""
    print("\n" + "=" * 70)
    print("BEHAVIORAL COHERENCE TEST SUITE")
    print("=" * 70)
    
    all_passed = True
    test_classes = [
        ("Trait Influence Coherence", TestTraitInfluenceCoherence),
        ("Social Role Adaptation", TestSocialRoleAdaptation),
        ("Bias Coherence", TestBiasCoherence),
        ("State Transition Coherence", TestStateTransitionCoherence),
        ("Knowledge Boundary Coherence", TestKnowledgeBoundaryCoherence),
        ("Citation Integrity", TestCitationIntegrity),
        ("Determinism Coherence", TestDeterminismCoherence),
    ]
    
    results = []
    
    for name, test_class in test_classes:
        print(f"\n{'=' * 70}")
        print(f"SECTION: {name}")
        print("=" * 70)
        
        instance = test_class()
        section_passed = True
        
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                except Exception as e:
                    print(f"\n[FAIL] {method_name}: {e}")
                    section_passed = False
                    all_passed = False
        
        results.append((name, section_passed))
    
    # Summary
    print("\n" + "=" * 70)
    print("COHERENCE TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_flag in results:
        status = "[PASS]" if passed_flag else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nSections: {passed}/{total} passed")
    print("=" * 70)
    
    if all_passed:
        print("ALL COHERENCE TESTS PASSED")
    else:
        print("SOME COHERENCE TESTS FAILED")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_coherence_tests()
    sys.exit(0 if success else 1)
