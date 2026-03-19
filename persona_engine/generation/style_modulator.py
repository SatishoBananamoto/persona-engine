"""
Style Modulator

Post-processes LLM output to ensure compliance with IR constraints.
Applies verbosity enforcement and constraint validation.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    Verbosity,
)


@dataclass
class ConstraintViolation:
    """Represents a detected constraint violation."""
    constraint: str
    expected: str
    actual: str
    severity: str  # "warning" or "error"


class StyleModulator:
    """
    Post-processes generated text to match IR constraints.
    
    Responsibilities:
    - Enforce verbosity limits
    - Validate constraint compliance
    - Report violations for debugging
    """
    
    # Verbosity targets (sentence counts)
    VERBOSITY_TARGETS = {
        Verbosity.MINIMAL: (0, 0),
        Verbosity.BRIEF: (1, 2),
        Verbosity.MEDIUM: (3, 5),
        Verbosity.DETAILED: (6, 15),
    }
    
    def enforce_verbosity(
        self,
        text: str,
        target: Verbosity,
        strict: bool = False
    ) -> str:
        """
        Enforce verbosity constraint on text.
        
        Args:
            text: Generated text
            target: Target verbosity level
            strict: If True, truncate/expand to exact range
            
        Returns:
            Adjusted text (or original if within range)
        """
        sentences = self._split_sentences(text)
        # MINIMAL: keep only first few words, not sentence-based
        if target == Verbosity.MINIMAL:
            words = text.strip().split()
            if strict and len(words) > 3:
                return " ".join(words[:3])
            return text
        min_sentences, max_sentences = self.VERBOSITY_TARGETS[target]
        
        if len(sentences) < min_sentences:
            # Too brief - can't expand without LLM, return as-is with warning
            return text
        
        if strict and len(sentences) > max_sentences:
            # Too long - truncate
            truncated = sentences[:max_sentences]
            return " ".join(truncated)
        
        return text
    
    def validate_constraints(
        self,
        text: str,
        ir: IntermediateRepresentation
    ) -> List[ConstraintViolation]:
        """
        Check if generated text respects IR constraints.
        
        Args:
            text: Generated text
            ir: Original IR with constraints
            
        Returns:
            List of detected violations
        """
        violations = []
        
        # Check verbosity
        verbosity_violation = self._check_verbosity(text, ir.communication_style.verbosity)
        if verbosity_violation:
            violations.append(verbosity_violation)
        
        # Check for safety violations (topics to avoid)
        safety_violations = self._check_safety(text, ir)
        violations.extend(safety_violations)
        
        # Check knowledge claim violations
        claim_violations = self._check_knowledge_claims(text, ir)
        violations.extend(claim_violations)
        
        return violations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting (handles common cases)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]
    
    def _check_verbosity(
        self,
        text: str,
        target: Verbosity
    ) -> Optional[ConstraintViolation]:
        """Check if text matches target verbosity."""
        # MINIMAL is word-based, not sentence-based (matches enforce_verbosity logic)
        if target == Verbosity.MINIMAL:
            words = text.strip().split()
            if len(words) > 10:
                return ConstraintViolation(
                    constraint="verbosity",
                    expected="minimal (≤10 words)",
                    actual=f"{len(words)} words",
                    severity="warning"
                )
            return None

        sentences = self._split_sentences(text)
        min_s, max_s = self.VERBOSITY_TARGETS[target]

        if len(sentences) < min_s:
            return ConstraintViolation(
                constraint="verbosity",
                expected=f"{target.value} ({min_s}-{max_s} sentences)",
                actual=f"{len(sentences)} sentences",
                severity="warning"
            )
        elif len(sentences) > max_s:
            return ConstraintViolation(
                constraint="verbosity",
                expected=f"{target.value} ({min_s}-{max_s} sentences)",
                actual=f"{len(sentences)} sentences",
                severity="warning"
            )
        
        return None
    
    def _check_safety(
        self,
        text: str,
        ir: IntermediateRepresentation
    ) -> List[ConstraintViolation]:
        """Check for safety constraint violations.

        Checks blocked_topics from the safety plan AND cannot_claim /
        must_avoid from the persona invariants (stored on the IR safety plan).
        """
        violations = []
        text_lower = text.lower()

        # Check blocked topics (from safety plan)
        if ir.safety_plan.blocked_topics:
            for topic in ir.safety_plan.blocked_topics:
                if topic.lower() in text_lower:
                    violations.append(ConstraintViolation(
                        constraint="topics_to_avoid",
                        expected=f"Avoid '{topic}'",
                        actual=f"Text contains '{topic}'",
                        severity="error"
                    ))

        # Check cannot_claim items
        for claim in getattr(ir.safety_plan, "cannot_claim", []):
            if claim.lower() in text_lower:
                violations.append(ConstraintViolation(
                    constraint="cannot_claim",
                    expected=f"Must not claim '{claim}'",
                    actual=f"Text contains forbidden claim '{claim}'",
                    severity="error"
                ))

        # Check must_avoid items
        for avoided in getattr(ir.safety_plan, "must_avoid", []):
            if avoided.lower() in text_lower:
                violations.append(ConstraintViolation(
                    constraint="must_avoid",
                    expected=f"Must avoid '{avoided}'",
                    actual=f"Text mentions avoided topic '{avoided}'",
                    severity="error"
                ))

        return violations
    
    def _check_knowledge_claims(
        self,
        text: str,
        ir: IntermediateRepresentation
    ) -> List[ConstraintViolation]:
        """Check for inappropriate knowledge claims."""
        violations = []
        text_lower = text.lower()
        
        from persona_engine.schema.ir_schema import KnowledgeClaimType
        
        claim_type = ir.knowledge_disclosure.knowledge_claim_type

        # Pre-compute strong assertion detection (used by multiple branches)
        strong_assertions = [
            "definitely", "certainly", "always", "never",
            "the fact is", "studies show", "research proves"
        ]
        has_strong = any(phrase in text_lower for phrase in strong_assertions)

        # If speculative, should have hedging
        if claim_type in (KnowledgeClaimType.SPECULATIVE, KnowledgeClaimType.NONE):
            hedging_phrases = [
                "i think", "i believe", "perhaps", "maybe", "might",
                "could be", "not sure", "uncertain", "my guess"
            ]
            has_hedging = any(phrase in text_lower for phrase in hedging_phrases)

            if has_strong and not has_hedging:
                violations.append(ConstraintViolation(
                    constraint="knowledge_claim",
                    expected=f"Speculative claim type should use hedging",
                    actual="Contains strong assertions without hedging",
                    severity="warning"
                ))

        # ANECDOTAL should use second-hand markers
        if claim_type == KnowledgeClaimType.ANECDOTAL:
            anecdotal_markers = [
                "i heard", "someone told me", "a friend", "they say",
                "apparently", "supposedly", "i was told",
            ]
            has_marker = any(phrase in text_lower for phrase in anecdotal_markers)
            if not has_marker and has_strong:
                violations.append(ConstraintViolation(
                    constraint="knowledge_claim",
                    expected="Anecdotal claim should use second-hand language",
                    actual="Contains strong assertions without anecdotal markers",
                    severity="warning"
                ))

        # HYPOTHETICAL should use conditional language
        if claim_type == KnowledgeClaimType.HYPOTHETICAL:
            conditional_markers = [
                "if ", "hypothetically", "in theory", "what if",
                "suppose", "assuming", "would ", "could ",
            ]
            has_conditional = any(phrase in text_lower for phrase in conditional_markers)
            if not has_conditional:
                violations.append(ConstraintViolation(
                    constraint="knowledge_claim",
                    expected="Hypothetical claim should use conditional language",
                    actual="Missing conditional/hypothetical framing",
                    severity="warning"
                ))

        # ACADEMIC_CITED should have research markers
        if claim_type == KnowledgeClaimType.ACADEMIC_CITED:
            research_markers = [
                "research", "study", "studies", "published",
                "according to", "evidence suggests", "findings",
            ]
            has_research = any(phrase in text_lower for phrase in research_markers)
            if not has_research:
                violations.append(ConstraintViolation(
                    constraint="knowledge_claim",
                    expected="Academic cited claim should reference research",
                    actual="No research-related language found",
                    severity="warning"
                ))

        return violations
    
    def get_sentence_count(self, text: str) -> int:
        """Get the number of sentences in text."""
        return len(self._split_sentences(text))
    
    def summarize_violations(
        self,
        violations: List[ConstraintViolation]
    ) -> str:
        """Create a human-readable summary of violations."""
        if not violations:
            return "No constraint violations detected."
        
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        summary = []
        if errors:
            summary.append(f"ERRORS ({len(errors)}):")
            for v in errors:
                summary.append(f"  - {v.constraint}: {v.actual} (expected: {v.expected})")
        
        if warnings:
            summary.append(f"WARNINGS ({len(warnings)}):")
            for v in warnings:
                summary.append(f"  - {v.constraint}: {v.actual} (expected: {v.expected})")
        
        return "\n".join(summary)
