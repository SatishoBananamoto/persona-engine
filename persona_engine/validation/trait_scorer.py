"""
Trait Marker Scorer — validates that generated text exhibits expected Big Five markers.

Given a response text and expected Big Five profile, scores how well
the text matches expected linguistic markers for each trait dimension.

Usage:
    scorer = TraitMarkerScorer()
    result = scorer.score(text, expected_big_five)
    print(result.overall_coherence)  # 0.0 - 1.0
    print(result.per_trait)          # {"openness": 0.85, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# Linguistic markers for each Big Five trait dimension
# =============================================================================

# High-trait markers: words/phrases that indicate HIGH levels of the trait
# Low-trait markers: words/phrases that indicate LOW levels

HIGH_OPENNESS_MARKERS = [
    "what if", "imagine", "creative", "innovative", "explore",
    "perspective", "interesting", "fascinating", "novel", "abstract",
    "theoretical", "possibility", "wonder", "curious", "unconventional",
    "alternative", "metaphor", "idea", "concept", "experiment",
]
LOW_OPENNESS_MARKERS = [
    "practical", "traditional", "proven", "standard", "conventional",
    "straightforward", "simple", "established", "concrete", "routine",
    "familiar", "typical", "usual", "normal", "basic",
]

HIGH_CONSCIENTIOUSNESS_MARKERS = [
    "first", "second", "third", "step", "plan", "organize",
    "specifically", "carefully", "precisely", "thorough", "detail",
    "systematic", "structured", "therefore", "consequently", "ensure",
    "schedule", "priority", "methodical", "accurate",
]
LOW_CONSCIENTIOUSNESS_MARKERS = [
    "whatever", "roughly", "approximately", "guess", "probably",
    "sort of", "kind of", "maybe", "anyway", "flexible",
    "spontaneous", "casual", "loosely", "vaguely", "general",
]

HIGH_EXTRAVERSION_MARKERS = [
    "i love", "excited", "amazing", "awesome", "fantastic",
    "!", "we", "together", "share", "everyone",
    "fun", "energy", "enthusiastic", "social", "team",
    "absolutely", "definitely", "totally", "great",
]
LOW_EXTRAVERSION_MARKERS = [
    "i prefer", "quiet", "alone", "independently", "private",
    "reserved", "solitary", "contemplative", "reflect", "internal",
    "myself", "personally", "individually", "calm",
]

HIGH_AGREEABLENESS_MARKERS = [
    "i understand", "that makes sense", "you're right", "good point",
    "i appreciate", "thank you", "helpful", "kind", "supportive",
    "agree", "empathize", "compassion", "care", "harmony",
    "please", "we could", "perhaps", "considerate",
]
LOW_AGREEABLENESS_MARKERS = [
    "actually", "however", "but", "disagree", "wrong",
    "incorrect", "challenge", "debate", "argue", "frankly",
    "bluntly", "straightforwardly", "no", "critical", "direct",
]

HIGH_NEUROTICISM_MARKERS = [
    "worried", "concerned", "anxious", "nervous", "uncertain",
    "i'm not sure", "difficult", "stressful", "overwhelming",
    "afraid", "unfortunately", "risk", "danger", "careful",
    "might go wrong", "what if something", "hesitant",
]
LOW_NEUROTICISM_MARKERS = [
    "confident", "certain", "sure", "no problem", "easy",
    "relaxed", "calm", "comfortable", "stable", "secure",
    "straightforward", "manageable", "fine", "okay", "steady",
]


@dataclass
class TraitScore:
    """Score for a single trait dimension."""
    trait: str
    expected_level: float  # 0-1 from persona's Big Five
    high_marker_count: int
    low_marker_count: int
    marker_ratio: float  # high / (high + low), or 0.5 if no markers
    coherence: float  # 0-1, how well text matches expected trait level


@dataclass
class TraitScorerResult:
    """Complete scoring result across all Big Five traits."""
    per_trait: dict[str, TraitScore] = field(default_factory=dict)
    overall_coherence: float = 0.0
    total_markers_found: int = 0

    def summary(self) -> str:
        lines = [f"Overall coherence: {self.overall_coherence:.2f}"]
        for name, score in self.per_trait.items():
            lines.append(
                f"  {name}: expected={score.expected_level:.2f}, "
                f"markers=H{score.high_marker_count}/L{score.low_marker_count}, "
                f"coherence={score.coherence:.2f}"
            )
        return "\n".join(lines)


TRAIT_MARKERS = {
    "openness": (HIGH_OPENNESS_MARKERS, LOW_OPENNESS_MARKERS),
    "conscientiousness": (HIGH_CONSCIENTIOUSNESS_MARKERS, LOW_CONSCIENTIOUSNESS_MARKERS),
    "extraversion": (HIGH_EXTRAVERSION_MARKERS, LOW_EXTRAVERSION_MARKERS),
    "agreeableness": (HIGH_AGREEABLENESS_MARKERS, LOW_AGREEABLENESS_MARKERS),
    "neuroticism": (HIGH_NEUROTICISM_MARKERS, LOW_NEUROTICISM_MARKERS),
}


class TraitMarkerScorer:
    """Scores generated text against expected Big Five trait markers."""

    def score(
        self,
        text: str,
        expected_big_five: dict[str, float],
    ) -> TraitScorerResult:
        """
        Score text against expected Big Five profile.

        Args:
            text: Generated response text
            expected_big_five: Dict with keys openness, conscientiousness,
                extraversion, agreeableness, neuroticism (values 0-1)

        Returns:
            TraitScorerResult with per-trait scores and overall coherence
        """
        text_lower = text.lower()
        result = TraitScorerResult()
        coherence_scores = []

        for trait_name, (high_markers, low_markers) in TRAIT_MARKERS.items():
            expected = expected_big_five.get(trait_name, 0.5)

            high_count = sum(1 for m in high_markers if m in text_lower)
            low_count = sum(1 for m in low_markers if m in text_lower)
            total = high_count + low_count

            if total > 0:
                marker_ratio = high_count / total
            else:
                # No markers found — neutral, assume moderate coherence
                marker_ratio = 0.5

            # Coherence: how well does marker ratio match expected level?
            # Expected high trait (>0.6) should have high marker ratio
            # Expected low trait (<0.4) should have low marker ratio
            # The closer marker_ratio is to expected, the higher coherence
            coherence = 1.0 - abs(marker_ratio - expected)

            trait_score = TraitScore(
                trait=trait_name,
                expected_level=expected,
                high_marker_count=high_count,
                low_marker_count=low_count,
                marker_ratio=marker_ratio,
                coherence=coherence,
            )

            result.per_trait[trait_name] = trait_score
            result.total_markers_found += total
            coherence_scores.append(coherence)

        result.overall_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        return result
