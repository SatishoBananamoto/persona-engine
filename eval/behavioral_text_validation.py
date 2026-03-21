"""
BV-2: Full Text Behavioral Validation

Generates real text via Anthropic API for 10 Big Five extreme profiles,
analyzes the text for LIWC-style markers, and compares against
Yarkoni (2010) correlation directions.

Usage:
    kv run -- python3 -m eval.behavioral_text_validation

Requires ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder


# =============================================================================
# Profiles
# =============================================================================

PROFILES = {
    "high_O": {"O": 0.9, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
    "low_O":  {"O": 0.1, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
    "high_E": {"O": 0.5, "C": 0.5, "E": 0.9, "A": 0.5, "N": 0.5},
    "low_E":  {"O": 0.5, "C": 0.5, "E": 0.1, "A": 0.5, "N": 0.5},
    "high_A": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.9, "N": 0.5},
    "low_A":  {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.1, "N": 0.5},
    "high_N": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.9},
    "low_N":  {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.1},
    "high_C": {"O": 0.5, "C": 0.9, "E": 0.5, "A": 0.5, "N": 0.5},
    "low_C":  {"O": 0.5, "C": 0.1, "E": 0.5, "A": 0.5, "N": 0.5},
}

PROMPTS = [
    "What do you think about remote work vs office work?",
    "Tell me about a time you had to deal with a difficult situation.",
    "How should we approach climate change as a society?",
]


# =============================================================================
# LIWC-style Text Analysis
# =============================================================================

# Marker word lists (simplified LIWC categories relevant to Big Five)
HEDGING_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could be", "i think",
    "it seems", "i guess", "kind of", "sort of", "i suppose",
    "not sure", "i believe", "in my opinion", "arguably",
]

CERTAINTY_WORDS = [
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "always", "never", "must", "without doubt", "undoubtedly",
    "for sure", "no question",
]

NEGATIVE_EMOTION = [
    "worried", "concerned", "afraid", "anxious", "frustrated",
    "disappointed", "upset", "stressed", "unfortunate", "sadly",
    "fear", "risk", "danger", "problem", "trouble", "difficult",
    "struggle", "challenge", "downside", "drawback",
]

POSITIVE_EMOTION = [
    "great", "wonderful", "excellent", "love", "enjoy", "excited",
    "happy", "fantastic", "amazing", "beautiful", "brilliant",
    "appreciate", "grateful", "thrilled", "delighted", "awesome",
]

SOCIAL_WORDS = [
    " we ", " us ", " our ", " together ", " people ", " community ",
    " team ", " everyone ", " folks ", " group ",
]

SELF_REFERENCE = [
    " i ", " me ", " my ", " myself ", " i'm ", " i've ", " i'd ",
]

STRUCTURE_WORDS = [
    "first", "second", "additionally", "furthermore", "in summary",
    "in conclusion", "moreover", "specifically", "therefore",
]


def count_markers(text: str, markers: list[str]) -> int:
    """Count occurrences of marker words/phrases in text."""
    text_lower = f" {text.lower()} "
    count = 0
    for marker in markers:
        count += text_lower.count(marker.lower())
    return count


@dataclass
class TextAnalysis:
    """LIWC-style analysis of generated text."""
    word_count: int = 0
    hedging: int = 0
    certainty: int = 0
    negative_emotion: int = 0
    positive_emotion: int = 0
    social_words: int = 0
    self_reference: int = 0
    structure_words: int = 0

    def to_dict(self) -> dict:
        return {
            "word_count": self.word_count,
            "hedging": self.hedging,
            "certainty": self.certainty,
            "negative_emotion": self.negative_emotion,
            "positive_emotion": self.positive_emotion,
            "social_words": self.social_words,
            "self_reference": self.self_reference,
            "structure_words": self.structure_words,
        }


def analyze_text(text: str) -> TextAnalysis:
    """Run LIWC-style analysis on text."""
    return TextAnalysis(
        word_count=len(text.split()),
        hedging=count_markers(text, HEDGING_WORDS),
        certainty=count_markers(text, CERTAINTY_WORDS),
        negative_emotion=count_markers(text, NEGATIVE_EMOTION),
        positive_emotion=count_markers(text, POSITIVE_EMOTION),
        social_words=count_markers(text, SOCIAL_WORDS),
        self_reference=count_markers(text, SELF_REFERENCE),
        structure_words=count_markers(text, STRUCTURE_WORDS),
    )


# =============================================================================
# Direction Checks (Yarkoni 2010)
# =============================================================================

DIRECTION_CHECKS = [
    # (high_profile, low_profile, marker_field, expected_direction, yarkoni_citation)
    ("high_N", "low_N", "hedging", ">", "N+ → tentative, hedging (r=.08-.14)"),
    ("high_N", "low_N", "negative_emotion", ">", "N+ → negative emotion words (r=.16)"),
    ("high_N", "low_N", "self_reference", ">", "N+ → first person singular (r=.12)"),
    ("high_E", "low_E", "positive_emotion", ">", "E+ → positive emotion words (r=.10)"),
    ("high_E", "low_E", "social_words", ">", "E+ → social process words (r=.15)"),
    ("high_E", "low_E", "word_count", ">", "E+ → higher word count"),
    ("high_A", "low_A", "positive_emotion", ">", "A+ → positive emotion (r=.18)"),
    ("high_A", "low_A", "negative_emotion", "<", "A+ → fewer anger/negative words (r=-.23)"),
    ("high_C", "low_C", "certainty", ">", "C+ → certainty words (r=.10)"),
    ("high_C", "low_C", "structure_words", ">", "C+ → structured, organized language"),
]


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Run with: kv run -- python3 -m eval.behavioral_text_validation")
        return 1

    print("\n" + "=" * 80)
    print("  BV-2: FULL TEXT BEHAVIORAL VALIDATION (Anthropic API)")
    print("=" * 80)

    # Generate text for each profile
    results: dict[str, TextAnalysis] = {}
    texts: dict[str, list[str]] = {}

    for name, traits in PROFILES.items():
        print(f"\n  Generating text for {name}...")
        builder = PersonaBuilder(
            name=name.replace("_", " ").title(),
            occupation="General Professional",
        )
        builder._big_five = {
            "openness": traits["O"], "conscientiousness": traits["C"],
            "extraversion": traits["E"], "agreeableness": traits["A"],
            "neuroticism": traits["N"],
        }
        persona = builder.build()

        try:
            engine = PersonaEngine(persona=persona, llm_provider="anthropic")
        except Exception as e:
            print(f"    ERROR creating engine: {e}")
            continue

        combined_analysis = TextAnalysis()
        profile_texts = []

        for prompt in PROMPTS:
            try:
                result = engine.chat(prompt)
                text = result.text
                profile_texts.append(text)
                analysis = analyze_text(text)

                combined_analysis.word_count += analysis.word_count
                combined_analysis.hedging += analysis.hedging
                combined_analysis.certainty += analysis.certainty
                combined_analysis.negative_emotion += analysis.negative_emotion
                combined_analysis.positive_emotion += analysis.positive_emotion
                combined_analysis.social_words += analysis.social_words
                combined_analysis.self_reference += analysis.self_reference
                combined_analysis.structure_words += analysis.structure_words

                print(f"    {prompt[:50]}... ({analysis.word_count} words)")
            except Exception as e:
                print(f"    ERROR: {e}")

        results[name] = combined_analysis
        texts[name] = profile_texts

    # Print comparison table
    print("\n" + "=" * 80)
    print("  MARKER COUNTS (summed across 3 prompts)")
    print("=" * 80)
    header = f"{'Profile':<12} {'words':>6} {'hedge':>6} {'cert':>6} {'neg':>6} {'pos':>6} {'social':>6} {'self':>6} {'struct':>6}"
    print(header)
    print("-" * 72)
    for name in PROFILES:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<12} {r.word_count:>6} {r.hedging:>6} {r.certainty:>6} "
              f"{r.negative_emotion:>6} {r.positive_emotion:>6} {r.social_words:>6} "
              f"{r.self_reference:>6} {r.structure_words:>6}")

    # Run direction checks
    print("\n" + "=" * 80)
    print("  DIRECTION CHECKS (vs Yarkoni 2010)")
    print("=" * 80)

    passed = 0
    failed = 0

    for high, low, marker, direction, citation in DIRECTION_CHECKS:
        if high not in results or low not in results:
            print(f"  ?? [{high} vs {low}] — missing data")
            continue

        high_val = getattr(results[high], marker)
        low_val = getattr(results[low], marker)

        if direction == ">":
            check_passed = high_val > low_val
        else:
            check_passed = high_val < low_val

        if check_passed:
            passed += 1
            marker_str = "  "
        else:
            failed += 1
            marker_str = ">>"

        status = "PASS" if check_passed else "FAIL"
        print(f"  {marker_str} [{status}] {high} {direction} {low} on {marker}: "
              f"{high_val} vs {low_val}")
        print(f"         {citation}")

    print(f"\n  Result: {passed}/{passed + failed} direction checks passed")
    if failed:
        print(f"  WARNING: {failed} checks failed — some text markers don't align with literature")
    else:
        print("  All text-level markers align with Yarkoni (2010) correlations!")

    # Save report
    report = {
        "profiles": {
            name: {
                "traits": PROFILES[name],
                "markers": results[name].to_dict() if name in results else None,
                "texts": texts.get(name, []),
            }
            for name in PROFILES
        },
        "direction_checks": [
            {
                "high": high, "low": low, "marker": marker,
                "direction": direction, "citation": citation,
                "high_val": getattr(results.get(high, TextAnalysis()), marker, 0),
                "low_val": getattr(results.get(low, TextAnalysis()), marker, 0),
                "passed": (getattr(results.get(high, TextAnalysis()), marker, 0) >
                          getattr(results.get(low, TextAnalysis()), marker, 0))
                          if direction == ">" else
                          (getattr(results.get(high, TextAnalysis()), marker, 0) <
                          getattr(results.get(low, TextAnalysis()), marker, 0)),
            }
            for high, low, marker, direction, citation in DIRECTION_CHECKS
        ],
        "summary": {"passed": passed, "failed": failed, "total": passed + failed},
    }
    report_path = Path(__file__).parent / "bv2_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report: {report_path}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
