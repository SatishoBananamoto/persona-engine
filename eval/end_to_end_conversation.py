"""
End-to-End Multi-Turn Conversation Validation

Tests the FULL pipeline: Persona → multi-turn conversation with real LLM →
verify that state accumulation (fatigue, mood, trust, stance cache) produces
observable behavioral changes in the actual generated TEXT.

Usage:
    kv run -- python3 -m eval.end_to_end_conversation

Budget: ~$0.25-0.50 with Claude Sonnet (3 personas x 12 turns)
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder


# =============================================================================
# Helpers
# =============================================================================

def _build_engine(name: str, **trait_overrides) -> PersonaEngine:
    builder = PersonaBuilder(name=name, occupation="General Professional")
    traits = {
        "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
        "agreeableness": 0.5, "neuroticism": 0.5,
    }
    traits.update(trait_overrides)
    builder._big_five = traits
    persona = builder.build()
    return PersonaEngine(persona=persona, llm_provider="anthropic")


def word_count(text: str) -> int:
    return len(text.split())


def count_markers(text: str, markers: list[str]) -> int:
    text_lower = f" {text.lower()} "
    return sum(text_lower.count(m.lower()) for m in markers)


HEDGING = ["maybe", "perhaps", "i think", "it seems", "i guess", "kind of",
           "sort of", "possibly", "might", "could be", "not sure"]
NEGATIVE = ["worried", "concerned", "anxious", "frustrated", "difficult",
            "struggle", "problem", "risk", "unfortunately", "afraid"]
POSITIVE = ["great", "wonderful", "love", "enjoy", "excited", "amazing",
            "fantastic", "happy", "excellent", "appreciate"]


@dataclass
class TurnData:
    turn: int
    prompt: str
    text: str
    word_count: int
    hedging: int
    negative: int
    positive: int
    ir_confidence: float
    ir_tone: str
    ir_verbosity: str
    ir_directness: float
    ir_disclosure: float


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# =============================================================================
# Test 1: Fatigue → shorter responses over many turns
# =============================================================================

def test_fatigue_text_shortening() -> tuple[list[TurnData], CheckResult]:
    """After many turns, fatigue should produce observably shorter text."""
    print("  Test 1: Fatigue → text shortening")
    engine = _build_engine("Fatigue Test", conscientiousness=0.3)

    neutral_prompts = [
        "What do you think about modern technology?",
        "Tell me about your views on education.",
        "How do you feel about travel?",
        "What's your take on social media?",
        "Do you have thoughts on work-life balance?",
        "What about the future of cities?",
        "How do you see entertainment changing?",
        "What matters to you in food culture?",
        "Your thoughts on community involvement?",
        "How do you approach learning new skills?",
        "What do you think about environmental policy?",
        "Any views on the role of art in society?",
    ]

    turns: list[TurnData] = []
    for i, prompt in enumerate(neutral_prompts):
        result = engine.chat(prompt)
        td = TurnData(
            turn=i + 1, prompt=prompt[:50], text=result.text,
            word_count=word_count(result.text),
            hedging=count_markers(result.text, HEDGING),
            negative=count_markers(result.text, NEGATIVE),
            positive=count_markers(result.text, POSITIVE),
            ir_confidence=result.ir.response_structure.confidence,
            ir_tone=result.ir.communication_style.tone.value,
            ir_verbosity=result.ir.communication_style.verbosity.value,
            ir_directness=result.ir.communication_style.directness,
            ir_disclosure=result.ir.knowledge_disclosure.disclosure_level,
        )
        turns.append(td)
        print(f"    Turn {i+1:>2}: {td.word_count:>4} words, verb={td.ir_verbosity}, conf={td.ir_confidence:.3f}")

    # Compare first 3 turns vs last 3 turns
    early_wc = sum(t.word_count for t in turns[:3]) / 3
    late_wc = sum(t.word_count for t in turns[-3:]) / 3

    passed = late_wc < early_wc
    detail = f"Early avg: {early_wc:.0f} words, Late avg: {late_wc:.0f} words (diff: {early_wc - late_wc:.0f})"
    if not passed:
        detail += " — late turns NOT shorter"

    return turns, CheckResult("fatigue_text_shortening", passed, detail)


# =============================================================================
# Test 2: Stance consistency on revisited topic
# =============================================================================

def test_stance_consistency() -> tuple[list[TurnData], CheckResult]:
    """Same topic revisited should produce consistent opinions in text."""
    print("  Test 2: Stance consistency on topic revisit")
    engine = _build_engine("Stance Test", openness=0.3)  # low-O = more rigid

    prompts = [
        "What do you think about AI in healthcare?",            # Topic A
        "How do you feel about remote work?",                    # Topic B
        "What's your view on sustainable energy?",               # Topic C
        "Going back to AI in healthcare — any new thoughts?",    # Topic A revisit
    ]

    turns: list[TurnData] = []
    for i, prompt in enumerate(prompts):
        result = engine.chat(prompt)
        td = TurnData(
            turn=i + 1, prompt=prompt[:50], text=result.text,
            word_count=word_count(result.text),
            hedging=count_markers(result.text, HEDGING),
            negative=count_markers(result.text, NEGATIVE),
            positive=count_markers(result.text, POSITIVE),
            ir_confidence=result.ir.response_structure.confidence,
            ir_tone=result.ir.communication_style.tone.value,
            ir_verbosity=result.ir.communication_style.verbosity.value,
            ir_directness=result.ir.communication_style.directness,
            ir_disclosure=result.ir.knowledge_disclosure.disclosure_level,
        )
        turns.append(td)
        print(f"    Turn {i+1}: {prompt[:50]}...")
        print(f"           {td.text[:100]}...")

    # Check stance IR matches between turn 1 and turn 4
    stance_1 = turns[0].text[:200].lower()
    stance_4 = turns[3].text[:200].lower()

    # They should share key phrases or sentiment direction
    # Check IR stance is identical (cache hit)
    ir_match = (engine._planner._prior_snapshot is not None)  # basic check that pipeline ran

    # Simple: check some shared content words (not exact match — LLM won't repeat verbatim)
    words_1 = set(stance_1.split()) - {"the", "a", "an", "is", "in", "to", "and", "of", "i", "it", "that", "this"}
    words_4 = set(stance_4.split()) - {"the", "a", "an", "is", "in", "to", "and", "of", "i", "it", "that", "this"}
    overlap = words_1 & words_4
    overlap_ratio = len(overlap) / max(len(words_1), 1)

    passed = overlap_ratio > 0.15  # At least 15% word overlap
    detail = f"Turn 1 vs Turn 4 word overlap: {overlap_ratio:.0%} ({len(overlap)} shared words)"

    return turns, CheckResult("stance_consistency", passed, detail)


# =============================================================================
# Test 3: High-N persona under escalating stress → text changes
# =============================================================================

def test_stress_text_markers() -> tuple[list[TurnData], CheckResult]:
    """High-N persona should show more hedging/negative markers under stress."""
    print("  Test 3: Stress → text marker changes (high-N persona)")
    engine = _build_engine("Stress Test", neuroticism=0.85, agreeableness=0.7)

    prompts = [
        # Calm start
        "What do you enjoy most about your work?",
        "Tell me about a hobby you like.",
        # Escalating stress
        "I need to challenge something you said — I think you're wrong about that.",
        "Actually, your entire approach seems flawed. Can you defend it?",
        "You don't seem very confident. Are you sure you know what you're talking about?",
    ]

    turns: list[TurnData] = []
    for i, prompt in enumerate(prompts):
        result = engine.chat(prompt)
        td = TurnData(
            turn=i + 1, prompt=prompt[:50], text=result.text,
            word_count=word_count(result.text),
            hedging=count_markers(result.text, HEDGING),
            negative=count_markers(result.text, NEGATIVE),
            positive=count_markers(result.text, POSITIVE),
            ir_confidence=result.ir.response_structure.confidence,
            ir_tone=result.ir.communication_style.tone.value,
            ir_verbosity=result.ir.communication_style.verbosity.value,
            ir_directness=result.ir.communication_style.directness,
            ir_disclosure=result.ir.knowledge_disclosure.disclosure_level,
        )
        turns.append(td)
        print(f"    Turn {i+1}: hedge={td.hedging}, neg={td.negative}, pos={td.positive}, "
              f"tone={td.ir_tone}, conf={td.ir_confidence:.3f}")

    # Compare calm (turns 1-2) vs stressed (turns 4-5)
    calm_hedging = sum(t.hedging for t in turns[:2])
    stress_hedging = sum(t.hedging for t in turns[3:5])
    calm_negative = sum(t.negative for t in turns[:2])
    stress_negative = sum(t.negative for t in turns[3:5])

    # At least one signal should increase
    hedging_increased = stress_hedging > calm_hedging
    negative_increased = stress_negative > calm_negative

    passed = hedging_increased or negative_increased
    detail = (f"Calm hedging={calm_hedging}, Stress hedging={stress_hedging} | "
              f"Calm neg={calm_negative}, Stress neg={stress_negative}")
    if not passed:
        detail += " — no observable text change under stress"

    return turns, CheckResult("stress_text_markers", passed, detail)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Run with: kv run -- python3 -m eval.end_to_end_conversation")
        return 1

    print("\n" + "=" * 80)
    print("  END-TO-END MULTI-TURN CONVERSATION VALIDATION")
    print("  (Real LLM, real state accumulation, real text analysis)")
    print("=" * 80 + "\n")

    all_results: list[CheckResult] = []
    all_turns: dict[str, list[TurnData]] = {}

    # Test 1: Fatigue
    turns1, result1 = test_fatigue_text_shortening()
    all_results.append(result1)
    all_turns["fatigue"] = turns1
    print(f"  → {'PASS' if result1.passed else 'FAIL'}: {result1.detail}\n")

    # Test 2: Stance
    turns2, result2 = test_stance_consistency()
    all_results.append(result2)
    all_turns["stance"] = turns2
    print(f"  → {'PASS' if result2.passed else 'FAIL'}: {result2.detail}\n")

    # Test 3: Stress
    turns3, result3 = test_stress_text_markers()
    all_results.append(result3)
    all_turns["stress"] = turns3
    print(f"  → {'PASS' if result3.passed else 'FAIL'}: {result3.detail}\n")

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print("=" * 80)
    print(f"  END-TO-END RESULT: {passed}/{total} conversation tests passed")
    for r in all_results:
        marker = "  " if r.passed else ">>"
        status = "PASS" if r.passed else "FAIL"
        print(f"  {marker} [{status}] {r.name}: {r.detail}")
    print("=" * 80)

    # Save report
    report = {
        "tests": [
            {"name": r.name, "passed": r.passed, "detail": r.detail}
            for r in all_results
        ],
        "conversations": {
            name: [
                {
                    "turn": t.turn, "prompt": t.prompt,
                    "word_count": t.word_count, "hedging": t.hedging,
                    "negative": t.negative, "positive": t.positive,
                    "ir_confidence": t.ir_confidence, "ir_tone": t.ir_tone,
                    "ir_verbosity": t.ir_verbosity,
                    "text_preview": t.text[:200],
                }
                for t in turns
            ]
            for name, turns in all_turns.items()
        },
        "summary": {"passed": passed, "total": total},
    }
    report_path = Path(__file__).parent / "e2e_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report: {report_path}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
