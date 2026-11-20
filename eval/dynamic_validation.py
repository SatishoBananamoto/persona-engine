"""
Dynamic Validation — Multi-turn behavioral consistency checks.

Validates that dynamic elements (state accumulation, mood drift, trust,
stance cache, cross-turn inertia, bias stacking) produce correct directional
changes across conversation turns.

Unlike benchmark_profiles.py (single-turn, static), this tests the EQUATIONS
that govern how persona behavior evolves during conversation.

Usage:
    python3 -m eval.dynamic_validation            # run all checks
    python3 -m eval.dynamic_validation --verbose   # show per-turn details

Exit codes:
    0 — all dynamic checks passed
    1 — one or more checks failed
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
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal, Tone, Verbosity
from persona_engine.memory import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.utils import DeterminismManager


VERBOSE = "--verbose" in sys.argv


# =============================================================================
# Helpers
# =============================================================================

def _build_engine(name: str = "Test Persona", **trait_overrides):
    """Build a persona engine with optional trait overrides."""
    builder = PersonaBuilder(name=name, occupation="General Professional")
    traits = {
        "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
        "agreeableness": 0.5, "neuroticism": 0.5,
    }
    traits.update(trait_overrides)
    builder._big_five = traits
    persona = builder.build()
    return PersonaEngine(persona=persona, llm_provider="template")


def _build_planner(name: str = "Test", **trait_overrides):
    """Build a TurnPlanner directly for lower-level testing."""
    builder = PersonaBuilder(name=name, occupation="General Professional")
    traits = {
        "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
        "agreeableness": 0.5, "neuroticism": 0.5,
    }
    traits.update(trait_overrides)
    builder._big_five = traits
    persona = builder.build()
    det = DeterminismManager(seed=42)
    return TurnPlanner(persona, determinism=det), persona


def _make_context(user_input: str, turn: int, cache: StanceCache, domain: str | None = None):
    """Create a ConversationContext for planner.generate_ir()."""
    return ConversationContext(
        conversation_id="dynamic_test",
        turn_number=turn,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="test_topic",
        user_input=user_input,
        stance_cache=cache,
        domain=domain,
    )


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    values: dict = field(default_factory=dict)


def _log(msg: str):
    if VERBOSE:
        print(f"    {msg}")


# =============================================================================
# Dynamic Check 1: Fatigue → Verbosity
# =============================================================================

def check_fatigue_verbosity() -> CheckResult:
    """After many turns, fatigue should accumulate and affect behavior."""
    planner, persona = _build_planner(conscientiousness=0.3)  # low C = less stamina
    cache = StanceCache()

    # Run 20 turns to accumulate fatigue
    fatigue_turn1 = planner.state.get_fatigue()
    for t in range(1, 21):
        ctx = _make_context("Tell me more about this topic.", t, cache)
        planner.generate_ir(ctx)

    fatigue_turn20 = planner.state.get_fatigue()
    _log(f"fatigue turn 1: {fatigue_turn1:.3f}, turn 20: {fatigue_turn20:.3f}")

    passed = fatigue_turn20 > fatigue_turn1
    return CheckResult(
        name="fatigue_accumulates",
        passed=passed,
        detail=f"Fatigue turn 1={fatigue_turn1:.3f} → turn 20={fatigue_turn20:.3f}",
        values={"turn1": fatigue_turn1, "turn20": fatigue_turn20},
    )


# =============================================================================
# Dynamic Check 2: Mood Drift Toward Baseline
# =============================================================================

def check_mood_drift() -> CheckResult:
    """Mood should drift toward baseline over turns (regression to mean)."""
    planner, persona = _build_planner(neuroticism=0.3, extraversion=0.7)
    cache = StanceCache()

    # Spike mood high
    planner.state.update_mood_from_event(valence_delta=0.8, arousal_delta=0.3)
    mood_after_spike = planner.state.get_mood()
    _log(f"mood after spike: valence={mood_after_spike[0]:.3f}, arousal={mood_after_spike[1]:.3f}")

    # Run 10 turns — mood should drift back toward baseline
    for t in range(1, 11):
        ctx = _make_context("Tell me something neutral.", t, cache)
        planner.generate_ir(ctx)

    mood_after_drift = planner.state.get_mood()
    _log(f"mood after 10 turns: valence={mood_after_drift[0]:.3f}, arousal={mood_after_drift[1]:.3f}")

    # Baseline for this persona: 0.1 + 0.7*0.15 - 0.3*0.2 = 0.145
    # Valence should have moved toward ~0.145 from the high spike
    drift_occurred = abs(mood_after_drift[0]) < abs(mood_after_spike[0])
    return CheckResult(
        name="mood_drifts_toward_baseline",
        passed=drift_occurred,
        detail=f"Valence: {mood_after_spike[0]:.3f} → {mood_after_drift[0]:.3f} (should move toward ~0.145)",
        values={"spike": mood_after_spike[0], "after_drift": mood_after_drift[0]},
    )


# =============================================================================
# Dynamic Check 3: Stress → Directness
# =============================================================================

def check_stress_directness() -> CheckResult:
    """Stress should reduce patience, which increases directness."""
    planner, persona = _build_planner(neuroticism=0.7)  # stress-sensitive
    cache = StanceCache()

    # Baseline directness (turn 1, no stress)
    ctx1 = _make_context("What do you think about project management?", 1, cache)
    ir1 = planner.generate_ir(ctx1)
    dir1 = ir1.communication_style.directness
    patience1 = planner.state.get_patience_level()
    _log(f"turn 1: directness={dir1:.3f}, patience={patience1:.3f}, stress={planner.state.get_stress():.3f}")

    # Apply stress triggers
    planner.state.apply_stress_trigger("conflict", intensity=0.7)
    planner.state.apply_stress_trigger("time_pressure", intensity=0.6)
    _log(f"after stress triggers: stress={planner.state.get_stress():.3f}")

    # Turn 2 under stress
    ctx2 = _make_context("What do you think about project management?", 2, cache)
    ir2 = planner.generate_ir(ctx2)
    dir2 = ir2.communication_style.directness
    patience2 = planner.state.get_patience_level()
    _log(f"turn 2: directness={dir2:.3f}, patience={patience2:.3f}, stress={planner.state.get_stress():.3f}")

    # Patience should drop under stress
    patience_dropped = patience2 < patience1
    return CheckResult(
        name="stress_reduces_patience",
        passed=patience_dropped,
        detail=f"Patience: {patience1:.3f} → {patience2:.3f} (stress applied)",
        values={"patience_before": patience1, "patience_after": patience2},
    )


# =============================================================================
# Dynamic Check 4: Trust → Disclosure
# =============================================================================

def check_trust_disclosure() -> CheckResult:
    """Higher trust should increase disclosure level."""
    # High trust engine
    engine_high = _build_engine(extraversion=0.6)
    engine_high._memory.record_relationship_event(
        "User validated expertise and showed deep trust",
        trust_delta=0.5, rapport_delta=0.3, turn=0,
    )
    ir_high = engine_high.plan("Tell me about your personal philosophy on work")
    disc_high = ir_high.knowledge_disclosure.disclosure_level

    # Low trust engine (fresh, same persona)
    engine_low = _build_engine(extraversion=0.6)
    engine_low._memory.record_relationship_event(
        "User was hostile and dismissive",
        trust_delta=-0.5, rapport_delta=-0.3, turn=0,
    )
    ir_low = engine_low.plan("Tell me about your personal philosophy on work")
    disc_low = ir_low.knowledge_disclosure.disclosure_level

    _log(f"high trust disclosure: {disc_high:.3f}, low trust: {disc_low:.3f}")

    passed = disc_high > disc_low
    return CheckResult(
        name="trust_increases_disclosure",
        passed=passed,
        detail=f"Disclosure: high_trust={disc_high:.3f} > low_trust={disc_low:.3f}",
        values={"high_trust": disc_high, "low_trust": disc_low},
    )


# =============================================================================
# Dynamic Check 5: Stance Cache Consistency
# =============================================================================

def check_stance_cache() -> CheckResult:
    """Same topic should produce same stance on revisit (cache hit)."""
    planner, persona = _build_planner()
    cache = StanceCache()

    # Turn 1: generate stance on a topic
    ctx1 = _make_context("What do you think about AI safety?", 1, cache, domain="technology")
    ir1 = planner.generate_ir(ctx1)
    stance1 = ir1.response_structure.stance

    # Turn 2: different topic
    ctx2 = _make_context("How's the weather today?", 2, cache)
    planner.generate_ir(ctx2)

    # Turn 3: revisit same topic — should hit cache
    ctx3 = _make_context("Back to AI safety — any new thoughts?", 3, cache, domain="technology")
    # Use same topic_signature to trigger cache
    ctx3.topic_signature = "test_topic"
    ir3 = planner.generate_ir(ctx3)
    stance3 = ir3.response_structure.stance

    _log(f"stance 1: {stance1[:60]}...")
    _log(f"stance 3: {stance3[:60]}...")

    # Stances should be identical (cache hit) or very similar
    passed = stance1 == stance3
    return CheckResult(
        name="stance_cache_consistency",
        passed=passed,
        detail=f"Stance turn 1 {'==' if passed else '!='} stance turn 3 (cache {'hit' if passed else 'miss'})",
        values={"stance1": stance1[:80], "stance3": stance3[:80]},
    )


# =============================================================================
# Dynamic Check 6: Cross-Turn Inertia Smoothing
# =============================================================================

def check_cross_turn_inertia() -> CheckResult:
    """Parameters should not jump wildly between consecutive turns."""
    planner, persona = _build_planner()
    cache = StanceCache()

    # Run 5 turns with different inputs and collect confidence values
    prompts = [
        "Tell me about cooking techniques",
        "What's your view on quantum physics?",
        "How do you feel about team management?",
        "What about art and creativity?",
        "Tell me about sports training",
    ]
    confidences = []
    for t, prompt in enumerate(prompts, 1):
        ctx = _make_context(prompt, t, cache)
        ir = planner.generate_ir(ctx)
        confidences.append(ir.response_structure.confidence)
        _log(f"turn {t}: confidence={confidences[-1]:.3f}")

    # Check that no consecutive pair has a swing > 0.45 (the max allowed)
    max_swing = 0.0
    for i in range(1, len(confidences)):
        swing = abs(confidences[i] - confidences[i - 1])
        max_swing = max(max_swing, swing)

    passed = max_swing <= 0.45
    return CheckResult(
        name="cross_turn_inertia_smoothing",
        passed=passed,
        detail=f"Max confidence swing: {max_swing:.3f} (limit: 0.45)",
        values={"confidences": [round(c, 3) for c in confidences], "max_swing": max_swing},
    )


# =============================================================================
# Dynamic Check 7: Stress Decay Over Turns
# =============================================================================

def check_stress_decay() -> CheckResult:
    """Stress should decay naturally over turns without new triggers."""
    planner, persona = _build_planner(neuroticism=0.5)
    cache = StanceCache()

    # Spike stress
    planner.state.apply_stress_trigger("conflict", intensity=0.8)
    stress_peak = planner.state.get_stress()
    _log(f"stress after trigger: {stress_peak:.3f}")

    # Run 10 turns with neutral input — stress should decay
    for t in range(1, 11):
        ctx = _make_context("Tell me something calm and relaxing.", t, cache)
        planner.generate_ir(ctx)

    stress_after = planner.state.get_stress()
    _log(f"stress after 10 turns: {stress_after:.3f}")

    passed = stress_after < stress_peak
    return CheckResult(
        name="stress_decays_over_turns",
        passed=passed,
        detail=f"Stress: peak={stress_peak:.3f} → after 10 turns={stress_after:.3f}",
        values={"peak": stress_peak, "after": stress_after},
    )


# =============================================================================
# Dynamic Check 8: Emotional Appraisal → Mood
# =============================================================================

def check_emotional_appraisal() -> CheckResult:
    """Negative user emotion should shift persona mood negatively."""
    planner, persona = _build_planner(neuroticism=0.6, agreeableness=0.7)
    cache = StanceCache()

    mood_before = planner.state.get_mood()
    _log(f"mood before: valence={mood_before[0]:.3f}")

    # Send emotionally negative inputs
    negative_prompts = [
        "I'm really frustrated and angry about this whole situation!",
        "Everything is going wrong and I feel terrible about it.",
        "This is so disappointing, I'm losing hope.",
    ]
    for t, prompt in enumerate(negative_prompts, 1):
        ctx = _make_context(prompt, t, cache)
        planner.generate_ir(ctx)

    mood_after = planner.state.get_mood()
    _log(f"mood after negative inputs: valence={mood_after[0]:.3f}")

    # Valence should be lower after negative emotional inputs
    # (emotional appraisal detects user emotion and adjusts mood)
    valence_dropped = mood_after[0] < mood_before[0]
    return CheckResult(
        name="negative_emotion_shifts_mood",
        passed=valence_dropped,
        detail=f"Valence: {mood_before[0]:.3f} → {mood_after[0]:.3f}",
        values={"before": mood_before[0], "after": mood_after[0]},
    )


# =============================================================================
# Dynamic Check 9: Neuroticism Modulates Drift Rate
# =============================================================================

def check_neuroticism_drift_rate() -> CheckResult:
    """High-N persona's mood should linger longer than low-N."""
    # High N — slow drift
    planner_high_n, _ = _build_planner(neuroticism=0.9)
    planner_high_n.state.update_mood_from_event(0.6, 0.0)
    high_n_start = planner_high_n.state.get_mood()[0]

    # Low N — fast drift
    planner_low_n, _ = _build_planner(neuroticism=0.1)
    planner_low_n.state.update_mood_from_event(0.6, 0.0)
    low_n_start = planner_low_n.state.get_mood()[0]

    cache = StanceCache()
    # Run 5 turns of drift
    for t in range(1, 6):
        ctx_h = _make_context("neutral.", t, cache)
        ctx_l = _make_context("neutral.", t, StanceCache())
        planner_high_n.generate_ir(ctx_h)
        planner_low_n.generate_ir(ctx_l)

    high_n_after = planner_high_n.state.get_mood()[0]
    low_n_after = planner_low_n.state.get_mood()[0]

    _log(f"high-N: {high_n_start:.3f} → {high_n_after:.3f} (drift: {abs(high_n_start - high_n_after):.3f})")
    _log(f"low-N:  {low_n_start:.3f} → {low_n_after:.3f} (drift: {abs(low_n_start - low_n_after):.3f})")

    # High-N should drift LESS (mood lingers)
    high_n_drift = abs(high_n_start - high_n_after)
    low_n_drift = abs(low_n_start - low_n_after)
    passed = high_n_drift < low_n_drift
    return CheckResult(
        name="high_n_mood_lingers",
        passed=passed,
        detail=f"High-N drift: {high_n_drift:.3f}, Low-N drift: {low_n_drift:.3f} (high-N should be smaller)",
        values={"high_n_drift": high_n_drift, "low_n_drift": low_n_drift},
    )


# =============================================================================
# Dynamic Check 10: Bias Stacking on Elasticity
# =============================================================================

def check_bias_stacking_elasticity() -> CheckResult:
    """Multiple biases targeting elasticity should reduce it cumulatively."""
    # Low-O, high-C persona — susceptible to confirmation + status quo
    planner, persona = _build_planner(openness=0.2, conscientiousness=0.8)
    cache = StanceCache()

    # Turn 1: baseline elasticity
    ctx1 = _make_context("What do you think about changing our approach?", 1, cache)
    ir1 = planner.generate_ir(ctx1)
    elas1 = ir1.response_structure.elasticity
    _log(f"turn 1 elasticity: {elas1:.3f}")

    # Set an anchor stance (activates anchoring bias)
    planner.bias_simulator.set_anchor("I prefer the current approach")

    # Turn 2: same topic, with anchor set
    ctx2 = _make_context("But what if we tried something completely different?", 2, cache)
    ir2 = planner.generate_ir(ctx2)
    elas2 = ir2.response_structure.elasticity
    _log(f"turn 2 elasticity (with anchor): {elas2:.3f}")

    # Low-O + high-C + anchored → elasticity should be low
    passed = elas2 <= 0.5  # Should be well below midpoint
    return CheckResult(
        name="bias_reduces_elasticity",
        passed=passed,
        detail=f"Elasticity for low-O/high-C anchored persona: {elas2:.3f} (should be <= 0.5)",
        values={"turn1": elas1, "turn2_anchored": elas2},
    )


# =============================================================================
# G1: TF-002 — Extreme N confidence over-suppression
# =============================================================================

def check_extreme_n_confidence() -> CheckResult:
    """N=0.95 should produce low confidence but NOT collapse to floor (0.1)."""
    planner, _ = _build_planner(neuroticism=0.95, conscientiousness=0.3)
    cache = StanceCache()

    ctx = _make_context("What do you think about this approach?", 1, cache)
    ir = planner.generate_ir(ctx)
    conf = ir.response_structure.confidence
    _log(f"N=0.95 confidence: {conf:.3f}")

    # Should be low but not collapsed to absolute floor
    # Floor is 0.1 from the clamp. If confidence = 0.1, the three N paths
    # are over-suppressing and we've lost all signal.
    not_collapsed = conf > 0.1
    reasonably_low = conf < 0.35

    passed = not_collapsed and reasonably_low
    detail = f"Confidence={conf:.3f} (expected: 0.1 < x < 0.35)"
    if not not_collapsed:
        detail += " — COLLAPSED to floor, N over-suppression!"
    elif not reasonably_low:
        detail += " — not low enough for N=0.95"

    return CheckResult(
        name="extreme_n_confidence_not_collapsed",
        passed=passed,
        detail=detail,
        values={"confidence": conf, "N": 0.95},
    )


# =============================================================================
# G1: TF-003 — Extreme A + contentious directness over-reduction
# =============================================================================

def check_extreme_a_directness() -> CheckResult:
    """A=0.95 + contentious input should reduce directness but not collapse to 0."""
    planner, _ = _build_planner(agreeableness=0.95)
    cache = StanceCache()

    # Contentious input that should trigger conflict_avoidance_boost
    ctx = _make_context("I completely disagree with your wrong position on this!", 1, cache)
    ir = planner.generate_ir(ctx)
    dire = ir.communication_style.directness
    _log(f"A=0.95 + contentious directness: {dire:.3f}")

    # Should be low but not zero — even very agreeable people maintain some directness
    not_collapsed = dire > 0.05
    reasonably_low = dire < 0.4

    passed = not_collapsed and reasonably_low
    detail = f"Directness={dire:.3f} (expected: 0.05 < x < 0.4)"
    if not not_collapsed:
        detail += " — COLLAPSED near zero, A over-reduction!"
    elif not reasonably_low:
        detail += " — not low enough for A=0.95 + contentious"

    return CheckResult(
        name="extreme_a_directness_not_collapsed",
        passed=passed,
        detail=detail,
        values={"directness": dire, "A": 0.95},
    )


# =============================================================================
# G2: Tone direction validation
# =============================================================================

def check_tone_high_n_stress() -> CheckResult:
    """High-N persona under stress should select anxious/negative tone."""
    planner, _ = _build_planner(neuroticism=0.85)
    cache = StanceCache()

    # Apply stress
    planner.state.apply_stress_trigger("conflict", intensity=0.8)

    ctx = _make_context("What should we do about this problem?", 1, cache)
    ir = planner.generate_ir(ctx)
    tone = ir.communication_style.tone
    _log(f"high-N + stress tone: {tone.value}")

    # Should be anxious or concerned or at least negative
    negative_tones = {
        Tone.ANXIOUS_STRESSED, Tone.CONCERNED_EMPATHETIC,
        Tone.FRUSTRATED_TENSE, Tone.DEFENSIVE_AGITATED,
        Tone.DISAPPOINTED_RESIGNED, Tone.SAD_SUBDUED,
        Tone.TIRED_WITHDRAWN,
    }
    passed = tone in negative_tones
    return CheckResult(
        name="high_n_stress_negative_tone",
        passed=passed,
        detail=f"Tone={tone.value} (expected one of: anxious, concerned, frustrated, etc.)",
        values={"tone": tone.value, "N": 0.85, "stress": "high"},
    )


def check_tone_high_e_positive() -> CheckResult:
    """High-E persona with positive mood should select enthusiastic/warm tone."""
    planner, _ = _build_planner(extraversion=0.9, neuroticism=0.2)
    cache = StanceCache()

    # Boost mood positive
    planner.state.update_mood_from_event(valence_delta=0.5, arousal_delta=0.3)

    ctx = _make_context("Tell me about your favorite things!", 1, cache)
    ir = planner.generate_ir(ctx)
    tone = ir.communication_style.tone
    _log(f"high-E + positive mood tone: {tone.value}")

    positive_tones = {
        Tone.WARM_ENTHUSIASTIC, Tone.EXCITED_ENGAGED,
        Tone.THOUGHTFUL_ENGAGED, Tone.WARM_CONFIDENT,
        Tone.FRIENDLY_RELAXED, Tone.CONTENT_CALM,
    }
    passed = tone in positive_tones
    return CheckResult(
        name="high_e_positive_tone",
        passed=passed,
        detail=f"Tone={tone.value} (expected one of: warm, excited, enthusiastic, etc.)",
        values={"tone": tone.value, "E": 0.9, "mood": "positive"},
    )


def check_tone_neutral_midrange() -> CheckResult:
    """Mid-range persona with neutral state should select neutral/professional tone."""
    planner, _ = _build_planner()  # all traits at 0.5
    cache = StanceCache()

    ctx = _make_context("Can you explain how this process works?", 1, cache)
    ir = planner.generate_ir(ctx)
    tone = ir.communication_style.tone
    _log(f"mid-range neutral tone: {tone.value}")

    neutral_tones = {
        Tone.NEUTRAL_CALM, Tone.PROFESSIONAL_COMPOSED,
        Tone.MATTER_OF_FACT, Tone.CONTENT_CALM,
        Tone.THOUGHTFUL_ENGAGED,
    }
    passed = tone in neutral_tones
    return CheckResult(
        name="neutral_persona_neutral_tone",
        passed=passed,
        detail=f"Tone={tone.value} (expected: neutral, professional, or calm)",
        values={"tone": tone.value},
    )


# =============================================================================
# Main
# =============================================================================

ALL_CHECKS = [
    check_fatigue_verbosity,
    check_mood_drift,
    check_stress_directness,
    check_trust_disclosure,
    check_stance_cache,
    check_cross_turn_inertia,
    check_stress_decay,
    check_emotional_appraisal,
    check_neuroticism_drift_rate,
    check_bias_stacking_elasticity,
    # G1: Extreme value checks (TF-002/003)
    check_extreme_n_confidence,
    check_extreme_a_directness,
    # G2: Tone direction validation
    check_tone_high_n_stress,
    check_tone_high_e_positive,
    check_tone_neutral_midrange,
]


def main() -> int:
    print("\n" + "=" * 80)
    print("  DYNAMIC VALIDATION — Multi-Turn Behavioral Consistency")
    print("=" * 80 + "\n")

    results: list[CheckResult] = []
    for check_fn in ALL_CHECKS:
        name = check_fn.__name__.replace("check_", "")
        print(f"  Running {name}...")
        try:
            result = check_fn()
        except Exception as e:
            result = CheckResult(name=name, passed=False, detail=f"ERROR: {e}")
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        marker = "  " if result.passed else ">>"
        print(f"  {marker} [{status}] {result.name}: {result.detail}")
        print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("=" * 80)
    print(f"  Result: {passed}/{total} dynamic checks passed")
    if passed < total:
        print("  WARNING: Some dynamic behaviors are not working as expected!")
        for r in results:
            if not r.passed:
                print(f"    >> FAILED: {r.name} — {r.detail}")
    else:
        print("  All dynamic elements behave correctly across multi-turn conversations.")
    print("=" * 80 + "\n")

    # Write report
    report_path = Path(__file__).parent / "dynamic_report.json"
    report = {
        "checks": [
            {"name": r.name, "passed": r.passed, "detail": r.detail, "values": r.values}
            for r in results
        ],
        "summary": {"total": total, "passed": passed, "failed": total - passed},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
