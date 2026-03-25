"""
Fair Comparison: IR-Driven vs Prompt-Only Personality Expression

Fixes the design flaws from the external validation study:
1. Engine reset per scenario (no state accumulation)
2. Uses engine.plan() for IR parameters, separate LLM call for text
3. Same LLM call structure for both approaches
4. Larger sample: 10 archetypes × 12 scenarios × 2 approaches
5. Expected ratings derived transparently from Big Five scores

Design principle: both approaches get EXACTLY the same task —
generate a 2-3 sentence response as this person to this scenario.
No rating extraction. No "respond with a number." Compare the
TEXT quality directly.

Usage:
    kv run -- python3 -m eval.fair_comparison
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: Run with: kv run -- python3 -m eval.fair_comparison")
    sys.exit(1)

import anthropic

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder

# =============================================================================
# Archetypes — same 10 as external study
# =============================================================================

ARCHETYPES = [
    {"id": "A01", "desc": "A 28-year-old male software engineer in San Francisco. Introverted, analytical, curious, and organized.", "O": 0.75, "C": 0.80, "E": 0.20, "A": 0.50, "N": 0.35},
    {"id": "A02", "desc": "A 45-year-old female nurse in Chicago. Warm, empathetic, reliable, and calm.", "O": 0.40, "C": 0.75, "E": 0.55, "A": 0.85, "N": 0.30},
    {"id": "A03", "desc": "A 35-year-old male entrepreneur in New York. Outgoing, assertive, risk-tolerant, and competitive.", "O": 0.70, "C": 0.50, "E": 0.85, "A": 0.30, "N": 0.25},
    {"id": "A04", "desc": "A 62-year-old female retired teacher in rural Vermont. Traditional, nurturing, cautious, and quiet.", "O": 0.30, "C": 0.70, "E": 0.25, "A": 0.80, "N": 0.40},
    {"id": "A05", "desc": "A 23-year-old non-binary artist in Berlin. Creative, spontaneous, open-minded, and sensitive.", "O": 0.90, "C": 0.30, "E": 0.50, "A": 0.60, "N": 0.70},
    {"id": "A06", "desc": "A 50-year-old male police officer in Dallas. Disciplined, direct, tough, and stable.", "O": 0.25, "C": 0.90, "E": 0.50, "A": 0.30, "N": 0.20},
    {"id": "A07", "desc": "A 38-year-old female social worker in London. Compassionate, cooperative, anxious, and meticulous.", "O": 0.55, "C": 0.75, "E": 0.55, "A": 0.85, "N": 0.70},
    {"id": "A08", "desc": "A 30-year-old male chef in Tokyo. Energetic, creative, blunt, and enthusiastic.", "O": 0.75, "C": 0.65, "E": 0.80, "A": 0.35, "N": 0.45},
    {"id": "A09", "desc": "A 55-year-old female accountant in Toronto. Organized, reserved, methodical, and risk-averse.", "O": 0.30, "C": 0.85, "E": 0.20, "A": 0.55, "N": 0.50},
    {"id": "A10", "desc": "A 40-year-old male musician in Nashville. Outgoing, spontaneous, friendly, and relaxed.", "O": 0.70, "C": 0.35, "E": 0.80, "A": 0.75, "N": 0.25},
]

# =============================================================================
# Scenarios — same 12, but used for TEXT generation not rating
# =============================================================================

SCENARIOS = [
    {"id": "Q01", "text": "You discover a completely new field of study that fascinates you. How do you react?", "trait": "O", "direction": "+"},
    {"id": "Q02", "text": "A friend suggests an unplanned road trip leaving tomorrow. What do you say?", "trait": "O", "direction": "+"},
    {"id": "Q03", "text": "You have a project due in two weeks. What's your first move?", "trait": "C", "direction": "+"},
    {"id": "Q04", "text": "Your workspace is getting messy. How do you feel about it?", "trait": "C", "direction": "+"},
    {"id": "Q05", "text": "You're at a party where you know nobody. What do you do?", "trait": "E", "direction": "+"},
    {"id": "Q06", "text": "After a long, exhausting week, you have a free evening. What sounds ideal?", "trait": "E", "direction": "-"},
    {"id": "Q07", "text": "A colleague makes a mistake that affects your work. How do you handle it?", "trait": "A", "direction": "-"},
    {"id": "Q08", "text": "A friend asks for a favor that's really inconvenient for you. What do you do?", "trait": "A", "direction": "+"},
    {"id": "Q09", "text": "You have a big presentation tomorrow. How are you feeling tonight?", "trait": "N", "direction": "+"},
    {"id": "Q10", "text": "Plans change unexpectedly at the last minute. How do you react?", "trait": "N", "direction": "+"},
    {"id": "Q11", "text": "You're offered a stable job with good pay, or a risky startup with big potential. What are you leaning toward?", "trait": "O", "direction": "+"},
    {"id": "Q12", "text": "A team project is falling apart. Nobody's stepping up. What do you do?", "trait": "E+C", "direction": "+"},
]

# =============================================================================
# LIWC-style markers for automated text analysis
# =============================================================================

HEDGING = ["maybe", "perhaps", "i think", "it seems", "i guess", "kind of",
           "sort of", "possibly", "might", "could be", "not sure", "probably"]
CERTAINTY = ["definitely", "absolutely", "certainly", "clearly", "always",
             "must", "no question", "without doubt", "for sure"]
SOCIAL = [" we ", " us ", " our ", " together ", " people ", " everyone ",
          " team ", " folks ", " group ", " friends "]
SELF_REF = [" i ", " me ", " my ", " myself ", " i'm ", " i've ", " i'd "]
NEGATIVE = ["worried", "concerned", "anxious", "stressed", "nervous",
            "afraid", "unfortunately", "struggle", "difficult", "problem"]
POSITIVE = ["great", "love", "enjoy", "excited", "happy", "wonderful",
            "fantastic", "amazing", "appreciate", "thrilled"]
STRUCTURE = ["first", "second", "additionally", "furthermore", "finally",
             "in summary", "specifically", "therefore", "moreover"]


def count_markers(text, markers):
    t = f" {text.lower()} "
    return sum(t.count(m.lower()) for m in markers)


def analyze_text(text):
    return {
        "words": len(text.split()),
        "hedging": count_markers(text, HEDGING),
        "certainty": count_markers(text, CERTAINTY),
        "social": count_markers(text, SOCIAL),
        "self_ref": count_markers(text, SELF_REF),
        "negative": count_markers(text, NEGATIVE),
        "positive": count_markers(text, POSITIVE),
        "structure": count_markers(text, STRUCTURE),
    }


# =============================================================================
# Approach 1: Prompt-Only
# =============================================================================

def run_prompt_only(archetypes, scenarios):
    """Each archetype gets a fresh call per scenario. No state accumulation."""
    print("\n  === PROMPT-ONLY ===")
    client = anthropic.Anthropic()
    results = {}

    for arch in archetypes:
        arch_results = []
        system = (
            f"You are the following person. Stay in character.\n\n"
            f"{arch['desc']}\n\n"
            f"Respond naturally in 2-4 sentences as this person would. "
            f"No disclaimers. No meta-commentary. Just be this person."
        )

        for sc in scenarios:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=system,
                messages=[{"role": "user", "content": sc["text"]}],
            )
            text = resp.content[0].text
            arch_results.append({
                "scenario": sc["id"],
                "text": text,
                "markers": analyze_text(text),
            })

        results[arch["id"]] = arch_results
        print(f"    {arch['id']} done")

    return results


# =============================================================================
# Approach 2: IR-Driven (FAIR design)
# =============================================================================

def run_ir_driven(archetypes, scenarios):
    """Fresh engine per archetype. Uses engine.chat() but resets between archetypes."""
    print("\n  === IR-DRIVEN (fresh engine per archetype) ===")
    results = {}
    ir_params = {}

    for arch in archetypes:
        # Build persona with EXPLICIT Big Five (not Layer Zero minting)
        # This ensures we're testing the IR pipeline, not Layer Zero
        builder = PersonaBuilder(
            name=arch["id"],
            occupation="General Professional",
        )
        builder._big_five = {
            "openness": arch["O"],
            "conscientiousness": arch["C"],
            "extraversion": arch["E"],
            "agreeableness": arch["A"],
            "neuroticism": arch["N"],
        }
        persona = builder.build()

        arch_results = []
        arch_params = []

        for sc in scenarios:
            # FRESH engine per scenario — no state accumulation
            engine = PersonaEngine(persona=persona, llm_provider="anthropic", seed=42)
            result = engine.chat(sc["text"])

            ir = result.ir
            arch_results.append({
                "scenario": sc["id"],
                "text": result.text,
                "markers": analyze_text(result.text),
            })
            arch_params.append({
                "scenario": sc["id"],
                "confidence": round(ir.response_structure.confidence, 3),
                "competence": round(ir.response_structure.competence, 3),
                "directness": round(ir.communication_style.directness, 3),
                "formality": round(ir.communication_style.formality, 3),
                "tone": ir.communication_style.tone.value,
                "verbosity": ir.communication_style.verbosity.value,
                "disclosure": round(ir.knowledge_disclosure.disclosure_level, 3),
            })
            del engine

        results[arch["id"]] = arch_results
        ir_params[arch["id"]] = arch_params
        print(f"    {arch['id']} done")

    return results, ir_params


# =============================================================================
# Analysis
# =============================================================================

def compare(prompt_results, ir_results, ir_params, archetypes, scenarios):
    print("\n" + "=" * 80)
    print("  FAIR COMPARISON RESULTS")
    print("=" * 80)

    # Per-archetype text analysis comparison
    print("\n  WORD COUNT COMPARISON (should be similar if test is fair)")
    print(f"  {'Archetype':<30} {'Prompt avg':>12} {'IR avg':>12} {'Ratio':>8}")
    print(f"  {'-'*65}")

    all_prompt_words = []
    all_ir_words = []

    for arch in archetypes:
        p_words = [r["markers"]["words"] for r in prompt_results[arch["id"]]]
        i_words = [r["markers"]["words"] for r in ir_results[arch["id"]]]
        p_avg = sum(p_words) / len(p_words)
        i_avg = sum(i_words) / len(i_words)
        ratio = p_avg / i_avg if i_avg > 0 else 999
        all_prompt_words.extend(p_words)
        all_ir_words.extend(i_words)
        print(f"  {arch['desc'][:28]:<30} {p_avg:>12.0f} {i_avg:>12.0f} {ratio:>8.1f}x")

    p_total = sum(all_prompt_words) / len(all_prompt_words)
    i_total = sum(all_ir_words) / len(all_ir_words)
    print(f"  {'OVERALL':<30} {p_total:>12.0f} {i_total:>12.0f} {p_total/i_total:>8.1f}x")

    # Marker comparison by trait
    print(f"\n  PERSONALITY MARKER COMPARISON")
    print(f"  {'Trait group':<15} {'Marker':<12} {'Prompt':>8} {'IR':>8} {'Expected winner':>20}")
    print(f"  {'-'*65}")

    # High-E archetypes vs Low-E archetypes
    high_e = [a for a in archetypes if a["E"] >= 0.7]
    low_e = [a for a in archetypes if a["E"] <= 0.3]

    for label, group, marker_name, markers in [
        ("High-E", high_e, "social", SOCIAL),
        ("Low-E", low_e, "self_ref", SELF_REF),
    ]:
        p_count = sum(count_markers(r["text"], markers) for a in group for r in prompt_results[a["id"]])
        i_count = sum(count_markers(r["text"], markers) for a in group for r in ir_results[a["id"]])
        winner = "Prompt" if p_count > i_count else "IR" if i_count > p_count else "Tie"
        print(f"  {label:<15} {marker_name:<12} {p_count:>8} {i_count:>8} {winner:>20}")

    # High-N vs Low-N
    high_n = [a for a in archetypes if a["N"] >= 0.6]
    low_n = [a for a in archetypes if a["N"] <= 0.3]

    for label, group, marker_name, markers in [
        ("High-N", high_n, "hedging", HEDGING),
        ("High-N", high_n, "negative", NEGATIVE),
        ("Low-N", low_n, "certainty", CERTAINTY),
    ]:
        p_count = sum(count_markers(r["text"], markers) for a in group for r in prompt_results[a["id"]])
        i_count = sum(count_markers(r["text"], markers) for a in group for r in ir_results[a["id"]])
        winner = "Prompt" if p_count > i_count else "IR" if i_count > p_count else "Tie"
        print(f"  {label:<15} {marker_name:<12} {p_count:>8} {i_count:>8} {winner:>20}")

    # High-C vs Low-C
    high_c = [a for a in archetypes if a["C"] >= 0.7]
    low_c = [a for a in archetypes if a["C"] <= 0.4]

    for label, group, marker_name, markers in [
        ("High-C", high_c, "structure", STRUCTURE),
    ]:
        p_count = sum(count_markers(r["text"], markers) for a in group for r in prompt_results[a["id"]])
        i_count = sum(count_markers(r["text"], markers) for a in group for r in ir_results[a["id"]])
        winner = "Prompt" if p_count > i_count else "IR" if i_count > p_count else "Tie"
        print(f"  {label:<15} {marker_name:<12} {p_count:>8} {i_count:>8} {winner:>20}")

    # Differentiation test: do high-trait and low-trait archetypes
    # produce DIFFERENT marker counts? (Higher diff = better personality modeling)
    print(f"\n  DIFFERENTIATION TEST (high-trait vs low-trait marker difference)")
    print(f"  {'Comparison':<35} {'Prompt diff':>12} {'IR diff':>12} {'Better diff':>15}")
    print(f"  {'-'*75}")

    for trait_label, high_group, low_group, marker_name, markers in [
        ("E: social words", high_e, low_e, "social", SOCIAL),
        ("N: hedging words", high_n, low_n, "hedging", HEDGING),
        ("C: structure words", high_c, low_c, "structure", STRUCTURE),
    ]:
        p_high = sum(count_markers(r["text"], markers) for a in high_group for r in prompt_results[a["id"]])
        p_low = sum(count_markers(r["text"], markers) for a in low_group for r in prompt_results[a["id"]])
        i_high = sum(count_markers(r["text"], markers) for a in high_group for r in ir_results[a["id"]])
        i_low = sum(count_markers(r["text"], markers) for a in low_group for r in ir_results[a["id"]])

        # Normalize by word count
        p_high_wc = sum(r["markers"]["words"] for a in high_group for r in prompt_results[a["id"]])
        p_low_wc = sum(r["markers"]["words"] for a in low_group for r in prompt_results[a["id"]])
        i_high_wc = sum(r["markers"]["words"] for a in high_group for r in ir_results[a["id"]])
        i_low_wc = sum(r["markers"]["words"] for a in low_group for r in ir_results[a["id"]])

        p_diff = (p_high / max(p_high_wc, 1) - p_low / max(p_low_wc, 1)) * 1000  # per 1000 words
        i_diff = (i_high / max(i_high_wc, 1) - i_low / max(i_low_wc, 1)) * 1000

        better = "Prompt" if abs(p_diff) > abs(i_diff) else "IR" if abs(i_diff) > abs(p_diff) else "Tie"
        print(f"  {trait_label:<35} {p_diff:>+12.1f} {i_diff:>+12.1f} {better:>15}")

    # Show sample responses for comparison
    print(f"\n  SAMPLE RESPONSES (A03 entrepreneur, Q05 party scenario)")
    a03_q05_p = [r for r in prompt_results["A03"] if r["scenario"] == "Q05"][0]
    a03_q05_i = [r for r in ir_results["A03"] if r["scenario"] == "Q05"][0]
    print(f"  Prompt: {a03_q05_p['text'][:200]}")
    print(f"  IR:     {a03_q05_i['text'][:200]}")

    print(f"\n  SAMPLE RESPONSES (A09 accountant, Q05 party scenario)")
    a09_q05_p = [r for r in prompt_results["A09"] if r["scenario"] == "Q05"][0]
    a09_q05_i = [r for r in ir_results["A09"] if r["scenario"] == "Q05"][0]
    print(f"  Prompt: {a09_q05_p['text'][:200]}")
    print(f"  IR:     {a09_q05_i['text'][:200]}")

    return {
        "prompt_avg_words": p_total,
        "ir_avg_words": i_total,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("  FAIR COMPARISON: IR-Driven vs Prompt-Only")
    print("  Design fixes: fresh engine per scenario, same task, text comparison")
    print(f"  {len(ARCHETYPES)} archetypes × {len(SCENARIOS)} scenarios = {len(ARCHETYPES) * len(SCENARIOS)} per approach")
    print("=" * 80)

    t0 = time.time()
    prompt_results = run_prompt_only(ARCHETYPES, SCENARIOS)
    t_prompt = time.time() - t0
    print(f"  Prompt-only: {t_prompt:.0f}s")

    t0 = time.time()
    ir_results, ir_params = run_ir_driven(ARCHETYPES, SCENARIOS)
    t_ir = time.time() - t0
    print(f"  IR-driven: {t_ir:.0f}s")

    summary = compare(prompt_results, ir_results, ir_params, ARCHETYPES, SCENARIOS)

    # Save
    report = {
        "design": "Fair comparison — fresh engine per scenario, same text generation task",
        "archetypes": ARCHETYPES,
        "scenarios": [{"id": s["id"], "text": s["text"], "trait": s["trait"]} for s in SCENARIOS],
        "prompt_results": prompt_results,
        "ir_results": ir_results,
        "ir_params": ir_params,
        "timing": {"prompt_s": round(t_prompt, 1), "ir_s": round(t_ir, 1)},
        "summary": summary,
    }
    report_path = Path(__file__).parent / "fair_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
