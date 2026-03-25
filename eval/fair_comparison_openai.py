"""
Fair Comparison (OpenAI): IR-Driven vs Prompt-Only

Same design as fair_comparison.py but uses OpenAI GPT-4o instead of Claude Sonnet.
Tests whether IR advantages are model-specific or generalizable.

Usage:
    kv run -- python3 -m eval.fair_comparison_openai
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Run with: kv run -- python3 -m eval.fair_comparison_openai")
    sys.exit(1)

import openai
from empath import Empath

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder

lexicon = Empath()

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

SCENARIOS = [
    {"id": "Q01", "text": "You discover a completely new field of study that fascinates you. How do you react?", "trait": "O"},
    {"id": "Q02", "text": "A friend suggests an unplanned road trip leaving tomorrow. What do you say?", "trait": "O"},
    {"id": "Q03", "text": "You have a project due in two weeks. What's your first move?", "trait": "C"},
    {"id": "Q04", "text": "Your workspace is getting messy. How do you feel about it?", "trait": "C"},
    {"id": "Q05", "text": "You're at a party where you know nobody. What do you do?", "trait": "E"},
    {"id": "Q06", "text": "After a long, exhausting week, you have a free evening. What sounds ideal?", "trait": "E"},
    {"id": "Q07", "text": "A colleague makes a mistake that affects your work. How do you handle it?", "trait": "A"},
    {"id": "Q08", "text": "A friend asks for a favor that's really inconvenient for you. What do you do?", "trait": "A"},
    {"id": "Q09", "text": "You have a big presentation tomorrow. How are you feeling tonight?", "trait": "N"},
    {"id": "Q10", "text": "Plans change unexpectedly at the last minute. How do you react?", "trait": "N"},
    {"id": "Q11", "text": "You're offered a stable job with good pay, or a risky startup with big potential. What are you leaning toward?", "trait": "O"},
    {"id": "Q12", "text": "A team project is falling apart. Nobody's stepping up. What do you do?", "trait": "E+C"},
]

TRAIT_CATEGORIES = {
    'E': {'high': ['positive_emotion', 'friends', 'communication', 'cheerfulness', 'warmth', 'speaking', 'optimism']},
    'N': {'high': ['nervousness', 'negative_emotion', 'sadness', 'emotional', 'weakness']},
    'A': {'high': ['politeness', 'help', 'warmth', 'trust', 'friends', 'positive_emotion']},
    'C': {'high': ['order', 'achievement', 'work']},
    'O': {'high': ['philosophy', 'science', 'art', 'reading']},
}


def run_prompt_only():
    print("\n  === PROMPT-ONLY (OpenAI GPT-4o) ===")
    client = openai.OpenAI()
    results = {}

    for arch in ARCHETYPES:
        arch_results = []
        system = (
            f"You are the following person. Stay in character.\n\n"
            f"{arch['desc']}\n\n"
            f"Respond naturally in 2-4 sentences as this person would. "
            f"No disclaimers. No meta-commentary. Just be this person."
        )

        for sc in SCENARIOS:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=200,
                    timeout=30,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": sc["text"]},
                    ],
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"      ERROR {arch['id']}/{sc['id']}: {e}")
                text = ""
            arch_results.append({
                "scenario": sc["id"],
                "text": text,
                "words": len(text.split()),
            })

        results[arch["id"]] = arch_results
        print(f"    {arch['id']} done")

    return results


def run_ir_driven():
    print("\n  === IR-DRIVEN (persona-engine → OpenAI GPT-4o) ===")
    results = {}

    for arch in ARCHETYPES:
        builder = PersonaBuilder(name=arch["id"], occupation="General Professional")
        builder._big_five = {
            "openness": arch["O"], "conscientiousness": arch["C"],
            "extraversion": arch["E"], "agreeableness": arch["A"],
            "neuroticism": arch["N"],
        }
        persona = builder.build()
        arch_results = []

        for sc in SCENARIOS:
            try:
                from persona_engine.generation.llm_adapter import OpenAIAdapter
                adapter = OpenAIAdapter(model="gpt-4o")
                engine = PersonaEngine(
                    persona=persona, adapter=adapter, seed=42,
                )
                result = engine.chat(sc["text"])
                arch_results.append({
                    "scenario": sc["id"],
                    "text": result.text,
                    "words": len(result.text.split()),
                })
                del engine
            except Exception as e:
                print(f"      ERROR {arch['id']}/{sc['id']}: {e}")
                arch_results.append({
                    "scenario": sc["id"],
                    "text": "",
                    "words": 0,
                })

        results[arch["id"]] = arch_results
        print(f"    {arch['id']} done")

    return results


def analyze(prompt_results, ir_results):
    print("\n" + "=" * 80)
    print("  OPENAI GPT-4o: FAIR COMPARISON RESULTS")
    print("=" * 80)

    # Word count
    p_all = [r["words"] for a in ARCHETYPES for r in prompt_results[a["id"]]]
    i_all = [r["words"] for a in ARCHETYPES for r in ir_results[a["id"]]]
    print(f"\n  Word count: Prompt avg={sum(p_all)/len(p_all):.0f}, IR avg={sum(i_all)/len(i_all):.0f}")

    # Empath differentiation
    print(f"\n  EMPATH DIFFERENTIATION (high-trait vs low-trait)")
    print(f"  {'Trait':<20} {'Prompt wins':>12} {'IR wins':>12} {'Trait winner':>15}")
    print(f"  {'-'*60}")

    total_p = 0
    total_i = 0

    for trait, cats in TRAIT_CATEGORIES.items():
        high_archs = [a for a in ARCHETYPES if a[trait] >= 0.7]
        low_archs = [a for a in ARCHETYPES if a[trait] <= 0.35]
        if not high_archs or not low_archs:
            continue

        p_wins = 0
        i_wins = 0

        for cat in cats['high']:
            def safe_empath(text, cat):
                if not text.strip():
                    return 0
                result = lexicon.analyze(text, categories=[cat], normalize=True)
                return result.get(cat, 0) if result else 0

            p_high = sum(
                safe_empath(r['text'], cat)
                for a in high_archs for r in prompt_results[a['id']]
            ) / (len(high_archs) * len(SCENARIOS))
            p_low = sum(
                safe_empath(r['text'], cat)
                for a in low_archs for r in prompt_results[a['id']]
            ) / (len(low_archs) * len(SCENARIOS))
            i_high = sum(
                safe_empath(r['text'], cat)
                for a in high_archs for r in ir_results[a['id']]
            ) / (len(high_archs) * len(SCENARIOS))
            i_low = sum(
                safe_empath(r['text'], cat)
                for a in low_archs for r in ir_results[a['id']]
            ) / (len(low_archs) * len(SCENARIOS))

            p_diff = abs(p_high - p_low)
            i_diff = abs(i_high - i_low)

            if p_diff > i_diff:
                p_wins += 1
            elif i_diff > p_diff:
                i_wins += 1

        trait_names = {'E': 'Extraversion', 'N': 'Neuroticism', 'A': 'Agreeableness', 'C': 'Conscientiousness', 'O': 'Openness'}
        winner = 'Prompt' if p_wins > i_wins else 'IR' if i_wins > p_wins else 'Tie'
        print(f"  {trait_names[trait]:<20} {p_wins:>12} {i_wins:>12} {winner:>15}")
        total_p += p_wins
        total_i += i_wins

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<20} {total_p:>12} {total_i:>12} {'Prompt' if total_p > total_i else 'IR' if total_i > total_p else 'Tie':>15}")

    return {"prompt_total": total_p, "ir_total": total_i}


def main():
    print("=" * 80)
    print("  FAIR COMPARISON — OpenAI GPT-4o")
    print(f"  {len(ARCHETYPES)} archetypes × {len(SCENARIOS)} scenarios = {len(ARCHETYPES) * len(SCENARIOS)} per approach")
    print("=" * 80)

    t0 = time.time()
    prompt_results = run_prompt_only()
    t_prompt = time.time() - t0
    print(f"  Prompt-only: {t_prompt:.0f}s")

    t0 = time.time()
    ir_results = run_ir_driven()
    t_ir = time.time() - t0
    print(f"  IR-driven: {t_ir:.0f}s")

    summary = analyze(prompt_results, ir_results)

    report = {
        "model": "gpt-4o",
        "design": "Fair comparison — fresh engine per scenario, same task",
        "prompt_results": prompt_results,
        "ir_results": ir_results,
        "summary": summary,
        "timing": {"prompt_s": round(t_prompt, 1), "ir_s": round(t_ir, 1)},
    }
    report_path = Path(__file__).parent / "fair_comparison_openai_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
