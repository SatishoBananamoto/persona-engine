#!/usr/bin/env python3
"""
Full Trace Test — 10-turn conversation with complete pipeline visibility.

Shows everything for each turn:
1. USER INPUT
2. IR VALUES (all computed fields)
3. FULL CITATION CHAIN (every mutation, every step)
4. EXACT PROMPT SENT TO LLM (system + user prompt)
5. LLM RESPONSE
6. VALIDATION RESULT

Usage:
    export ANTHROPIC_API_KEY=...
    python run_full_trace.py                    # prints to stdout
    python run_full_trace.py > trace.txt        # saves to file
"""

import os
import sys
import yaml
from datetime import datetime

from persona_engine.memory import MemoryManager
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation import PipelineValidator
from persona_engine.generation.response_generator import ResponseGenerator
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.llm_adapter import create_adapter

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
PERSONA_PATH = "personas/chef.yaml"

TURNS = [
    # Expert domain — deep knowledge
    {"input": "What makes a perfect French mother sauce?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "french_cuisine"},

    # Expert domain — practical experience
    {"input": "How do you handle a kitchen that's falling behind on a Friday rush?",
     "mode": InteractionMode.INTERVIEW,
     "goal": ConversationGoal.GATHER_INFO,
     "topic": "restaurant_management"},

    # Adjacent domain — knows a bit (health/nutrition)
    {"input": "Is a keto diet actually healthy long-term?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "nutrition"},

    # Completely outside domain — should be low competence
    {"input": "Can you explain how blockchain works?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "blockchain"},

    # Values + personal experience
    {"input": "Do you think culinary school is worth the money?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "culinary_education"},

    # Sensitive topic — should trigger privacy constraints
    {"input": "What restaurant do you work at? I want to visit.",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.BUILD_RAPPORT,
     "topic": "professional_work"},

    # Disagreement / challenge
    {"input": "I think frozen food is just as good as fresh. Change my mind.",
     "mode": InteractionMode.DEBATE,
     "goal": ConversationGoal.PERSUADE,
     "topic": "food_quality"},

    # Return to expert domain after gap — memory should kick in
    {"input": "Going back to sauces — how do you feel about molecular gastronomy?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "french_cuisine"},

    # Another unknown domain
    {"input": "What do you think about abstract expressionist painting?",
     "mode": InteractionMode.CASUAL_CHAT,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "art"},

    # Philosophical / values-driven
    {"input": "Is cooking an art or a craft?",
     "mode": InteractionMode.DEBATE,
     "goal": ConversationGoal.EXPLORE_IDEAS,
     "topic": "philosophy_of_food"},
]


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
SEP_HEAVY = "=" * 80
SEP_LIGHT = "-" * 80
SEP_DOT = "·" * 80


def p(text=""):
    print(text)


def section(title):
    p(SEP_LIGHT)
    p("  %s" % title)
    p(SEP_LIGHT)


def format_citation(i, c):
    """Format a single citation as multi-line text."""
    lines = []
    lines.append("  %2d. [%s:%s]" % (i, c.source_type, c.source_id))
    if c.target_field:
        lines.append("      field:     %s" % c.target_field)
    if c.operation:
        lines.append("      operation: %s" % c.operation)
    before = c.value_before if c.value_before is not None else "(init)"
    after = c.value_after if c.value_after is not None else "-"
    if c.value_before is not None or c.value_after is not None:
        delta = ""
        if c.value_before is not None and c.value_after is not None:
            try:
                d = float(c.value_after) - float(c.value_before)
                delta = "  (delta: %+.4f)" % d
            except (ValueError, TypeError):
                pass
        lines.append("      before:    %s" % before)
        lines.append("      after:     %s%s" % (after, delta))
    lines.append("      effect:    %s" % c.effect)
    if c.weight != 1.0:
        lines.append("      weight:    %s" % c.weight)
    if c.reason:
        lines.append("      reason:    %s" % c.reason)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # Load persona
    with open(PERSONA_PATH) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    persona = Persona(**data)

    # Initialize pipeline
    determinism = DeterminismManager(seed=42)
    memory = MemoryManager()
    planner = TurnPlanner(persona, determinism, memory_manager=memory)
    validator = PipelineValidator(persona)
    prompt_builder = IRPromptBuilder()
    adapter = create_adapter("anthropic")
    generator = ResponseGenerator(persona, adapter=adapter)

    # System prompt (same for all turns)
    system_prompt = prompt_builder.build_system_prompt(persona)

    # ─── Header ──────────────────────────────────────────────
    p(SEP_HEAVY)
    p("  FULL PIPELINE TRACE — 10 TURNS")
    p("  Persona:   %s" % persona.label)
    p("  Model:     %s" % adapter.get_model_name())
    p("  Timestamp: %s" % datetime.now().isoformat())
    p("  Seed:      42")
    p(SEP_HEAVY)

    # ─── System Prompt ───────────────────────────────────────
    p()
    section("SYSTEM PROMPT (sent with every turn)")
    p()
    for line in system_prompt.strip().split("\n"):
        p("  %s" % line)
    p()

    # ─── Persona Summary ─────────────────────────────────────
    section("PERSONA PROFILE SUMMARY")
    p()
    bf = persona.psychology.big_five
    p("  Big Five:")
    p("    Openness:          %.2f" % bf.openness)
    p("    Conscientiousness: %.2f" % bf.conscientiousness)
    p("    Extraversion:      %.2f" % bf.extraversion)
    p("    Agreeableness:     %.2f" % bf.agreeableness)
    p("    Neuroticism:       %.2f" % bf.neuroticism)
    p()
    p("  Knowledge Domains:")
    for kd in persona.knowledge_domains:
        p("    %-15s  proficiency=%.2f  subdomains=%s" % (
            kd.domain, kd.proficiency, kd.subdomains))
    p()
    cs = persona.psychology.cognitive_style
    p("  Cognitive Style:")
    p("    analytical_intuitive: %.2f" % cs.analytical_intuitive)
    p("    risk_tolerance:       %.2f" % cs.risk_tolerance)
    p("    need_for_closure:     %.2f" % cs.need_for_closure)
    p("    cognitive_complexity:  %.2f" % cs.cognitive_complexity)
    p()
    comm = persona.psychology.communication
    p("  Communication Base:")
    p("    formality=%.2f  directness=%.2f  verbosity=%.2f  expressiveness=%.2f" % (
        comm.formality, comm.directness, comm.verbosity, comm.emotional_expressiveness))
    p()

    # ─── Turns ───────────────────────────────────────────────
    cache = StanceCache()

    for i, turn in enumerate(TURNS, 1):
        p()
        p(SEP_HEAVY)
        p("  TURN %d of %d" % (i, len(TURNS)))
        p(SEP_HEAVY)

        # 1. USER INPUT
        section("1. USER INPUT")
        p()
        p("  \"%s\"" % turn["input"])
        p("  mode=%s  goal=%s  topic=%s" % (
            turn["mode"].value, turn["goal"].value, turn["topic"]))
        p()

        # Generate IR
        context = ConversationContext(
            conversation_id="full_trace",
            turn_number=i,
            interaction_mode=turn["mode"],
            goal=turn["goal"],
            topic_signature=turn["topic"],
            user_input=turn["input"],
            stance_cache=cache,
        )
        ir = planner.generate_ir(context)

        # 2. IR VALUES
        section("2. IR VALUES (computed by TurnPlanner)")
        p()
        p("  conversation_frame:")
        p("    interaction_mode:    %s" % ir.conversation_frame.interaction_mode.value)
        p("    goal:                %s" % ir.conversation_frame.goal.value)
        p()
        p("  response_structure:")
        p("    intent:     %s" % ir.response_structure.intent)
        p("    stance:     %s" % (ir.response_structure.stance or "(none)"))
        p("    rationale:  %s" % (ir.response_structure.rationale or "(none)"))
        p("    elasticity: %.4f" % (ir.response_structure.elasticity or 0))
        p("    confidence: %.4f" % ir.response_structure.confidence)
        p("    competence: %.4f" % ir.response_structure.competence)
        p()
        p("  communication_style:")
        p("    tone:       %s" % ir.communication_style.tone.value)
        p("    verbosity:  %s" % ir.communication_style.verbosity.value)
        p("    formality:  %.4f" % ir.communication_style.formality)
        p("    directness: %.4f" % ir.communication_style.directness)
        p()
        p("  knowledge_disclosure:")
        p("    disclosure_level:     %.4f" % ir.knowledge_disclosure.disclosure_level)
        p("    uncertainty_action:   %s" % ir.knowledge_disclosure.uncertainty_action.value)
        p("    knowledge_claim_type: %s" % ir.knowledge_disclosure.knowledge_claim_type.value)
        p()
        p("  safety_plan:")
        p("    blocked_topics: %s" % (ir.safety_plan.blocked_topics or []))
        p("    clamped_fields: %d" % len(ir.safety_plan.clamped_fields))
        p("    active_constraints: %s" % (ir.safety_plan.active_constraints or []))
        p()
        p("  memory_ops:")
        p("    write_intents: %d" % len(ir.memory_ops.write_intents))
        for wi in ir.memory_ops.write_intents:
            p("      - [%s] %s (conf=%.1f)" % (wi.content_type, wi.content[:60], wi.confidence))
        p()
        p("  seed: %s" % ir.seed)
        p()

        # 3. FULL CITATION CHAIN
        section("3. FULL CITATION CHAIN (%d citations)" % len(ir.citations))
        p()
        for j, c in enumerate(ir.citations, 1):
            p(format_citation(j, c))
            p()

        # 4. EXACT PROMPT TO LLM
        memory_ctx = memory.get_context_for_turn(turn["topic"], turn["input"])
        user_prompt = prompt_builder.build_generation_prompt(
            ir, turn["input"], persona=persona, memory_context=memory_ctx)

        section("4. EXACT PROMPT SENT TO LLM")
        p()
        p("  [system prompt: see above — same for all turns]")
        p()
        p("  [user prompt:]")
        for line in user_prompt.strip().split("\n"):
            p("  | %s" % line)
        p()

        # 5. VALIDATION
        result = validator.validate(ir, turn_number=i, topic=turn["topic"])

        section("5. VALIDATION RESULT")
        p()
        p("  passed: %s" % result.passed)
        p("  violations: %d" % len(result.violations))
        for v in result.violations:
            p("    [%s] %s" % (v.severity, v.violation_type))
            p("      %s" % v.message)
            if v.suggested_fix:
                p("      fix: %s" % v.suggested_fix)
        p("  invariants_checked: %d" % len(result.checked_invariants))
        p()

        # 6. LLM RESPONSE
        response = generator.generate(ir, turn["input"], memory_context=memory_ctx)

        section("6. LLM RESPONSE")
        p()
        p("  model:  %s" % response.model)
        p("  tokens: ~%s" % response.estimated_tokens)
        p("  valid:  %s" % response.is_valid())
        p()
        for line in response.text.split("\n"):
            p("  %s" % line)
        p()

    # ─── Summary ─────────────────────────────────────────────
    p(SEP_HEAVY)
    p("  TRACE COMPLETE — %d turns" % len(TURNS))
    p("  Cross-turn snapshots: %d" % validator.turn_count)
    p(SEP_HEAVY)


if __name__ == "__main__":
    main()
