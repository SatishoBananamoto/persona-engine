#!/usr/bin/env python3
"""
Full Pipeline Demo - Persona Engine

Demonstrates the complete flow:
  1. Load persona from YAML
  2. Generate IR via TurnPlanner
  3. Convert IR to text via ResponseGenerator
  4. Display result with IR constraints

USAGE:
    # Template mode (no API key needed):
    python demo_full_pipeline.py

    # With Anthropic API:
    export ANTHROPIC_API_KEY='your-key'
    python demo_full_pipeline.py --backend anthropic

    # Mock mode (for testing):
    python demo_full_pipeline.py --backend mock
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.response import ResponseGenerator, ResponseConfig, GenerationBackend
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def demo_conversation(backend: str = "template", persona_path: str = "personas/ux_researcher.yaml"):
    print(f"\n{'='*20} PERSONA ENGINE - FULL PIPELINE DEMO {'='*20}\n")

    # Load persona
    persona = load_persona(persona_path)
    print(f"Loading persona from: {persona_path}")
    print(f"  Name: {persona.label}")
    print(f"  Occupation: {persona.identity.occupation}")
    print(f"  Age: {persona.identity.age}")

    # Initialize
    print(f"\nInitializing with backend: {backend}")
    determinism = DeterminismManager(seed=42)
    planner = TurnPlanner(persona, determinism)

    api_key = None
    if backend == "anthropic":
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\nERROR: ANTHROPIC_API_KEY not set.")
            print("  export ANTHROPIC_API_KEY='your-key'")
            print("  Or run without --backend anthropic for template mode.")
            return

    config = ResponseConfig(
        backend=GenerationBackend(backend),
        api_key=api_key,
        model_id="claude-haiku-4-5-20251001",
        max_tokens=500,
    )
    generator = ResponseGenerator(config=config, persona=persona)
    print(f"  Backend: {config.backend.value}")
    if backend == "anthropic":
        print(f"  Model: {config.model_id}")

    print(f"\n{'='*20} DEMO CONVERSATIONS {'='*20}\n")

    # Demo turns
    demo_inputs = [
        {
            "input": "Hi Sarah! Tell me about your work as a UX researcher.",
            "mode": InteractionMode.CASUAL_CHAT,
            "goal": ConversationGoal.BUILD_RAPPORT,
            "topic": "ux_research",
        },
        {
            "input": "What do you think about remote usability testing?",
            "mode": InteractionMode.INTERVIEW,
            "goal": ConversationGoal.GATHER_INFO,
            "topic": "usability_testing",
        },
        {
            "input": "Can you explain quantum computing?",
            "mode": InteractionMode.CASUAL_CHAT,
            "goal": ConversationGoal.EXPLORE_IDEAS,
            "topic": "quantum_computing",
        },
    ]

    cache = StanceCache()

    for turn, demo in enumerate(demo_inputs, 1):
        print(f"--- Turn {turn} ---")
        print(f"USER: {demo['input']}\n")

        # Generate IR
        context = ConversationContext(
            conversation_id="demo_session",
            turn_number=turn,
            interaction_mode=demo["mode"],
            goal=demo["goal"],
            topic_signature=demo["topic"],
            user_input=demo["input"],
            stance_cache=cache,
        )
        ir = planner.generate_ir(context)

        # Show IR constraints
        print("IR CONSTRAINTS:")
        print(f"  Tone: {ir.communication_style.tone.value}")
        print(f"  Formality: {ir.communication_style.formality:.2f}")
        print(f"  Verbosity: {ir.communication_style.verbosity.value}")
        print(f"  Confidence: {ir.response_structure.confidence:.2f}")
        print(f"  Claim Type: {ir.knowledge_disclosure.knowledge_claim_type.value}")
        print(f"  Uncertainty: {ir.knowledge_disclosure.uncertainty_action.value}")
        if ir.response_structure.stance:
            print(f"  Stance: {ir.response_structure.stance[:80]}...")
        print()

        # Generate response
        response = generator.generate(ir, demo["input"])

        print(f"SARAH:")
        print(response.text)
        print()

        if response.token_usage:
            tokens = response.token_usage["input_tokens"] + response.token_usage["output_tokens"]
            print(f"Tokens: {tokens}")
        print(f"Backend: {response.backend.value}")
        print()

    print(f"{'='*20} DEMO COMPLETE {'='*20}")
    print(f"Total turns: {len(demo_inputs)}")
    print(f"Backend: {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persona Engine Full Pipeline Demo")
    parser.add_argument(
        "--backend",
        choices=["template", "anthropic", "mock"],
        default="template",
        help="Response generation backend (default: template)",
    )
    parser.add_argument(
        "--persona",
        default="personas/ux_researcher.yaml",
        help="Path to persona YAML file",
    )
    args = parser.parse_args()
    demo_conversation(backend=args.backend, persona_path=args.persona)
