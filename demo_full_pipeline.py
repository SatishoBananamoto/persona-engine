#!/usr/bin/env python3
"""
Full Pipeline Demo - Persona Engine

Demonstrates the complete flow from user input to persona response:
1. Load persona from YAML
2. Generate IR (Intermediate Representation) via Turn Planner
3. Convert IR to text via Response Generator
4. Display result with citations and validation

USAGE:
    # Edit .env file with your API key, then run:
    python demo_full_pipeline.py
    
    # Or use mock mode (no API key needed)
    python demo_full_pipeline.py --mock
"""

import argparse
import yaml
import sys
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use env vars directly

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from persona_engine.schema.persona_schema import Persona
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.generation import create_adapter
from persona_engine.generation.response_generator import ResponseGenerator


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    """Load persona from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def create_context(
    user_input: str,
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    topic: str = "general",
    turn: int = 1,
) -> ConversationContext:
    """Create a conversation context."""
    return ConversationContext(
        conversation_id="demo_session",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=StanceCache(),
    )


def print_separator(title: str = ""):
    """Print a visual separator."""
    print()
    if title:
        print(f"{'='*20} {title} {'='*20}")
    else:
        print("=" * 60)
    print()


def demo_conversation(provider: str = "anthropic", persona_path: str = "personas/ux_researcher.yaml"):
    """Run an interactive demo conversation."""
    
    print_separator("PERSONA ENGINE - FULL PIPELINE DEMO")
    
    # Load persona
    print(f"Loading persona from: {persona_path}")
    persona = load_persona(persona_path)
    print(f"  Name: {persona.label}")
    print(f"  Occupation: {persona.identity.occupation}")
    print(f"  Age: {persona.identity.age}")
    
    # Initialize components
    print(f"\nInitializing with provider: {provider}")
    determinism = DeterminismManager(seed=42)
    planner = TurnPlanner(persona, determinism)
    
    try:
        adapter = create_adapter(provider)
        generator = ResponseGenerator(persona, adapter)
        print(f"  Model: {adapter.get_model_name()}")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nTo fix this:")
        print("  1. Set environment variable: $env:ANTHROPIC_API_KEY = 'your-key'")
        print("  2. Or run with --mock flag: python demo_full_pipeline.py --mock")
        return
    
    print_separator("DEMO CONVERSATIONS")
    
    # Demo conversations
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
    
    turn = 1
    for demo in demo_inputs:
        print(f"--- Turn {turn} ---")
        print(f"USER: {demo['input']}")
        print()
        
        # Generate IR
        context = create_context(
            user_input=demo["input"],
            mode=demo["mode"],
            goal=demo["goal"],
            topic=demo["topic"],
            turn=turn,
        )
        ir = planner.generate_ir(context)
        
        # Show key IR constraints
        print("IR CONSTRAINTS:")
        print(f"  Tone: {ir.communication_style.tone.value}")
        print(f"  Formality: {ir.communication_style.formality:.2f}")
        print(f"  Verbosity: {ir.communication_style.verbosity.value}")
        print(f"  Confidence: {ir.response_structure.confidence:.2f}")
        print(f"  Claim Type: {ir.knowledge_disclosure.knowledge_claim_type.value}")
        if ir.response_structure.stance:
            print(f"  Stance: {ir.response_structure.stance[:80]}...")
        print()
        
        # Generate response
        response = generator.generate(ir, demo["input"])
        
        print(f"SARAH ({persona.label.split('-')[0].strip()}):") 
        print(response.text)
        print()
        
        # Show validation
        if response.violations:
            print("VALIDATION:")
            for v in response.violations:
                print(f"  [{v.severity.upper()}] {v.constraint}: {v.actual}")
        else:
            print("VALIDATION: All constraints satisfied [OK]")
        
        print(f"Est. tokens: ~{response.estimated_tokens}")
        print()
        
        turn += 1
    
    print_separator("DEMO COMPLETE")
    print(f"Total turns: {turn - 1}")
    print(f"Provider: {provider}")
    print(f"Model: {generator.adapter.get_model_name()}")


def main():
    parser = argparse.ArgumentParser(
        description="Persona Engine Full Pipeline Demo"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no API key needed)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "mock"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)"
    )
    parser.add_argument(
        "--persona",
        default="personas/ux_researcher.yaml",
        help="Path to persona YAML file"
    )
    
    args = parser.parse_args()
    
    provider = "mock" if args.mock else args.provider
    
    demo_conversation(provider=provider, persona_path=args.persona)


if __name__ == "__main__":
    main()
