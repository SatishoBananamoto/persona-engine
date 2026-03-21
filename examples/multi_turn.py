#!/usr/bin/env python3
"""Multi-Turn Conversation — Showing memory and state across turns.

Demonstrates the Conversation wrapper and how the engine tracks
facts, preferences, and relationship signals over multiple turns.
"""

from persona_engine import PersonaEngine, Conversation

engine = PersonaEngine.from_yaml("personas/ux_researcher.yaml", llm_provider="mock")
convo = Conversation(engine, metadata={"experiment": "multi_turn_demo"})

# Send a sequence of messages — each turn builds on prior context
messages = [
    "Hi! I'm working on a mobile banking app redesign.",
    "We're seeing a 40% drop-off at the onboarding screen.",
    "What research methods would you recommend?",
    "Good ideas. We have a tight budget though.",
    "Thanks, that's really helpful!",
]

for msg in messages:
    result = convo.say(msg)
    print(f"[Turn {result.turn_number}] User: {msg}")
    print(f"  Response: {result.text[:120]}...")
    print(f"  Confidence={result.confidence:.2f}  Competence={result.competence:.2f}")
    print()

# Conversation summary with aggregate stats
print("=== Conversation Summary ===")
summary = convo.summary()
for key, val in summary.items():
    if key != "metadata":
        print(f"  {key}: {val}")

# Memory stats show what the engine remembered across turns
print("\n=== Memory Stats ===")
stats = engine.memory_stats()
for key, val in stats.items():
    print(f"  {key}: {val}")

# Iterate back over turns to see confidence trajectory
print("\n=== Confidence Trajectory ===")
for turn in convo:
    tone = turn.ir.communication_style.tone.value
    print(f"  Turn {turn.turn_number}: conf={turn.confidence:.2f} tone={tone}")
