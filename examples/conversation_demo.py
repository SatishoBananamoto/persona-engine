"""
Conversation Demo — Multi-turn dialogue with the Conversation class.

Shows the high-level Conversation API with iteration, summary, and export.
"""

from persona_engine import PersonaEngine, Conversation

# Create engine and wrap in Conversation
engine = PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock")
convo = Conversation(engine, metadata={"purpose": "demo", "topic": "physics"})

# Send messages using say()
messages = [
    "What's your area of research?",
    "Can you explain quantum entanglement simply?",
    "Do you think quantum computing will change the world?",
    "What advice would you give a physics student?",
]

for msg in messages:
    result = convo.say(msg)
    print(f"Turn {result.turn_number}: {msg}")
    print(f"  → {result.text[:100]}...")
    print()

# Or send all at once with say_all()
# results = convo.say_all(messages)

# Iterate over turns
print("=== Conversation Summary ===")
for turn in convo:
    print(f"  Turn {turn.turn_number}: confidence={turn.confidence:.2f}, "
          f"competence={turn.competence:.2f}")

# Get summary
summary = convo.summary()
print(f"\nPersona: {summary['persona']}")
print(f"Turns: {summary['turn_count']}")
print(f"Avg Confidence: {summary['avg_confidence']:.2f}")
print(f"Avg Competence: {summary['avg_competence']:.2f}")
print(f"All Passed: {summary['all_passed_validation']}")

# Access last turn
last = convo.last()
if last:
    print(f"\nLast response: {last.text[:80]}...")

# Export transcript
transcript = convo.export_transcript()
print(f"\n{transcript[:300]}...")

# Export to JSON/YAML (commented out to avoid file creation in demo)
# convo.export_json("conversation.json")
# convo.export_yaml("conversation.yaml")
