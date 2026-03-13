"""
Multi-Turn Conversation — 10 turns showing personality consistency.

Demonstrates that the persona maintains consistent behavior across
multiple turns, with memory and cross-turn validation active.
"""

from persona_engine import PersonaEngine

engine = PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock")

messages = [
    "What's your area of research?",
    "That sounds fascinating. Can you explain it simply?",
    "What got you interested in physics?",
    "Do you think AI will change physics research?",
    "What's the biggest unsolved problem in your field?",
    "Have you published any papers recently?",
    "What do you do outside of work?",
    "Do you think physics is getting harder to do?",
    "What advice would you give a physics student?",
    "Thanks for chatting! Any final thoughts?",
]

for msg in messages:
    result = engine.chat(msg)
    print(f"Turn {result.turn_number}: {msg}")
    print(f"  → {result.text[:100]}...")
    print(f"  Confidence: {result.confidence:.2f} | Competence: {result.competence:.2f}")
    print()

# Show memory stats after conversation
stats = engine.memory_stats()
print("Memory stats:", stats)
