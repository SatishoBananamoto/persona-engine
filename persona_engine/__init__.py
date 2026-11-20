"""
Persona Engine - Universal Conversational Persona System

A psychologically-grounded system for creating behaviorally coherent synthetic personas.

Quick start::

    from persona_engine import PersonaEngine, Conversation

    engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
    result = engine.chat("What makes a good sauce?")
    print(result.text)

    # Multi-turn with Conversation wrapper
    convo = Conversation(engine)
    convo.say("Tell me more about French cuisine")
    convo.say("What about Italian?")
    print(convo.summary())
"""

__version__ = "0.4.0"

from persona_engine.conversation import Conversation
from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.persona_builder import PersonaBuilder
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    ResponseStructure,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    CognitiveStyle,
    CommunicationPreferences,
    DomainKnowledge,
    Goal,
    Persona,
    PersonalityProfile,
    SchwartzValues,
)

__all__ = [
    # Core SDK
    "PersonaEngine",
    "Conversation",
    "PersonaBuilder",
    "ChatResult",
    # Persona schema
    "Persona",
    "PersonalityProfile",
    "BigFiveTraits",
    "SchwartzValues",
    "CognitiveStyle",
    "CommunicationPreferences",
    "DomainKnowledge",
    "Goal",
    # IR schema
    "IntermediateRepresentation",
    "ConversationFrame",
    "ResponseStructure",
    "CommunicationStyle",
    "KnowledgeAndDisclosure",
    # IR enums
    "InteractionMode",
    "ConversationGoal",
    "Tone",
    "Verbosity",
    "UncertaintyAction",
]
