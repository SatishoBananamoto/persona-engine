"""
Persona Engine - Universal Conversational Persona System

A psychologically-grounded system for creating behaviorally coherent synthetic personas.
"""

__version__ = "0.1.0"

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.persona_builder import PersonaBuilder
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    ResponseStructure,
)
from persona_engine.schema.persona_schema import (
    CognitiveStyle,
    CommunicationPreferences,
    Persona,
    PersonalityProfile,
)

__all__ = [
    "PersonaEngine",
    "PersonaBuilder",
    "ChatResult",
    "Persona",
    "PersonalityProfile",
    "CognitiveStyle",
    "CommunicationPreferences",
    "IntermediateRepresentation",
    "ConversationFrame",
    "ResponseStructure",
    "CommunicationStyle",
    "KnowledgeAndDisclosure",
]
