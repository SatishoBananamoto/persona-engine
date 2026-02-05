"""
Persona Engine - Universal Conversational Persona System

A psychologically-grounded system for creating behaviorally coherent synthetic personas.
"""

__version__ = "0.1.0"

from persona_engine.schema.persona_schema import (
    Persona,
    PersonalityProfile,
    CognitiveStyle,
    CommunicationPreferences,
)
from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    ConversationFrame,
    ResponseStructure,
    CommunicationStyle,
    KnowledgeAndDisclosure,
)

__all__ = [
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
