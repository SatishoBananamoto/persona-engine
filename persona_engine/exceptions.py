"""
Persona Engine — Custom Exception Hierarchy

Typed exceptions so callers can distinguish between missing dependencies,
configuration errors, LLM failures, validation issues, and memory problems.

Usage::

    from persona_engine.exceptions import LLMAPIKeyError, PersonaValidationError

    try:
        engine = PersonaEngine.from_yaml("persona.yaml")
        result = engine.chat("Hello")
    except LLMAPIKeyError:
        print("Set ANTHROPIC_API_KEY or pass api_key=")
    except PersonaValidationError as e:
        print(f"Invalid persona: {e}")
"""


class PersonaEngineError(Exception):
    """Base exception for all persona engine errors."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class PersonaValidationError(PersonaEngineError):
    """Schema or builder validation failure.

    Raised when a persona definition fails Pydantic validation,
    builder constraints are violated, or input is malformed.
    """


class InputValidationError(PersonaValidationError):
    """User input fails validation (too long, empty, etc.)."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLMError(PersonaEngineError):
    """Base for all LLM-related failures."""


class LLMAPIKeyError(LLMError):
    """Missing or invalid API key for an LLM provider."""


class LLMConnectionError(LLMError):
    """Network or timeout failure when calling LLM."""


class LLMResponseError(LLMError):
    """Malformed, empty, or unexpected LLM response."""


# ---------------------------------------------------------------------------
# IR Generation
# ---------------------------------------------------------------------------

class IRGenerationError(PersonaEngineError):
    """Turn planner failed to generate a valid IR."""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class PersonaMemoryError(PersonaEngineError):
    """Base for memory store failures."""


class MemoryCapacityError(PersonaMemoryError):
    """Store full and eviction failed."""


class MemoryCorruptionError(PersonaMemoryError):
    """Inconsistent state detected in memory stores."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(PersonaEngineError):
    """Missing dependencies, bad configuration, or unsupported provider."""
