"""
Validation Module — end-to-end coherence checks for the persona engine.

Three validation layers:
1. IR Coherence: Internal field consistency
2. Persona Compliance: IR vs persona profile
3. Cross-Turn Consistency: Multi-turn behavior coherence
"""

from persona_engine.validation.cross_turn import CrossTurnTracker, TurnSnapshot
from persona_engine.validation.ir_coherence import validate_ir_coherence
from persona_engine.validation.persona_compliance import validate_persona_compliance
from persona_engine.validation.pipeline_validator import PipelineValidator

__all__ = [
    "PipelineValidator",
    "CrossTurnTracker",
    "TurnSnapshot",
    "validate_ir_coherence",
    "validate_persona_compliance",
]
