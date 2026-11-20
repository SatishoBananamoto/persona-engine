"""
Pipeline stages for the TurnPlanner.

Each stage encapsulates a discrete phase of IR generation:
1. Foundation  — trace setup, per-turn seed, memory context
2. Interpretation — topic relevance, bias, state, intent, domain, expert eligibility
3. Behavioral metrics — elasticity, stance, confidence, competence, tone, verbosity, comm style
4. Knowledge & safety — disclosure, uncertainty, claim type, patterns, constraints
5. Finalization — memory writes, IR assembly, stance cache, snapshot
"""

from persona_engine.planner.stages.foundation import FoundationStage
from persona_engine.planner.stages.interpretation import InterpretationStage
from persona_engine.planner.stages.behavioral import BehavioralMetricsStage
from persona_engine.planner.stages.knowledge import KnowledgeSafetyStage
from persona_engine.planner.stages.finalization import FinalizationStage
from persona_engine.planner.stages.stage_results import (
    BehavioralMetricsResult,
    InterpretationResult,
    KnowledgeSafetyResult,
)

__all__ = [
    "FoundationStage",
    "InterpretationStage",
    "BehavioralMetricsStage",
    "KnowledgeSafetyStage",
    "FinalizationStage",
    "InterpretationResult",
    "BehavioralMetricsResult",
    "KnowledgeSafetyResult",
]
