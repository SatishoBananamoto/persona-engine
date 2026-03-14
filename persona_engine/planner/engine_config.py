"""
Engine Configuration — consolidated tuning constants.

All magic numbers from the planner pipeline are centralized here
for easier tuning, testing, and per-deployment overrides.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineConfig:
    """Central configuration for the persona engine planner pipeline."""

    # Domain & Expertise
    default_proficiency: float = 0.3
    expert_threshold: float = 0.7

    # Topic Relevance
    default_topic_relevance: float = 0.5

    # Communication Style
    formality_role_weight: float = 0.7
    formality_base_weight: float = 0.3
    directness_impatience_bump: float = 0.1
    patience_threshold: float = 0.3

    # Elasticity
    elasticity_min: float = 0.1
    elasticity_max: float = 0.9

    # Disclosure
    disclosure_min: float = 0.0
    disclosure_max: float = 1.0

    # Evidence / Stress
    evidence_stress_threshold: float = 0.4

    # Competence
    unknown_domain_base: float = 0.10
    openness_competence_weight: float = 0.1
    fact_boost_per_fact: float = 0.03
    fact_boost_cap: float = 0.10

    # Cross-Turn Dynamics
    cross_turn_inertia: float = 0.15
    personality_field_inertia: float = 0.08  # Phase R2: lower inertia for personality-driven fields
    familiarity_boost_per_episode: float = 0.05
    familiarity_boost_cap: float = 0.15

    # Dynamic Time Pressure
    time_pressure_turn_threshold: int = 5
    time_pressure_per_turn: float = 0.03
    time_pressure_max_buildup: float = 0.15

    # Debate mode overlay
    debate_directness_bonus: float = 0.15


# Module-level default instance for backward compatibility
DEFAULT_CONFIG = EngineConfig()
