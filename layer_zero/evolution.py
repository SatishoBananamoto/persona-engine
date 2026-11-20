"""
Persona Evolution — event-driven trait and state updates.

Personas are not static. Life events shift personality traits, values,
and state in psychologically consistent ways.

Based on:
- Whole Trait Theory (Fleeson & Jayawickreme): traits are density distributions
- Research on personality change after major life events
- SYNTHIA temporal evolution patterns

Usage:
    evolved = evolve(persona, event="job_loss")
    evolved = evolve(persona, event="promotion", intensity=0.8)
    evolved = evolve(persona, years=5)  # natural aging
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from layer_zero.models import BIG_FIVE_TRAITS, SCHWARTZ_VALUES, MintedPersona


# =============================================================================
# Life event definitions
# =============================================================================

@dataclass
class LifeEvent:
    """Definition of how a life event shifts personality and values."""
    name: str
    description: str
    # Big Five shifts (additive, clamped to [0,1])
    trait_shifts: dict[str, float]
    # Schwartz value shifts
    value_shifts: dict[str, float]
    # State impacts
    state_shifts: dict[str, float]
    # How much variance the shift has (some people react more than others)
    variance: float = 0.3


LIFE_EVENTS: dict[str, LifeEvent] = {
    "job_loss": LifeEvent(
        name="Job Loss",
        description="Involuntary job loss — financial stress, identity disruption",
        trait_shifts={"neuroticism": 0.08, "extraversion": -0.03, "openness": -0.02, "conscientiousness": -0.03},
        value_shifts={"security": 0.10, "power": -0.05, "achievement": -0.05, "stimulation": -0.03},
        state_shifts={"stress": 0.3, "mood_valence": -0.3, "engagement": -0.2},
    ),
    "promotion": LifeEvent(
        name="Promotion",
        description="Career advancement — increased confidence, status",
        trait_shifts={"neuroticism": -0.03, "extraversion": 0.03, "conscientiousness": 0.02},
        value_shifts={"achievement": 0.05, "power": 0.03, "self_direction": 0.02},
        state_shifts={"mood_valence": 0.2, "engagement": 0.2, "stress": -0.1},
    ),
    "new_child": LifeEvent(
        name="New Child",
        description="Becoming a parent — increased responsibility, empathy",
        trait_shifts={"agreeableness": 0.05, "conscientiousness": 0.05, "neuroticism": 0.03, "extraversion": -0.02},
        value_shifts={"benevolence": 0.10, "security": 0.08, "tradition": 0.03, "stimulation": -0.05, "hedonism": -0.05},
        state_shifts={"fatigue": 0.2, "stress": 0.1, "mood_valence": 0.1},
    ),
    "divorce": LifeEvent(
        name="Divorce",
        description="Relationship dissolution — emotional upheaval, identity shift",
        trait_shifts={"neuroticism": 0.05, "agreeableness": -0.03, "extraversion": -0.02},
        value_shifts={"security": -0.05, "self_direction": 0.05, "tradition": -0.03},
        state_shifts={"stress": 0.3, "mood_valence": -0.3},
    ),
    "retirement": LifeEvent(
        name="Retirement",
        description="End of career — freedom, identity restructuring",
        trait_shifts={"neuroticism": -0.03, "extraversion": -0.02, "conscientiousness": -0.03, "openness": 0.02, "agreeableness": 0.03},
        value_shifts={"self_direction": 0.05, "stimulation": -0.03, "achievement": -0.08, "security": 0.05, "benevolence": 0.03},
        state_shifts={"stress": -0.2, "fatigue": -0.1, "engagement": -0.1},
    ),
    "relocation": LifeEvent(
        name="Major Relocation",
        description="Moving to a new city/country — openness challenge, social disruption",
        trait_shifts={"openness": 0.05, "neuroticism": 0.03, "extraversion": -0.02},
        value_shifts={"stimulation": 0.05, "security": -0.05, "self_direction": 0.03, "tradition": -0.03},
        state_shifts={"stress": 0.15, "engagement": 0.1},
    ),
    "health_crisis": LifeEvent(
        name="Health Crisis",
        description="Serious illness or injury — vulnerability, perspective shift",
        trait_shifts={"neuroticism": 0.08, "openness": 0.03, "conscientiousness": 0.02},
        value_shifts={"security": 0.10, "benevolence": 0.05, "universalism": 0.03, "power": -0.05, "achievement": -0.03},
        state_shifts={"stress": 0.3, "fatigue": 0.3, "mood_valence": -0.2},
    ),
    "education_milestone": LifeEvent(
        name="Education Milestone",
        description="Completing a degree or major certification",
        trait_shifts={"openness": 0.03, "conscientiousness": 0.03, "neuroticism": -0.02},
        value_shifts={"achievement": 0.05, "self_direction": 0.03},
        state_shifts={"mood_valence": 0.15, "engagement": 0.1, "stress": -0.1},
    ),
    "bereavement": LifeEvent(
        name="Bereavement",
        description="Loss of someone close — grief, perspective shift",
        trait_shifts={"neuroticism": 0.05, "agreeableness": 0.03, "openness": 0.02},
        value_shifts={"benevolence": 0.05, "universalism": 0.03, "tradition": 0.03, "hedonism": -0.03},
        state_shifts={"mood_valence": -0.4, "stress": 0.2, "fatigue": 0.15},
    ),
    "financial_windfall": LifeEvent(
        name="Financial Windfall",
        description="Inheritance, lottery, or major financial gain",
        trait_shifts={"neuroticism": -0.03, "openness": 0.02, "extraversion": 0.02},
        value_shifts={"security": 0.05, "power": 0.03, "hedonism": 0.03, "stimulation": 0.02},
        state_shifts={"stress": -0.2, "mood_valence": 0.2},
    ),
}


# =============================================================================
# Evolution functions
# =============================================================================

def evolve(
    persona: MintedPersona,
    event: str | None = None,
    intensity: float = 1.0,
    years: int = 0,
    seed: int = 42,
) -> MintedPersona:
    """Evolve a persona based on a life event or natural aging.

    Creates a new MintedPersona with updated traits, values, and state.
    The original is not modified.

    Args:
        persona: The persona to evolve.
        event: Life event name (from LIFE_EVENTS).
        intensity: How strongly the event affects this person (0-2, default 1.0).
        years: Natural aging in years (shifts traits per age research).
        seed: Random seed for individual variation.

    Returns:
        New MintedPersona with evolved traits.
    """
    rng = np.random.default_rng(seed)
    p = persona.persona

    # Deep copy the persona data
    new_data = p.model_dump()

    # --- Apply life event ---
    if event and event in LIFE_EVENTS:
        le = LIFE_EVENTS[event]

        # Trait shifts with individual variance
        for trait, base_shift in le.trait_shifts.items():
            individual_shift = base_shift * intensity * (1.0 + rng.normal(0, le.variance))
            current = new_data["psychology"]["big_five"][trait]
            new_data["psychology"]["big_five"][trait] = float(np.clip(current + individual_shift, 0.01, 0.99))

        # Value shifts
        for val, base_shift in le.value_shifts.items():
            individual_shift = base_shift * intensity * (1.0 + rng.normal(0, le.variance))
            current = new_data["psychology"]["values"][val]
            new_data["psychology"]["values"][val] = float(np.clip(current + individual_shift, 0.0, 1.0))

        # State shifts
        for state_field, shift in le.state_shifts.items():
            if state_field in new_data["initial_state"]:
                current = new_data["initial_state"][state_field]
                lo, hi = (-1.0, 1.0) if state_field == "mood_valence" else (0.0, 1.0)
                new_data["initial_state"][state_field] = float(np.clip(current + shift * intensity, lo, hi))

    # --- Natural aging ---
    if years > 0:
        # Per year: A +0.002, C +0.002, N -0.002, E -0.001, O -0.001
        aging_rates = {
            "agreeableness": 0.002, "conscientiousness": 0.002,
            "neuroticism": -0.002, "extraversion": -0.001, "openness": -0.001,
        }
        for trait, rate in aging_rates.items():
            current = new_data["psychology"]["big_five"][trait]
            new_data["psychology"]["big_five"][trait] = float(np.clip(current + rate * years, 0.01, 0.99))

        new_data["identity"]["age"] = min(100, new_data["identity"]["age"] + years)

        # Value aging: conservation up, openness-to-change down
        value_aging = {
            "tradition": 0.002, "conformity": 0.002, "security": 0.002,
            "stimulation": -0.002, "self_direction": -0.001,
            "benevolence": 0.001, "universalism": 0.001,
        }
        for val, rate in value_aging.items():
            current = new_data["psychology"]["values"][val]
            new_data["psychology"]["values"][val] = float(np.clip(current + rate * years, 0.0, 1.0))

    # Rebuild persona from updated data
    from persona_engine.schema.persona_schema import Persona
    new_persona = Persona(**new_data)

    # Update provenance
    new_provenance = dict(persona.provenance)
    evolution_note = f"Evolved: event={event}, intensity={intensity}, years={years}"

    return MintedPersona(
        persona=new_persona,
        provenance=new_provenance,
        warnings=persona.warnings + [evolution_note],
    )


def list_events() -> list[dict[str, str]]:
    """List all available life events."""
    return [
        {"name": name, "description": le.description}
        for name, le in LIFE_EVENTS.items()
    ]
