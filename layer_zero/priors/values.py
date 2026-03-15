"""
Schwartz value prior engine — circumplex-structured generation.

Instead of sampling 10 values independently and policing post-hoc,
generates values from the circumplex structure directly:
1. Sample a peak angle θ (persona's "North Star" value)
2. Sample an amplitude A (how strongly differentiated)
3. Generate all 10 values: value_i = baseline + A × cos(θ - position_i)
4. Apply demographic prior shifts
5. Add calibrated residual per value
6. Clamp to [0, 1]

Sources:
- Schwartz (2006, 2011) — Theory of basic human values
- Inglehart-Welzel cultural map
- PMC5549227 — Values and adult age
"""

from __future__ import annotations

import math

import numpy as np

from layer_zero.models import (
    MintRequest,
    SCHWARTZ_VALUES,
    SCHWARTZ_POSITIONS,
    TraitPrior,
)


# =============================================================================
# Constants
# =============================================================================

BASELINE_VALUE = 0.5
VALUE_RESIDUAL_SD = 0.04  # per-value residual noise
DEFAULT_AMPLITUDE_MEAN = 0.15  # how differentiated the average profile is
DEFAULT_AMPLITUDE_SD = 0.05

# Age-based value shifts (from Schwartz 2011, PMC5549227)
# Conservation increases with age, Openness-to-Change decreases
AGE_VALUE_SHIFTS: dict[str, list[tuple[int, int, float]]] = {
    # (age_min, age_max, delta)
    "self_direction":  [(18, 30, 0.03), (31, 50, 0.0), (51, 70, -0.02), (71, 100, -0.03)],
    "stimulation":     [(18, 30, 0.05), (31, 50, 0.0), (51, 70, -0.05), (71, 100, -0.07)],
    "hedonism":        [(18, 30, 0.03), (31, 50, 0.0), (51, 70, -0.02), (71, 100, -0.03)],
    "achievement":     [(18, 30, 0.02), (31, 50, 0.0), (51, 70, -0.03), (71, 100, -0.05)],
    "power":           [(18, 30, 0.01), (31, 50, 0.0), (51, 70, -0.02), (71, 100, -0.03)],
    "security":        [(18, 30, -0.03), (31, 50, 0.0), (51, 70, 0.03), (71, 100, 0.05)],
    "conformity":      [(18, 30, -0.03), (31, 50, 0.0), (51, 70, 0.03), (71, 100, 0.05)],
    "tradition":       [(18, 30, -0.05), (31, 50, 0.0), (51, 70, 0.05), (71, 100, 0.08)],
    "benevolence":     [(18, 30, -0.01), (31, 50, 0.02), (51, 70, 0.03), (71, 100, 0.04)],
    "universalism":    [(18, 30, 0.0), (31, 50, 0.01), (51, 70, 0.02), (71, 100, 0.03)],
}

# Occupation → value profile shifts (from Springer 2007, ESS data)
OCCUPATION_VALUE_SHIFTS: dict[str, dict[str, float]] = {
    "nurse": {"benevolence": 0.08, "universalism": 0.05, "security": 0.03, "power": -0.05, "achievement": -0.02},
    "doctor": {"achievement": 0.05, "benevolence": 0.05, "self_direction": 0.03, "power": 0.02},
    "teacher": {"benevolence": 0.08, "universalism": 0.05, "self_direction": 0.03, "conformity": 0.02},
    "engineer": {"achievement": 0.05, "self_direction": 0.05, "security": 0.02},
    "software engineer": {"self_direction": 0.08, "achievement": 0.05, "stimulation": 0.03},
    "data scientist": {"self_direction": 0.08, "achievement": 0.05, "universalism": 0.02},
    "researcher": {"self_direction": 0.10, "universalism": 0.05, "stimulation": 0.03},
    "artist": {"self_direction": 0.10, "stimulation": 0.08, "hedonism": 0.03, "conformity": -0.08, "tradition": -0.05},
    "musician": {"self_direction": 0.08, "stimulation": 0.08, "hedonism": 0.05, "tradition": -0.03},
    "entrepreneur": {"achievement": 0.10, "self_direction": 0.08, "power": 0.05, "stimulation": 0.03, "security": -0.05},
    "lawyer": {"power": 0.05, "achievement": 0.05, "security": 0.05, "conformity": 0.03},
    "accountant": {"security": 0.08, "conformity": 0.05, "achievement": 0.03, "stimulation": -0.05},
    "salesperson": {"achievement": 0.05, "hedonism": 0.03, "stimulation": 0.02, "power": 0.02},
    "social worker": {"benevolence": 0.10, "universalism": 0.08, "self_direction": 0.03, "power": -0.08},
    "journalist": {"self_direction": 0.08, "stimulation": 0.05, "universalism": 0.03},
    "chef": {"achievement": 0.08, "stimulation": 0.05, "hedonism": 0.03, "self_direction": 0.03},
    "consultant": {"achievement": 0.05, "self_direction": 0.05, "power": 0.03},
    "fitness coach": {"benevolence": 0.05, "achievement": 0.05, "stimulation": 0.05, "hedonism": 0.03},
    "scientist": {"self_direction": 0.10, "universalism": 0.05, "achievement": 0.05},
    "product manager": {"achievement": 0.05, "self_direction": 0.05, "stimulation": 0.02},
    "police officer": {"security": 0.10, "conformity": 0.08, "tradition": 0.05, "self_direction": -0.05},
    "pilot": {"stimulation": 0.05, "security": 0.05, "achievement": 0.05, "self_direction": 0.03},
    "developer": {"self_direction": 0.08, "achievement": 0.05, "stimulation": 0.03},
    "programmer": {"self_direction": 0.08, "achievement": 0.05, "stimulation": 0.03},
    "architect": {"self_direction": 0.08, "achievement": 0.05, "stimulation": 0.03},
    "data analyst": {"achievement": 0.05, "security": 0.03, "self_direction": 0.03},
    "project manager": {"achievement": 0.05, "conformity": 0.03, "security": 0.03},
    "writer": {"self_direction": 0.10, "stimulation": 0.05, "hedonism": 0.03, "conformity": -0.05},
    "editor": {"self_direction": 0.05, "achievement": 0.05, "conformity": 0.02},
    "photographer": {"self_direction": 0.08, "stimulation": 0.05, "hedonism": 0.03},
    "financial analyst": {"security": 0.08, "achievement": 0.05, "conformity": 0.03},
    "marketing manager": {"achievement": 0.05, "stimulation": 0.03, "power": 0.03},
    "counselor": {"benevolence": 0.10, "universalism": 0.05, "self_direction": 0.03},
    "dentist": {"security": 0.05, "achievement": 0.05, "benevolence": 0.03},
    "veterinarian": {"benevolence": 0.08, "universalism": 0.05, "self_direction": 0.03},
    "mechanic": {"self_direction": 0.03, "security": 0.05, "achievement": 0.02},
    "electrician": {"security": 0.05, "self_direction": 0.03, "achievement": 0.02},
    "paramedic": {"benevolence": 0.08, "security": 0.05, "stimulation": 0.03},
    "librarian": {"universalism": 0.05, "self_direction": 0.05, "tradition": 0.03, "security": 0.03},
    "biologist": {"self_direction": 0.10, "universalism": 0.05, "achievement": 0.03},
    "chemist": {"self_direction": 0.08, "achievement": 0.05, "security": 0.03},
    "physicist": {"self_direction": 0.10, "universalism": 0.05, "stimulation": 0.05},
}


# =============================================================================
# Circumplex generation
# =============================================================================

def generate_schwartz_values(
    request: MintRequest,
    count: int = 1,
    seed: int | None = None,
    culture_value_shifts: dict[str, float] | None = None,
) -> np.ndarray:
    """Generate Schwartz value profiles from circumplex structure.

    Args:
        request: MintRequest with optional value overrides, occupation, age.
        count: Number of value profiles to generate.
        seed: Random seed for reproducibility.
        culture_value_shifts: Optional culture-specific shifts (from big_five.get_culture_value_shifts).

    Returns:
        np.ndarray of shape (count, 10) with values in [0, 1].
        Column order matches SCHWARTZ_VALUES.
    """
    rng = np.random.default_rng(seed)
    positions = np.array([SCHWARTZ_POSITIONS[v] for v in SCHWARTZ_VALUES])

    # Compute demographic shifts per value
    demo_shifts = _compute_demographic_shifts(request, culture_value_shifts)

    # Sample circumplex parameters
    # Peak angle: uniform over [0, 2π] — no demographic prior on which value is "peak"
    # But if occupation has strong value profile, bias the angle toward the dominant value
    peak_bias = _compute_peak_bias(request)
    if peak_bias is not None:
        # Von Mises distribution centered on dominant value position
        peak_angles = rng.vonmises(peak_bias, kappa=1.5, size=count)
    else:
        peak_angles = rng.uniform(0, 2 * math.pi, size=count)

    # Amplitude: how differentiated the profile is
    amplitudes = rng.normal(DEFAULT_AMPLITUDE_MEAN, DEFAULT_AMPLITUDE_SD, size=count)
    amplitudes = np.clip(amplitudes, 0.05, 0.35)

    # Generate profiles
    profiles = np.zeros((count, 10))
    for i in range(count):
        # Base circumplex pattern
        cosine_pattern = amplitudes[i] * np.cos(peak_angles[i] - positions)
        base_values = BASELINE_VALUE + cosine_pattern

        # Add demographic shifts
        for j, val_name in enumerate(SCHWARTZ_VALUES):
            base_values[j] += demo_shifts.get(val_name, 0.0)

        # Add residual noise per value
        residuals = rng.normal(0, VALUE_RESIDUAL_SD, size=10)
        base_values += residuals

        # Apply user overrides (pin specific values)
        for val_name, override_val in request.values_overrides.items():
            if val_name in SCHWARTZ_VALUES:
                idx = SCHWARTZ_VALUES.index(val_name)
                base_values[idx] = override_val

        # Clamp to [0, 1]
        profiles[i] = np.clip(base_values, 0.0, 1.0)

    return profiles


def _compute_demographic_shifts(
    request: MintRequest,
    culture_shifts: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute per-value shifts from age, occupation, and culture."""
    shifts: dict[str, float] = {}

    # Age shifts
    if request.age:
        for val_name, brackets in AGE_VALUE_SHIFTS.items():
            for age_min, age_max, delta in brackets:
                if age_min <= request.age <= age_max:
                    shifts[val_name] = shifts.get(val_name, 0.0) + delta
                    break

    # Occupation shifts
    if request.occupation:
        occ_lower = request.occupation.lower().strip()
        occ_shifts = OCCUPATION_VALUE_SHIFTS.get(occ_lower)
        if not occ_shifts:
            # Partial match
            for key in OCCUPATION_VALUE_SHIFTS:
                if key in occ_lower:
                    occ_shifts = OCCUPATION_VALUE_SHIFTS[key]
                    break
        if occ_shifts:
            for val_name, delta in occ_shifts.items():
                shifts[val_name] = shifts.get(val_name, 0.0) + delta

    # Culture shifts
    if culture_shifts:
        for val_name, delta in culture_shifts.items():
            shifts[val_name] = shifts.get(val_name, 0.0) + delta

    # Cap total shift per value at ±0.20
    MAX_VALUE_SHIFT = 0.20
    for val_name in shifts:
        shifts[val_name] = max(-MAX_VALUE_SHIFT, min(MAX_VALUE_SHIFT, shifts[val_name]))

    return shifts


def _compute_peak_bias(request: MintRequest) -> float | None:
    """If occupation has a clear dominant value, return its circle position as bias."""
    if not request.occupation:
        return None

    occ_lower = request.occupation.lower().strip()
    occ_shifts = OCCUPATION_VALUE_SHIFTS.get(occ_lower)
    if not occ_shifts:
        for key in OCCUPATION_VALUE_SHIFTS:
            if key in occ_lower:
                occ_shifts = OCCUPATION_VALUE_SHIFTS[key]
                break

    if not occ_shifts:
        return None

    # Find the value with the largest positive shift
    best_val = max(occ_shifts, key=occ_shifts.get)  # type: ignore
    if occ_shifts[best_val] < 0.05:
        return None  # No strong signal

    return SCHWARTZ_POSITIONS.get(best_val)
