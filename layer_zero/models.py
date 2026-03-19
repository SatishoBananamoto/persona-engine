"""
Core data models for Layer Zero.

Three categories:
- Input models: MintRequest, SegmentRequest
- Output models: FieldProvenance, MintedPersona
- Internal models: TraitPrior, ResidualConfig
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Input Models
# =============================================================================

@dataclass
class MintRequest:
    """Normalized input for single-persona or small-batch minting (Tiers 1-4).

    Fields set to None are inferred downstream. Explicit values always win.
    """

    # Identity (None = infer or generate)
    name: str | None = None
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    industry: str | None = None
    location: str | None = None
    education: str | None = None
    culture_region: str | None = None

    # Trait hints — adjectives like "analytical", "warm", "cautious"
    trait_hints: list[str] = field(default_factory=list)

    # Direct overrides (user-specified psychological parameters)
    # These REPLACE priors — they are not combined with them.
    big_five_overrides: dict[str, float] = field(default_factory=dict)
    values_overrides: dict[str, float] = field(default_factory=dict)
    cognitive_overrides: dict[str, float] = field(default_factory=dict)
    communication_overrides: dict[str, float] = field(default_factory=dict)

    # Goals and domains (explicit)
    goals: list[str] = field(default_factory=list)
    domains: list[dict[str, Any]] = field(default_factory=list)

    # Generation settings
    count: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.age is not None and not (18 <= self.age <= 100):
            raise ValueError(f"age must be 18-100, got {self.age}")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        for trait, val in self.big_five_overrides.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"big_five override '{trait}' must be 0.0-1.0, got {val}")
        for val_name, val in self.values_overrides.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"values override '{val_name}' must be 0.0-1.0, got {val}")


@dataclass
class SegmentRequest:
    """Input for CSV-based segment minting (Tier 3).

    Supports ranges and distributions, not just point values.
    """

    segment_name: str = ""
    age_range: tuple[int, int] = (25, 55)
    gender_distribution: dict[str, float] = field(
        default_factory=lambda: {"female": 0.5, "male": 0.5}
    )
    occupations: list[str] = field(default_factory=list)
    location: str | None = None
    culture_region: str | None = None

    # Optional trait constraints (ranges, not point values)
    trait_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    count: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        lo, hi = self.age_range
        if not (18 <= lo <= hi <= 100):
            raise ValueError(f"age_range must be (18-100, 18-100) with lo <= hi, got {self.age_range}")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        total = sum(self.gender_distribution.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"gender_distribution must sum to ~1.0, got {total}")


# =============================================================================
# Provenance
# =============================================================================

SOURCE_TYPES = ("explicit", "sampled", "derived", "template", "default")
DEPTH_DECAY = 0.85  # confidence multiplier per inferential hop


@dataclass(frozen=True)
class FieldProvenance:
    """Provenance metadata for a single generated field.

    Tracks how the value was produced so it can be audited, debugged,
    and assessed for confidence.
    """

    value: Any
    source: Literal["explicit", "sampled", "derived", "template", "default"]
    confidence: float  # 0.0-1.0
    mapping_strength: float = 1.0  # strength of empirical mapping used (0-1)
    inferential_depth: int = 0  # steps from direct evidence
    parent_fields: tuple[str, ...] = ()  # what this was derived from
    notes: str = ""

    @staticmethod
    def compute_confidence(
        source: str,
        mapping_strength: float = 1.0,
        inferential_depth: int = 0,
    ) -> float:
        """Compute confidence from source type, mapping strength, and depth."""
        base = {
            "explicit": 0.95,
            "sampled": 0.70,
            "derived": 0.60,
            "template": 0.40,
            "default": 0.30,
        }.get(source, 0.30)
        return base * mapping_strength * (DEPTH_DECAY ** inferential_depth)


# =============================================================================
# Output Models
# =============================================================================

@dataclass
class MintedPersona:
    """A generated persona with full provenance metadata.

    Attributes:
        persona: Engine-compatible Persona object (from persona_engine.schema).
        provenance: Mapping of field path → FieldProvenance.
        warnings: Validation warnings (if any).
    """

    persona: Any  # persona_engine.schema.persona_schema.Persona
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.persona.label

    @property
    def persona_id(self) -> str:
        return self.persona.persona_id


# =============================================================================
# Internal Models
# =============================================================================

@dataclass
class TraitPrior:
    """Prior distribution for a single trait: (mean, std_dev) plus metadata."""

    mean: float
    std_dev: float
    source: str = "default"  # "occupation", "age", "gender", "culture", "override"
    mapping_strength: float = 0.5  # strength of the mapping that produced this prior

    def __post_init__(self) -> None:
        self.mean = max(0.01, min(0.99, self.mean))  # keep away from 0/1 for logit
        self.std_dev = max(0.01, self.std_dev)


@dataclass
class ResidualConfig:
    """Configuration for calibrated residual variance at a derivation step."""

    field_name: str
    std_dev: float = 0.08
    shared_group: str | None = None  # fields in the same group share a latent residual


# Big Five trait names (canonical order)
BIG_FIVE_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

# Schwartz value names (circle order, clockwise)
SCHWARTZ_VALUES = (
    "self_direction", "stimulation", "hedonism", "achievement", "power",
    "security", "conformity", "tradition", "benevolence", "universalism",
)

# Schwartz circle positions (radians, 36° apart, starting at 0 for self_direction)
SCHWARTZ_POSITIONS = {
    name: i * (2 * math.pi / 10)
    for i, name in enumerate(SCHWARTZ_VALUES)
}

# Schwartz opposing pairs (values ~180° apart on the circle)
SCHWARTZ_OPPOSING_PAIRS = [
    ("self_direction", "conformity"),
    ("self_direction", "tradition"),
    ("stimulation", "conformity"),
    ("stimulation", "tradition"),
    ("hedonism", "conformity"),
    ("achievement", "benevolence"),
    ("power", "universalism"),
    ("power", "benevolence"),
    ("security", "self_direction"),
    ("security", "stimulation"),
]

# Schwartz adjacent pairs (values next to each other on circle)
SCHWARTZ_ADJACENT_PAIRS = [
    (SCHWARTZ_VALUES[i], SCHWARTZ_VALUES[(i + 1) % 10])
    for i in range(10)
]
