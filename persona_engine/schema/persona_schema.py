"""
Persona Schema - Complete Pydantic v2 Model

Defines the full persona profile including psychology, knowledge, behavior,
constraints, and invariants.

IMPORTANT: This module contains ONLY data schemas. No runtime objects,
callables, or engine dependencies should be stored in these models.
"""

from enum import StrEnum
from typing import Literal

import warnings

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Enums and Constants
# ============================================================================

class SchwartzValueType(StrEnum):
    """Schwartz's 10 basic human values"""
    SELF_DIRECTION = "self_direction"
    STIMULATION = "stimulation"
    HEDONISM = "hedonism"
    ACHIEVEMENT = "achievement"
    POWER = "power"
    SECURITY = "security"
    CONFORMITY = "conformity"
    TRADITION = "tradition"
    BENEVOLENCE = "benevolence"
    UNIVERSALISM = "universalism"


# ============================================================================
# Psychology Models
# ============================================================================

class BigFiveTraits(BaseModel):
    """
    Big Five personality traits (OCEAN model)
    Each trait ranges from 0.0 (low) to 1.0 (high)
    """

    openness: float = Field(
        ge=0.0,
        le=1.0,
        description="Curiosity, creativity, preference for novelty"
    )

    conscientiousness: float = Field(
        ge=0.0,
        le=1.0,
        description="Organization, dependability, discipline"
    )

    extraversion: float = Field(
        ge=0.0,
        le=1.0,
        description="Sociability, assertiveness, energy in social settings"
    )

    agreeableness: float = Field(
        ge=0.0,
        le=1.0,
        description="Compassion, cooperation, trust in others"
    )

    neuroticism: float = Field(
        ge=0.0,
        le=1.0,
        description="Emotional instability, anxiety, stress response"
    )


class SchwartzValues(BaseModel):
    """
    Schwartz values theory - 10 basic human values
    Values are weighted 0.0-1.0 indicating priority
    """

    self_direction: float = Field(ge=0.0, le=1.0, description="Independence, autonomy")
    stimulation: float = Field(ge=0.0, le=1.0, description="Excitement, novelty, challenge")
    hedonism: float = Field(ge=0.0, le=1.0, description="Pleasure, gratification")
    achievement: float = Field(ge=0.0, le=1.0, description="Personal success, competence")
    power: float = Field(ge=0.0, le=1.0, description="Status, control over resources")
    security: float = Field(ge=0.0, le=1.0, description="Safety, stability, harmony")
    conformity: float = Field(ge=0.0, le=1.0, description="Restraint, obedience to norms")
    tradition: float = Field(ge=0.0, le=1.0, description="Respect for customs, commitment")
    benevolence: float = Field(ge=0.0, le=1.0, description="Preserving welfare of close others")
    universalism: float = Field(ge=0.0, le=1.0, description="Understanding, tolerance, protecting all people")


class CognitiveStyle(BaseModel):
    """
    How the persona thinks and processes information
    """

    analytical_intuitive: float = Field(
        ge=0.0,
        le=1.0,
        description="0=highly intuitive, 1=highly analytical"
    )

    systematic_heuristic: float = Field(
        ge=0.0,
        le=1.0,
        description="0=heuristic shortcuts, 1=systematic processing"
    )

    risk_tolerance: float = Field(
        ge=0.0,
        le=1.0,
        description="Willingness to take chances"
    )

    need_for_closure: float = Field(
        ge=0.0,
        le=1.0,
        description="Desire for definite answers (intolerance of ambiguity)"
    )

    cognitive_complexity: float = Field(
        ge=0.0,
        le=1.0,
        description="Ability to hold nuanced, multifaceted views"
    )


class CommunicationPreferences(BaseModel):
    """
    Base communication style (before context adjustments)
    """

    verbosity: float = Field(
        ge=0.0,
        le=1.0,
        description="0=very brief, 1=very detailed"
    )

    formality: float = Field(
        ge=0.0,
        le=1.0,
        description="0=casual, 1=very formal"
    )

    directness: float = Field(
        ge=0.0,
        le=1.0,
        description="0=very indirect/diplomatic, 1=very direct/blunt"
    )

    emotional_expressiveness: float = Field(
        ge=0.0,
        le=1.0,
        description="0=reserved, 1=emotionally expressive"
    )


class PersonalityProfile(BaseModel):
    """Complete psychological profile"""

    big_five: BigFiveTraits
    values: SchwartzValues
    cognitive_style: CognitiveStyle
    communication: CommunicationPreferences


# ============================================================================
# Identity & Knowledge Models
# ============================================================================

class Identity(BaseModel):
    """Core demographic and background information"""

    age: int = Field(ge=18, le=100)
    gender: str | None = None
    location: str = Field(description="City, Country or region")
    education: str
    occupation: str
    background: str = Field(description="Brief life story / context")


class DomainKnowledge(BaseModel):
    """Knowledge in a specific domain"""

    domain: str = Field(examples=["Psychology", "Technology", "Finance"])
    proficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="0=no knowledge, 0.7+=expert level"
    )
    subdomains: list[str] = Field(default_factory=list)


class LanguageKnowledge(BaseModel):
    """Language proficiency"""

    language: str
    proficiency: float = Field(ge=0.0, le=1.0)
    accent: str | None = None


class CulturalKnowledge(BaseModel):
    """Cultural familiarity"""

    primary_culture: str
    exposure_level: dict[str, float] = Field(
        description="Culture region -> exposure level (0-1)",
        examples=[{"european": 0.8, "american": 0.6, "asian": 0.3}]
    )


# ============================================================================
# Behavioral & Constraint Models
# ============================================================================

class Goal(BaseModel):
    """Life/work goal with importance weight"""

    goal: str
    weight: float = Field(ge=0.0, le=1.0)


class SocialRole(BaseModel):
    """Communication style adjustments for social context"""

    formality: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)
    emotional_expressiveness: float = Field(ge=0.0, le=1.0)


class UncertaintyPolicy(BaseModel):
    """How persona handles uncertain knowledge"""

    admission_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Below this confidence, admits uncertainty"
    )
    hedging_frequency: float = Field(ge=0.0, le=1.0)
    clarification_tendency: float = Field(ge=0.0, le=1.0)
    knowledge_boundary_strictness: float = Field(ge=0.0, le=1.0)


class ClaimPolicy(BaseModel):
    """
    Prevents hallucination by enforcing knowledge boundaries.

    NOTE: This is a DATA-ONLY model. The enforcement logic lives in
    the validator module, not here.
    """

    allowed_claim_types: list[str] = Field(
        default=["personal_experience", "general_common_knowledge"],
        description="Types of claims this persona can make"
    )

    citation_required_when: dict[str, float | bool] = Field(
        default_factory=dict,
        description="Conditions requiring citations",
        examples=[{"proficiency_below": 0.6, "factual_or_time_sensitive": True}]
    )

    lookup_behavior: Literal["ask", "hedge", "refuse", "speculate"] = Field(
        default="hedge",
        description="What to do when knowledge is uncertain"
    )

    expert_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum domain proficiency for expert-level claims",
    )


class PersonaInvariants(BaseModel):
    """
    Explicit constraints that make validation simple
    """

    identity_facts: list[str] = Field(
        description="Immutable facts about persona identity",
        examples=[["Lives in London, UK", "Age 34", "UX Researcher"]]
    )

    cannot_claim: list[str] = Field(
        default_factory=list,
        description="Roles/credentials persona cannot claim",
        examples=[["medical doctor", "licensed therapist", "lawyer"]]
    )

    must_avoid: list[str] = Field(
        default_factory=list,
        description="Topics/actions persona must not engage in",
        examples=[["revealing employer name", "sharing participant data"]]
    )


class TopicSensitivity(BaseModel):
    """Sensitivity level for specific topics"""

    topic: str
    sensitivity: float = Field(ge=0.0, le=1.0, description="0=open, 1=very sensitive")


class DisclosurePolicy(BaseModel):
    """
    Function-based disclosure calculation (data only).

    The actual computation happens in the behavioral engine,
    not in this schema. This just stores the parameters.
    """

    base_openness: float = Field(ge=0.0, le=1.0)
    factors: dict[str, float] = Field(
        description="Multipliers for disclosure calculation",
        examples=[{
            "topic_sensitivity": -0.3,
            "trust_level": 0.4,
            "formal_context": -0.2,
            "positive_mood": 0.15
        }]
    )
    bounds: tuple[float, float] = Field(default=(0.1, 0.9))


class DecisionPolicy(BaseModel):
    """Conditional decision rule"""

    condition: str
    approach: str
    time_needed: str | None = None


class ResponsePattern(BaseModel):
    """Behavioral pattern for specific triggers"""

    trigger: str
    response: str
    emotionality: float = Field(ge=0.0, le=1.0)


class Bias(BaseModel):
    """Subtle, bounded human bias"""

    type: str
    strength: float = Field(ge=0.0, le=1.0, description="How strong this bias is")


# ============================================================================
# State Models
# ============================================================================

class DynamicState(BaseModel):
    """
    Changing state during interaction
    """

    mood_valence: float = Field(ge=-1.0, le=1.0, description="-1=negative, +1=positive")
    mood_arousal: float = Field(ge=0.0, le=1.0, description="Energy level")
    fatigue: float = Field(ge=0.0, le=1.0)
    stress: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0, description="Interest in current topic")


# ============================================================================
# Main Persona Model
# ============================================================================

class Persona(BaseModel):
    """
    Complete persona specification with full psychological framework.

    CRITICAL: This model contains ONLY serializable data. No callables,
    no engine objects, no runtime dependencies.
    """

    # Metadata
    persona_id: str
    version: str = "1.0"
    label: str = Field(description="Human-readable name/description")

    # Core identity
    identity: Identity

    # Psychological profile
    psychology: PersonalityProfile

    # Knowledge
    knowledge_domains: list[DomainKnowledge] = Field(default_factory=list)
    languages: list[LanguageKnowledge] = Field(
        default_factory=list,
        description="Reserved for future multi-language support. Currently unused by the planner pipeline.",
    )
    cultural_knowledge: CulturalKnowledge | None = None

    # Goals & motivations
    primary_goals: list[Goal] = Field(default_factory=list)
    secondary_goals: list[Goal] = Field(default_factory=list)

    # Social roles
    social_roles: dict[str, SocialRole] = Field(
        description="Context-specific communication adjustments"
    )

    # Uncertainty handling
    uncertainty: UncertaintyPolicy

    # Knowledge claim policy
    claim_policy: ClaimPolicy

    # Invariants (explicit constraints)
    invariants: PersonaInvariants

    # Constraints
    time_scarcity: float = Field(ge=0.0, le=1.0)
    privacy_sensitivity: float = Field(ge=0.0, le=1.0)
    disclosure_policy: DisclosurePolicy
    topic_sensitivities: list[TopicSensitivity] = Field(default_factory=list)

    # Behavioral rules
    decision_policies: list[DecisionPolicy] = Field(default_factory=list)
    response_patterns: list[ResponsePattern] = Field(default_factory=list)
    biases: list[Bias] = Field(default_factory=list)

    # Dynamic state (initial values)
    initial_state: DynamicState

    model_config = {
        "json_schema_extra": {
            "title": "Persona",
            "description": "Complete persona profile with psychological framework"
        }
    }

    @field_validator('social_roles')
    @classmethod
    def ensure_default_role(cls, v: dict[str, "SocialRole"]) -> dict[str, "SocialRole"]:
        """Ensure 'default' social role exists"""
        if 'default' not in v:
            raise ValueError("social_roles must include a 'default' role")
        return v

    @model_validator(mode='after')
    def warn_unused_languages(self) -> "Persona":
        """Emit a warning if languages[] is populated since it's not yet wired."""
        if self.languages:
            warnings.warn(
                f"Persona '{self.persona_id}': languages field is populated with "
                f"{len(self.languages)} entries but is currently unused by the planner "
                f"pipeline. Multi-language support is planned for a future release.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def to_dict(self) -> dict:
        """Export persona as a plain dictionary (YAML-compatible)."""
        return self.model_dump(mode="python")

    def to_yaml(self, path: str | None = None) -> str:
        """Export persona as YAML string. Optionally write to file.

        Args:
            path: If provided, write YAML to this file path.

        Returns:
            The YAML string.
        """
        import yaml as _yaml
        from pathlib import Path as _Path

        data = self.model_dump(mode="json")
        yaml_str = _yaml.dump(
            data, default_flow_style=False, sort_keys=False, allow_unicode=True,
        )
        if path is not None:
            _Path(path).write_text(yaml_str)
        return yaml_str
