"""
Persona Builder — Create personas without hand-crafting YAML.

Three ways to create a persona:

1. **Builder API** (no LLM needed)::

    from persona_engine import PersonaBuilder

    persona = (
        PersonaBuilder("Marcus", "Chef")
        .age(41)
        .location("Chicago, IL")
        .traits("passionate", "direct", "opinionated")
        .archetype("expert")
        .build()
    )

2. **From natural-language description** (no LLM needed)::

    persona = PersonaBuilder.from_description(
        "A 45-year-old French chef named Marcus, passionate and direct, "
        "lives in Chicago"
    )

3. **Archetype shortcut**::

    persona = PersonaBuilder.archetype_persona("expert", name="Dr. Lee", occupation="Physicist")

All approaches produce a fully valid ``Persona`` object with sensible defaults
for every field the user doesn't explicitly set.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from persona_engine.exceptions import PersonaValidationError

from persona_engine.schema.persona_schema import (
    Bias,
    BigFiveTraits,
    ClaimPolicy,
    CognitiveStyle,
    CommunicationPreferences,
    CulturalKnowledge,
    DecisionPolicy,
    DisclosurePolicy,
    DomainKnowledge,
    DynamicState,
    Goal,
    Identity,
    LanguageKnowledge,
    Persona,
    PersonaInvariants,
    PersonalityProfile,
    ResponsePattern,
    SchwartzValues,
    SocialRole,
    TopicSensitivity,
    UncertaintyPolicy,
)


# ============================================================================
# Trait Adjective Mappings
# ============================================================================

# Maps personality adjectives to deltas on specific fields.
# Format: "adjective" -> { "section.field": delta }
# Deltas are additive and clamped to [0, 1] at build time.

TRAIT_MODIFIERS: dict[str, dict[str, float]] = {
    # Temperament
    "passionate": {
        "big_five.extraversion": +0.12,
        "big_five.openness": +0.08,
        "communication.emotional_expressiveness": +0.15,
    },
    "calm": {
        "big_five.neuroticism": -0.15,
        "values.security": +0.10,
        "communication.emotional_expressiveness": -0.10,
    },
    "anxious": {
        "big_five.neuroticism": +0.20,
        "values.security": +0.15,
    },
    "confident": {
        "big_five.neuroticism": -0.15,
        "values.power": +0.10,
        "communication.directness": +0.10,
    },
    "shy": {
        "big_five.extraversion": -0.20,
        "communication.emotional_expressiveness": -0.15,
        "communication.verbosity": -0.10,
    },

    # Communication style
    "direct": {
        "communication.directness": +0.20,
        "big_five.agreeableness": -0.10,
    },
    "blunt": {
        "communication.directness": +0.25,
        "big_five.agreeableness": -0.15,
    },
    "diplomatic": {
        "communication.directness": -0.15,
        "big_five.agreeableness": +0.10,
    },
    "formal": {
        "communication.formality": +0.20,
    },
    "casual": {
        "communication.formality": -0.20,
        "communication.directness": +0.05,
    },
    "verbose": {
        "communication.verbosity": +0.20,
    },
    "concise": {
        "communication.verbosity": -0.20,
    },
    "reserved": {
        "big_five.extraversion": -0.15,
        "communication.emotional_expressiveness": -0.15,
        "communication.verbosity": -0.10,
    },
    "expressive": {
        "communication.emotional_expressiveness": +0.20,
        "big_five.extraversion": +0.10,
    },

    # Social
    "friendly": {
        "big_five.agreeableness": +0.15,
        "big_five.extraversion": +0.10,
    },
    "warm": {
        "big_five.agreeableness": +0.15,
        "communication.emotional_expressiveness": +0.10,
        "values.benevolence": +0.15,
    },
    "empathetic": {
        "big_five.agreeableness": +0.15,
        "values.benevolence": +0.15,
        "values.universalism": +0.10,
    },
    "nurturing": {
        "values.benevolence": +0.20,
        "big_five.agreeableness": +0.15,
    },
    "competitive": {
        "values.achievement": +0.15,
        "values.power": +0.10,
        "big_five.agreeableness": -0.10,
    },
    "stubborn": {
        "big_five.agreeableness": -0.15,
        "cognitive_style.need_for_closure": +0.10,
    },

    # Cognitive
    "analytical": {
        "cognitive_style.analytical_intuitive": +0.20,
        "cognitive_style.cognitive_complexity": +0.15,
    },
    "intuitive": {
        "cognitive_style.analytical_intuitive": -0.20,
    },
    "systematic": {
        "cognitive_style.systematic_heuristic": +0.20,
        "cognitive_style.analytical_intuitive": +0.15,
    },
    "practical": {
        "cognitive_style.analytical_intuitive": -0.10,
        "cognitive_style.systematic_heuristic": -0.10,
    },
    "creative": {
        "big_five.openness": +0.20,
        "values.stimulation": +0.15,
        "values.self_direction": +0.10,
    },
    "curious": {
        "big_five.openness": +0.15,
        "values.self_direction": +0.10,
    },
    "meticulous": {
        "big_five.conscientiousness": +0.20,
        "cognitive_style.systematic_heuristic": +0.15,
    },
    "organized": {
        "big_five.conscientiousness": +0.15,
        "cognitive_style.systematic_heuristic": +0.10,
    },
    "detail-oriented": {
        "big_five.conscientiousness": +0.15,
        "cognitive_style.systematic_heuristic": +0.10,
    },
    "big-picture": {
        "cognitive_style.cognitive_complexity": +0.15,
        "cognitive_style.systematic_heuristic": -0.10,
    },
    "opinionated": {
        "communication.directness": +0.15,
        "cognitive_style.need_for_closure": +0.15,
    },
    "open-minded": {
        "big_five.openness": +0.15,
        "cognitive_style.need_for_closure": -0.15,
    },

    # Risk / adventure
    "cautious": {
        "cognitive_style.risk_tolerance": -0.20,
        "values.security": +0.15,
        "big_five.neuroticism": +0.10,
    },
    "adventurous": {
        "big_five.openness": +0.15,
        "cognitive_style.risk_tolerance": +0.15,
        "values.stimulation": +0.15,
    },
    "risk-taking": {
        "cognitive_style.risk_tolerance": +0.20,
    },

    # Values-driven
    "ambitious": {
        "values.achievement": +0.20,
        "values.power": +0.10,
    },
    "humble": {
        "values.power": -0.15,
        "big_five.agreeableness": +0.10,
    },
    "traditional": {
        "values.tradition": +0.20,
        "values.conformity": +0.15,
    },
    "progressive": {
        "values.universalism": +0.15,
        "values.self_direction": +0.10,
        "values.tradition": -0.10,
    },
}


# ============================================================================
# Occupation → Domain Mappings
# ============================================================================

# Maps occupation keywords to (domain, proficiency) tuples.
# Multiple keywords can map to the same occupation profile.

OCCUPATION_DOMAINS: dict[str, list[tuple[str, float]]] = {
    "chef": [("Food", 0.90), ("Business", 0.50), ("Health", 0.35)],
    "cook": [("Food", 0.75), ("Health", 0.30)],
    "baker": [("Food", 0.80)],
    "bartender": [("Food", 0.65), ("Business", 0.35)],
    "researcher": [("Science", 0.75), ("Technology", 0.50)],
    "ux researcher": [("Psychology", 0.85), ("Technology", 0.70), ("Business", 0.50)],
    "physicist": [("Science", 0.90), ("Technology", 0.60), ("Education", 0.50)],
    "scientist": [("Science", 0.85), ("Technology", 0.55)],
    "chemist": [("Science", 0.85), ("Health", 0.40)],
    "biologist": [("Science", 0.85), ("Health", 0.45)],
    "musician": [("Arts", 0.85), ("Education", 0.40)],
    "artist": [("Arts", 0.85)],
    "writer": [("Arts", 0.80), ("Education", 0.35)],
    "actor": [("Arts", 0.80)],
    "filmmaker": [("Arts", 0.80), ("Technology", 0.40)],
    "photographer": [("Arts", 0.75), ("Technology", 0.40)],
    "designer": [("Arts", 0.75), ("Technology", 0.50)],
    "lawyer": [("Law", 0.90), ("Business", 0.55)],
    "attorney": [("Law", 0.90), ("Business", 0.55)],
    "judge": [("Law", 0.90)],
    "paralegal": [("Law", 0.65), ("Business", 0.40)],
    "fitness coach": [("Sports", 0.80), ("Health", 0.70)],
    "personal trainer": [("Sports", 0.80), ("Health", 0.65)],
    "coach": [("Sports", 0.75), ("Health", 0.55)],
    "athlete": [("Sports", 0.85), ("Health", 0.55)],
    "teacher": [("Education", 0.80)],
    "professor": [("Education", 0.85), ("Science", 0.60)],
    "tutor": [("Education", 0.70)],
    "doctor": [("Health", 0.90), ("Science", 0.60)],
    "physician": [("Health", 0.90), ("Science", 0.55)],
    "nurse": [("Health", 0.80)],
    "therapist": [("Psychology", 0.85), ("Health", 0.50)],
    "psychologist": [("Psychology", 0.90), ("Health", 0.45), ("Education", 0.40)],
    "counselor": [("Psychology", 0.75), ("Health", 0.40)],
    "engineer": [("Technology", 0.80), ("Science", 0.50)],
    "software engineer": [("Technology", 0.85)],
    "developer": [("Technology", 0.80)],
    "data scientist": [("Technology", 0.80), ("Science", 0.60)],
    "accountant": [("Finance", 0.85), ("Business", 0.60)],
    "financial advisor": [("Finance", 0.85), ("Business", 0.55)],
    "banker": [("Finance", 0.80), ("Business", 0.60)],
    "trader": [("Finance", 0.80)],
    "consultant": [("Business", 0.75)],
    "manager": [("Business", 0.70)],
    "entrepreneur": [("Business", 0.80), ("Finance", 0.50)],
    "ceo": [("Business", 0.85), ("Finance", 0.55)],
    "marketer": [("Business", 0.70), ("Psychology", 0.40)],
    "journalist": [("Arts", 0.60), ("Business", 0.40)],
    "nutritionist": [("Health", 0.80), ("Food", 0.55), ("Science", 0.40)],
    "dietitian": [("Health", 0.80), ("Food", 0.55)],
    "pharmacist": [("Health", 0.80), ("Science", 0.55)],
    "veterinarian": [("Health", 0.75), ("Science", 0.55)],
    "architect": [("Arts", 0.65), ("Technology", 0.55), ("Business", 0.40)],
    "pilot": [("Technology", 0.60), ("Science", 0.40)],
    "mechanic": [("Technology", 0.65)],
    "electrician": [("Technology", 0.60)],
}

# Maps (occupation_key, domain) → subdomains for richer domain detection.
OCCUPATION_SUBDOMAINS: dict[str, dict[str, list[str]]] = {
    "chef": {
        "Food": ["French cuisine", "Fermentation", "Butchery", "Pastry", "Farm-to-table", "Food safety"],
        "Business": ["Restaurant management", "Supply chain", "Cost control"],
        "Health": ["Nutrition basics", "Food allergies", "Dietary restrictions"],
    },
    "lawyer": {
        "Law": ["Corporate Law", "Contract Law", "Litigation", "International Law"],
        "Business": ["Corporate finance", "Governance", "Compliance"],
    },
    "software engineer": {
        "Technology": ["Programming", "System architecture", "Testing", "DevOps"],
    },
    "developer": {
        "Technology": ["Web development", "APIs", "Databases", "Version control"],
    },
    "data scientist": {
        "Technology": ["Machine learning", "Data pipelines", "Visualization"],
        "Science": ["Statistics", "Experimental design"],
    },
    "researcher": {
        "Science": ["Research methods", "Data analysis", "Experimental design"],
        "Technology": ["Scientific computing", "Data tools"],
    },
    "ux researcher": {
        "Psychology": ["UX research", "Behavioral science", "Cognitive psychology"],
        "Technology": ["UX tools", "Data analysis", "Web technologies"],
        "Business": ["Project management", "Stakeholder management"],
    },
    "physicist": {
        "Science": ["Quantum Physics", "Theoretical Physics", "Mathematics"],
        "Technology": ["Scientific computing", "Simulation"],
    },
    "doctor": {
        "Health": ["Diagnosis", "Treatment", "Patient care", "Clinical research"],
        "Science": ["Medical research", "Pharmacology"],
    },
    "nurse": {
        "Health": ["Patient care", "Clinical procedures", "Health education"],
    },
    "therapist": {
        "Psychology": ["Counseling", "CBT", "Trauma therapy", "Group therapy"],
        "Health": ["Mental health", "Wellness"],
    },
    "psychologist": {
        "Psychology": ["Clinical assessment", "Research methods", "Behavioral analysis"],
        "Health": ["Mental health", "Neuropsychology"],
    },
    "teacher": {
        "Education": ["Curriculum design", "Classroom management", "Assessment"],
    },
    "professor": {
        "Education": ["University teaching", "Research supervision", "Academic publishing"],
        "Science": ["Research methodology", "Peer review"],
    },
    "accountant": {
        "Finance": ["Tax preparation", "Auditing", "Financial reporting"],
        "Business": ["Compliance", "Payroll", "Budgeting"],
    },
    "financial advisor": {
        "Finance": ["Portfolio management", "Retirement planning", "Tax strategy"],
        "Business": ["Client relations", "Regulatory compliance"],
    },
    "musician": {
        "Arts": ["Jazz", "Music theory", "Improvisation", "Music history"],
    },
    "designer": {
        "Arts": ["Visual design", "Typography", "Branding", "UI/UX"],
        "Technology": ["Design tools", "Prototyping", "Web design"],
    },
    "journalist": {
        "Arts": ["Investigative reporting", "Feature writing", "Interviewing"],
        "Business": ["Media industry", "Publishing"],
    },
    "architect": {
        "Arts": ["Architectural design", "Urban planning", "Sustainability"],
        "Technology": ["CAD", "Building information modeling"],
        "Business": ["Project management", "Client relations"],
    },
    "consultant": {
        "Business": ["Strategy", "Process improvement", "Change management"],
    },
    "entrepreneur": {
        "Business": ["Startup strategy", "Fundraising", "Product development"],
        "Finance": ["Venture capital", "Cash flow management"],
    },
    "fitness coach": {
        "Sports": ["Strength training", "Nutrition", "Exercise science"],
        "Health": ["Injury prevention", "Wellness"],
    },
    "veterinarian": {
        "Health": ["Animal medicine", "Surgery", "Preventive care"],
        "Science": ["Zoology", "Pathology"],
    },
    "pharmacist": {
        "Health": ["Medication management", "Drug interactions", "Patient counseling"],
        "Science": ["Pharmacology", "Chemistry"],
    },
    "pilot": {
        "Technology": ["Avionics", "Navigation systems", "Flight planning"],
        "Science": ["Aerodynamics", "Meteorology"],
    },
    "mechanic": {
        "Technology": ["Engine repair", "Diagnostics", "Electrical systems"],
    },
    "electrician": {
        "Technology": ["Wiring", "Circuit design", "Safety codes", "Troubleshooting"],
    },
}


# ============================================================================
# Archetype Profiles
# ============================================================================

# Each archetype provides overrides on top of the neutral baseline (0.5).
# Only non-default values need to be specified.

ARCHETYPES: dict[str, dict[str, dict[str, float]]] = {
    "expert": {
        "big_five": {"conscientiousness": 0.75, "openness": 0.60},
        "values": {"achievement": 0.80, "self_direction": 0.65},
        "cognitive_style": {
            "analytical_intuitive": 0.75,
            "cognitive_complexity": 0.70,
            "systematic_heuristic": 0.70,
        },
        "communication": {"directness": 0.70, "formality": 0.55},
    },
    "coach": {
        "big_five": {"agreeableness": 0.80, "extraversion": 0.70, "openness": 0.65},
        "values": {"benevolence": 0.85, "universalism": 0.70},
        "cognitive_style": {"cognitive_complexity": 0.65},
        "communication": {"emotional_expressiveness": 0.70, "verbosity": 0.60},
    },
    "creative": {
        "big_five": {"openness": 0.85, "extraversion": 0.65, "conformity": 0.30},
        "values": {
            "self_direction": 0.80,
            "stimulation": 0.75,
            "conformity": 0.20,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.30,
            "cognitive_complexity": 0.75,
        },
        "communication": {"emotional_expressiveness": 0.75},
    },
    "analyst": {
        "big_five": {"conscientiousness": 0.80, "openness": 0.55},
        "values": {"achievement": 0.70, "security": 0.65},
        "cognitive_style": {
            "analytical_intuitive": 0.85,
            "systematic_heuristic": 0.80,
            "cognitive_complexity": 0.75,
        },
        "communication": {
            "directness": 0.65,
            "emotional_expressiveness": 0.30,
            "formality": 0.60,
        },
    },
    "caregiver": {
        "big_five": {"agreeableness": 0.85, "neuroticism": 0.40, "extraversion": 0.60},
        "values": {"benevolence": 0.90, "security": 0.65, "universalism": 0.75},
        "cognitive_style": {"cognitive_complexity": 0.55},
        "communication": {
            "emotional_expressiveness": 0.75,
            "directness": 0.40,
            "formality": 0.40,
        },
    },
    "leader": {
        "big_five": {
            "extraversion": 0.75,
            "conscientiousness": 0.75,
            "openness": 0.60,
        },
        "values": {"achievement": 0.75, "power": 0.65, "self_direction": 0.70},
        "cognitive_style": {
            "cognitive_complexity": 0.70,
            "risk_tolerance": 0.65,
        },
        "communication": {"directness": 0.75, "formality": 0.55, "verbosity": 0.55},
    },
}


# ============================================================================
# PersonaBuilder
# ============================================================================


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class PersonaBuilder:
    """Fluent builder for creating Persona objects with sensible defaults.

    Minimal usage::

        persona = PersonaBuilder("Marcus", "Chef").build()

    Everything else is optional — the builder fills in psychologically
    coherent defaults for ~50 fields.
    """

    def __init__(self, name: str, occupation: str) -> None:
        self._name = name
        self._occupation = occupation

        # Identity
        self._age: int = 30
        self._gender: str | None = None
        self._location: str = "United States"
        self._education: str = ""
        self._background: str = ""

        # Psychology — start at neutral 0.5 baseline
        self._big_five: dict[str, float] = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
        self._values: dict[str, float] = {
            "self_direction": 0.5,
            "stimulation": 0.5,
            "hedonism": 0.5,
            "achievement": 0.5,
            "power": 0.5,
            "security": 0.5,
            "conformity": 0.5,
            "tradition": 0.5,
            "benevolence": 0.5,
            "universalism": 0.5,
        }
        self._cognitive_style: dict[str, float] = {
            "analytical_intuitive": 0.5,
            "systematic_heuristic": 0.5,
            "risk_tolerance": 0.5,
            "need_for_closure": 0.5,
            "cognitive_complexity": 0.5,
        }
        self._communication: dict[str, float] = {
            "verbosity": 0.5,
            "formality": 0.5,
            "directness": 0.5,
            "emotional_expressiveness": 0.5,
        }

        # Optional overrides
        self._knowledge_domains: list[tuple[str, float, list[str]]] | None = None
        self._languages: list[tuple[str, float, str | None]] | None = None
        self._primary_culture: str | None = None
        self._goals: list[tuple[str, float]] = []
        self._archetype_name: str | None = None
        self._applied_traits: list[str] = []

        # Fine-tuning overrides
        self._time_scarcity: float | None = None
        self._privacy_sensitivity: float | None = None
        self._lookup_behavior: Literal["ask", "hedge", "refuse", "speculate"] | None = None

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def age(self, value: int) -> PersonaBuilder:
        """Set persona age (18-100)."""
        self._age = max(18, min(100, value))
        return self

    def gender(self, value: str) -> PersonaBuilder:
        self._gender = value
        return self

    def location(self, value: str) -> PersonaBuilder:
        self._location = value
        return self

    def education(self, value: str) -> PersonaBuilder:
        self._education = value
        return self

    def background(self, value: str) -> PersonaBuilder:
        self._background = value
        return self

    def culture(self, primary: str) -> PersonaBuilder:
        self._primary_culture = primary
        return self

    def language(
        self, lang: str, proficiency: float = 1.0, accent: str | None = None
    ) -> PersonaBuilder:
        if self._languages is None:
            self._languages = []
        self._languages.append((lang, proficiency, accent))
        return self

    def domain(
        self, name: str, proficiency: float = 0.7, subdomains: list[str] | None = None
    ) -> PersonaBuilder:
        """Add a knowledge domain manually."""
        if self._knowledge_domains is None:
            self._knowledge_domains = []
        self._knowledge_domains.append((name, proficiency, subdomains or []))
        return self

    def goal(self, description: str, weight: float = 0.7) -> PersonaBuilder:
        self._goals.append((description, weight))
        return self

    def trait(self, *adjectives: str) -> PersonaBuilder:
        """Apply one or more personality adjectives.

        Known adjectives are mapped to Big Five, Schwartz values, cognitive
        style, and communication parameters. Unknown adjectives are silently
        ignored (use ``traits()`` alias for multiple).

        Example::

            builder.trait("passionate", "direct", "analytical")
        """
        for adj in adjectives:
            adj_lower = adj.lower().strip()
            if adj_lower in TRAIT_MODIFIERS:
                self._apply_trait_deltas(TRAIT_MODIFIERS[adj_lower])
                self._applied_traits.append(adj_lower)
        return self

    # Alias
    traits = trait

    def archetype(self, name: str) -> PersonaBuilder:
        """Apply an archetype personality profile.

        Available: expert, coach, creative, analyst, caregiver, leader.
        Archetype values override the neutral baseline (0.5) but are then
        further modified by any ``trait()`` calls.
        """
        name_lower = name.lower().strip()
        if name_lower not in ARCHETYPES:
            available = ", ".join(sorted(ARCHETYPES))
            raise PersonaValidationError(
                f"Unknown archetype '{name}'. Available: {available}"
            )
        self._archetype_name = name_lower
        profile = ARCHETYPES[name_lower]
        for section, overrides in profile.items():
            target = self._get_section(section)
            if target is not None:
                for field, value in overrides.items():
                    if field in target:
                        target[field] = value
        return self

    def time_scarcity(self, value: float) -> PersonaBuilder:
        self._time_scarcity = _clamp(value)
        return self

    def privacy_sensitivity(self, value: float) -> PersonaBuilder:
        self._privacy_sensitivity = _clamp(value)
        return self

    def lookup_behavior(self, value: str) -> PersonaBuilder:
        """Set claim policy lookup behavior: ask, hedge, refuse, speculate."""
        if value not in ("ask", "hedge", "refuse", "speculate"):
            raise PersonaValidationError(f"lookup_behavior must be ask/hedge/refuse/speculate, got '{value}'")
        self._lookup_behavior = value  # type: ignore[assignment]
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Persona:
        """Construct the full Persona object with all defaults filled in."""
        persona_id = self._generate_id()
        identity = self._build_identity()
        psychology = self._build_psychology()
        domains = self._build_domains()
        languages = self._build_languages()
        cultural = self._build_cultural()
        goals = self._build_goals()
        social_roles = self._build_social_roles()
        uncertainty = self._build_uncertainty()
        claim_policy = self._build_claim_policy(domains)
        invariants = self._build_invariants(identity, domains)
        ts = self._derive_time_scarcity()
        ps = self._derive_privacy_sensitivity()
        disclosure = self._build_disclosure()
        topic_sens = self._build_topic_sensitivities(ps)
        decision_policies = self._build_decision_policies()
        response_patterns = self._build_response_patterns()
        biases = self._build_biases()
        initial_state = self._build_initial_state()

        return Persona(
            persona_id=persona_id,
            version="1.0",
            label=f"{self._name} - {self._occupation}, {self._location}",
            identity=identity,
            psychology=psychology,
            knowledge_domains=domains,
            languages=languages,
            cultural_knowledge=cultural,
            primary_goals=[Goal(goal=g, weight=w) for g, w in goals[:3]],
            secondary_goals=[Goal(goal=g, weight=w) for g, w in goals[3:]],
            social_roles=social_roles,
            uncertainty=uncertainty,
            claim_policy=claim_policy,
            invariants=invariants,
            time_scarcity=ts,
            privacy_sensitivity=ps,
            disclosure_policy=disclosure,
            topic_sensitivities=topic_sens,
            decision_policies=decision_policies,
            response_patterns=response_patterns,
            biases=biases,
            initial_state=initial_state,
        )

    def save_yaml(self, path: str) -> Persona:
        """Build the persona and write it to a YAML file.

        Args:
            path: File path to write YAML output.

        Returns:
            The built Persona object.
        """
        persona = self.build()
        persona.to_yaml(path=path)
        return persona

    # ------------------------------------------------------------------
    # from_description — parse natural language into builder calls
    # ------------------------------------------------------------------

    @classmethod
    def from_description(cls, description: str) -> Persona:
        """Create a persona from a natural-language description.

        Uses heuristic parsing (no LLM required) to extract name,
        occupation, age, location, and personality adjectives.

        Example::

            persona = PersonaBuilder.from_description(
                "A 45-year-old French chef named Marcus, passionate and direct"
            )
        """
        name = _extract_name(description)
        occupation = _extract_occupation(description)
        builder = cls(name, occupation)

        # Extract age
        age = _extract_age(description)
        if age:
            builder.age(age)

        # Extract location
        loc = _extract_location(description)
        if loc:
            builder.location(loc)

        # Extract personality adjectives
        adjectives = _extract_adjectives(description)
        if adjectives:
            builder.trait(*adjectives)

        # Extract archetype hints
        archetype = _extract_archetype(description)
        if archetype:
            builder.archetype(archetype)

        # Use description as background if no explicit background
        builder.background(description)

        return builder.build()

    # ------------------------------------------------------------------
    # Archetype shortcut
    # ------------------------------------------------------------------

    @classmethod
    def archetype_persona(
        cls,
        archetype_name: str,
        *,
        name: str = "Alex",
        occupation: str = "Professional",
        **kwargs: Any,
    ) -> Persona:
        """Quick creation from an archetype.

        Args:
            archetype_name: One of: expert, coach, creative, analyst, caregiver, leader
            name: Persona name
            occupation: Persona occupation
            **kwargs: Forwarded as builder method calls (age=35, location="NYC", etc.)
        """
        builder = cls(name, occupation).archetype(archetype_name)
        for key, value in kwargs.items():
            method = getattr(builder, key, None)
            if callable(method):
                method(value)
        return builder.build()

    # ------------------------------------------------------------------
    # Internal: apply trait deltas
    # ------------------------------------------------------------------

    def _apply_trait_deltas(self, deltas: dict[str, float]) -> None:
        for path, delta in deltas.items():
            section_name, field_name = path.split(".", 1)
            target = self._get_section(section_name)
            if target is not None and field_name in target:
                target[field_name] = _clamp(target[field_name] + delta)

    def _get_section(self, name: str) -> dict[str, float] | None:
        return {
            "big_five": self._big_five,
            "values": self._values,
            "cognitive_style": self._cognitive_style,
            "communication": self._communication,
        }.get(name)

    # ------------------------------------------------------------------
    # Internal: build sub-objects
    # ------------------------------------------------------------------

    def _generate_id(self) -> str:
        raw = f"{self._name}:{self._occupation}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:8]
        return f"P_GEN_{h.upper()}"

    def _build_identity(self) -> Identity:
        education = self._education or _infer_education(self._occupation)
        background = self._background or (
            f"{self._name} is a {self._age}-year-old {self._occupation} "
            f"based in {self._location}."
        )
        return Identity(
            age=self._age,
            gender=self._gender,
            location=self._location,
            education=education,
            occupation=self._occupation,
            background=background,
        )

    def _build_psychology(self) -> PersonalityProfile:
        return PersonalityProfile(
            big_five=BigFiveTraits(**self._big_five),
            values=SchwartzValues(**self._values),
            cognitive_style=CognitiveStyle(**self._cognitive_style),
            communication=CommunicationPreferences(**self._communication),
        )

    def _build_domains(self) -> list[DomainKnowledge]:
        if self._knowledge_domains is not None:
            return [
                DomainKnowledge(domain=d, proficiency=p, subdomains=s)
                for d, p, s in self._knowledge_domains
            ]
        # Auto-detect from occupation
        return _infer_domains(self._occupation)

    def _build_languages(self) -> list[LanguageKnowledge]:
        if self._languages is not None:
            return [
                LanguageKnowledge(language=l, proficiency=p, accent=a)
                for l, p, a in self._languages
            ]
        return [LanguageKnowledge(language="English", proficiency=1.0)]

    def _build_cultural(self) -> CulturalKnowledge:
        culture = self._primary_culture or _infer_culture(self._location)
        return CulturalKnowledge(
            primary_culture=culture,
            exposure_level={culture.lower(): 0.9, "general": 0.5},
        )

    def _build_goals(self) -> list[tuple[str, float]]:
        if self._goals:
            return self._goals
        return [
            (f"Excel as a {self._occupation}", 0.80),
            ("Maintain work-life balance", 0.60),
        ]

    def _build_social_roles(self) -> dict[str, SocialRole]:
        f = self._communication["formality"]
        d = self._communication["directness"]
        e = self._communication["emotional_expressiveness"]
        return {
            "default": SocialRole(
                formality=f, directness=d, emotional_expressiveness=e
            ),
            "at_work": SocialRole(
                formality=_clamp(f + 0.15),
                directness=_clamp(d + 0.05),
                emotional_expressiveness=_clamp(e - 0.10),
            ),
            "friend": SocialRole(
                formality=_clamp(f - 0.20),
                directness=d,
                emotional_expressiveness=_clamp(e + 0.15),
            ),
        }

    def _build_uncertainty(self) -> UncertaintyPolicy:
        # More conscientious / analytical → stricter boundaries
        conscientiousness = self._big_five["conscientiousness"]
        analytical = self._cognitive_style["analytical_intuitive"]
        return UncertaintyPolicy(
            admission_threshold=_clamp(0.3 + conscientiousness * 0.2),
            hedging_frequency=_clamp(0.2 + (1 - self._communication["directness"]) * 0.3),
            clarification_tendency=_clamp(0.3 + analytical * 0.2),
            knowledge_boundary_strictness=_clamp(0.4 + conscientiousness * 0.3),
        )

    def _build_claim_policy(self, domains: list[DomainKnowledge]) -> ClaimPolicy:
        types = ["personal_experience", "general_common_knowledge"]
        if any(d.proficiency >= 0.7 for d in domains):
            types.append("domain_expert")
        behavior: Literal["ask", "hedge", "refuse", "speculate"] = (
            "hedge" if self._big_five["conscientiousness"] > 0.6 else "speculate"
        )
        if self._lookup_behavior is not None:
            behavior = self._lookup_behavior
        return ClaimPolicy(
            allowed_claim_types=types,
            citation_required_when={"proficiency_below": 0.6, "factual_or_time_sensitive": True},
            lookup_behavior=behavior,
        )

    def _build_invariants(
        self, identity: Identity, domains: list[DomainKnowledge]
    ) -> PersonaInvariants:
        facts = [
            f"Lives in {identity.location}",
            f"Age {identity.age}" + (f", {identity.gender}" if identity.gender else ""),
            f"{identity.occupation}",
            f"Education: {identity.education}",
        ]
        # cannot_claim: professions outside their domain
        cannot_claim = _infer_cannot_claim(self._occupation, domains)
        return PersonaInvariants(
            identity_facts=facts,
            cannot_claim=cannot_claim,
            must_avoid=["revealing private personal information"],
        )

    def _derive_time_scarcity(self) -> float:
        if self._time_scarcity is not None:
            return self._time_scarcity
        # Higher conscientiousness = busier; extraversion = more social commitments
        return _clamp(
            0.4 + self._big_five["conscientiousness"] * 0.3
            + self._big_five["extraversion"] * 0.1
        )

    def _derive_privacy_sensitivity(self) -> float:
        if self._privacy_sensitivity is not None:
            return self._privacy_sensitivity
        # Higher neuroticism = more guarded; lower agreeableness = less open
        return _clamp(
            0.3 + self._big_five["neuroticism"] * 0.3
            + (1 - self._big_five["agreeableness"]) * 0.2
        )

    def _build_disclosure(self) -> DisclosurePolicy:
        agreeableness = self._big_five["agreeableness"]
        openness = self._big_five["openness"]
        base = _clamp(0.3 + agreeableness * 0.2 + openness * 0.15)
        return DisclosurePolicy(
            base_openness=round(base, 2),
            factors={
                "topic_sensitivity": -0.25,
                "trust_level": 0.35,
                "formal_context": -0.15,
                "positive_mood": 0.15,
            },
            bounds=(0.1, 0.9),
        )

    def _build_topic_sensitivities(
        self, privacy: float
    ) -> list[TopicSensitivity]:
        # Scale sensitivities by overall privacy level
        base = [
            ("personal_finances", 0.6),
            ("family_details", 0.5),
            ("professional_work", 0.2),
            ("political_views", 0.5),
            ("mental_health_personal", 0.65),
        ]
        return [
            TopicSensitivity(
                topic=t, sensitivity=_clamp(s * (0.7 + privacy * 0.6))
            )
            for t, s in base
        ]

    def _build_decision_policies(self) -> list[DecisionPolicy]:
        # Derive from cognitive style
        analytical = self._cognitive_style["analytical_intuitive"]
        approach = "analytical_systematic" if analytical > 0.6 else "intuitive_quick"
        return [
            DecisionPolicy(
                condition="high_stakes_decision",
                approach=approach,
                time_needed="extended" if analytical > 0.6 else "moderate",
            ),
            DecisionPolicy(condition="low_stakes_decision", approach="quick_decisive"),
            DecisionPolicy(condition="unfamiliar_topic", approach="ask_questions_first"),
        ]

    def _build_response_patterns(self) -> list[ResponsePattern]:
        directness = self._communication["directness"]
        agreeableness = self._big_five["agreeableness"]

        disagreement_response = (
            "acknowledge_then_explain" if agreeableness > 0.6 else "direct_pushback"
        )
        intrusion_response = (
            "polite_deflect" if agreeableness > 0.6 else "blunt_redirect"
        )

        return [
            ResponsePattern(
                trigger="disagreement",
                response=disagreement_response,
                emotionality=_clamp(0.3 + directness * 0.3),
            ),
            ResponsePattern(
                trigger="personal_question_intrusive",
                response=intrusion_response,
                emotionality=_clamp(0.2 + (1 - agreeableness) * 0.2),
            ),
            ResponsePattern(
                trigger="professional_expertise_request",
                response="confident_helpful",
                emotionality=0.5,
            ),
        ]

    def _build_biases(self) -> list[Bias]:
        return [
            Bias(type="confirmation_bias", strength=0.4),
            Bias(type="availability_heuristic", strength=0.5),
            Bias(
                type="social_desirability",
                strength=_clamp(0.1 + self._big_five["agreeableness"] * 0.3),
            ),
        ]

    def _build_initial_state(self) -> DynamicState:
        return DynamicState(
            mood_valence=0.1,
            mood_arousal=_clamp(0.3 + self._big_five["extraversion"] * 0.3),
            fatigue=0.3,
            stress=_clamp(0.2 + self._big_five["neuroticism"] * 0.3),
            engagement=0.6,
        )


# ============================================================================
# Description Parsing Helpers (no LLM required)
# ============================================================================


def _extract_name(text: str) -> str:
    """Extract a name from description text."""
    # "named X", "called X", "name is X"
    match = re.search(r"(?:named|called|name is|name\'s)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    if match:
        return match.group(1).strip()

    # "X is a ..." or "X, a ..."
    match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|,)", text)
    if match:
        return match.group(1).strip()

    # First capitalized word that isn't a common starter
    skip = {"A", "An", "The", "This", "My", "Our", "She", "He", "They", "Meet", "Imagine"}
    for word in text.split():
        cleaned = word.strip(".,;:'\"")
        if cleaned and cleaned[0].isupper() and cleaned not in skip and cleaned.isalpha():
            return cleaned

    return "Alex"


def _extract_occupation(text: str) -> str:
    """Extract occupation from description text."""
    text_lower = text.lower()

    # Try multi-word occupations first (longest match)
    for occ in sorted(OCCUPATION_DOMAINS, key=len, reverse=True):
        if occ in text_lower:
            return occ.title()

    # Common occupation patterns
    match = re.search(
        r"(?:works as|profession is|occupation is|who is)\s+(?:a |an )?([a-z][a-z\s]{2,25}?)(?:[,.\s]|$)",
        text_lower,
    )
    if match:
        return match.group(1).strip().title()

    return "Professional"


def _extract_age(text: str) -> int | None:
    """Extract age from description."""
    match = re.search(r"(\d{2})\s*[-–]?\s*years?\s*[-–]?\s*old", text.lower())
    if match:
        age = int(match.group(1))
        if 18 <= age <= 100:
            return age

    match = re.search(r"age\s*:?\s*(\d{2})", text.lower())
    if match:
        age = int(match.group(1))
        if 18 <= age <= 100:
            return age

    return None


def _extract_location(text: str) -> str | None:
    """Extract location from description."""
    # "from X", "based in X", "lives in X", "in X"
    match = re.search(
        r"(?:from|based in|lives in|living in|located in)\s+([A-Z][A-Za-z\s,]+?)(?:\.|,\s*[a-z]|\s*$)",
        text,
    )
    if match:
        loc = match.group(1).strip().rstrip(",.")
        if len(loc) > 2:
            return loc
    return None


def _extract_adjectives(text: str) -> list[str]:
    """Extract known personality adjectives from description."""
    text_lower = text.lower()
    found = []
    for adj in TRAIT_MODIFIERS:
        # Match whole word (handle hyphenated adjectives too)
        pattern = r"\b" + re.escape(adj) + r"\b"
        if re.search(pattern, text_lower):
            found.append(adj)
    return found


def _extract_archetype(text: str) -> str | None:
    """Extract archetype hint from description."""
    text_lower = text.lower()
    for name in ARCHETYPES:
        if name in text_lower:
            return name
    return None


# ============================================================================
# Inference Helpers
# ============================================================================


def _infer_education(occupation: str) -> str:
    """Guess education level from occupation."""
    occ_lower = occupation.lower()
    phd_roles = {"professor", "physicist", "scientist", "researcher", "psychologist"}
    masters_roles = {"therapist", "counselor", "architect", "data scientist"}
    professional_roles = {"doctor", "physician", "lawyer", "attorney", "pharmacist", "veterinarian"}
    trade_roles = {"chef", "cook", "mechanic", "electrician", "baker", "bartender", "pilot"}

    for role in phd_roles:
        if role in occ_lower:
            return "PhD"
    for role in masters_roles:
        if role in occ_lower:
            return "Master's degree"
    for role in professional_roles:
        if role in occ_lower:
            return "Professional degree"
    for role in trade_roles:
        if role in occ_lower:
            return "Vocational / trade certification"
    return "Bachelor's degree"


def _infer_domains(occupation: str) -> list[DomainKnowledge]:
    """Map occupation to knowledge domains with subdomains."""
    occ_lower = occupation.lower()

    # Try exact match first, then substring (longest match wins)
    for key in sorted(OCCUPATION_DOMAINS, key=len, reverse=True):
        if key in occ_lower:
            subdomain_map = OCCUPATION_SUBDOMAINS.get(key, {})
            return [
                DomainKnowledge(
                    domain=d,
                    proficiency=p,
                    subdomains=subdomain_map.get(d, []),
                )
                for d, p in OCCUPATION_DOMAINS[key]
            ]

    # Fallback: general domain
    return [DomainKnowledge(domain="General", proficiency=0.5, subdomains=[])]


def _infer_culture(location: str) -> str:
    """Guess primary culture from location."""
    loc_lower = location.lower()
    culture_map = {
        "american": ["us", "usa", "united states", "america", "new york", "chicago",
                      "los angeles", "san francisco", "seattle", "boston", "miami",
                      "texas", "california"],
        "British": ["uk", "united kingdom", "britain", "london", "england", "scotland", "wales"],
        "French": ["france", "paris", "lyon", "marseille"],
        "German": ["germany", "berlin", "munich"],
        "Japanese": ["japan", "tokyo", "osaka"],
        "Chinese": ["china", "beijing", "shanghai"],
        "Indian": ["india", "mumbai", "delhi", "bangalore"],
        "Brazilian": ["brazil", "sao paulo", "rio"],
        "Australian": ["australia", "sydney", "melbourne"],
        "Canadian": ["canada", "toronto", "vancouver", "montreal"],
        "Mexican": ["mexico", "mexico city"],
        "Italian": ["italy", "rome", "milan"],
        "Spanish": ["spain", "madrid", "barcelona"],
        "Korean": ["korea", "seoul"],
    }
    for culture, keywords in culture_map.items():
        if any(kw in loc_lower for kw in keywords):
            return culture
    return "International"


def _infer_cannot_claim(
    occupation: str, domains: list[DomainKnowledge]
) -> list[str]:
    """Generate cannot_claim list based on what's NOT in their domains."""
    domain_names = {d.domain.lower() for d in domains}
    all_restricted = {
        "health": ["medical doctor", "licensed therapist", "pharmacist"],
        "law": ["licensed attorney", "judge"],
        "psychology": ["clinical psychologist", "psychiatrist"],
        "finance": ["certified financial advisor", "licensed broker"],
        "science": ["research scientist"],
    }
    cannot: list[str] = []
    for domain, roles in all_restricted.items():
        if domain not in domain_names:
            cannot.extend(roles)
    return cannot
