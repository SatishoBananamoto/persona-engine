"""
Argument-Based Stance Generator

Generates persona stances using a compositional approach:
- Topic Axis: What is being discussed (from input + domain detection)
- Value Axis: What the persona cares about (Schwartz values, filtered by domain)
- Competence Axis: How the persona relates to the topic (expert/moderate/low)

Each axis contributes fragments composed into a topic-grounded, value-driven stance.
Fully deterministic, zero LLM calls.
"""

import re
from dataclasses import dataclass

from persona_engine.behavioral.cognitive_interpreter import CognitiveStyleInterpreter
from persona_engine.behavioral.values_interpreter import ValuesInterpreter
from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.persona_schema import BigFiveTraits, Persona


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class TopicExtraction:
    """What the user is talking about."""
    topic_hint: str | None
    domain: str
    is_question: bool
    is_opinion_share: bool
    is_request: bool


@dataclass(frozen=True)
class InvariantCheck:
    """Result of checking persona invariants against the topic."""
    is_blocked: bool
    reason: str | None
    caveat: str | None


# =============================================================================
# Value-Topic Mapping Table
# =============================================================================
# VALUE_TOPIC_TABLE[value_name][domain] -> (stance_fragment, rationale_fragment)
# "general" key is the fallback for unknown domains.

VALUE_TOPIC_TABLE: dict[str, dict[str, tuple[str, str]]] = {
    "achievement": {
        "technology": (
            "I favor well-tested, performant solutions that ship",
            "Achievement value: competence measured by working software",
        ),
        "business": (
            "I focus on measurable outcomes and clear success metrics",
            "Achievement value: results-oriented business approach",
        ),
        "food": (
            "I favor techniques that reliably produce excellent results",
            "Achievement value: mastery of culinary skill",
        ),
        "health": (
            "I focus on evidence-based approaches with trackable progress",
            "Achievement value: measurable health outcomes",
        ),
        "psychology": (
            "I value rigorous methodology and replicable findings",
            "Achievement value: competence in research practice",
        ),
        "science": (
            "I value precise methodology and reproducible results",
            "Achievement value: scientific rigor",
        ),
        "law": (
            "I focus on thorough preparation and winning arguments",
            "Achievement value: professional excellence in legal work",
        ),
        "music": (
            "I value technical mastery and putting in the hours to get it right",
            "Achievement value: musical excellence through practice",
        ),
        "fitness": (
            "I focus on measurable gains and consistent improvement",
            "Achievement value: trackable fitness progress",
        ),
        "general": (
            "I favor approaches that demonstrate competence and produce results",
            "Achievement value: general drive for competence",
        ),
    },
    "benevolence": {
        "technology": (
            "I prioritize technology that serves people's actual needs",
            "Benevolence value: tech should improve lives",
        ),
        "business": (
            "I favor decisions that look after the team and the people involved",
            "Benevolence value: people over profit",
        ),
        "food": (
            "I enjoy cooking that brings people together and nourishes them",
            "Benevolence value: food as care and connection",
        ),
        "health": (
            "I focus on approaches that consider the whole person",
            "Benevolence value: holistic care",
        ),
        "general": (
            "I prefer solutions that prioritize people's wellbeing",
            "Benevolence value: care for others' welfare",
        ),
    },
    "security": {
        "technology": (
            "I prefer proven, stable tools over bleeding-edge choices",
            "Security value: reliability in tech infrastructure",
        ),
        "business": (
            "I lean toward strategies that protect what we've built",
            "Security value: stability and risk management",
        ),
        "food": (
            "I stick with methods I know produce consistent results",
            "Security value: reliability in food preparation",
        ),
        "health": (
            "I trust established treatments over experimental ones",
            "Security value: proven safety in health decisions",
        ),
        "general": (
            "I gravitate toward stable, proven methods that minimize risk",
            "Security value: preference for safety and stability",
        ),
    },
    "self_direction": {
        "technology": (
            "I value tools and approaches that give developers autonomy",
            "Self-direction value: independence in technical choices",
        ),
        "business": (
            "I favor strategies that preserve creative freedom and ownership",
            "Self-direction value: autonomy in business decisions",
        ),
        "food": (
            "I like adapting recipes and improvising based on my own instincts",
            "Self-direction value: creative independence in cooking",
        ),
        "music": (
            "I value finding my own sound over following what's trending",
            "Self-direction value: artistic independence",
        ),
        "general": (
            "I value autonomy and the freedom to choose my own approach",
            "Self-direction value: preference for independence",
        ),
    },
    "universalism": {
        "technology": (
            "I care about accessibility and whether tech works for everyone",
            "Universalism value: equitable technology access",
        ),
        "business": (
            "I think business decisions should account for broader impact",
            "Universalism value: considering all stakeholders",
        ),
        "food": (
            "I appreciate diverse food traditions and think everyone deserves good food",
            "Universalism value: food equity and cultural respect",
        ),
        "general": (
            "I lean toward solutions that consider broader societal impact",
            "Universalism value: concern for equity and justice",
        ),
    },
    "hedonism": {
        "technology": (
            "I appreciate tools that are genuinely pleasant to use",
            "Hedonism value: user experience and enjoyment matter",
        ),
        "food": (
            "I believe food should be a pleasure, not just fuel",
            "Hedonism value: sensory enjoyment of food",
        ),
        "general": (
            "I appreciate approaches that balance effectiveness with enjoyment",
            "Hedonism value: pleasure and personal gratification",
        ),
    },
    "stimulation": {
        "technology": (
            "I'm drawn to novel approaches and emerging tools",
            "Stimulation value: excitement in technical exploration",
        ),
        "food": (
            "I love trying unfamiliar cuisines and experimental techniques",
            "Stimulation value: novelty and adventure in food",
        ),
        "music": (
            "I'm drawn to genre-bending work and unexpected combinations",
            "Stimulation value: musical novelty and experimentation",
        ),
        "general": (
            "I favor dynamic, novel approaches over conventional ones",
            "Stimulation value: variety and challenge",
        ),
    },
    "conformity": {
        "technology": (
            "I prefer following established conventions and standards",
            "Conformity value: consistency and team norms in tech",
        ),
        "business": (
            "I believe in following established processes and team norms",
            "Conformity value: organizational discipline",
        ),
        "general": (
            "I tend to prefer established norms and proven processes",
            "Conformity value: respect for social expectations",
        ),
    },
    "tradition": {
        "food": (
            "I respect traditional recipes and classical techniques",
            "Tradition value: honoring culinary heritage",
        ),
        "music": (
            "I respect the roots and traditions of the genre",
            "Tradition value: honoring musical heritage",
        ),
        "general": (
            "I value methods that respect established practices and heritage",
            "Tradition value: commitment to customs",
        ),
    },
    "power": {
        "business": (
            "I favor approaches that build leverage and strategic position",
            "Power value: influence and capability in business",
        ),
        "technology": (
            "I favor architectures that give us control over critical systems",
            "Power value: control and capability in tech",
        ),
        "general": (
            "I favor approaches that demonstrate influence and capability",
            "Power value: status and control over resources",
        ),
    },
}


# =============================================================================
# Competence Frame Templates
# =============================================================================

COMPETENCE_FRAMES: dict[str, dict[str, str]] = {
    "expert": {
        "question": "From my experience with {topic},",
        "opinion_share": "Having worked extensively with {topic},",
        "request": "Drawing on my background in {topic},",
        "statement": "Based on what I know about {topic},",
    },
    "moderate": {
        "question": "I've encountered {topic} enough to have a view:",
        "opinion_share": "I have some experience with {topic} —",
        "request": "I have a working familiarity with {topic}, so",
        "statement": "From what I've seen of {topic},",
    },
    "low": {
        "question": "I'm not deeply familiar with {topic}, but",
        "opinion_share": "I don't have deep expertise in {topic}, though",
        "request": "I can't speak to {topic} with authority, but",
        "statement": "Without deep knowledge of {topic},",
    },
}


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_stance_safe(
    persona: Persona,
    values: ValuesInterpreter,
    cognitive: CognitiveStyleInterpreter,
    user_input: str,
    topic_signature: str,
    proficiency: float,
    expert_allowed: bool,
    ctx: TraceContext,
    domain: str = "general",
) -> tuple[str, str]:
    """
    Generate a topic-grounded, value-driven stance with invariant safety.

    Args:
        persona: Full persona object
        values: Values interpreter for value priorities
        cognitive: Cognitive interpreter for nuance
        user_input: User's question/statement
        topic_signature: Topic identifier
        proficiency: Domain proficiency (0-1)
        expert_allowed: Whether persona can make expert claims
        ctx: TraceContext for recording decisions
        domain: Detected domain (from domain_detection)

    Returns:
        (stance_text, rationale)
    """
    # Step 1: Extract topic structure
    topic = _extract_topic(user_input, domain)

    # Step 2: Check invariants
    invariant = _check_invariants(
        topic, persona.invariants.cannot_claim, persona.invariants.must_avoid
    )
    if invariant.is_blocked:
        ctx.enum(
            source_type="constraint",
            source_id="invariant_block",
            target_field="response_structure.stance",
            operation="set",
            before="none",
            after="blocked",
            effect=f"Topic touches must_avoid: {invariant.reason}",
            weight=1.0,
            reason="invariant_safety",
        )
        return (
            "I'd rather not weigh in on that particular topic",
            f"Blocked by invariant: {invariant.reason}",
        )

    # Step 3: Map values to this topic
    top_values = values.get_top_values(n=3)
    primary_value, primary_weight = top_values[0]
    secondary = top_values[1] if len(top_values) > 1 else None

    primary_stance, primary_rationale = _get_value_topic_mapping(primary_value, domain)
    secondary_stance = None
    secondary_rationale = None
    if secondary:
        secondary_stance, secondary_rationale = _get_value_topic_mapping(secondary[0], domain)

    # Step 4: Select competence frame
    competence_frame = _select_competence_frame(
        proficiency, expert_allowed, topic, cognitive.get_reasoning_approach()
    )

    # Step 5: Build nuance qualifier
    nuance = None
    if cognitive.style.cognitive_complexity > 0.7:
        if secondary and secondary_stance:
            # Check for value conflict
            conflict = values.resolve_conflict_detailed(
                primary_value, secondary[0],
                context=_domain_to_context(domain),
            )
            if conflict.is_opposing and conflict.confidence < 0.6:
                nuance = f"though I'm genuinely torn because {secondary_stance.lower()}"
            else:
                nuance = f"though I also value that {secondary_stance.lower()}"
        else:
            nuance = "though context matters here"

    # Step 6: Assemble stance
    stance = _assemble_stance(
        competence_frame=competence_frame,
        value_position=primary_stance,
        nuance=nuance,
        invariant_caveat=invariant.caveat,
    )

    # Step 7: Assemble rationale (include raw value key for traceability)
    rationale_parts = [f"{primary_value}: {primary_rationale}"]
    if secondary and secondary_rationale:
        rationale_parts.append(f"{secondary[0]}: {secondary_rationale}")
    rationale_parts.append(
        f"Proficiency: {proficiency:.2f} ({'expert' if expert_allowed else 'non-expert'})"
    )
    rationale = " | ".join(rationale_parts)

    # Trace citation
    ctx.enum(
        source_type="value",
        source_id=primary_value,
        target_field="response_structure.stance",
        operation="set",
        before="none",
        after=stance[:60],
        effect=f"Stance driven by {primary_value} value applied to {domain} domain",
        weight=0.9,
        reason=f"topic={topic.topic_hint or domain}, proficiency={proficiency:.2f}",
    )

    # Phase R1: Detect and resolve value conflicts
    value_conflicts = values.detect_value_conflicts(threshold=0.6)
    if value_conflicts:
        top_conflict = value_conflicts[0]
        resolution = values.resolve_conflict_detailed(
            top_conflict["value_1"], top_conflict["value_2"],
            context=_domain_to_context(domain),
        )
        if resolution.confidence < 0.6:
            stance += (
                f", though I feel conflicted because I value both "
                f"{top_conflict['value_1'].replace('_', ' ')} and "
                f"{top_conflict['value_2'].replace('_', ' ')}"
            )
        for cite in resolution.citations:
            ctx.add_basic_citation(
                source_type=cite.get("source_type", "value"),
                source_id=cite.get("source_id", "conflict"),
                effect=cite.get("effect", "Value conflict detected"),
                weight=cite.get("weight", 0.5),
            )

    # Phase R5: Personality modulation — same value, different voice
    stance = _modulate_stance_by_personality(stance, persona.psychology.big_five)

    return stance, rationale


# =============================================================================
# Internal Functions
# =============================================================================

def _extract_topic(user_input: str, domain: str) -> TopicExtraction:
    """Extract topic structure from user input."""
    user_lower = user_input.lower()

    # Topic hint via regex
    topic_hint = None
    for pattern in [
        r"(?:about|regarding|on|of)\s+(.{5,40}?)(?:\?|$)",
        r"(?:approach|handle|deal with|manage|make|cook|prepare)\s+(.{5,30})",
        r"(?:best way to|how to|how do you)\s+(.{5,30})",
    ]:
        match = re.search(pattern, user_lower)
        if match:
            topic_hint = match.group(1).strip().rstrip("?.,")
            break

    return TopicExtraction(
        topic_hint=topic_hint,
        domain=domain,
        is_question="?" in user_input,
        is_opinion_share=any(
            phrase in user_lower for phrase in ["i think", "i feel", "my view", "i believe"]
        ),
        is_request=any(
            phrase in user_lower for phrase in ["please", "could you", "can you", "would you"]
        ),
    )


def _check_invariants(
    topic: TopicExtraction,
    cannot_claim: list[str],
    must_avoid: list[str],
) -> InvariantCheck:
    """Check if topic collides with persona invariants."""
    topic_text = (topic.topic_hint or "").lower()
    domain_lower = topic.domain.lower()

    # Hard block: must_avoid
    for avoid_item in must_avoid:
        if avoid_item.lower() in topic_text or avoid_item.lower() in domain_lower:
            return InvariantCheck(is_blocked=True, reason=avoid_item, caveat=None)

    # Soft caveat: cannot_claim
    for claim_item in cannot_claim:
        claim_lower = claim_item.lower()
        claim_tokens = set(claim_lower.split())
        topic_tokens = set(topic_text.split()) | {domain_lower}
        overlap = topic_tokens & claim_tokens
        if overlap and len(overlap) / max(len(claim_tokens), 1) >= 0.4:
            return InvariantCheck(
                is_blocked=False, reason=None,
                caveat=f"Speaking as a non-{claim_item}",
            )

    return InvariantCheck(is_blocked=False, reason=None, caveat=None)


def _get_value_topic_mapping(value_name: str, domain: str) -> tuple[str, str]:
    """Look up value x domain stance fragment with fallback to general."""
    value_entries = VALUE_TOPIC_TABLE.get(value_name, {})
    if domain in value_entries:
        return value_entries[domain]
    return value_entries.get("general", (
        "I have a considered perspective on this",
        f"{value_name} value: general stance",
    ))


def _select_competence_frame(
    proficiency: float,
    expert_allowed: bool,
    topic: TopicExtraction,
    reasoning_approach: str,
) -> str:
    """Select opening frame based on competence level and input type."""
    # Determine tier
    if expert_allowed and proficiency >= 0.6:
        tier = "expert"
    elif proficiency >= 0.4:
        tier = "moderate"
    else:
        tier = "low"

    # Determine input type
    if topic.is_question:
        input_type = "question"
    elif topic.is_opinion_share:
        input_type = "opinion_share"
    elif topic.is_request:
        input_type = "request"
    else:
        input_type = "statement"

    template = COMPETENCE_FRAMES[tier][input_type]
    topic_str = topic.topic_hint or "this area"
    return template.format(topic=topic_str)


def _domain_to_context(domain: str) -> str:
    """Map domain to Schwartz value resolution context."""
    mapping = {
        "technology": "work",
        "business": "work",
        "law": "work",
        "food": "personal",
        "health": "personal",
        "fitness": "personal",
        "music": "personal",
        "psychology": "work",
        "science": "work",
    }
    return mapping.get(domain, "general")


def _assemble_stance(
    competence_frame: str,
    value_position: str,
    nuance: str | None,
    invariant_caveat: str | None,
) -> str:
    """Compose final stance string from fragments."""
    parts = []
    if invariant_caveat:
        parts.append(f"{invariant_caveat}:")
    parts.append(f"{competence_frame} {value_position.lower()}")
    if nuance:
        parts.append(f"— {nuance}")
    return " ".join(parts)


def _modulate_stance_by_personality(stance: str, traits: BigFiveTraits) -> str:
    """Modulate a base stance expression based on personality traits.

    Same value, different voice depending on who's speaking.
    Ported from target branch Phase R5.
    """
    modulations: list[str] = []

    # High-N: cautious, hedged expression
    if traits.neuroticism > 0.65:
        modulations.append(
            "I worry that we might not be considering all the risks, but "
        )
    # Low-N + low-A: confident, assertive framing
    elif traits.neuroticism < 0.25 and traits.agreeableness < 0.4:
        modulations.append("Let me be clear: ")

    # High-O: exploratory framing (only if N didn't already prefix)
    if traits.openness > 0.7 and not modulations:
        modulations.append(
            "Looking at this from a broader perspective, "
        )

    # Apply prefix modulation (at most one)
    if modulations:
        prefix = modulations[0]
        stance = prefix + stance[0].lower() + stance[1:]

    # High-A: add empathetic acknowledgment suffix
    if traits.agreeableness > 0.75:
        stance += " — and I understand others may see this differently"

    return stance
