"""
Bias Simulator - Subtle, Bounded Cognitive Bias Injection

Implements bounded cognitive biases that subtly influence persona behavior:
- Confirmation Bias: Reduced elasticity when input aligns with values
- Negativity Bias: Increased arousal/attention to negative input (Neuroticism-driven)
- Authority Bias: Increased confidence when citing authorities (Conformity-driven)

Design Principles:
- Biases are observable (appear in citations)
- Biases are bounded (max impact ±0.15)
- Biases are deterministic (same input + persona = same bias)
- Biases never override safety constraints or expertise boundaries
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Optional

from persona_engine.behavioral.negation import (
    NEGATION_WORDS,
    count_unnegated_markers,
)

if TYPE_CHECKING:
    from persona_engine.planner.trace_context import TraceContext


# =============================================================================
# Constants
# =============================================================================

# Maximum modifier magnitude (prevents biases from dominating decisions)
MAX_BIAS_IMPACT = 0.15

# Thresholds for bias activation
CONFIRMATION_ALIGNMENT_THRESHOLD = 0.6  # Value alignment must exceed this
NEGATIVITY_NEUROTICISM_THRESHOLD = 0.5  # Neuroticism must exceed this
AUTHORITY_CONFORMITY_THRESHOLD = 0.5    # Conformity/Tradition must exceed this

# Authority markers in user input
AUTHORITY_MARKERS = frozenset([
    "research shows", "studies show", "according to experts",
    "scientists say", "data shows", "research indicates",
    "experts agree", "evidence suggests", "studies indicate",
    "the research", "peer reviewed", "academic consensus"
])

# Negative sentiment markers
NEGATIVE_MARKERS = frozenset([
    "problem", "issue", "concern", "worried", "frustrated",
    "angry", "upset", "disappointed", "failed", "broken",
    "terrible", "awful", "horrible", "bad", "wrong",
    "risk", "danger", "threat", "fear", "anxiety"
])

# Re-exported for backward compatibility (canonical source: negation.py)
# NEGATION_WORDS is imported from persona_engine.behavioral.negation

# Change proposal markers (for status quo bias)
CHANGE_MARKERS = frozenset([
    "change", "switch", "replace", "new approach", "different way",
    "instead", "alternative", "rethink", "overhaul", "transform",
    "upgrade", "redesign", "migrate", "pivot",
])

# Emotional content markers (for empathy gap)
EMOTIONAL_CONTENT_MARKERS = frozenset([
    "feel", "feeling", "hurt", "painful", "heartbroken", "devastated",
    "overwhelmed", "emotional", "crying", "struggling", "suffering",
    "grief", "lonely", "depressed", "anxious",
])


# =============================================================================
# Data Structures
# =============================================================================

class BiasType(StrEnum):
    """Types of cognitive biases simulated"""
    CONFIRMATION = "confirmation_bias"
    NEGATIVITY = "negativity_bias"
    AUTHORITY = "authority_bias"
    ANCHORING = "anchoring_bias"
    STATUS_QUO = "status_quo_bias"
    AVAILABILITY = "availability_bias"
    EMPATHY_GAP = "empathy_gap"
    DUNNING_KRUGER = "dunning_kruger_bias"


@dataclass
class BiasModifier:
    """A single bias modifier to be applied to an IR field"""
    bias_type: BiasType
    target_field: str  # e.g., "response_structure.elasticity"
    operation: Literal["add", "multiply"]
    modifier: float  # The actual adjustment value (bounded by MAX_BIAS_IMPACT)
    trigger: str  # What triggered this bias
    strength: float  # 0-1, how strongly the bias is active


# =============================================================================
# Negation Detection (delegates to shared negation.py utility)
# =============================================================================

def _count_unnegated_markers(text: str) -> int:
    """Count NEGATIVE_MARKERS in *text* that are not negated.

    Thin wrapper around the shared ``count_unnegated_markers`` utility,
    hard-wired to the ``NEGATIVE_MARKERS`` set used by the bias simulator.
    """
    return count_unnegated_markers(text, NEGATIVE_MARKERS)


# =============================================================================
# Bias Simulator
# =============================================================================

class BiasSimulator:
    """
    Computes subtle cognitive bias modifiers based on persona traits and input.

    Usage:
        simulator = BiasSimulator(traits, values)
        modifiers = simulator.compute_modifiers(user_input, topic_alignment)

        for mod in modifiers:
            if mod.target_field == "response_structure.elasticity":
                elasticity = ctx.num(..., before=elasticity, after=elasticity + mod.modifier)
    """

    def __init__(
        self,
        traits: dict[str, float],
        value_priorities: dict[str, float],
        persona_biases: list[dict] | None = None,
        knowledge_boundary_strictness: float = 0.5,
    ):
        """
        Args:
            traits: Big 5 traits dict (openness, conscientiousness, etc.)
            value_priorities: Schwartz values normalized priorities
            persona_biases: Optional list of persona-declared biases
                [{"type": "anchoring_bias", "strength": 0.4}, ...]
            knowledge_boundary_strictness: From UncertaintyPolicy — higher values
                suppress Dunning-Kruger (persona is self-aware about boundaries)
        """
        self.traits = traits
        self.values = value_priorities
        self._knowledge_boundary_strictness = knowledge_boundary_strictness

        # Precompute trait accessors
        self.neuroticism = traits.get("neuroticism", 0.5)
        self.openness = traits.get("openness", 0.5)
        self.conscientiousness = traits.get("conscientiousness", 0.5)
        self.agreeableness = traits.get("agreeableness", 0.5)
        self.extraversion = traits.get("extraversion", 0.5)

        # Precompute value accessors for conformity-related values
        self.conformity = value_priorities.get("conformity", 0.0)
        self.tradition = value_priorities.get("tradition", 0.0)
        self.security = value_priorities.get("security", 0.0)

        # Phase R6.2: Persona-declared bias strength overrides
        self._strength_overrides: dict[str, float] = {}
        if persona_biases:
            for bias in persona_biases:
                self._strength_overrides[bias["type"]] = bias["strength"]

        # Anchoring state: tracks first claim for anchoring bias
        self._anchor_stance: str | None = None

    def _apply_override(self, bias_type: str, computed_strength: float) -> float:
        """Apply persona-declared strength override if available."""
        if bias_type in self._strength_overrides:
            # Blend: persona declaration dominates (70% override, 30% computed)
            return self._strength_overrides[bias_type] * 0.7 + computed_strength * 0.3
        return computed_strength

    def compute_modifiers(
        self,
        user_input: str,
        value_alignment: float = 0.0,
        ctx: Optional["TraceContext"] = None,
        proficiency: float = 0.5,
    ) -> list[BiasModifier]:
        """
        Compute all applicable bias modifiers for current context.

        Args:
            user_input: The user's input text
            value_alignment: How aligned is input with persona's values (0-1)
            ctx: TraceContext for citation (optional)
            proficiency: Domain proficiency (0-1) for Dunning-Kruger bias

        Returns:
            List of BiasModifier objects to apply
        """
        modifiers: list[BiasModifier] = []
        input_lower = user_input.lower()

        # 1. Confirmation Bias (affects elasticity)
        conf_mod = self._compute_confirmation_bias(value_alignment, ctx)
        if conf_mod:
            modifiers.append(conf_mod)

        # 2. Negativity Bias (affects arousal/tone)
        neg_mod = self._compute_negativity_bias(input_lower, ctx)
        if neg_mod:
            modifiers.append(neg_mod)

        # 3. Authority Bias (affects confidence)
        auth_mod = self._compute_authority_bias(input_lower, ctx)
        if auth_mod:
            modifiers.append(auth_mod)

        # 4. Anchoring Bias (affects elasticity — anchors to first stance)
        anchor_mod = self._compute_anchoring_bias(ctx)
        if anchor_mod:
            modifiers.append(anchor_mod)

        # 5. Status Quo Bias (affects elasticity — resists change proposals)
        sq_mod = self._compute_status_quo_bias(input_lower, ctx)
        if sq_mod:
            modifiers.append(sq_mod)

        # 6. Availability Bias (affects arousal — overweights recent negative info)
        avail_mod = self._compute_availability_bias(input_lower, ctx)
        if avail_mod:
            modifiers.append(avail_mod)

        # 7. Empathy Gap (affects disclosure — underestimates emotional reactions)
        eg_mod = self._compute_empathy_gap(input_lower, ctx)
        if eg_mod:
            modifiers.append(eg_mod)

        # 8. Dunning-Kruger Bias (affects confidence — overconfident in low-proficiency)
        dk_mod = self._compute_dunning_kruger_bias(proficiency, ctx)
        if dk_mod:
            modifiers.append(dk_mod)

        return modifiers

    def _compute_confirmation_bias(
        self,
        value_alignment: float,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """
        Confirmation Bias: Reduced elasticity when input aligns with existing beliefs.

        Mechanism: High value alignment → less open to changing position.
        Counter: High Openness reduces this effect.
        """
        if value_alignment < CONFIRMATION_ALIGNMENT_THRESHOLD:
            return None  # Not enough alignment to trigger

        # Base strength from alignment (0.6-1.0 maps to 0-1 strength)
        raw_strength = (value_alignment - CONFIRMATION_ALIGNMENT_THRESHOLD) / (1.0 - CONFIRMATION_ALIGNMENT_THRESHOLD)

        # Openness counters confirmation bias (high openness = less bias)
        openness_counter = self.openness * 0.5  # Max 50% reduction
        adjusted_strength = raw_strength * (1.0 - openness_counter)

        if adjusted_strength < 0.1:
            return None  # Too weak to matter

        # Compute modifier (negative = reduces elasticity)
        modifier = -min(adjusted_strength * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="confirmation_bias",
                effect=f"Value alignment ({value_alignment:.2f}) triggers confirmation bias → elasticity {modifier:+.3f}",
                weight=adjusted_strength,
            )

        return BiasModifier(
            bias_type=BiasType.CONFIRMATION,
            target_field="response_structure.elasticity",
            operation="add",
            modifier=modifier,
            trigger=f"value_alignment={value_alignment:.2f}",
            strength=adjusted_strength,
        )

    def _compute_negativity_bias(
        self,
        input_lower: str,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """
        Negativity Bias: Increased attention/arousal to negative input.

        Mechanism: Neuroticism amplifies response to negative stimuli.
        """
        if self.neuroticism < NEGATIVITY_NEUROTICISM_THRESHOLD:
            return None  # Not neurotic enough to trigger

        # Count negative markers, filtering out negated ones
        negative_count = _count_unnegated_markers(input_lower)

        if negative_count == 0:
            return None  # No negative content (or all negated)

        # Strength based on marker density and neuroticism
        marker_strength = min(negative_count / 3.0, 1.0)  # Cap at 3 markers
        neuroticism_factor = (self.neuroticism - NEGATIVITY_NEUROTICISM_THRESHOLD) / (1.0 - NEGATIVITY_NEUROTICISM_THRESHOLD)
        adjusted_strength = marker_strength * neuroticism_factor

        if adjusted_strength < 0.1:
            return None

        # Modifier: increases arousal component in tone selection
        # Positive modifier = higher arousal
        modifier = min(adjusted_strength * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        # Find first matched marker for citation
        matched = next((m for m in NEGATIVE_MARKERS if m in input_lower), "negative_content")

        if ctx:
            ctx.add_basic_citation(
                source_type="trait",
                source_id="negativity_bias",
                effect=f"Negative content ('{matched}') + neuroticism ({self.neuroticism:.2f}) → arousal {modifier:+.3f}",
                weight=adjusted_strength,
            )

        return BiasModifier(
            bias_type=BiasType.NEGATIVITY,
            target_field="communication_style.arousal",
            operation="add",
            modifier=modifier,
            trigger=f"negative_markers={negative_count}, neuroticism={self.neuroticism:.2f}",
            strength=adjusted_strength,
        )

    def _compute_authority_bias(
        self,
        input_lower: str,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """
        Authority Bias: Increased confidence when input cites authorities.

        Mechanism: Conformity/Tradition values increase deference to authority.
        """
        # Combine conformity-related values
        authority_susceptibility = (self.conformity + self.tradition + self.security) / 3.0

        if authority_susceptibility < AUTHORITY_CONFORMITY_THRESHOLD:
            return None  # Not susceptible to authority bias

        # Check for authority markers
        matched_markers = [m for m in AUTHORITY_MARKERS if m in input_lower]

        if not matched_markers:
            return None

        # Strength based on marker presence and susceptibility
        marker_strength = min(len(matched_markers) / 2.0, 1.0)  # Cap at 2 markers
        susceptibility_factor = (authority_susceptibility - AUTHORITY_CONFORMITY_THRESHOLD) / (1.0 - AUTHORITY_CONFORMITY_THRESHOLD)
        adjusted_strength = marker_strength * susceptibility_factor

        if adjusted_strength < 0.1:
            return None

        # Modifier: increases confidence
        modifier = min(adjusted_strength * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="value",
                source_id="authority_bias",
                effect=f"Authority marker ('{matched_markers[0]}') + conformity ({authority_susceptibility:.2f}) → confidence {modifier:+.3f}",
                weight=adjusted_strength,
            )

        return BiasModifier(
            bias_type=BiasType.AUTHORITY,
            target_field="response_structure.confidence",
            operation="add",
            modifier=modifier,
            trigger=f"authority_markers={len(matched_markers)}, conformity={authority_susceptibility:.2f}",
            strength=adjusted_strength,
        )

    # ---- Phase R6 New Biases ----

    def set_anchor(self, stance: str) -> None:
        """Record first stance as anchor for anchoring bias."""
        if self._anchor_stance is None and stance:
            self._anchor_stance = stance

    def _compute_anchoring_bias(
        self,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """Anchoring Bias: Once a stance is set, resist changing it.

        Trigger: Having a prior anchor stance.
        Personality: Low-O (less flexible) amplifies; High-O counters.
        Effect: Reduces elasticity (anchored to first position).
        """
        if self._anchor_stance is None:
            return None

        # Low-O = more anchored
        raw_strength = (1 - self.openness) * 0.8
        adjusted = self._apply_override("anchoring_bias", raw_strength)

        if adjusted < 0.1:
            return None

        modifier = -min(adjusted * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="anchoring_bias",
                effect=f"Anchored to prior stance → elasticity {modifier:+.3f}",
                weight=adjusted,
            )

        return BiasModifier(
            bias_type=BiasType.ANCHORING,
            target_field="response_structure.elasticity",
            operation="add",
            modifier=modifier,
            trigger=f"anchor_set=True, openness={self.openness:.2f}",
            strength=adjusted,
        )

    def _compute_status_quo_bias(
        self,
        input_lower: str,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """Status Quo Bias: Resist proposed changes.

        Trigger: Change-proposal markers in input.
        Personality: Low-O + High-C (prefer established ways).
        Effect: Reduces elasticity.
        """
        matched = [m for m in CHANGE_MARKERS if m in input_lower]
        if not matched:
            return None

        # Low-O + High-C = more status quo bias
        susceptibility = ((1 - self.openness) + self.conscientiousness) / 2.0
        if susceptibility < 0.5:
            return None

        marker_strength = min(len(matched) / 2.0, 1.0)
        raw_strength = marker_strength * (susceptibility - 0.5) / 0.5
        adjusted = self._apply_override("status_quo_bias", raw_strength)

        if adjusted < 0.1:
            return None

        modifier = -min(adjusted * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="status_quo_bias",
                effect=f"Change proposal ('{matched[0]}') + low-O/high-C → elasticity {modifier:+.3f}",
                weight=adjusted,
            )

        return BiasModifier(
            bias_type=BiasType.STATUS_QUO,
            target_field="response_structure.elasticity",
            operation="add",
            modifier=modifier,
            trigger=f"change_markers={len(matched)}, susceptibility={susceptibility:.2f}",
            strength=adjusted,
        )

    def _compute_availability_bias(
        self,
        input_lower: str,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """Availability Bias: Overweight negative examples (availability heuristic).

        Trigger: Negative content in input.
        Personality: High-N (more available negative examples in memory).
        Effect: Increases arousal (overreacts to negative information).
        """
        negative_count = _count_unnegated_markers(input_lower)
        if negative_count == 0:
            return None

        # High-N = more negative examples available in memory
        if self.neuroticism < 0.5:
            return None

        marker_strength = min(negative_count / 3.0, 1.0)
        raw_strength = marker_strength * (self.neuroticism - 0.5) / 0.5
        adjusted = self._apply_override("availability_bias", raw_strength)

        if adjusted < 0.1:
            return None

        modifier = min(adjusted * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="trait",
                source_id="availability_bias",
                effect=f"Negative info + high-N availability → arousal {modifier:+.3f}",
                weight=adjusted,
            )

        return BiasModifier(
            bias_type=BiasType.AVAILABILITY,
            target_field="communication_style.arousal",
            operation="add",
            modifier=modifier,
            trigger=f"negative_markers={negative_count}, neuroticism={self.neuroticism:.2f}",
            strength=adjusted,
        )

    def _compute_empathy_gap(
        self,
        input_lower: str,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """Empathy Gap: Underestimates others' emotional reactions.

        Trigger: Emotional content in user input.
        Personality: Low-A (primary — less emotionally attuned to others).
        High-N actually increases emotional sensitivity, so it counteracts
        empathy gap rather than amplifying it.
        Effect: Reduces disclosure (doesn't engage emotionally).
        """
        matched = [m for m in EMOTIONAL_CONTENT_MARKERS if m in input_lower]
        if not matched:
            return None

        # Low-A = primary driver of empathy gap (less attuned to others)
        # High-N counteracts: neurotic individuals are MORE sensitive to emotions
        empathy_deficit = (1 - self.agreeableness) * 0.8 + (1 - self.neuroticism) * 0.2
        if empathy_deficit < 0.6:
            return None

        marker_strength = min(len(matched) / 2.0, 1.0)
        raw_strength = marker_strength * (empathy_deficit - 0.6) / 0.4
        adjusted = self._apply_override("empathy_gap", raw_strength)

        if adjusted < 0.1:
            return None

        modifier = -min(adjusted * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="trait",
                source_id="empathy_gap",
                effect=f"Emotional content ('{matched[0]}') + low-A/low-N → disclosure {modifier:+.3f}",
                weight=adjusted,
            )

        return BiasModifier(
            bias_type=BiasType.EMPATHY_GAP,
            target_field="knowledge_disclosure.disclosure_level",
            operation="add",
            modifier=modifier,
            trigger=f"emotional_markers={len(matched)}, empathy_deficit={empathy_deficit:.2f}",
            strength=adjusted,
        )

    def _compute_dunning_kruger_bias(
        self,
        proficiency: float,
        ctx: Optional["TraceContext"] = None,
    ) -> BiasModifier | None:
        """Dunning-Kruger Bias: Overconfident when unknowledgeable.

        Trigger: Low proficiency in current domain.
        Personality: Low-O + High-C (less self-aware, more certain).
        Effect: Increases confidence despite low expertise.
        """
        if proficiency > 0.35:
            return None  # Only triggers for genuinely low proficiency

        # High knowledge_boundary_strictness = persona is self-aware about limits
        # This suppresses DK effect (you can't be overconfident if you know you don't know)
        if self._knowledge_boundary_strictness > 0.7:
            return None

        # Low-O + High-C = more susceptible to DK effect
        susceptibility = ((1 - self.openness) + self.conscientiousness) / 2.0
        if susceptibility < 0.5:
            return None

        # Strength inversely proportional to proficiency
        raw_strength = (0.35 - proficiency) / 0.35 * (susceptibility - 0.5) / 0.5
        adjusted = self._apply_override("dunning_kruger_bias", raw_strength)

        if adjusted < 0.1:
            return None

        modifier = min(adjusted * MAX_BIAS_IMPACT, MAX_BIAS_IMPACT)

        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="dunning_kruger_bias",
                effect=f"Low proficiency ({proficiency:.2f}) + low-O/high-C → confidence {modifier:+.3f}",
                weight=adjusted,
            )

        return BiasModifier(
            bias_type=BiasType.DUNNING_KRUGER,
            target_field="response_structure.confidence",
            operation="add",
            modifier=modifier,
            trigger=f"proficiency={proficiency:.2f}, susceptibility={susceptibility:.2f}",
            strength=adjusted,
        )

    def get_modifier_for_field(
        self,
        modifiers: list[BiasModifier],
        field: str,
    ) -> BiasModifier | None:
        """
        Get the first modifier targeting a specific field, if any.

        Args:
            modifiers: List from compute_modifiers()
            field: Target field name (e.g., "response_structure.elasticity")

        Returns:
            The matching BiasModifier or None
        """
        for mod in modifiers:
            if mod.target_field == field:
                return mod
        return None

    def get_total_modifier_for_field(
        self,
        modifiers: list[BiasModifier],
        field: str,
    ) -> float:
        """Sum all modifiers targeting a specific field.

        Phase R6: Multiple biases may target the same field (e.g., confirmation
        + anchoring + status_quo all reduce elasticity). Sum them, bounded by
        MAX_BIAS_IMPACT.
        """
        total = 0.0
        for mod in modifiers:
            if mod.target_field == field:
                total += mod.modifier
        # Bound total to [-MAX_BIAS_IMPACT, MAX_BIAS_IMPACT] per field
        return max(-MAX_BIAS_IMPACT * 2, min(MAX_BIAS_IMPACT * 2, total))


# =============================================================================
# Factory Function
# =============================================================================

def create_bias_simulator(
    traits: dict[str, float],
    value_priorities: dict[str, float],
) -> BiasSimulator:
    """Factory function to create a BiasSimulator instance."""
    return BiasSimulator(traits, value_priorities)
