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
from typing import List, Optional, Literal, Dict, Any, TYPE_CHECKING
from enum import Enum

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


# =============================================================================
# Data Structures
# =============================================================================

class BiasType(str, Enum):
    """Types of cognitive biases simulated"""
    CONFIRMATION = "confirmation_bias"
    NEGATIVITY = "negativity_bias"
    AUTHORITY = "authority_bias"


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
        traits: Dict[str, float],
        value_priorities: Dict[str, float],
    ):
        """
        Args:
            traits: Big 5 traits dict (openness, conscientiousness, etc.)
            value_priorities: Schwartz values normalized priorities
        """
        self.traits = traits
        self.values = value_priorities
        
        # Precompute trait accessors
        self.neuroticism = traits.get("neuroticism", 0.5)
        self.openness = traits.get("openness", 0.5)
        
        # Precompute value accessors for conformity-related values
        self.conformity = value_priorities.get("conformity", 0.0)
        self.tradition = value_priorities.get("tradition", 0.0)
        self.security = value_priorities.get("security", 0.0)
    
    def compute_modifiers(
        self,
        user_input: str,
        value_alignment: float = 0.0,
        ctx: Optional["TraceContext"] = None,
    ) -> List[BiasModifier]:
        """
        Compute all applicable bias modifiers for current context.
        
        Args:
            user_input: The user's input text
            value_alignment: How aligned is input with persona's values (0-1)
            ctx: TraceContext for citation (optional)
        
        Returns:
            List of BiasModifier objects to apply
        """
        modifiers: List[BiasModifier] = []
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
        
        return modifiers
    
    def _compute_confirmation_bias(
        self,
        value_alignment: float,
        ctx: Optional["TraceContext"] = None,
    ) -> Optional[BiasModifier]:
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
    ) -> Optional[BiasModifier]:
        """
        Negativity Bias: Increased attention/arousal to negative input.
        
        Mechanism: Neuroticism amplifies response to negative stimuli.
        """
        if self.neuroticism < NEGATIVITY_NEUROTICISM_THRESHOLD:
            return None  # Not neurotic enough to trigger
        
        # Count negative markers
        negative_count = sum(1 for marker in NEGATIVE_MARKERS if marker in input_lower)
        
        if negative_count == 0:
            return None  # No negative content
        
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
    ) -> Optional[BiasModifier]:
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
    
    def get_modifier_for_field(
        self,
        modifiers: List[BiasModifier],
        field: str,
    ) -> Optional[BiasModifier]:
        """
        Get the modifier targeting a specific field, if any.
        
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


# =============================================================================
# Factory Function
# =============================================================================

def create_bias_simulator(
    traits: Dict[str, float],
    value_priorities: Dict[str, float],
) -> BiasSimulator:
    """Factory function to create a BiasSimulator instance."""
    return BiasSimulator(traits, value_priorities)
