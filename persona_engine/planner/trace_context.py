"""
TraceContext - Centralized Citation and Safety Tracking

Production-grade context object for Turn Planner that ensures:
- Every mutation goes through proper citation
- All clamps recorded in both citations AND safety plan
- No "forgot to cite" bugs
- Per-turn seed management
"""

from typing import List, Optional, Literal
from persona_engine.schema.ir_schema import (
    Citation,
    SafetyPlan,
    ClampRecord,
    MemoryOps,
    apply_numeric_modifier,
    apply_enum_modifier,
)


class TraceContext:
    """
    Centralized citation and safety tracking for Turn Planner.
    
    Provides convenience methods that ensure every mutation
    goes through proper citation and safety plan recording.
    
    Features:
    - Auto-generates delta citations
    - Records clamps in SafetyPlan
    - Supports multiple clamps per field
    - Constraint name tracking
    """
    
    def __init__(self):
        self.citations: List[Citation] = []
        self.safety_plan: SafetyPlan = SafetyPlan()
        self.memory_ops: MemoryOps = MemoryOps()
    
    # ========================================================================
    # Convenience Wrappers (shorter names for planner code)
    # ========================================================================
    
    def num(
        self,
        *,
        source_type: str,
        source_id: str,
        target_field: str,
        operation: Literal["set", "add", "multiply", "clamp", "override", "blend"],
        before: float,
        after: float,
        effect: str,
        weight: float = 1.0,
        reason: Optional[str] = None,
    ) -> float:
        """Apply numeric modifier with auto-citation"""
        return apply_numeric_modifier(
            citations=self.citations,
            source_type=source_type,
            source_id=source_id,
            target_field=target_field,
            operation=operation,
            before=before,
            after=after,
            effect=effect,
            weight=weight,
            reason=reason,
        )
    
    def enum(
        self,
        *,
        source_type: str,
        source_id: str,
        target_field: str,
        operation: Literal["set", "override"],
        before: str,
        after: str,
        effect: str,
        weight: float = 1.0,
        reason: Optional[str] = None,
    ) -> str:
        """Apply enum/string modifier with auto-citation"""
        return apply_enum_modifier(
            citations=self.citations,
            source_type=source_type,
            source_id=source_id,
            target_field=target_field,
            operation=operation,
            before=before,
            after=after,
            effect=effect,
            weight=weight,
            reason=reason,
        )
    
    def clamp(
        self,
        *,
        field_name: str,
        target_field: str,
        proposed: float,
        minimum: Optional[float],
        maximum: Optional[float],
        constraint_name: str,  # CRITICAL: not hardcoded to "bounds_check"
        reason: str,
    ) -> float:
        """
        Clamp value and record in both citations and safety plan.
        
        Supports multiple clamps per field (stored as List[ClampRecord]).
        Only records if actual clamping occurred.
        
        Args:
            field_name: Short name for safety_plan key (e.g., "disclosure_level")
            target_field: Full IR field path (e.g., "knowledge_disclosure.disclosure_level")
            proposed: Value before clamping
            minimum: Lower bound (None = no lower bound)
            maximum: Upper bound (None = no upper bound)
            constraint_name: Which constraint is clamping (e.g., "privacy_filter", "bounds_check")
            reason: Why this clamp is happening
            
        Returns:
            Clamped value
        """
        actual = proposed
        
        # Apply clamps
        if minimum is not None and actual < minimum:
            actual = minimum
        if maximum is not None and actual > maximum:
            actual = maximum
        
        # Only record if clamping happened
        if abs(actual - proposed) > 1e-9:
            # Add citation
            self.num(
                source_type="constraint",
                source_id=constraint_name,
                target_field=target_field,
                operation="clamp",
                before=proposed,
                after=actual,
                effect=f"Clamped to [{minimum}, {maximum}]",
                weight=1.0,
                reason=reason
            )
            
            # Add safety plan record (supports multiple clamps per field)
            if field_name not in self.safety_plan.clamped_fields:
                self.safety_plan.clamped_fields[field_name] = []
            
            self.safety_plan.clamped_fields[field_name].append(ClampRecord(
                proposed=proposed,
                actual=actual,
                minimum=minimum,
                maximum=maximum,
                reason=f"{constraint_name}: {reason}"
            ))
            
            # Mark constraint as active
            self.activate_constraint(constraint_name)
        
        return actual
    
    def base(
        self,
        *,
        field_name: str,
        target_field: str,
        value: float,
        effect: str,
    ) -> float:
        """Initialize numeric value with base citation (for clean trails)"""
        return self.num(
            source_type="base",
            source_id=field_name,
            target_field=target_field,
            operation="set",
            before=0.0,
            after=value,
            effect=effect,
            weight=1.0
        )
    
    def base_enum(
        self,
        *,
        field_name: str,
        target_field: str,
        value: str,
        effect: str,
    ) -> str:
        """Initialize enum value with base citation"""
        return self.enum(
            source_type="base",
            source_id=field_name,
            target_field=target_field,
            operation="set",
            before="none",
            after=value,
            effect=effect,
            weight=1.0
        )
    
    def block_topic(self, topic: str):
        """Record blocked topic in safety plan"""
        if topic not in self.safety_plan.blocked_topics:
            self.safety_plan.blocked_topics.append(topic)
        
        self.activate_constraint("must_avoid")
    
    def block_pattern(self, pattern_trigger: str, reason: str):
        """Record blocked pattern in safety plan"""
        block_msg = f"Pattern '{pattern_trigger}' blocked: {reason}"
        if block_msg not in self.safety_plan.pattern_blocks:
            self.safety_plan.pattern_blocks.append(block_msg)
        
        self.activate_constraint("pattern_safety")
    
    def activate_constraint(self, constraint_name: str):
        """Mark constraint as active (idempotent)"""
        if constraint_name not in self.safety_plan.active_constraints:
            self.safety_plan.active_constraints.append(constraint_name)
    
    def add_basic_citation(
        self,
        *,
        source_type: str,
        source_id: str,
        effect: str,
        weight: float = 1.0
    ):
        """Add basic citation without delta tracking (for info/context)"""
        self.citations.append(Citation(
            source_type=source_type,  # type: ignore
            source_id=source_id,
            effect=effect,
            weight=weight
        ))


# ============================================================================
# Utility Functions
# ============================================================================

def clamp01(
    ctx: TraceContext,
    field_name: str,
    target_field: str,
    value: float,
    reason: str = "Ensure [0,1] bounds"
) -> float:
    """
    Convenience: clamp numeric value to [0, 1].
    
    Only records if actual clamping occurs.
    """
    return ctx.clamp(
        field_name=field_name,
        target_field=target_field,
        proposed=value,
        minimum=0.0,
        maximum=1.0,
        constraint_name="bounds_check",
        reason=reason
    )


def create_turn_seed(base_seed: int, conversation_id: str, turn_number: int) -> int:
    """
    Create deterministic per-turn seed.
    
    Ensures same conversation+turn always gets same seed for reproducibility.
    
    Args:
        base_seed: Base random seed
        conversation_id: Unique conversation identifier
        turn_number: Turn number in conversation
        
    Returns:
        Deterministic seed for this turn
    """
    import hashlib
    
    # Combine inputs deterministically
    seed_input = f"{base_seed}:{conversation_id}:{turn_number}"
    hash_digest = hashlib.sha256(seed_input.encode()).digest()
    
    # Convert first 4 bytes to int
    turn_seed = int.from_bytes(hash_digest[:4], byteorder='big')
    
    return turn_seed
