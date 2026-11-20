"""
Schwartz Values Interpreter

Translates Schwartz's 10 basic values into motivational priorities and
resolves conflicts between competing values.
"""

from dataclasses import dataclass, field
from typing import Any

from persona_engine.schema.persona_schema import Persona, SchwartzValues

# Value conflict pairs (theoretically opposed in Schwartz circumplex)
VALUE_CONFLICTS = {
    "self_direction": ["conformity", "tradition"],
    "stimulation": ["security", "conformity"],
    "hedonism": ["conformity", "tradition"],
    "achievement": ["benevolence", "universalism"],
    "power": ["universalism", "benevolence"],
    "security": ["stimulation", "self_direction"],
    "conformity": ["self_direction", "stimulation", "hedonism"],
    "tradition": ["self_direction", "hedonism", "stimulation"],
    "benevolence": ["power", "achievement"],
    "universalism": ["power", "achievement"],
}

# Schwartz circumplex adjacency: values that are neighbors on the circle
# reinforce each other and resolve more easily when in tension.
SCHWARTZ_ADJACENCY: dict[str, list[str]] = {
    "self_direction": ["stimulation", "universalism"],
    "stimulation": ["self_direction", "hedonism"],
    "hedonism": ["stimulation", "achievement"],
    "achievement": ["hedonism", "power"],
    "power": ["achievement", "security"],
    "security": ["power", "conformity"],
    "conformity": ["security", "tradition"],
    "tradition": ["conformity", "benevolence"],
    "benevolence": ["tradition", "universalism"],
    "universalism": ["benevolence", "self_direction"],
}


@dataclass
class ConflictResolution:
    """Result of a value conflict resolution."""
    winner: str
    confidence: float  # 0-1, how clean the resolution was
    is_adjacent: bool
    is_opposing: bool
    citations: list[dict[str, Any]] = field(default_factory=list)


class ValuesInterpreter:
    """
    Interprets Schwartz values and resolves value conflicts.

    Values influence:
    - Stance formation (what persona believes)
    - Rationale (why they believe it)
    - Goal prioritization
    - Decision-making under uncertainty
    """

    def __init__(self, values: SchwartzValues):
        self.values = values
        self._value_dict = self._to_dict()

    def _to_dict(self) -> dict[str, float]:
        """Convert SchwartzValues model to dict for easier processing"""
        return {
            "self_direction": self.values.self_direction,
            "stimulation": self.values.stimulation,
            "hedonism": self.values.hedonism,
            "achievement": self.values.achievement,
            "power": self.values.power,
            "security": self.values.security,
            "conformity": self.values.conformity,
            "tradition": self.values.tradition,
            "benevolence": self.values.benevolence,
            "universalism": self.values.universalism,
        }

    def get_value_priorities(self) -> dict[str, float]:
        """
        Get all value priorities as a dictionary.

        Returns:
            Dict mapping value name -> weight (0-1)
        """
        return dict(self._value_dict)

    def get_top_values(self, n: int = 3) -> list[tuple[str, float]]:
        """
        Get the N highest-priority values.

        Args:
            n: Number of top values to return

        Returns:
            List of (value_name, weight) sorted by weight descending
        """
        sorted_values = sorted(
            self._value_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_values[:n]

    def detect_value_conflicts(self, threshold: float = 0.6) -> list[dict[str, Any]]:
        """
        Detect when persona has high weights on conflicting values.

        Args:
            threshold: Minimum value weight to consider (default 0.6)

        Returns:
            List of conflicts with tension scores
        """
        conflicts = []

        for value, weight in self._value_dict.items():
            if weight >= threshold:
                # Check if any conflicting values are also high
                for conflicting_value in VALUE_CONFLICTS.get(value, []):
                    conflicting_weight = self._value_dict[conflicting_value]

                    if conflicting_weight >= threshold:
                        # Calculate tension (both values high)
                        tension = min(weight, conflicting_weight)

                        conflicts.append({
                            "value_1": value,
                            "value_2": conflicting_value,
                            "weight_1": weight,
                            "weight_2": conflicting_weight,
                            "tension": tension,
                            "description": f"{value} ({weight:.2f}) conflicts with {conflicting_value} ({conflicting_weight:.2f})"
                        })

        # Remove duplicates (A-B and B-A)
        seen = set()
        unique_conflicts = []
        for conflict in conflicts:
            pair = tuple(sorted([str(conflict["value_1"]), str(conflict["value_2"])]))
            if pair not in seen:
                seen.add(pair)
                unique_conflicts.append(conflict)

        return sorted(unique_conflicts, key=lambda x: x["tension"], reverse=True)

    def resolve_conflict(
        self,
        value_1: str,
        value_2: str,
        context: str = "general"
    ) -> str:
        """
        When two conflicting values are triggered, decide which dominates.

        Args:
            value_1: First value name
            value_2: Second value name
            context: Situational context ("work", "personal", "social", "general")

        Returns:
            Dominant value name
        """
        resolution = self.resolve_conflict_detailed(value_1, value_2, context)
        return resolution.winner

    def resolve_conflict_detailed(
        self,
        value_1: str,
        value_2: str,
        context: str = "general"
    ) -> ConflictResolution:
        """
        Resolve a value conflict with full detail including adjacency effects.

        Adjacent values on the Schwartz circumplex resolve more easily.
        Opposing values produce lower confidence and conflict citations.

        Args:
            value_1: First value name
            value_2: Second value name
            context: Situational context ("work", "personal", "social", "general")

        Returns:
            ConflictResolution with winner, confidence, and citations
        """
        weight_1 = self._value_dict[value_1]
        weight_2 = self._value_dict[value_2]

        # Determine circumplex relationship
        is_adjacent = value_2 in SCHWARTZ_ADJACENCY.get(value_1, [])
        is_opposing = value_2 in VALUE_CONFLICTS.get(value_1, [])

        # Context biases (some values stronger in certain contexts)
        context_biases = {
            "work": {"achievement": 0.1, "power": 0.05, "conformity": 0.05},
            "personal": {"self_direction": 0.1, "hedonism": 0.1, "security": 0.05},
            "social": {"benevolence": 0.1, "universalism": 0.08, "tradition": 0.05},
        }

        # Apply context bias
        bias_1 = context_biases.get(context, {}).get(value_1, 0)
        bias_2 = context_biases.get(context, {}).get(value_2, 0)

        adjusted_1 = weight_1 + bias_1
        adjusted_2 = weight_2 + bias_2

        citations: list[dict[str, Any]] = []

        # Apply adjacency effects
        if is_adjacent:
            # Adjacent values reinforce — boost the winner slightly for smooth resolution
            if adjusted_1 >= adjusted_2:
                adjusted_1 += 0.05
            else:
                adjusted_2 += 0.05
            citations.append({
                "source_type": "value",
                "source_id": "schwartz_adjacency",
                "effect": f"{value_1} and {value_2} are adjacent on Schwartz circumplex — easy resolution",
                "weight": 0.3,
            })
        elif is_opposing:
            # Opposing values create genuine tension — reduce both slightly
            citations.append({
                "source_type": "value",
                "source_id": "schwartz_opposition",
                "effect": f"{value_1} and {value_2} are opposing on Schwartz circumplex — conflicted resolution",
                "weight": 0.7,
            })

        winner = value_1 if adjusted_1 >= adjusted_2 else value_2

        # Compute resolution confidence
        margin = abs(adjusted_1 - adjusted_2)
        if is_adjacent:
            # Adjacent: high confidence even with small margins
            confidence = min(1.0, 0.7 + margin)
        elif is_opposing:
            # Opposing: lower confidence, especially with small margins
            confidence = min(0.8, 0.3 + margin)
        else:
            # Neutral relationship
            confidence = min(1.0, 0.5 + margin)

        return ConflictResolution(
            winner=winner,
            confidence=confidence,
            is_adjacent=is_adjacent,
            is_opposing=is_opposing,
            citations=citations,
        )

    def get_value_influence_on_stance(
        self,
        topic: str,
        options: list[str]
    ) -> dict[str, float]:
        """
        Score stance options based on value alignment.

        Args:
            topic: Topic being discussed
            options: List of possible stance descriptions

        Returns:
            Dict mapping option -> alignment score (0-1)
        """
        # This is a simplified version - in production would use
        # semantic similarity between option text and value definitions

        # For now, return even scores (to be enhanced with LLM or embedding matching)
        return {option: 0.5 for option in options}

    def get_rationale_influences(self) -> list[tuple[str, float, str]]:
        """
        Get values that should appear in rationale explanations.

        Returns:
            List of (value_name, weight, influence_description)
        """
        influences = []

        # Only include values above threshold
        for value, weight in self._value_dict.items():
            if weight >= 0.6:
                influence = self._get_value_influence_description(value)
                influences.append((value, weight, influence))

        return sorted(influences, key=lambda x: x[1], reverse=True)

    def _get_value_influence_description(self, value: str) -> str:
        """Get description of how value influences behavior"""
        descriptions = {
            "self_direction": "Emphasizes autonomy and independent thinking",
            "stimulation": "Seeks novelty and challenges conventional approaches",
            "hedonism": "Prioritizes enjoyment and personal satisfaction",
            "achievement": "Focuses on competence and demonstrable success",
            "power": "Values influence and status (use cautiously)",
            "security": "Prioritizes safety, stability, and predictability",
            "conformity": "Respects rules and social expectations",
            "tradition": "Values customs and established practices",
            "benevolence": "Emphasizes caring for others and human welfare",
            "universalism": "Advocates for justice, equality, and protecting all people",
        }
        return descriptions.get(value, "Influences decision-making")

    def should_include_in_citation(self, value: str) -> bool:
        """
        Determine if a value is strong enough to cite in IR.

        Args:
            value: Value name

        Returns:
            True if value should be cited as influencing decision
        """
        return self._value_dict[value] >= 0.65

    def get_value_markers_for_validation(self) -> dict[str, Any]:
        """
        Returns expected value markers for validation.

        Returns:
            Dict of value markers to check in IR/text
        """
        top_values = self.get_top_values(n=3)
        conflicts = self.detect_value_conflicts()

        return {
            "top_values": [{"name": v, "weight": w} for v, w in top_values],
            "active_conflicts": conflicts,
            "citation_worthy": [
                v for v in self._value_dict.keys()
                if self.should_include_in_citation(v)
            ],
            "dominant_value": top_values[0][0] if top_values else None
        }


def create_values_interpreter(persona: Persona) -> ValuesInterpreter:
    """Factory function to create values interpreter from persona"""
    return ValuesInterpreter(persona.psychology.values)
