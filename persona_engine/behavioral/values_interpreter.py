"""
Schwartz Values Interpreter

Translates Schwartz's 10 basic values into motivational priorities and 
resolves conflicts between competing values.
"""

from typing import Dict, List, Tuple, Optional
from persona_engine.schema.persona_schema import SchwartzValues, Persona


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
    
    def _to_dict(self) -> Dict[str, float]:
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
    
    def get_value_priorities(self) -> Dict[str, float]:
        """
        Get all value priorities as a dictionary.
        
        Returns:
            Dict mapping value name -> weight (0-1)
        """
        return dict(self._value_dict)
    
    def get_top_values(self, n: int = 3) -> List[Tuple[str, float]]:
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
    
    def detect_value_conflicts(self, threshold: float = 0.6) -> List[Dict[str, any]]:
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
            pair = tuple(sorted([conflict["value_1"], conflict["value_2"]]))
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
        weight_1 = self._value_dict[value_1]
        weight_2 = self._value_dict[value_2]
        
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
        
        return value_1 if adjusted_1 >= adjusted_2 else value_2
    
    def get_value_influence_on_stance(
        self, 
        topic: str, 
        options: List[str]
    ) -> Dict[str, float]:
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
    
    def get_rationale_influences(self) -> List[Tuple[str, float, str]]:
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
    
    def get_value_markers_for_validation(self) -> Dict[str, any]:
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
