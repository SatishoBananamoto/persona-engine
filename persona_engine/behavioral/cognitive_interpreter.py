"""
Cognitive Style Interpreter

Translates cognitive style dimensions into decision-making patterns
and reasoning approaches.
"""

from typing import Literal
from persona_engine.schema.persona_schema import CognitiveStyle, Persona


class CognitiveStyleInterpreter:
    """
    Maps cognitive style to decision pathways and reasoning patterns.
    
    Cognitive style influences:
    - How persona proces information (analytical vs intuitive)
    - Risk-taking in uncertain situations
    - Tolerance for ambiguity
    - Complexity of reasoning in rationale
    """
    
    def __init__(self, style: CognitiveStyle):
        self.style = style
    
    # ========================================================================
    # Analytical vs Intuitive
    # ========================================================================
    
    def get_reasoning_approach(self) -> Literal["analytical", "intuitive", "mixed"]:
        """
        Determine primary reasoning style.
        
        Returns:
            "analytical": Systematic, data-driven
            "intuitive": Pattern-based, gut-feel
            "mixed": Balanced approach
        """
        if self.style.analytical_intuitive > 0.7:
            return "analytical"
        elif self.style.analytical_intuitive < 0.3:
            return "intuitive"
        else:
            return "mixed"
    
    def get_rationale_depth(self) -> int:
        """
        How many reasoning steps to include in rationale.
        
        High analytical: More steps, explicit logic
        High intuitive: Fewer steps, holistic reasoning
        
        Returns:
            Number of reasoning steps (1-5)
        """
        # Analytical thinkers provide more detailed reasoning
        if self.style.analytical_intuitive > 0.7:
            return 4 if self.style.systematic_heuristic > 0.7 else 3
        elif self.style.analytical_intuitive < 0.3:
            return 1 if self.style.systematic_heuristic < 0.3 else 2
        else:
            return 2
    
    # ========================================================================
    # Systematic vs Heuristic
    # ========================================================================
    
    def prefers_systematic_processing(self) -> bool:
        """Does persona prefer systematic over heuristic processing?"""
        return self.style.systematic_heuristic > 0.6
    
    def get_decision_time_modifier(self) -> Literal["quick", "moderate", "extended"]:
        """
        How much time persona needs for decisions.
        
        High systematic: More time needed
        High heuristic: Quick decisions
        """
        if self.style.systematic_heuristic > 0.7:
            return "extended"
        elif self.style.systematic_heuristic < 0.3:
            return "quick"
        else:
            return "moderate"
    
    # ========================================================================
    # Risk Tolerance
    # ========================================================================
    
    def get_risk_stance_modifier(self, base_confidence: float) -> float:
        """
        Adjusts willingness to take stances based on risk tolerance.
        
        High risk tolerance: More willing to take bold stances even with uncertainty
        Low risk tolerance: Only takes clear stances when confident
        
        Args:
            base_confidence: Confidence from knowledge proficiency
            
        Returns:
            Modified willingness to commit to stance (0-1)
        """
        # Low confidence + low risk tolerance = hedging/refusing
        # Low confidence + high risk tolerance = still willing to commit
        
        if base_confidence < 0.4:
            # Risk tolerance matters more when uncertain
            return base_confidence + (self.style.risk_tolerance * 0.3)
        else:
            # When confident, risk tolerance has less impact
            return base_confidence + (self.style.risk_tolerance * 0.1)
    
    def influences_uncertainty_action(
        self, 
        confidence: float
    ) -> Literal["answer", "hedge", "ask", "refuse"]:
        """
        Determines how to handle uncertainty.
        
        Args:
            confidence: Current confidence level (0-1)
            
        Returns:
            Recommended uncertainty action
        """
        # High need for closure + low confidence → ask clarifying questions
        # Low need for closure + low confidence → speculate/hedge
        # High risk tolerance + moderate confidence → answer
        
        if confidence > 0.7:
            return "answer"
        
        elif confidence > 0.4:
            if self.style.risk_tolerance > 0.6:
                return "answer"  # Willing to commit despite uncertainty
            else:
                return "hedge"   # Play it safe
        
        else:  # Low confidence
            if self.style.need_for_closure > 0.6:
                return "ask"     # Need definiteness → ask questions
            elif self.style.risk_tolerance < 0.3:
                return "refuse"  # Too risky → decline to answer
            else:
                return "hedge"   # Speculate with caveats
    
    # ========================================================================
    # Need for Closure
    # ========================================================================
    
    def get_ambiguity_tolerance(self) -> float:
        """
        How comfortable persona is with ambiguity.
        
        High need for closure = low ambiguity tolerance
        
        Returns:
            Ambiguity tolerance (0-1)
        """
        return 1.0 - self.style.need_for_closure
    
    def prefers_definite_answers(self) -> bool:
        """High need for closure → prefers yes/no over maybe"""
        return self.style.need_for_closure > 0.6
    
    # ========================================================================
    # Cognitive Complexity
    # ========================================================================
    
    def get_nuance_capacity(self) -> Literal["low", "moderate", "high"]:
        """
        Ability to hold nuanced, multifaceted views.
        
        Returns:
            Nuance capacity level
        """
        if self.style.cognitive_complexity > 0.7:
            return "high"
        elif self.style.cognitive_complexity < 0.3:
            return "low"
        else:
            return "moderate"
    
    def should_acknowledge_tradeoffs(self) -> bool:
        """
        High cognitive complexity: Acknowledges tradeoffs and counterarguments
        Low cognitive complexity: More black-and-white thinking
        """
        return self.style.cognitive_complexity > 0.6
    
    def get_stance_complexity_level(self) -> int:
        """
        How many dimensions/qualifications to include in stance.
        
        High complexity: "X is good for Y but problematic for Z"
        Low complexity: "X is good"
        
        Returns:
            Number of dimensions (1-3)
        """
        if self.style.cognitive_complexity > 0.7:
            return 3
        elif self.style.cognitive_complexity < 0.3:
            return 1
        else:
            return 2
    
    # ========================================================================
    # Multi-Dimension Interactions
    # ========================================================================
    
    def get_elasticity_from_cognitive_style(self) -> float:
        """
        Cognitive complexity and analytical thinking affect elasticity.
        
        High cognitive complexity: More elastic (sees multiple perspectives)
        Low need for closure: More elastic (comfortable with revision)
        
        Returns:
            Elasticity contribution (0-1)
        """
        complexity_factor = self.style.cognitive_complexity * 0.6
        closure_factor = (1.0 - self.style.need_for_closure) * 0.4
        
        return complexity_factor + closure_factor
    
    def get_confidence_adjustment(self, base_confidence: float) -> float:
        """
        Adjust confidence based on cognitive style.
        
        High analytical: Slightly lower confidence (sees caveats)
        High need for closure: Slightly higher confidence (needs certainty)
        
        Args:
            base_confidence: Base confidence level
            
        Returns:
            Adjusted confidence
        """
        analytical_penalty = 0 if self.style.analytical_intuitive < 0.5 else -0.05
        closure_boost = 0 if self.style.need_for_closure < 0.7 else 0.08
        
        adjusted = base_confidence + analytical_penalty + closure_boost
        return max(0.1, min(0.95, adjusted))
    
    def get_cognitive_markers_for_validation(self) -> dict:
        """
        Returns expected cognitive markers for validation.
        
        Returns:
            Dict of cognitive style → expected patterns
        """
        return {
            "reasoning_approach": self.get_reasoning_approach(),
            "rationale_depth": self.get_rationale_depth(),
            "systematic_preference": self.prefers_systematic_processing(),
            "risk_tolerance": {
                "level": self.style.risk_tolerance,
                "affects_low_confidence_stances": self.style.risk_tolerance > 0.6
            },
            "ambiguity_tolerance": self.get_ambiguity_tolerance(),
            "nuance_capacity": self.get_nuance_capacity(),
            "should_acknowledge_tradeoffs": self.should_acknowledge_tradeoffs(),
            "stance_complexity_level": self.get_stance_complexity_level()
        }


def create_cognitive_interpreter(persona: Persona) -> CognitiveStyleInterpreter:
    """Factory function to create cognitive interpreter from persona"""
    return CognitiveStyleInterpreter(persona.psychology.cognitive_style)
