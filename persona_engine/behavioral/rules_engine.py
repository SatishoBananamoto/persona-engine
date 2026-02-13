"""
Behavioral Rules Engine

Applies persona-specific decision policies, social role adjustments,
and response patterns.
"""

from typing import Any

from persona_engine.schema.ir_schema import InteractionMode
from persona_engine.schema.persona_schema import DecisionPolicy, Persona, ResponsePattern


class BehavioralRulesEngine:
    """
    Applies behavioral rules and policies to modify IR parameters.

    Rules include:
    - Social role-based communication adjustments
    - Decision policies (conditional logic)
    - Response patterns (trigger → reaction)
    """

    def __init__(self, persona: Persona):
        self.persona = persona
        self.social_roles = persona.social_roles
        self.decision_policies = persona.decision_policies
        self.response_patterns = persona.response_patterns
        self.base_communication = persona.psychology.communication

    # ========================================================================
    # Social Role Application
    # ========================================================================

    def get_social_role_mode(
        self,
        interaction_mode: InteractionMode
    ) -> str:
        """
        Map interaction mode to social role mode.

        Args:
            interaction_mode: Type of interaction

        Returns:
            Social role key (e.g., "at_work", "friend", "default")
        """
        # Mapping from interaction modes to social roles
        mode_to_role = {
            InteractionMode.CUSTOMER_SUPPORT: "at_work",
            InteractionMode.INTERVIEW: "at_work",
            InteractionMode.CASUAL_CHAT: "friend",
            InteractionMode.SMALL_TALK: "friend",
            InteractionMode.SURVEY: "default",
            InteractionMode.COACHING: "default",
            InteractionMode.DEBATE: "debate",
        }

        role = mode_to_role.get(interaction_mode, "default")

        # Synthesize debate role from default if not defined
        if role == "debate" and role not in self.social_roles:
            if "default" in self.social_roles:
                default_role = self.social_roles["default"]
                from persona_engine.schema.persona_schema import SocialRole
                self.social_roles["debate"] = SocialRole(
                    formality=min(1.0, default_role.formality + 0.10),
                    directness=min(1.0, default_role.directness + 0.15),
                    emotional_expressiveness=min(1.0, default_role.emotional_expressiveness + 0.10),
                )
            else:
                role = "default"

        # Ensure role exists in persona's social_roles
        if role not in self.social_roles:
            role = "default"

        return role

    def apply_social_role_adjustments(
        self,
        interaction_mode: InteractionMode,
        base_formality: float,
        base_directness: float,
        base_emotional_expressiveness: float
    ) -> dict[str, float]:
        """
        Adjust communication style based on social role.

        Args:
            interaction_mode: Current interaction mode
            base_formality: Base formality from trait interpreter
            base_directness: Base directness from trait interpreter
            base_emotional_expressiveness: Base from communication preferences

        Returns:
            Dict with adjusted values
        """
        role_mode = self.get_social_role_mode(interaction_mode)
        role = self.social_roles[role_mode]

        # Social role overrides base communication style
        # Blend 70% role, 30% base personality
        adjusted_formality = (role.formality * 0.7) + (base_formality * 0.3)
        adjusted_directness = (role.directness * 0.7) + (base_directness * 0.3)
        adjusted_expressiveness = (role.emotional_expressiveness * 0.7) + (base_emotional_expressiveness * 0.3)

        return {
            "formality": adjusted_formality,
            "directness": adjusted_directness,
            "emotional_expressiveness": adjusted_expressiveness
        }

    # ========================================================================
    # Decision Policies
    # ========================================================================

    def check_decision_policy(
        self,
        situation: str
    ) -> DecisionPolicy | None:
        """
        Check if any decision policy applies to situation.

        Args:
            situation: Description of current situation/question

        Returns:
            Matching DecisionPolicy or None
        """
        # In production, this would use semantic matching (embeddings)
        # For MVP, use simple keyword matching

        situation_lower = situation.lower()

        for policy in self.decision_policies:
            # Simple keyword check (to be improved)
            if policy.condition.lower() in situation_lower:
                return policy

        return None

    def apply_decision_policy(
        self,
        policy: DecisionPolicy
    ) -> dict[str, Any]:
        """
        Apply decision policy to modify IR generation.

        Args:
            policy: Decision policy to apply

        Returns:
            Dict of IR modifications
        """
        modifications = {
            "policy_triggered": True,
            "policy_name": policy.condition,
            "suggested_approach": policy.approach
        }

        # Parse time_needed if specified
        if policy.time_needed:
            time_map = {
                "immediate": 0.1,
                "brief": 0.3,
                "moderate": 0.5,
                "extended": 0.8
            }
            modifications["decision_time"] = time_map.get(policy.time_needed, 0.5)

        return modifications

    # ========================================================================
    # Response Patterns
    # ========================================================================

    def check_response_pattern(
        self,
        input_text: str
    ) -> ResponsePattern | None:
        """
        Check if input matches any response pattern triggers.

        Args:
            input_text: User's input

        Returns:
            Matching ResponsePattern or None
        """
        input_lower = input_text.lower()

        for pattern in self.response_patterns:
            if pattern.trigger.lower() in input_lower:
                return pattern

        return None

    def apply_response_pattern(
        self,
        pattern: ResponsePattern
    ) -> dict[str, Any]:
        """
        Apply response pattern to influence IR.

        Args:
            pattern: Response pattern to apply

        Returns:
            Dict of IR modifications
        """
        return {
            "pattern_triggered": True,
            "trigger": pattern.trigger,
            "suggested_response": pattern.response,
            "emotional_intensity": pattern.emotionality,
            "arousal_boost": pattern.emotionality * 0.3  # Higher emotionality → higher arousal
        }

    # ========================================================================
    # Combined Rule Application
    # ========================================================================

    def apply_all_rules(
        self,
        interaction_mode: InteractionMode,
        input_text: str,
        base_style: dict[str, float]
    ) -> dict[str, Any]:
        """
        Apply all applicable behavioral rules.

        Args:
            interaction_mode: Current interaction mode
            input_text: User's input
            base_style: Base communication style (formality, directness, expressiveness)

        Returns:
            Dict of all modifications to apply
        """
        modifications = {}

        # 1. Social role adjustments
        role_adjustments = self.apply_social_role_adjustments(
            interaction_mode,
            base_style.get("formality", 0.5),
            base_style.get("directness", 0.5),
            base_style.get("emotional_expressiveness", 0.5)
        )
        modifications["social_role"] = role_adjustments

        # 2. Check decision policies
        policy = self.check_decision_policy(input_text)
        if policy:
            modifications["decision_policy"] = self.apply_decision_policy(policy)

        # 3. Check response patterns
        pattern = self.check_response_pattern(input_text)
        if pattern:
            modifications["response_pattern"] = self.apply_response_pattern(pattern)

        return modifications

    def get_privacy_filter_level(self, topic: str) -> float:
        """
        Get privacy sensitivity for topic.

        Args:
            topic: Topic being discussed

        Returns:
            Privacy filter strength (0=open, 1=very private)
        """
        # Check topic sensitivities
        for sensitivity in self.persona.topic_sensitivities:
            if sensitivity.topic.lower() in topic.lower():
                return sensitivity.sensitivity

        # Default to base privacy sensitivity
        return self.persona.privacy_sensitivity

    def should_apply_time_constraint(self, conversation_length: int) -> bool:
        """
        Check if time scarcity should affect response.

        Args:
            conversation_length: Number of turns

        Returns:
            True if time pressure applies
        """
        # High time scarcity → pressure builds faster
        threshold = int(10 / max(0.1, self.persona.time_scarcity))
        return conversation_length > threshold


def create_behavioral_rules_engine(persona: Persona) -> BehavioralRulesEngine:
    """Factory function to create behavioral rules engine"""
    return BehavioralRulesEngine(persona)
