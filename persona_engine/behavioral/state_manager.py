"""
Dynamic State Manager

Manages persona's changing state during conversation:
- Mood (valence + arousal)
- Fatigue
- Stress
- Engagement

State evolves based on conversation context and personality traits.
"""


from persona_engine.schema.persona_schema import BigFiveTraits, DynamicState, Persona
from persona_engine.utils.determinism import DeterminismManager


class StateManager:
    """
    Tracks and updates dynamic state during conversation.

    State influences:
    - Tone selection
    - Disclosure level
    - Verbosity
    - Patience/directness
    """

    def __init__(
        self,
        initial_state: DynamicState,
        traits: BigFiveTraits,
        determinism: DeterminismManager | None = None
    ):
        self.state = DynamicState(**initial_state.model_dump())
        self.traits = traits
        self.determinism = determinism or DeterminismManager()

        # State evolution rates (influenced by neuroticism)
        # High N → mood lingers (slower drift), low N → mood stabilizes quickly
        self.mood_drift_rate = 0.12 - (traits.neuroticism * 0.08)
        self.fatigue_accumulation_rate = 0.02
        # High N → slower stress recovery
        self.stress_decay_rate = 0.08 + (1.0 - traits.neuroticism) * 0.04

    # ========================================================================
    # State Getters
    # ========================================================================

    def get_current_state(self) -> DynamicState:
        """Returns current state snapshot"""
        return DynamicState(**self.state.model_dump())

    def get_mood(self) -> tuple[float, float]:
        """Returns (valence, arousal)"""
        return (self.state.mood_valence, self.state.mood_arousal)

    def get_fatigue(self) -> float:
        """Returns current fatigue level"""
        return self.state.fatigue

    def get_stress(self) -> float:
        """Returns current stress level"""
        return self.state.stress

    def get_engagement(self) -> float:
        """Returns current engagement level"""
        return self.state.engagement

    # ========================================================================
    # Mood Updates
    # ========================================================================

    def update_mood_from_event(
        self,
        valence_delta: float,
        arousal_delta: float
    ) -> None:
        """
        Update mood based on conversation event.

        Args:
            valence_delta: Change in valence (-1 to +1)
            arousal_delta: Change in arousal (-1 to +1)
        """
        # Apply deltas
        self.state.mood_valence += valence_delta
        self.state.mood_arousal += arousal_delta

        # Bound values
        self.state.mood_valence = max(-1.0, min(1.0, self.state.mood_valence))
        self.state.mood_arousal = max(0.0, min(1.0, self.state.mood_arousal))

    def apply_mood_drift(self) -> None:
        """
        Mood drifts toward baseline (regression to mean).

        High neuroticism: Slower drift (mood lingers)
        Low neuroticism: Faster drift (mood stabilizes quickly)
        """
        # Baseline mood (extraversion → positive baseline, neuroticism → negative)
        baseline_valence = 0.1 + (self.traits.extraversion * 0.15) - (self.traits.neuroticism * 0.2)
        baseline_arousal = 0.5

        # Drift toward baseline
        valence_diff = baseline_valence - self.state.mood_valence
        arousal_diff = baseline_arousal - self.state.mood_arousal

        self.state.mood_valence += valence_diff * self.mood_drift_rate
        self.state.mood_arousal += arousal_diff * self.mood_drift_rate

    def mood_from_topic_relevance(self, topic_relevance: float) -> None:
        """
        Adjust mood based on topic relevance to persona interests.

        Args:
            topic_relevance: 0-1, how relevant topic is to persona's interests
        """
        # High relevance → positive valence, increased arousal
        if topic_relevance > 0.7:
            self.update_mood_from_event(
                valence_delta=0.1 * self.traits.openness,  # Openness amplifies positive effect
                arousal_delta=0.15
            )
        # Low relevance → slight negative valence, decreased arousal
        elif topic_relevance < 0.3:
            self.update_mood_from_event(
                valence_delta=-0.05,
                arousal_delta=-0.1
            )

    # ========================================================================
    # Fatigue Updates
    # ========================================================================

    def increment_fatigue(self, conversation_length: int) -> None:
        """
        Increase fatigue based on conversation length.

        Args:
            conversation_length: Number of turns in conversation
        """
        # Fatigue accumulates gradually
        fatigue_increase = self.fatigue_accumulation_rate * (conversation_length / 10)

        # Conscientiousness reduces fatigue (more stamina for detail)
        stamina_modifier = 1.0 - (self.traits.conscientiousness * 0.2)

        self.state.fatigue += fatigue_increase * stamina_modifier
        self.state.fatigue = min(1.0, self.state.fatigue)

        # High fatigue reduces engagement and arousal
        if self.state.fatigue > 0.7:
            self.state.engagement = max(0.2, self.state.engagement - 0.1)
            self.state.mood_arousal = max(0.1, self.state.mood_arousal - 0.1)

    # ========================================================================
    # Stress Updates
    # ========================================================================

    def apply_stress_trigger(self, trigger_type: str, intensity: float = 0.3) -> None:
        """
        Apply stress from specific trigger.

        Args:
            trigger_type: "time_pressure", "conflict", "uncertainty", "complexity"
            intensity: How intense the stressor is (0-1)
        """
        # Neuroticism amplifies stress response
        stress_sensitivity = 1.0 + (self.traits.neuroticism * 0.5)

        # Different triggers affect different people differently
        trigger_multipliers = {
            "time_pressure": 1.0,
            "conflict": 1.2 if self.traits.agreeableness > 0.6 else 0.8,
            "uncertainty": 1.3 if self.traits.neuroticism > 0.6 else 0.9,
            "complexity": 0.7  # Less stressful for high cognitive complexity
        }

        multiplier = trigger_multipliers.get(trigger_type, 1.0)
        stress_increase = intensity * stress_sensitivity * multiplier

        self.state.stress += stress_increase
        self.state.stress = min(1.0, self.state.stress)

        # High stress affects mood
        if self.state.stress > 0.6:
            self.state.mood_valence -= 0.15
            self.state.mood_valence = max(-1.0, self.state.mood_valence)
            self.state.mood_arousal += 0.2  # Stress increases arousal
            self.state.mood_arousal = min(1.0, self.state.mood_arousal)

    def reduce_stress(self) -> None:
        """Gradual stress decay (recovery)"""
        self.state.stress -= self.stress_decay_rate
        self.state.stress = max(0.0, self.state.stress)

    # ========================================================================
    # Engagement Updates
    # ========================================================================

    def update_engagement(
        self,
        topic_relevance: float,
        conversation_turn: int
    ) -> None:
        """
        Update engagement based on topic relevance and conversation flow.

        Args:
            topic_relevance: How relevant topic is to persona (0-1)
            conversation_turn: Current turn number
        """
        # Engagement is driven by relevance and novelty
        base_engagement = topic_relevance

        # Openness to experience increases engagement with novel topics
        novelty_bonus = self.traits.openness * 0.2 if conversation_turn < 5 else 0

        # Fatigue reduces engagement
        fatigue_penalty = self.state.fatigue * 0.3

        target_engagement = base_engagement + novelty_bonus - fatigue_penalty
        target_engagement = max(0.1, min(1.0, target_engagement))

        # Smooth transition to target
        diff = target_engagement - self.state.engagement
        self.state.engagement += diff * 0.3

    # ========================================================================
    # Turn-Level State Evolution
    # ========================================================================

    def evolve_state_post_turn(
        self,
        conversation_length: int,
        topic_relevance: float = 0.5
    ) -> None:
        """
        Apply all state updates after a conversation turn.

        Args:
            conversation_length: Total turns so far
            topic_relevance: Relevance of current topic
        """
        # 1. Mood drifts toward baseline
        self.apply_mood_drift()

        # 2. Fatigue accumulates
        self.increment_fatigue(conversation_length)

        # 3. Stress decays (if no active stressors)
        self.reduce_stress()

        # 4. Engagement adjusts to topic
        self.update_engagement(topic_relevance, conversation_length)

        # 5. Add tiny random noise to prevent deterministic loops
        self._add_subtle_noise()

    def _add_subtle_noise(self) -> None:
        """Add small random variations to state (human unpredictability)"""
        noise_budget = 0.03

        self.state.mood_valence = max(-1.0, min(1.0, self.determinism.add_noise(
            self.state.mood_valence,
            noise_budget
        )))
        self.state.mood_arousal = max(0.0, min(1.0, self.determinism.add_noise(
            self.state.mood_arousal,
            noise_budget
        )))

    # ========================================================================
    # State Influence on Behavior
    # ========================================================================

    def get_disclosure_modifier(self) -> float:
        """
        State influence on disclosure level.

        Positive mood: More open (+)
        High stress: Less open (-)
        High fatigue: Less open (-)

        Returns:
            Modifier to add to base disclosure (-0.2 to +0.2)
        """
        mood_effect = self.state.mood_valence * 0.15
        stress_penalty = -self.state.stress * 0.2
        fatigue_penalty = -self.state.fatigue * 0.1

        return mood_effect + stress_penalty + fatigue_penalty

    def get_verbosity_modifier(self) -> int:
        """
        State influence on verbosity.

        High fatigue: Briefer responses (-1)
        High engagement: More detailed (+1)

        Returns:
            Modifier to apply to verbosity level (-1, 0, +1)
        """
        if self.state.fatigue > 0.7:
            return -1
        elif self.state.engagement > 0.7:
            return 1
        else:
            return 0

    def get_patience_level(self) -> float:
        """
        Current patience level (affects directness, tolerance).

        High stress + high fatigue = low patience

        Returns:
            Patience (0=impatient, 1=very patient)
        """
        base_patience = 0.7
        stress_drain = self.state.stress * 0.3
        fatigue_drain = self.state.fatigue * 0.2

        patience = base_patience - stress_drain - fatigue_drain
        return max(0.1, min(1.0, patience))


def create_state_manager(
    persona: Persona,
    determinism: DeterminismManager | None = None
) -> StateManager:
    """Factory function to create state manager from persona"""
    return StateManager(
        initial_state=persona.initial_state,
        traits=persona.psychology.big_five,
        determinism=determinism
    )
