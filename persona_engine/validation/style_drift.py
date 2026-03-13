"""
Style Drift Detection — Track behavioral consistency over multiple turns.

Detects when a persona's behavioral outputs drift without contextual
justification (e.g., a formal persona becoming casual without stress/engagement
changes represents a coherence failure).

Uses a sliding window of recent turns to compute drift scores.
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnMetrics:
    """Behavioral metrics from a single turn."""
    turn_number: int
    formality: float
    directness: float
    disclosure_level: float
    confidence: float
    # Optional state context (for justified drift detection)
    stress: float = 0.0
    engagement: float = 0.5
    mood_valence: float = 0.0


@dataclass
class DriftReport:
    """Report of style drift analysis."""
    drift_scores: dict[str, float]  # field -> standard deviation
    overall_drift: float  # mean of all drift scores
    flagged_fields: list[str]  # fields exceeding threshold
    justified: bool  # whether drift is explained by state changes
    details: list[str] = field(default_factory=list)


class StyleDriftDetector:
    """
    Track and detect unjustified behavioral drift over a sliding window.

    Usage:
        detector = StyleDriftDetector(window_size=10, drift_threshold=0.15)
        detector.record_turn(turn_metrics)
        report = detector.analyze()
    """

    def __init__(
        self,
        window_size: int = 10,
        drift_threshold: float = 0.15,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._history: list[TurnMetrics] = []

    def record_turn(self, metrics: TurnMetrics) -> None:
        """Record metrics from a turn. Only keeps the last window_size turns."""
        self._history.append(metrics)
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

    @property
    def turn_count(self) -> int:
        return len(self._history)

    def analyze(self) -> DriftReport:
        """
        Analyze behavioral drift over the recorded window.

        Returns:
            DriftReport with per-field drift scores and flags.
        """
        if len(self._history) < 2:
            return DriftReport(
                drift_scores={},
                overall_drift=0.0,
                flagged_fields=[],
                justified=True,
                details=["Insufficient turns for drift analysis"],
            )

        # Compute per-field standard deviation
        fields = ["formality", "directness", "disclosure_level", "confidence"]
        drift_scores: dict[str, float] = {}

        for f in fields:
            values = [getattr(m, f) for m in self._history]
            drift_scores[f] = _stddev(values)

        # Check state context for justified drift
        state_drift = self._compute_state_drift()
        flagged = [f for f, score in drift_scores.items() if score > self.drift_threshold]
        justified = state_drift > self.drift_threshold if flagged else True

        details = []
        for f in flagged:
            if justified:
                details.append(
                    f"Field '{f}' drifted (σ={drift_scores[f]:.3f}) but state changes "
                    f"(σ={state_drift:.3f}) may justify it"
                )
            else:
                details.append(
                    f"Field '{f}' drifted (σ={drift_scores[f]:.3f}) without "
                    f"proportionate state changes (σ={state_drift:.3f})"
                )

        overall = sum(drift_scores.values()) / len(drift_scores) if drift_scores else 0.0

        return DriftReport(
            drift_scores=drift_scores,
            overall_drift=overall,
            flagged_fields=flagged,
            justified=justified,
            details=details,
        )

    def _compute_state_drift(self) -> float:
        """Compute how much dynamic state changed over the window."""
        if len(self._history) < 2:
            return 0.0

        state_fields = ["stress", "engagement", "mood_valence"]
        state_stddevs = []
        for f in state_fields:
            values = [getattr(m, f) for m in self._history]
            state_stddevs.append(_stddev(values))

        return sum(state_stddevs) / len(state_stddevs) if state_stddevs else 0.0


def _stddev(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
