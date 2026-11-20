"""Tests for the Trait Marker Scorer."""

from persona_engine.validation.trait_scorer import TraitMarkerScorer


class TestTraitMarkerScorer:
    """Verify trait marker scoring works correctly."""

    def setup_method(self):
        self.scorer = TraitMarkerScorer()

    def test_high_openness_text(self):
        """Text with creative/abstract language scores high on openness."""
        text = (
            "What if we explored this from a completely different perspective? "
            "I find it fascinating how novel ideas emerge when we think abstractly. "
            "Imagine the possibilities of an unconventional approach."
        )
        result = self.scorer.score(text, {"openness": 0.9})
        score = result.per_trait["openness"]
        assert score.high_marker_count > score.low_marker_count
        assert score.marker_ratio > 0.5

    def test_low_openness_text(self):
        """Text with practical/conventional language scores low on openness."""
        text = (
            "The practical and proven approach is the most straightforward path. "
            "We should use established, conventional methods that are familiar "
            "and have a standard track record."
        )
        result = self.scorer.score(text, {"openness": 0.2})
        score = result.per_trait["openness"]
        assert score.low_marker_count > score.high_marker_count
        assert score.marker_ratio < 0.5

    def test_high_neuroticism_text(self):
        """Worried/uncertain text matches high neuroticism."""
        text = (
            "I'm worried about this and I'm not sure it will work. "
            "The risk is concerning and it could be overwhelming. "
            "I'm a bit anxious about what might go wrong."
        )
        result = self.scorer.score(text, {"neuroticism": 0.9})
        score = result.per_trait["neuroticism"]
        assert score.high_marker_count > 0
        assert score.coherence > 0.5

    def test_low_neuroticism_text(self):
        """Confident/calm text matches low neuroticism."""
        text = (
            "I'm confident this will work out fine. No problem at all, "
            "it's straightforward and manageable. I feel comfortable "
            "and secure with this approach."
        )
        result = self.scorer.score(text, {"neuroticism": 0.1})
        score = result.per_trait["neuroticism"]
        assert score.low_marker_count > score.high_marker_count

    def test_high_extraversion_text(self):
        """Enthusiastic/social language scores high on extraversion."""
        text = (
            "I love this! It's absolutely amazing and I'm so excited! "
            "We should definitely share this with everyone on the team."
        )
        result = self.scorer.score(text, {"extraversion": 0.9})
        score = result.per_trait["extraversion"]
        assert score.high_marker_count > 0

    def test_coherence_high_when_markers_match_expectation(self):
        """High coherence when text markers align with expected trait level."""
        # High openness text + high openness expectation → high coherence
        text = "Let's explore fascinating new perspectives and imagine creative alternatives."
        result = self.scorer.score(text, {"openness": 0.9})
        assert result.per_trait["openness"].coherence > 0.5

    def test_coherence_low_when_markers_mismatch(self):
        """Low coherence when text markers contradict expected trait level."""
        # Low openness text + high openness expectation → lower coherence
        text = "The practical and proven conventional approach is straightforward and standard."
        result = self.scorer.score(text, {"openness": 0.9})
        openness_score = result.per_trait["openness"]
        # Marker ratio will be low (mostly low-openness markers), but expected is 0.9
        assert openness_score.marker_ratio < 0.5  # mostly low markers
        # Coherence should be low because ratio doesn't match expectation
        assert openness_score.coherence < 0.7

    def test_no_markers_gives_neutral(self):
        """Text with no trait markers gives neutral 0.5 ratio."""
        text = "The weather is nice today."
        result = self.scorer.score(text, {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        })
        # With neutral expectations and neutral ratio, coherence should be high
        assert result.overall_coherence >= 0.9

    def test_overall_coherence_is_average(self):
        """Overall coherence is the mean of per-trait coherences."""
        text = "I think this is interesting."
        big5 = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
        result = self.scorer.score(text, big5)
        expected_avg = sum(s.coherence for s in result.per_trait.values()) / 5
        assert abs(result.overall_coherence - expected_avg) < 0.001

    def test_summary_output(self):
        """Summary string includes all traits."""
        text = "I'm excited about this fascinating idea!"
        result = self.scorer.score(text, {"openness": 0.8, "extraversion": 0.7})
        summary = result.summary()
        assert "Overall coherence" in summary
        assert "openness" in summary
