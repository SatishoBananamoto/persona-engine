"""
Tests for ValuesInterpreter (Schwartz Values)

Covers:
- Value priority retrieval and dict construction
- Top-N value selection (sorted by weight descending)
- Conflict detection with deduplication and custom thresholds
- Conflict resolution with context biases (work, personal, social, general)
- Stub stance influence scoring
- Rationale influence filtering and descriptions
- Citation inclusion boundary (0.65)
- Validation marker structure
- Factory function create_values_interpreter
"""

import pytest

from persona_engine.schema.persona_schema import SchwartzValues
from persona_engine.behavioral.values_interpreter import (
    ValuesInterpreter,
    VALUE_CONFLICTS,
    create_values_interpreter,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_values(**overrides: float) -> SchwartzValues:
    """Helper to build SchwartzValues with sensible defaults and overrides."""
    defaults = {
        "self_direction": 0.5,
        "stimulation": 0.5,
        "hedonism": 0.5,
        "achievement": 0.5,
        "power": 0.5,
        "security": 0.5,
        "conformity": 0.5,
        "tradition": 0.5,
        "benevolence": 0.5,
        "universalism": 0.5,
    }
    defaults.update(overrides)
    return SchwartzValues(**defaults)


@pytest.fixture
def uniform_values() -> SchwartzValues:
    """All values at 0.5 -- no conflicts at default threshold."""
    return _make_values()


@pytest.fixture
def polarized_values() -> SchwartzValues:
    """High self_direction and high conformity -- classic Schwartz conflict."""
    return _make_values(self_direction=0.9, conformity=0.8, tradition=0.3)


@pytest.fixture
def diverse_values() -> SchwartzValues:
    """Varied weights for sorting / filtering tests."""
    return _make_values(
        self_direction=0.9,
        stimulation=0.2,
        hedonism=0.7,
        achievement=0.85,
        power=0.3,
        security=0.6,
        conformity=0.4,
        tradition=0.1,
        benevolence=0.75,
        universalism=0.65,
    )


# ============================================================================
# _to_dict / __init__
# ============================================================================


class TestToDictAndInit:
    def test_value_dict_has_all_ten_keys(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        assert set(vi._value_dict.keys()) == {
            "self_direction", "stimulation", "hedonism", "achievement",
            "power", "security", "conformity", "tradition",
            "benevolence", "universalism",
        }

    def test_value_dict_reflects_input(self):
        sv = _make_values(self_direction=0.1, power=0.99)
        vi = ValuesInterpreter(sv)
        assert vi._value_dict["self_direction"] == pytest.approx(0.1)
        assert vi._value_dict["power"] == pytest.approx(0.99)


# ============================================================================
# get_value_priorities
# ============================================================================


class TestGetValuePriorities:
    def test_returns_correct_dict(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        priorities = vi.get_value_priorities()
        assert priorities["self_direction"] == pytest.approx(0.9)
        assert priorities["stimulation"] == pytest.approx(0.2)
        assert priorities["achievement"] == pytest.approx(0.85)
        assert len(priorities) == 10

    def test_returns_copy_not_reference(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        p1 = vi.get_value_priorities()
        p1["self_direction"] = 999.0
        p2 = vi.get_value_priorities()
        assert p2["self_direction"] == pytest.approx(0.5)


# ============================================================================
# get_top_values
# ============================================================================


class TestGetTopValues:
    def test_default_n_is_3(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        top = vi.get_top_values()
        assert len(top) == 3

    def test_sorted_descending(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        top = vi.get_top_values(n=5)
        weights = [w for _, w in top]
        assert weights == sorted(weights, reverse=True)

    def test_top_1_is_highest(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        top = vi.get_top_values(n=1)
        assert top[0] == ("self_direction", pytest.approx(0.9))

    def test_top_values_correct_order(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        top = vi.get_top_values(n=4)
        names = [n for n, _ in top]
        # 0.9, 0.85, 0.75, 0.7
        assert names == ["self_direction", "achievement", "benevolence", "hedonism"]

    def test_n_larger_than_total(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        top = vi.get_top_values(n=20)
        assert len(top) == 10

    def test_n_zero_returns_empty(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        top = vi.get_top_values(n=0)
        assert top == []


# ============================================================================
# detect_value_conflicts
# ============================================================================


class TestDetectValueConflicts:
    def test_no_conflicts_when_all_below_threshold(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        conflicts = vi.detect_value_conflicts(threshold=0.6)
        assert conflicts == []

    def test_single_conflict_detected(self, polarized_values: SchwartzValues):
        vi = ValuesInterpreter(polarized_values)
        conflicts = vi.detect_value_conflicts()
        # self_direction(0.9) conflicts with conformity(0.8) -- both above 0.6
        conflict_pairs = [(c["value_1"], c["value_2"]) for c in conflicts]
        # Should find at least this one pair (order may be either direction)
        found = any(
            set(p) == {"self_direction", "conformity"} for p in conflict_pairs
        )
        assert found, f"Expected self_direction-conformity conflict; got {conflict_pairs}"

    def test_deduplication(self, polarized_values: SchwartzValues):
        """A-B and B-A should appear only once."""
        vi = ValuesInterpreter(polarized_values)
        conflicts = vi.detect_value_conflicts()
        pairs = [
            tuple(sorted([c["value_1"], c["value_2"]])) for c in conflicts
        ]
        assert len(pairs) == len(set(pairs)), "Duplicate conflict pairs detected"

    def test_sorted_by_tension_descending(self):
        """When multiple conflicts exist, they are sorted by tension desc."""
        sv = _make_values(
            self_direction=0.9,
            conformity=0.7,
            stimulation=0.8,
            security=0.75,
        )
        vi = ValuesInterpreter(sv)
        conflicts = vi.detect_value_conflicts()
        tensions = [c["tension"] for c in conflicts]
        assert tensions == sorted(tensions, reverse=True)

    def test_custom_threshold_lower(self, uniform_values: SchwartzValues):
        """Lowering threshold to 0.5 should surface conflicts at default weights."""
        vi = ValuesInterpreter(uniform_values)
        conflicts = vi.detect_value_conflicts(threshold=0.5)
        # All values are 0.5 so any opposing pair qualifies
        assert len(conflicts) > 0

    def test_custom_threshold_higher_filters_more(self, polarized_values: SchwartzValues):
        """Higher threshold filters out more conflicts."""
        vi = ValuesInterpreter(polarized_values)
        # self_direction=0.9, conformity=0.8
        conflicts_low = vi.detect_value_conflicts(threshold=0.6)
        conflicts_high = vi.detect_value_conflicts(threshold=0.85)
        assert len(conflicts_high) <= len(conflicts_low)

    def test_tension_equals_min_of_weights(self):
        sv = _make_values(self_direction=0.9, conformity=0.7)
        vi = ValuesInterpreter(sv)
        conflicts = vi.detect_value_conflicts()
        for c in conflicts:
            if set([c["value_1"], c["value_2"]]) == {"self_direction", "conformity"}:
                assert c["tension"] == pytest.approx(0.7)  # min(0.9, 0.7)

    def test_conflict_dict_keys(self, polarized_values: SchwartzValues):
        vi = ValuesInterpreter(polarized_values)
        conflicts = vi.detect_value_conflicts()
        assert len(conflicts) > 0
        c = conflicts[0]
        expected_keys = {"value_1", "value_2", "weight_1", "weight_2", "tension", "description"}
        assert set(c.keys()) == expected_keys

    def test_multiple_conflicts(self):
        """Multiple distinct conflict pairs when many opposing values are high."""
        sv = _make_values(
            self_direction=0.8,
            conformity=0.8,
            tradition=0.8,
            hedonism=0.8,
            stimulation=0.8,
            security=0.8,
        )
        vi = ValuesInterpreter(sv)
        conflicts = vi.detect_value_conflicts()
        # Many Schwartz-opposing pairs should emerge
        assert len(conflicts) >= 3


# ============================================================================
# resolve_conflict
# ============================================================================


class TestResolveConflict:
    def test_general_context_picks_higher_weight(self):
        sv = _make_values(self_direction=0.9, conformity=0.4)
        vi = ValuesInterpreter(sv)
        winner = vi.resolve_conflict("self_direction", "conformity", context="general")
        assert winner == "self_direction"

    def test_general_context_picks_higher_weight_reversed(self):
        sv = _make_values(self_direction=0.4, conformity=0.9)
        vi = ValuesInterpreter(sv)
        winner = vi.resolve_conflict("self_direction", "conformity", context="general")
        assert winner == "conformity"

    def test_tie_goes_to_value_1(self):
        """When weights and biases are equal, value_1 wins (>= comparison)."""
        sv = _make_values(self_direction=0.7, conformity=0.7)
        vi = ValuesInterpreter(sv)
        winner = vi.resolve_conflict("self_direction", "conformity", context="general")
        assert winner == "self_direction"

    def test_work_context_boosts_achievement(self):
        """Work context gives achievement +0.1, which can flip the outcome."""
        sv = _make_values(achievement=0.55, benevolence=0.6)
        vi = ValuesInterpreter(sv)
        # Without bias: achievement 0.55 < benevolence 0.6 -> benevolence wins
        assert vi.resolve_conflict("achievement", "benevolence", context="general") == "benevolence"
        # With work bias: achievement 0.55+0.1=0.65 > benevolence 0.6 -> achievement wins
        assert vi.resolve_conflict("achievement", "benevolence", context="work") == "achievement"

    def test_work_context_boosts_power(self):
        sv = _make_values(power=0.55, universalism=0.58)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("power", "universalism", context="general") == "universalism"
        # power 0.55+0.05=0.60 > universalism 0.58
        assert vi.resolve_conflict("power", "universalism", context="work") == "power"

    def test_work_context_boosts_conformity(self):
        sv = _make_values(conformity=0.55, self_direction=0.58)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("conformity", "self_direction", context="general") == "self_direction"
        # conformity 0.55+0.05=0.60 > self_direction 0.58
        assert vi.resolve_conflict("conformity", "self_direction", context="work") == "conformity"

    def test_personal_context_boosts_self_direction(self):
        sv = _make_values(self_direction=0.55, conformity=0.6)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("self_direction", "conformity", context="general") == "conformity"
        # self_direction 0.55+0.1=0.65 > conformity 0.6
        assert vi.resolve_conflict("self_direction", "conformity", context="personal") == "self_direction"

    def test_personal_context_boosts_hedonism(self):
        sv = _make_values(hedonism=0.55, conformity=0.6)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("hedonism", "conformity", context="general") == "conformity"
        # hedonism 0.55+0.1=0.65 > conformity 0.6
        assert vi.resolve_conflict("hedonism", "conformity", context="personal") == "hedonism"

    def test_personal_context_boosts_security(self):
        sv = _make_values(security=0.56, stimulation=0.58)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("security", "stimulation", context="general") == "stimulation"
        # security 0.56+0.05=0.61 > stimulation 0.58
        assert vi.resolve_conflict("security", "stimulation", context="personal") == "security"

    def test_social_context_boosts_benevolence(self):
        sv = _make_values(benevolence=0.55, power=0.6)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("benevolence", "power", context="general") == "power"
        # benevolence 0.55+0.1=0.65 > power 0.6
        assert vi.resolve_conflict("benevolence", "power", context="social") == "benevolence"

    def test_social_context_boosts_universalism(self):
        sv = _make_values(universalism=0.55, achievement=0.6)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("universalism", "achievement", context="general") == "achievement"
        # universalism 0.55+0.08=0.63 > achievement 0.6
        assert vi.resolve_conflict("universalism", "achievement", context="social") == "universalism"

    def test_social_context_boosts_tradition(self):
        sv = _make_values(tradition=0.56, self_direction=0.58)
        vi = ValuesInterpreter(sv)
        assert vi.resolve_conflict("tradition", "self_direction", context="general") == "self_direction"
        # tradition 0.56+0.05=0.61 > self_direction 0.58
        assert vi.resolve_conflict("tradition", "self_direction", context="social") == "tradition"

    def test_unknown_context_uses_no_bias(self):
        sv = _make_values(achievement=0.6, benevolence=0.55)
        vi = ValuesInterpreter(sv)
        winner = vi.resolve_conflict("achievement", "benevolence", context="random_ctx")
        assert winner == "achievement"


# ============================================================================
# get_value_influence_on_stance
# ============================================================================


class TestGetValueInfluenceOnStance:
    def test_returns_even_scores(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        result = vi.get_value_influence_on_stance("technology", ["option_a", "option_b"])
        assert result == {"option_a": 0.5, "option_b": 0.5}

    def test_single_option(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        result = vi.get_value_influence_on_stance("politics", ["agree"])
        assert result == {"agree": 0.5}

    def test_empty_options(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        result = vi.get_value_influence_on_stance("anything", [])
        assert result == {}

    def test_many_options_all_half(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        options = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree"]
        result = vi.get_value_influence_on_stance("ethics", options)
        assert len(result) == 5
        for opt in options:
            assert result[opt] == pytest.approx(0.5)


# ============================================================================
# get_rationale_influences
# ============================================================================


class TestGetRationaleInfluences:
    def test_filters_below_0_6(self):
        sv = _make_values(self_direction=0.59, achievement=0.61)
        vi = ValuesInterpreter(sv)
        influences = vi.get_rationale_influences()
        names = [name for name, _, _ in influences]
        assert "self_direction" not in names
        assert "achievement" in names

    def test_exact_0_6_included(self):
        sv = _make_values(security=0.6)
        vi = ValuesInterpreter(sv)
        influences = vi.get_rationale_influences()
        names = [name for name, _, _ in influences]
        assert "security" in names

    def test_sorted_by_weight_descending(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        influences = vi.get_rationale_influences()
        weights = [w for _, w, _ in influences]
        assert weights == sorted(weights, reverse=True)

    def test_contains_descriptions(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        influences = vi.get_rationale_influences()
        for name, weight, desc in influences:
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_all_below_threshold_returns_empty(self):
        sv = _make_values(
            self_direction=0.1, stimulation=0.2, hedonism=0.3,
            achievement=0.4, power=0.1, security=0.2,
            conformity=0.3, tradition=0.1, benevolence=0.2,
            universalism=0.1,
        )
        vi = ValuesInterpreter(sv)
        assert vi.get_rationale_influences() == []

    def test_tuple_structure(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        influences = vi.get_rationale_influences()
        assert len(influences) > 0
        for item in influences:
            assert len(item) == 3
            name, weight, desc = item
            assert isinstance(name, str)
            assert isinstance(weight, float)
            assert isinstance(desc, str)


# ============================================================================
# _get_value_influence_description
# ============================================================================


class TestGetValueInfluenceDescription:
    def test_known_values_have_descriptions(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        all_values = [
            "self_direction", "stimulation", "hedonism", "achievement",
            "power", "security", "conformity", "tradition",
            "benevolence", "universalism",
        ]
        for value in all_values:
            desc = vi._get_value_influence_description(value)
            assert isinstance(desc, str)
            assert len(desc) > 10  # meaningful description

    def test_unknown_value_returns_default(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        desc = vi._get_value_influence_description("nonexistent_value")
        assert desc == "Influences decision-making"

    def test_specific_descriptions(self, uniform_values: SchwartzValues):
        vi = ValuesInterpreter(uniform_values)
        assert "autonomy" in vi._get_value_influence_description("self_direction")
        assert "novelty" in vi._get_value_influence_description("stimulation")
        assert "enjoyment" in vi._get_value_influence_description("hedonism") or \
               "pleasure" in vi._get_value_influence_description("hedonism").lower()
        assert "competence" in vi._get_value_influence_description("achievement") or \
               "success" in vi._get_value_influence_description("achievement")
        assert "safety" in vi._get_value_influence_description("security") or \
               "stability" in vi._get_value_influence_description("security")
        assert "rules" in vi._get_value_influence_description("conformity") or \
               "social" in vi._get_value_influence_description("conformity")
        assert "customs" in vi._get_value_influence_description("tradition") or \
               "established" in vi._get_value_influence_description("tradition")
        assert "caring" in vi._get_value_influence_description("benevolence") or \
               "welfare" in vi._get_value_influence_description("benevolence")
        assert "justice" in vi._get_value_influence_description("universalism") or \
               "equality" in vi._get_value_influence_description("universalism")


# ============================================================================
# should_include_in_citation
# ============================================================================


class TestShouldIncludeInCitation:
    def test_above_threshold_returns_true(self):
        sv = _make_values(self_direction=0.8)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("self_direction") is True

    def test_below_threshold_returns_false(self):
        sv = _make_values(self_direction=0.64)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("self_direction") is False

    def test_exactly_at_threshold_returns_true(self):
        sv = _make_values(self_direction=0.65)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("self_direction") is True

    def test_just_below_threshold_returns_false(self):
        sv = _make_values(achievement=0.649)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("achievement") is False

    def test_zero_returns_false(self):
        sv = _make_values(power=0.0)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("power") is False

    def test_one_returns_true(self):
        sv = _make_values(benevolence=1.0)
        vi = ValuesInterpreter(sv)
        assert vi.should_include_in_citation("benevolence") is True


# ============================================================================
# get_value_markers_for_validation
# ============================================================================


class TestGetValueMarkersForValidation:
    def test_structure_keys(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        assert set(markers.keys()) == {
            "top_values", "active_conflicts", "citation_worthy", "dominant_value"
        }

    def test_top_values_format(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        top_values = markers["top_values"]
        assert len(top_values) == 3
        for entry in top_values:
            assert "name" in entry
            assert "weight" in entry

    def test_top_values_match_get_top_values(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        top_direct = vi.get_top_values(n=3)
        for entry, (name, weight) in zip(markers["top_values"], top_direct):
            assert entry["name"] == name
            assert entry["weight"] == pytest.approx(weight)

    def test_dominant_value_is_top_value(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        assert markers["dominant_value"] == "self_direction"

    def test_citation_worthy_values(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        citation_worthy = markers["citation_worthy"]
        # diverse_values: self_direction=0.9, hedonism=0.7, achievement=0.85,
        #   benevolence=0.75, universalism=0.65 are all >= 0.65
        for val in citation_worthy:
            assert vi._value_dict[val] >= 0.65
        # Verify expected values are present
        assert "self_direction" in citation_worthy
        assert "achievement" in citation_worthy
        assert "benevolence" in citation_worthy
        assert "hedonism" in citation_worthy
        assert "universalism" in citation_worthy
        # Values below 0.65 should not be present
        assert "stimulation" not in citation_worthy  # 0.2
        assert "power" not in citation_worthy  # 0.3
        assert "tradition" not in citation_worthy  # 0.1
        assert "conformity" not in citation_worthy  # 0.4

    def test_active_conflicts_is_list(self, diverse_values: SchwartzValues):
        vi = ValuesInterpreter(diverse_values)
        markers = vi.get_value_markers_for_validation()
        assert isinstance(markers["active_conflicts"], list)

    def test_with_no_conflicts(self):
        sv = _make_values(
            self_direction=0.9, stimulation=0.1, hedonism=0.1,
            achievement=0.1, power=0.1, security=0.1,
            conformity=0.1, tradition=0.1, benevolence=0.1,
            universalism=0.1,
        )
        vi = ValuesInterpreter(sv)
        markers = vi.get_value_markers_for_validation()
        assert markers["active_conflicts"] == []
        assert markers["dominant_value"] == "self_direction"
        assert markers["citation_worthy"] == ["self_direction"]


# ============================================================================
# VALUE_CONFLICTS constant
# ============================================================================


class TestValueConflicts:
    def test_all_ten_values_present(self):
        expected_values = {
            "self_direction", "stimulation", "hedonism", "achievement",
            "power", "security", "conformity", "tradition",
            "benevolence", "universalism",
        }
        assert set(VALUE_CONFLICTS.keys()) == expected_values

    def test_conflicts_are_lists_of_strings(self):
        for value, opponents in VALUE_CONFLICTS.items():
            assert isinstance(opponents, list), f"{value} conflicts not a list"
            for opp in opponents:
                assert isinstance(opp, str), f"{value} opponent {opp} not a string"

    def test_opponents_are_valid_value_names(self):
        valid_names = set(VALUE_CONFLICTS.keys())
        for value, opponents in VALUE_CONFLICTS.items():
            for opp in opponents:
                assert opp in valid_names, f"{opp} (opponent of {value}) is not a valid value name"


# ============================================================================
# create_values_interpreter factory
# ============================================================================


class TestCreateValuesInterpreter:
    def _make_minimal_persona(self, values: SchwartzValues):
        """Build a minimal Persona object with given SchwartzValues."""
        from persona_engine.schema.persona_schema import (
            Persona,
            Identity,
            PersonalityProfile,
            BigFiveTraits,
            CognitiveStyle,
            CommunicationPreferences,
            SocialRole,
            UncertaintyPolicy,
            ClaimPolicy,
            PersonaInvariants,
            DisclosurePolicy,
            DynamicState,
        )
        return Persona(
            persona_id="test-persona",
            version="1.0",
            label="Test Persona",
            identity=Identity(
                age=30, location="London, UK",
                education="BSc", occupation="Engineer",
                background="Test background",
            ),
            psychology=PersonalityProfile(
                big_five=BigFiveTraits(
                    openness=0.5, conscientiousness=0.5,
                    extraversion=0.5, agreeableness=0.5, neuroticism=0.5,
                ),
                values=values,
                cognitive_style=CognitiveStyle(
                    analytical_intuitive=0.5, systematic_heuristic=0.5,
                    risk_tolerance=0.5, need_for_closure=0.5,
                    cognitive_complexity=0.5,
                ),
                communication=CommunicationPreferences(
                    verbosity=0.5, formality=0.5,
                    directness=0.5, emotional_expressiveness=0.5,
                ),
            ),
            social_roles={"default": SocialRole(
                formality=0.5, directness=0.5, emotional_expressiveness=0.5,
            )},
            uncertainty=UncertaintyPolicy(
                admission_threshold=0.5, hedging_frequency=0.5,
                clarification_tendency=0.5, knowledge_boundary_strictness=0.5,
            ),
            claim_policy=ClaimPolicy(),
            invariants=PersonaInvariants(identity_facts=["Test fact"]),
            time_scarcity=0.5,
            privacy_sensitivity=0.5,
            disclosure_policy=DisclosurePolicy(
                base_openness=0.5, factors={"topic_sensitivity": -0.3},
            ),
            initial_state=DynamicState(
                mood_valence=0.0, mood_arousal=0.5,
                fatigue=0.2, stress=0.2, engagement=0.7,
            ),
        )

    def test_returns_values_interpreter(self, diverse_values: SchwartzValues):
        persona = self._make_minimal_persona(diverse_values)
        vi = create_values_interpreter(persona)
        assert isinstance(vi, ValuesInterpreter)

    def test_uses_persona_values(self, diverse_values: SchwartzValues):
        persona = self._make_minimal_persona(diverse_values)
        vi = create_values_interpreter(persona)
        assert vi._value_dict["self_direction"] == pytest.approx(0.9)
        assert vi._value_dict["stimulation"] == pytest.approx(0.2)

    def test_interpreter_functional(self, diverse_values: SchwartzValues):
        persona = self._make_minimal_persona(diverse_values)
        vi = create_values_interpreter(persona)
        top = vi.get_top_values(n=1)
        assert top[0][0] == "self_direction"


# ============================================================================
# Edge cases and integration
# ============================================================================


class TestEdgeCases:
    def test_all_values_zero(self):
        sv = _make_values(
            self_direction=0.0, stimulation=0.0, hedonism=0.0,
            achievement=0.0, power=0.0, security=0.0,
            conformity=0.0, tradition=0.0, benevolence=0.0,
            universalism=0.0,
        )
        vi = ValuesInterpreter(sv)
        assert vi.get_top_values(n=3) is not None
        assert len(vi.get_top_values(n=3)) == 3
        assert vi.detect_value_conflicts() == []
        assert vi.get_rationale_influences() == []

    def test_all_values_one(self):
        sv = _make_values(
            self_direction=1.0, stimulation=1.0, hedonism=1.0,
            achievement=1.0, power=1.0, security=1.0,
            conformity=1.0, tradition=1.0, benevolence=1.0,
            universalism=1.0,
        )
        vi = ValuesInterpreter(sv)
        top = vi.get_top_values(n=3)
        assert all(w == pytest.approx(1.0) for _, w in top)
        conflicts = vi.detect_value_conflicts()
        assert len(conflicts) > 0  # Many conflicts with everything at 1.0
        influences = vi.get_rationale_influences()
        assert len(influences) == 10  # All above 0.6

    def test_conflict_description_format(self):
        sv = _make_values(self_direction=0.85, conformity=0.72)
        vi = ValuesInterpreter(sv)
        conflicts = vi.detect_value_conflicts()
        assert len(conflicts) > 0
        desc = conflicts[0]["description"]
        # Should contain both value names and weights formatted to 2 decimals
        assert "0.85" in desc or "0.72" in desc
