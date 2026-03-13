"""
Tests for PersonaBuilder — persona creation without hand-crafted YAML.

Covers:
- Minimal build (name + occupation only)
- Trait adjective application
- Archetype profiles
- from_description() natural language parsing
- Domain inference from occupation
- Education inference
- Culture inference
- All derived fields (uncertainty, claim_policy, disclosure, etc.)
- PersonaEngine.from_description() integration
"""

import pytest

from persona_engine.persona_builder import (
    ARCHETYPES,
    OCCUPATION_DOMAINS,
    OCCUPATION_SUBDOMAINS,
    TRAIT_MODIFIERS,
    PersonaBuilder,
    _extract_adjectives,
    _extract_age,
    _extract_archetype,
    _extract_location,
    _extract_name,
    _extract_occupation,
    _infer_culture,
    _infer_domains,
    _infer_education,
)
from persona_engine.schema.persona_schema import Persona


# ============================================================================
# Minimal Build
# ============================================================================


class TestMinimalBuild:
    """Verify that just name + occupation produces a valid Persona."""

    def test_minimal_build_produces_valid_persona(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        assert isinstance(persona, Persona)
        assert persona.persona_id.startswith("P_GEN_")
        assert persona.identity.occupation == "Chef"

    def test_minimal_build_has_default_age(self):
        persona = PersonaBuilder("Alex", "Engineer").build()
        assert persona.identity.age == 30

    def test_minimal_build_has_default_location(self):
        persona = PersonaBuilder("Alex", "Engineer").build()
        assert persona.identity.location == "United States"

    def test_minimal_build_has_neutral_big_five(self):
        persona = PersonaBuilder("Alex", "Professional").build()
        bf = persona.psychology.big_five
        assert bf.openness == 0.5
        assert bf.conscientiousness == 0.5
        assert bf.extraversion == 0.5

    def test_minimal_build_has_default_social_role(self):
        persona = PersonaBuilder("Alex", "Professional").build()
        assert "default" in persona.social_roles

    def test_minimal_build_has_knowledge_domains(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        domain_names = [d.domain for d in persona.knowledge_domains]
        assert "Food" in domain_names

    def test_minimal_build_has_invariants(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        assert len(persona.invariants.identity_facts) > 0

    def test_minimal_build_has_disclosure_policy(self):
        persona = PersonaBuilder("Alex", "Engineer").build()
        assert 0.0 < persona.disclosure_policy.base_openness < 1.0
        assert "trust_level" in persona.disclosure_policy.factors

    def test_minimal_build_has_initial_state(self):
        persona = PersonaBuilder("Alex", "Engineer").build()
        assert persona.initial_state.engagement == 0.6

    def test_label_includes_name_and_occupation(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        assert "Marcus" in persona.label
        assert "Chef" in persona.label


# ============================================================================
# Fluent Setters
# ============================================================================


class TestFluentSetters:
    def test_age_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").age(45).build()
        assert persona.identity.age == 45

    def test_age_clamped(self):
        persona = PersonaBuilder("Alex", "Engineer").age(150).build()
        assert persona.identity.age == 100

    def test_location_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").location("Berlin, Germany").build()
        assert persona.identity.location == "Berlin, Germany"

    def test_gender_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").gender("female").build()
        assert persona.identity.gender == "female"

    def test_education_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").education("MIT PhD").build()
        assert persona.identity.education == "MIT PhD"

    def test_background_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").background("A self-taught dev").build()
        assert "self-taught" in persona.identity.background

    def test_culture_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").culture("Japanese").build()
        assert persona.cultural_knowledge.primary_culture == "Japanese"

    def test_language_setter(self):
        persona = (
            PersonaBuilder("Alex", "Engineer")
            .language("English", 1.0, "American")
            .language("Spanish", 0.6)
            .build()
        )
        assert len(persona.languages) == 2
        assert persona.languages[0].language == "English"
        assert persona.languages[1].proficiency == 0.6

    def test_domain_setter(self):
        persona = (
            PersonaBuilder("Alex", "Professional")
            .domain("Finance", 0.9, ["Stocks", "Bonds"])
            .build()
        )
        assert persona.knowledge_domains[0].domain == "Finance"
        assert persona.knowledge_domains[0].proficiency == 0.9

    def test_goal_setter(self):
        persona = (
            PersonaBuilder("Alex", "Engineer")
            .goal("Build great software", 0.9)
            .goal("Learn new technologies", 0.7)
            .build()
        )
        assert len(persona.primary_goals) == 2

    def test_lookup_behavior_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").lookup_behavior("refuse").build()
        assert persona.claim_policy.lookup_behavior == "refuse"

    def test_lookup_behavior_invalid(self):
        with pytest.raises(ValueError, match="lookup_behavior"):
            PersonaBuilder("Alex", "Engineer").lookup_behavior("answer")

    def test_time_scarcity_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").time_scarcity(0.9).build()
        assert persona.time_scarcity == 0.9

    def test_privacy_sensitivity_setter(self):
        persona = PersonaBuilder("Alex", "Engineer").privacy_sensitivity(0.8).build()
        assert persona.privacy_sensitivity == 0.8

    def test_chaining(self):
        """All setters return self for chaining."""
        persona = (
            PersonaBuilder("Marcus", "Chef")
            .age(41)
            .gender("male")
            .location("Chicago, IL")
            .education("Culinary Arts")
            .background("20 years in kitchens")
            .culture("American")
            .language("English", 1.0)
            .domain("Food", 0.95)
            .goal("Create great food", 0.9)
            .trait("passionate", "direct")
            .time_scarcity(0.8)
            .privacy_sensitivity(0.4)
            .lookup_behavior("speculate")
            .build()
        )
        assert persona.identity.age == 41
        assert persona.identity.location == "Chicago, IL"


# ============================================================================
# Trait Application
# ============================================================================


class TestTraits:
    def test_passionate_increases_extraversion(self):
        baseline = PersonaBuilder("Alex", "Professional").build()
        modified = PersonaBuilder("Alex", "Professional").trait("passionate").build()
        assert modified.psychology.big_five.extraversion > baseline.psychology.big_five.extraversion

    def test_analytical_increases_analytical_intuitive(self):
        baseline = PersonaBuilder("Alex", "Professional").build()
        modified = PersonaBuilder("Alex", "Professional").trait("analytical").build()
        assert modified.psychology.cognitive_style.analytical_intuitive > baseline.psychology.cognitive_style.analytical_intuitive

    def test_direct_increases_directness(self):
        baseline = PersonaBuilder("Alex", "Professional").build()
        modified = PersonaBuilder("Alex", "Professional").trait("direct").build()
        assert modified.psychology.communication.directness > baseline.psychology.communication.directness

    def test_warm_increases_agreeableness(self):
        baseline = PersonaBuilder("Alex", "Professional").build()
        modified = PersonaBuilder("Alex", "Professional").trait("warm").build()
        assert modified.psychology.big_five.agreeableness > baseline.psychology.big_five.agreeableness

    def test_multiple_traits_stack(self):
        baseline = PersonaBuilder("Alex", "Professional").build()
        modified = PersonaBuilder("Alex", "Professional").trait("passionate", "adventurous").build()
        # Both boost openness
        assert modified.psychology.big_five.openness > baseline.psychology.big_five.openness

    def test_traits_alias_works(self):
        p1 = PersonaBuilder("A", "Pro").trait("calm").build()
        p2 = PersonaBuilder("A", "Pro").traits("calm").build()
        assert p1.psychology.big_five.neuroticism == p2.psychology.big_five.neuroticism

    def test_opposing_traits_partially_cancel(self):
        p = PersonaBuilder("Alex", "Professional").trait("cautious", "adventurous").build()
        # Both modify risk_tolerance in opposite directions
        # Should be close to baseline
        assert 0.3 < p.psychology.cognitive_style.risk_tolerance < 0.7

    def test_traits_clamped_to_bounds(self):
        # Apply many extraversion-boosting traits
        p = (
            PersonaBuilder("Alex", "Professional")
            .trait("passionate", "friendly", "expressive")
            .build()
        )
        assert p.psychology.big_five.extraversion <= 1.0

    def test_unknown_traits_silently_ignored(self):
        p = PersonaBuilder("Alex", "Professional").trait("nonexistent_trait").build()
        assert p.psychology.big_five.openness == 0.5  # unchanged

    def test_trait_modifiers_dict_completeness(self):
        """All trait modifiers reference valid sections."""
        valid_sections = {"big_five", "values", "cognitive_style", "communication"}
        for adj, deltas in TRAIT_MODIFIERS.items():
            for path in deltas:
                section = path.split(".")[0]
                assert section in valid_sections, f"Trait '{adj}' has invalid section '{section}'"


# ============================================================================
# Archetypes
# ============================================================================


class TestArchetypes:
    def test_expert_archetype_high_conscientiousness(self):
        p = PersonaBuilder("Alex", "Scientist").archetype("expert").build()
        assert p.psychology.big_five.conscientiousness == 0.75

    def test_coach_archetype_high_agreeableness(self):
        p = PersonaBuilder("Alex", "Trainer").archetype("coach").build()
        assert p.psychology.big_five.agreeableness == 0.80

    def test_creative_archetype_high_openness(self):
        p = PersonaBuilder("Alex", "Artist").archetype("creative").build()
        assert p.psychology.big_five.openness == 0.85

    def test_analyst_archetype_high_analytical(self):
        p = PersonaBuilder("Alex", "Researcher").archetype("analyst").build()
        assert p.psychology.cognitive_style.analytical_intuitive == 0.85

    def test_caregiver_archetype(self):
        p = PersonaBuilder("Alex", "Nurse").archetype("caregiver").build()
        assert p.psychology.big_five.agreeableness == 0.85

    def test_leader_archetype(self):
        p = PersonaBuilder("Alex", "CEO").archetype("leader").build()
        assert p.psychology.big_five.extraversion == 0.75

    def test_unknown_archetype_raises(self):
        with pytest.raises(ValueError, match="Unknown archetype"):
            PersonaBuilder("Alex", "Pro").archetype("wizard")

    def test_archetype_then_trait_stacks(self):
        p = PersonaBuilder("Alex", "Scientist").archetype("expert").trait("passionate").build()
        # expert sets conscientiousness=0.75, passionate boosts extraversion
        assert p.psychology.big_five.conscientiousness == 0.75
        assert p.psychology.big_five.extraversion > 0.5

    def test_archetype_persona_shortcut(self):
        p = PersonaBuilder.archetype_persona("expert", name="Dr. Lee", occupation="Physicist", age=55)
        assert isinstance(p, Persona)
        assert p.identity.occupation == "Physicist"
        assert p.identity.age == 55

    def test_all_archetypes_produce_valid_personas(self):
        for name in ARCHETYPES:
            p = PersonaBuilder("Test", "Professional").archetype(name).build()
            assert isinstance(p, Persona), f"Archetype '{name}' failed"


# ============================================================================
# from_description
# ============================================================================


class TestFromDescription:
    def test_basic_description(self):
        p = PersonaBuilder.from_description(
            "A 45-year-old chef named Marcus, passionate and direct"
        )
        assert isinstance(p, Persona)
        assert p.identity.age == 45
        assert "Marcus" in p.label

    def test_description_extracts_occupation(self):
        p = PersonaBuilder.from_description("Sarah is a UX researcher based in London")
        assert "researcher" in p.identity.occupation.lower() or "ux" in p.identity.occupation.lower()

    def test_description_extracts_location(self):
        p = PersonaBuilder.from_description(
            "A lawyer named Catherine based in New York"
        )
        assert "New York" in p.identity.location

    def test_description_extracts_traits(self):
        p = PersonaBuilder.from_description(
            "An analytical and cautious scientist named Dr. Lee"
        )
        assert p.psychology.cognitive_style.analytical_intuitive > 0.5
        assert p.psychology.cognitive_style.risk_tolerance < 0.5

    def test_description_with_archetype(self):
        p = PersonaBuilder.from_description(
            "An expert physicist named Dr. Priya, analytical and meticulous"
        )
        assert p.psychology.big_five.conscientiousness > 0.5

    def test_minimal_description(self):
        p = PersonaBuilder.from_description("A chef")
        assert isinstance(p, Persona)


# ============================================================================
# Name Extraction
# ============================================================================


class TestNameExtraction:
    def test_named_pattern(self):
        assert _extract_name("A chef named Marcus") == "Marcus"

    def test_called_pattern(self):
        assert _extract_name("A physicist called Priya") == "Priya"

    def test_is_a_pattern(self):
        assert _extract_name("Sarah is a UX researcher") == "Sarah"

    def test_comma_pattern(self):
        assert _extract_name("Marcus, a chef from Chicago") == "Marcus"

    def test_first_capital_fallback(self):
        assert _extract_name("Meet Marcus the chef") == "Marcus"

    def test_skips_common_starters(self):
        assert _extract_name("A passionate chef from Chicago") != "A"

    def test_no_name_gives_default(self):
        assert _extract_name("a small quiet person") == "Alex"


# ============================================================================
# Occupation Extraction
# ============================================================================


class TestOccupationExtraction:
    def test_known_occupation(self):
        result = _extract_occupation("A 45-year-old chef named Marcus")
        assert "chef" in result.lower()

    def test_multi_word_occupation(self):
        result = _extract_occupation("She is a fitness coach")
        assert "fitness coach" in result.lower()

    def test_ux_researcher(self):
        result = _extract_occupation("A UX researcher based in London")
        assert "researcher" in result.lower()

    def test_works_as_pattern(self):
        result = _extract_occupation("He works as a consultant for tech firms")
        assert "consultant" in result.lower()

    def test_default_when_unknown(self):
        result = _extract_occupation("A random person doing random things")
        assert result == "Professional"


# ============================================================================
# Age Extraction
# ============================================================================


class TestAgeExtraction:
    def test_standard_pattern(self):
        assert _extract_age("A 45-year-old chef") == 45

    def test_year_old_pattern(self):
        assert _extract_age("She is 34 years old") == 34

    def test_age_colon_pattern(self):
        assert _extract_age("Age: 28") == 28

    def test_no_age(self):
        assert _extract_age("A passionate chef") is None

    def test_invalid_age_too_young(self):
        assert _extract_age("A 5-year-old child") is None


# ============================================================================
# Location Extraction
# ============================================================================


class TestLocationExtraction:
    def test_based_in(self):
        assert _extract_location("A chef based in Chicago") == "Chicago"

    def test_from_location(self):
        result = _extract_location("A scientist from Berlin, Germany")
        assert "Berlin" in result

    def test_lives_in(self):
        result = _extract_location("She lives in London")
        assert "London" in result

    def test_no_location(self):
        assert _extract_location("A passionate chef") is None


# ============================================================================
# Adjective Extraction
# ============================================================================


class TestAdjectiveExtraction:
    def test_finds_known_adjectives(self):
        adjs = _extract_adjectives("A passionate and direct chef")
        assert "passionate" in adjs
        assert "direct" in adjs

    def test_hyphenated_adjectives(self):
        adjs = _extract_adjectives("A detail-oriented and open-minded researcher")
        assert "detail-oriented" in adjs
        assert "open-minded" in adjs

    def test_no_adjectives(self):
        adjs = _extract_adjectives("A chef from Chicago")
        assert len(adjs) == 0


# ============================================================================
# Domain Inference
# ============================================================================


class TestDomainInference:
    def test_chef_gets_food_domain(self):
        domains = _infer_domains("Chef")
        domain_names = [d.domain for d in domains]
        assert "Food" in domain_names

    def test_chef_food_proficiency_high(self):
        domains = _infer_domains("Chef")
        food = next(d for d in domains if d.domain == "Food")
        assert food.proficiency >= 0.85

    def test_lawyer_gets_law_domain(self):
        domains = _infer_domains("Lawyer")
        domain_names = [d.domain for d in domains]
        assert "Law" in domain_names

    def test_software_engineer_gets_tech(self):
        domains = _infer_domains("Software Engineer")
        domain_names = [d.domain for d in domains]
        assert "Technology" in domain_names

    def test_unknown_occupation_gets_general(self):
        domains = _infer_domains("Space Cowboy")
        assert domains[0].domain == "General"

    def test_multi_word_match(self):
        domains = _infer_domains("Fitness Coach")
        domain_names = [d.domain for d in domains]
        assert "Sports" in domain_names


# ============================================================================
# Education Inference
# ============================================================================


class TestEducationInference:
    def test_professor_gets_phd(self):
        assert "PhD" in _infer_education("Professor")

    def test_chef_gets_vocational(self):
        assert "Vocational" in _infer_education("Chef") or "trade" in _infer_education("Chef").lower()

    def test_doctor_gets_professional(self):
        assert "Professional" in _infer_education("Doctor")

    def test_therapist_gets_masters(self):
        assert "Master" in _infer_education("Therapist")

    def test_unknown_gets_bachelors(self):
        assert "Bachelor" in _infer_education("Consultant")


# ============================================================================
# Culture Inference
# ============================================================================


class TestCultureInference:
    def test_chicago_is_american(self):
        assert _infer_culture("Chicago, IL").lower() == "american"

    def test_london_is_british(self):
        assert _infer_culture("London, UK") == "British"

    def test_paris_is_french(self):
        assert _infer_culture("Paris, France") == "French"

    def test_tokyo_is_japanese(self):
        assert _infer_culture("Tokyo, Japan") == "Japanese"

    def test_unknown_is_international(self):
        assert _infer_culture("Atlantis") == "International"


# ============================================================================
# Derived Fields
# ============================================================================


class TestDerivedFields:
    def test_high_conscientiousness_higher_time_scarcity(self):
        neutral = PersonaBuilder("A", "Pro").build()
        meticulous = PersonaBuilder("A", "Pro").trait("meticulous", "organized").build()
        assert meticulous.time_scarcity > neutral.time_scarcity

    def test_high_neuroticism_higher_privacy(self):
        neutral = PersonaBuilder("A", "Pro").build()
        anxious = PersonaBuilder("A", "Pro").trait("anxious").build()
        assert anxious.privacy_sensitivity > neutral.privacy_sensitivity

    def test_agreeable_persona_acknowledges_disagreement(self):
        p = PersonaBuilder("A", "Pro").trait("warm", "empathetic").build()
        patterns = {rp.trigger: rp.response for rp in p.response_patterns}
        assert patterns["disagreement"] == "acknowledge_then_explain"

    def test_disagreeable_persona_pushes_back(self):
        p = PersonaBuilder("A", "Pro").trait("direct", "blunt", "stubborn").build()
        patterns = {rp.trigger: rp.response for rp in p.response_patterns}
        assert patterns["disagreement"] == "direct_pushback"

    def test_expert_domain_enables_domain_expert_claim(self):
        p = PersonaBuilder("A", "Chef").build()  # Chef gets Food at 0.90
        assert "domain_expert" in p.claim_policy.allowed_claim_types

    def test_no_expert_domain_no_domain_expert_claim(self):
        p = PersonaBuilder("A", "Professional").build()  # Gets General at 0.5
        assert "domain_expert" not in p.claim_policy.allowed_claim_types

    def test_high_analytical_gets_systematic_decision_approach(self):
        p = PersonaBuilder("A", "Pro").trait("analytical", "systematic").build()
        high_stakes = next(dp for dp in p.decision_policies if dp.condition == "high_stakes_decision")
        assert high_stakes.approach == "analytical_systematic"

    def test_cannot_claim_excludes_own_domain(self):
        p = PersonaBuilder("A", "Doctor").build()
        cannot = p.invariants.cannot_claim
        # Doctor has Health domain, so "medical doctor" should NOT be in cannot_claim
        assert "medical doctor" not in cannot

    def test_cannot_claim_includes_other_domains(self):
        p = PersonaBuilder("A", "Chef").build()
        cannot = p.invariants.cannot_claim
        # Chef doesn't have law domain
        assert "licensed attorney" in cannot

    def test_social_roles_derived_from_communication(self):
        p = PersonaBuilder("A", "Pro").trait("formal").build()
        # Default role should have formality matching communication
        assert p.social_roles["default"].formality == p.psychology.communication.formality
        # at_work should be even more formal
        assert p.social_roles["at_work"].formality > p.social_roles["default"].formality

    def test_topic_sensitivities_scale_with_privacy(self):
        low_priv = PersonaBuilder("A", "Pro").privacy_sensitivity(0.2).build()
        high_priv = PersonaBuilder("A", "Pro").privacy_sensitivity(0.8).build()
        # Personal finances should be more sensitive for high-privacy persona
        low_fin = next(t for t in low_priv.topic_sensitivities if t.topic == "personal_finances")
        high_fin = next(t for t in high_priv.topic_sensitivities if t.topic == "personal_finances")
        assert high_fin.sensitivity > low_fin.sensitivity


# ============================================================================
# PersonaEngine.from_description integration
# ============================================================================


class TestEngineFromDescription:
    def test_from_description_creates_engine(self):
        from persona_engine import PersonaEngine
        engine = PersonaEngine.from_description(
            "A 45-year-old chef named Marcus, passionate and direct",
            adapter=None,
            llm_provider="mock",
        )
        assert engine.persona.identity.occupation.lower() == "chef"
        assert engine.turn_count == 0

    def test_from_description_can_plan(self):
        from persona_engine import PersonaEngine
        engine = PersonaEngine.from_description(
            "A calm analytical scientist named Dr. Lee",
            llm_provider="mock",
        )
        ir = engine.plan("What do you think about quantum computing?")
        assert ir is not None
        assert ir.response_structure.confidence > 0

    def test_from_description_kwargs_forwarded(self):
        from persona_engine import PersonaEngine
        engine = PersonaEngine.from_description(
            "A friendly teacher named Sarah",
            llm_provider="mock",
            seed=123,
            validate=False,
        )
        assert engine.validator is None  # validate=False


# ============================================================================
# Coverage: all occupation domains produce valid personas
# ============================================================================


class TestOccupationCoverage:
    @pytest.mark.parametrize("occupation", list(OCCUPATION_DOMAINS.keys())[:20])
    def test_occupation_produces_valid_persona(self, occupation: str):
        p = PersonaBuilder("Test", occupation.title()).build()
        assert isinstance(p, Persona)
        assert len(p.knowledge_domains) > 0


# ============================================================================
# Coverage: all archetypes in archetype_persona
# ============================================================================


class TestArchetypeCoverage:
    @pytest.mark.parametrize("archetype", list(ARCHETYPES.keys()))
    def test_archetype_shortcut(self, archetype: str):
        p = PersonaBuilder.archetype_persona(archetype)
        assert isinstance(p, Persona)


# ============================================================================
# Subdomain Inference
# ============================================================================


class TestSubdomainInference:
    def test_chef_food_has_subdomains(self):
        domains = _infer_domains("Chef")
        food = next(d for d in domains if d.domain == "Food")
        assert len(food.subdomains) > 0
        assert "French cuisine" in food.subdomains

    def test_chef_business_has_subdomains(self):
        domains = _infer_domains("Chef")
        biz = next(d for d in domains if d.domain == "Business")
        assert "Restaurant management" in biz.subdomains

    def test_lawyer_law_has_subdomains(self):
        domains = _infer_domains("Lawyer")
        law = next(d for d in domains if d.domain == "Law")
        assert len(law.subdomains) > 0
        assert "Corporate Law" in law.subdomains

    def test_software_engineer_has_subdomains(self):
        domains = _infer_domains("Software Engineer")
        tech = next(d for d in domains if d.domain == "Technology")
        assert "Programming" in tech.subdomains

    def test_unknown_occupation_has_no_subdomains(self):
        domains = _infer_domains("Space Cowboy")
        assert domains[0].subdomains == []

    def test_occupation_without_subdomain_entry_has_empty_subdomains(self):
        """Occupations in OCCUPATION_DOMAINS but not in OCCUPATION_SUBDOMAINS get empty lists."""
        # Find an occupation key that has no subdomain entry
        missing = set(OCCUPATION_DOMAINS) - set(OCCUPATION_SUBDOMAINS)
        if missing:
            key = next(iter(missing))
            domains = _infer_domains(key)
            for d in domains:
                assert d.subdomains == []

    @pytest.mark.parametrize("occ", list(OCCUPATION_SUBDOMAINS.keys())[:15])
    def test_subdomain_keys_match_domain_keys(self, occ: str):
        """Every domain key in OCCUPATION_SUBDOMAINS must also appear in the
        domains returned by OCCUPATION_DOMAINS for the same occupation."""
        if occ not in OCCUPATION_DOMAINS:
            pytest.skip(f"{occ} not in OCCUPATION_DOMAINS")
        domain_names = {d for d, _ in OCCUPATION_DOMAINS[occ]}
        subdomain_domains = set(OCCUPATION_SUBDOMAINS[occ].keys())
        assert subdomain_domains.issubset(domain_names), (
            f"Subdomain domains {subdomain_domains - domain_names} "
            f"not in OCCUPATION_DOMAINS for '{occ}'"
        )


# ============================================================================
# YAML Export
# ============================================================================


class TestYAMLExport:
    def test_to_dict_returns_dict(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        d = persona.to_dict()
        assert isinstance(d, dict)
        assert "persona_id" in d
        assert "identity" in d

    def test_to_yaml_returns_string(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        yaml_str = persona.to_yaml()
        assert isinstance(yaml_str, str)
        assert "persona_id:" in yaml_str
        assert "Marcus" in yaml_str

    def test_to_yaml_roundtrips(self):
        import yaml
        persona = PersonaBuilder("Marcus", "Chef").age(41).build()
        yaml_str = persona.to_yaml()
        data = yaml.safe_load(yaml_str)
        assert data["identity"]["age"] == 41
        assert data["identity"]["occupation"] == "Chef"

    def test_to_yaml_writes_file(self, tmp_path):
        persona = PersonaBuilder("Marcus", "Chef").build()
        out = tmp_path / "persona.yaml"
        persona.to_yaml(path=str(out))
        assert out.exists()
        content = out.read_text()
        assert "Marcus" in content

    def test_save_yaml_builds_and_writes(self, tmp_path):
        out = tmp_path / "chef.yaml"
        persona = PersonaBuilder("Marcus", "Chef").age(41).save_yaml(str(out))
        assert isinstance(persona, Persona)
        assert out.exists()
        import yaml
        data = yaml.safe_load(out.read_text())
        assert data["identity"]["age"] == 41

    def test_to_dict_subdomains_present(self):
        persona = PersonaBuilder("Marcus", "Chef").build()
        d = persona.to_dict()
        domains = d["knowledge_domains"]
        food = next(dom for dom in domains if dom["domain"] == "Food")
        assert len(food["subdomains"]) > 0


# ============================================================================
# Builder → Engine integration
# ============================================================================


class TestBuilderEngineIntegration:
    def test_builder_persona_works_with_engine(self):
        from persona_engine import PersonaEngine
        persona = PersonaBuilder("Marcus", "Chef").age(41).trait("passionate", "direct").build()
        engine = PersonaEngine(persona=persona, llm_provider="mock")
        ir = engine.plan("What's the best way to cook a steak?")
        assert ir is not None
        assert ir.response_structure.confidence > 0

    def test_archetype_persona_works_with_engine(self):
        from persona_engine import PersonaEngine
        persona = PersonaBuilder("Dr. Lee", "Physicist").archetype("expert").build()
        engine = PersonaEngine(persona=persona, llm_provider="mock")
        ir = engine.plan("Tell me about quantum mechanics")
        assert ir is not None

    def test_builder_subdomains_visible_in_engine_persona(self):
        from persona_engine import PersonaEngine
        persona = PersonaBuilder("Marcus", "Chef").build()
        engine = PersonaEngine(persona=persona, llm_provider="mock")
        food = next(d for d in engine.persona.knowledge_domains if d.domain == "Food")
        assert len(food.subdomains) > 0
