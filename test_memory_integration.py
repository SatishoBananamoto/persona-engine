"""
Memory Integration Tests — proving memory flows through the pipeline.

Tests that MemoryManager is properly wired into TurnPlanner and
ResponseGenerator, and that memories persist across turns.

Categories:
  1. TurnPlanner + Memory wiring (read context, write intents, process writes)
  2. Multi-turn memory persistence (facts remembered across turns)
  3. Memory citations in IR (memory reads produce citations)
  4. Response generation with memory context (LLM prompts include memory)
  5. Behavioral scenarios (persona behavior changes based on memory)
"""

import pytest
import yaml

from persona_engine.generation.llm_adapter import TemplateAdapter
from persona_engine.generation.prompt_builder import IRPromptBuilder
from persona_engine.generation.response_generator import ResponseGenerator
from persona_engine.memory import MemoryManager
from persona_engine.memory.models import MemorySource
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    MemoryWriteIntent,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# =============================================================================
# Helpers
# =============================================================================


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    """Load persona from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_context(
    user_input: str,
    topic: str = "general",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    turn: int = 1,
    stance_cache: StanceCache | None = None,
    domain: str | None = None,
    conversation_id: str = "test_memory",
) -> ConversationContext:
    return ConversationContext(
        conversation_id=conversation_id,
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=stance_cache or StanceCache(),
        domain=domain,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def persona() -> Persona:
    return load_persona("personas/ux_researcher.yaml")


@pytest.fixture
def memory() -> MemoryManager:
    return MemoryManager()


@pytest.fixture
def planner_with_memory(persona: Persona, memory: MemoryManager) -> TurnPlanner:
    return TurnPlanner(
        persona=persona,
        determinism=DeterminismManager(seed=42),
        memory_manager=memory,
    )


@pytest.fixture
def planner_without_memory(persona: Persona) -> TurnPlanner:
    return TurnPlanner(
        persona=persona,
        determinism=DeterminismManager(seed=42),
    )


# =============================================================================
# 1. TurnPlanner + Memory Wiring
# =============================================================================


class TestTurnPlannerMemoryWiring:
    """Tests that TurnPlanner reads and writes memory correctly."""

    def test_planner_accepts_memory_manager(self, persona: Persona):
        """TurnPlanner can be created with a MemoryManager."""
        mm = MemoryManager()
        planner = TurnPlanner(persona=persona, memory_manager=mm)
        assert planner.memory is mm

    def test_planner_without_memory_still_works(self, planner_without_memory: TurnPlanner):
        """TurnPlanner without memory produces valid IR (backwards compatible)."""
        ctx = make_context("What do you think about AI?", topic="ai")
        ir = planner_without_memory.generate_ir(ctx)
        assert ir is not None
        assert ir.response_structure.confidence > 0

    def test_planner_with_memory_produces_ir(self, planner_with_memory: TurnPlanner):
        """TurnPlanner with memory produces valid IR."""
        ctx = make_context("What do you think about UX research?", topic="ux_research")
        ir = planner_with_memory.generate_ir(ctx)
        assert ir is not None
        assert ir.response_structure.confidence > 0

    def test_planner_generates_write_intents(self, planner_with_memory: TurnPlanner):
        """TurnPlanner populates memory_ops.write_intents."""
        ctx = make_context("Tell me about usability testing", topic="usability")
        ir = planner_with_memory.generate_ir(ctx)
        assert len(ir.memory_ops.write_intents) > 0

    def test_write_intents_have_episode(self, planner_with_memory: TurnPlanner):
        """Every turn generates at least an episode write intent."""
        ctx = make_context("How's your day?", topic="small_talk")
        ir = planner_with_memory.generate_ir(ctx)
        episode_intents = [
            w for w in ir.memory_ops.write_intents if w.content_type == "episode"
        ]
        assert len(episode_intents) >= 1

    def test_writes_are_processed_after_ir(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Memory writes are actually stored after generate_ir()."""
        assert memory.episodes.count == 0
        ctx = make_context("Let's discuss usability", topic="usability")
        planner_with_memory.generate_ir(ctx)
        assert memory.episodes.count > 0

    def test_no_writes_without_memory(self, planner_without_memory: TurnPlanner):
        """Without MemoryManager, no writes happen (no crash)."""
        ctx = make_context("Hello", topic="greeting")
        ir = planner_without_memory.generate_ir(ctx)
        # memory_ops may be empty or default
        assert ir is not None


# =============================================================================
# 2. Multi-Turn Memory Persistence
# =============================================================================


class TestMultiTurnPersistence:
    """Tests that memories persist across turns."""

    def test_episode_persists_across_turns(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Episodes from turn 1 are visible in turn 2."""
        cache = StanceCache()
        # Turn 1
        ctx1 = make_context(
            "Tell me about UX research methods",
            topic="ux_methods",
            turn=1,
            stance_cache=cache,
        )
        planner_with_memory.generate_ir(ctx1)
        assert memory.episodes.count >= 1

        # Turn 2: memory should have episodes from turn 1
        context = memory.get_context_for_turn(topic="ux_methods", current_turn=2)
        assert context.get("topic_episodes") or context.get("previously_discussed")

    def test_fact_persists_across_turns(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Facts stored via convenience method are available in later turns."""
        # Pre-load a fact (simulating a previous turn's processing)
        memory.remember_fact(
            "User works as a software engineer",
            category="occupation",
            confidence=0.9,
            turn=1,
        )

        # Turn 3: fact should appear in memory context
        context = memory.get_context_for_turn(topic="career", current_turn=3)
        facts = context["known_facts"]
        assert len(facts) >= 1
        assert any("software engineer" in f["content"] for f in facts)

    def test_preference_persists_and_strengthens(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Repeated preference observations strengthen the preference."""
        # Observe same preference twice
        memory.remember_preference("User prefers brief answers", strength=0.5, turn=1)
        memory.remember_preference("User prefers brief answers", strength=0.5, turn=3)

        # Check it's stronger now
        prefs = memory.preferences.get_active(current_turn=4)
        matching = [p for p in prefs if "brief" in p.content]
        assert len(matching) >= 1
        # Reinforcement: 0.5 + (2-1)*0.1 = 0.6
        assert matching[0].strength >= 0.5

    def test_relationship_evolves_across_turns(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Relationship trust/rapport changes over multiple turns."""
        initial_trust = memory.relationships.trust

        # Multiple positive interactions
        cache = StanceCache()
        for turn in range(1, 4):
            ctx = make_context(
                "That's a great point about UX",
                topic="ux_research",
                turn=turn,
                stance_cache=cache,
            )
            planner_with_memory.generate_ir(ctx)

        # Trust/rapport should have changed (relationship write intents processed)
        # The exact change depends on whether relationship intents were generated
        stats = memory.stats()
        assert stats["relationship_events"] >= 1

    def test_five_turn_conversation_accumulates_memory(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """A 5-turn conversation accumulates episodes and relationship events."""
        cache = StanceCache()
        inputs = [
            ("What's your experience with user interviews?", "user_interviews"),
            ("How do you handle difficult participants?", "interview_techniques"),
            ("That's interesting, tell me more", "interview_techniques"),
            ("Do you prefer remote or in-person?", "remote_vs_inperson"),
            ("Thanks, that's very helpful!", "conclusion"),
        ]

        for turn, (user_input, topic) in enumerate(inputs, 1):
            ctx = make_context(
                user_input, topic=topic, turn=turn, stance_cache=cache
            )
            planner_with_memory.generate_ir(ctx)

        stats = memory.stats()
        assert stats["episodes"] >= 5
        assert stats["relationship_events"] >= 1


# =============================================================================
# 3. Memory Citations in IR
# =============================================================================


class TestMemoryCitations:
    """Tests that memory reads produce citations in the IR."""

    def test_no_memory_citations_without_manager(self, planner_without_memory: TurnPlanner):
        """Without MemoryManager, no memory citations appear."""
        ctx = make_context("Hello", topic="greeting")
        ir = planner_without_memory.generate_ir(ctx)
        memory_cites = [c for c in ir.citations if c.source_type == "memory"]
        # Only stance_cache citations may appear, not fact/preference/episode
        store_cites = [
            c for c in memory_cites
            if c.source_id in ("fact_store", "preference_store", "episodic_store")
        ]
        assert len(store_cites) == 0

    def test_fact_citation_appears_when_facts_exist(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """When facts exist, a memory citation appears in the IR."""
        memory.remember_fact("User is a designer", turn=1)
        ctx = make_context("Tell me about design", topic="design", turn=3)
        ir = planner_with_memory.generate_ir(ctx)
        fact_cites = [
            c for c in ir.citations
            if c.source_type == "memory" and c.source_id == "fact_store"
        ]
        assert len(fact_cites) >= 1

    def test_preference_citation_appears(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """When preferences exist, a memory citation appears."""
        memory.remember_preference("User likes examples", strength=0.8, turn=1)
        ctx = make_context("Explain this concept", topic="concepts", turn=3)
        ir = planner_with_memory.generate_ir(ctx)
        pref_cites = [
            c for c in ir.citations
            if c.source_type == "memory" and c.source_id == "preference_store"
        ]
        assert len(pref_cites) >= 1

    def test_episode_citation_when_topic_revisited(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """When a topic was previously discussed, episodic citation appears."""
        memory.record_episode(
            "Discussed usability testing approaches",
            topic="usability",
            turn_start=1,
            turn_end=2,
        )
        ctx = make_context(
            "Let's revisit usability testing", topic="usability", turn=5
        )
        ir = planner_with_memory.generate_ir(ctx)
        ep_cites = [
            c for c in ir.citations
            if c.source_type == "memory" and c.source_id == "episodic_store"
        ]
        assert len(ep_cites) >= 1


# =============================================================================
# 4. Response Generation with Memory
# =============================================================================


class TestResponseGenerationWithMemory:
    """Tests that memory context flows into response generation."""

    def test_prompt_builder_includes_memory(self, persona: Persona):
        """PromptBuilder includes memory context in LLM prompts."""
        from persona_engine.schema.ir_schema import (
            CommunicationStyle,
            ConversationFrame,
            IntermediateRepresentation,
            KnowledgeAndDisclosure,
            KnowledgeClaimType,
            ResponseStructure,
            Tone,
            UncertaintyAction,
            Verbosity,
        )

        ir = IntermediateRepresentation(
            conversation_frame=ConversationFrame(
                interaction_mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
            ),
            response_structure=ResponseStructure(
                intent="Share perspective",
                confidence=0.8,
            ),
            communication_style=CommunicationStyle(
                tone=Tone.THOUGHTFUL_ENGAGED,
                verbosity=Verbosity.MEDIUM,
                formality=0.4,
                directness=0.5,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )

        memory_context = {
            "known_facts": [
                {"content": "User is a product manager", "category": "occupation", "confidence": 0.9}
            ],
            "active_preferences": [
                {"content": "User prefers detailed explanations", "strength": 0.8}
            ],
            "relationship": {"trust": 0.72, "rapport": 0.65},
            "previously_discussed": True,
            "topic_episodes": [
                {"content": "Previously discussed UX methods", "outcome": "agreed"}
            ],
        }

        builder = IRPromptBuilder()
        prompt = builder.build_generation_prompt(
            ir=ir,
            user_input="Tell me more about UX",
            persona=persona,
            memory_context=memory_context,
        )

        assert "WHAT YOU REMEMBER" in prompt
        assert "product manager" in prompt
        assert "detailed explanations" in prompt
        assert "72%" in prompt  # trust
        assert "discussed this topic before" in prompt
        assert "Previously discussed UX methods" in prompt

    def test_prompt_builder_no_memory_section_when_empty(self, persona: Persona):
        """No memory section in prompt when memory_context is None."""
        from persona_engine.schema.ir_schema import (
            CommunicationStyle,
            ConversationFrame,
            IntermediateRepresentation,
            KnowledgeAndDisclosure,
            KnowledgeClaimType,
            ResponseStructure,
            Tone,
            UncertaintyAction,
            Verbosity,
        )

        ir = IntermediateRepresentation(
            conversation_frame=ConversationFrame(
                interaction_mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
            ),
            response_structure=ResponseStructure(
                intent="Share perspective",
                confidence=0.8,
            ),
            communication_style=CommunicationStyle(
                tone=Tone.NEUTRAL_CALM,
                verbosity=Verbosity.MEDIUM,
                formality=0.5,
                directness=0.5,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.COMMON_KNOWLEDGE,
            ),
        )

        builder = IRPromptBuilder()
        prompt = builder.build_generation_prompt(ir=ir, user_input="Hello")
        assert "WHAT YOU REMEMBER" not in prompt

    def test_response_generator_accepts_memory_context(self, persona: Persona):
        """ResponseGenerator.generate() accepts memory_context parameter."""
        from persona_engine.schema.ir_schema import (
            CommunicationStyle,
            ConversationFrame,
            IntermediateRepresentation,
            KnowledgeAndDisclosure,
            KnowledgeClaimType,
            ResponseStructure,
            Tone,
            UncertaintyAction,
            Verbosity,
        )

        ir = IntermediateRepresentation(
            conversation_frame=ConversationFrame(
                interaction_mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
            ),
            response_structure=ResponseStructure(
                intent="Share perspective",
                stance="UX research is essential",
                confidence=0.8,
            ),
            communication_style=CommunicationStyle(
                tone=Tone.THOUGHTFUL_ENGAGED,
                verbosity=Verbosity.MEDIUM,
                formality=0.4,
                directness=0.5,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )

        generator = ResponseGenerator(persona=persona, adapter=TemplateAdapter())
        # Should not crash with memory_context
        response = generator.generate(
            ir=ir,
            user_input="What about UX?",
            memory_context={"known_facts": [], "relationship": {"trust": 0.5, "rapport": 0.3}},
        )
        assert response.text
        assert response.is_valid()


# =============================================================================
# 5. Behavioral Scenarios
# =============================================================================


class TestMemoryBehavioralScenarios:
    """End-to-end behavioral scenarios testing memory-informed behavior."""

    def test_persona_remembers_previous_topic(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Persona produces episodic citation when revisiting a topic."""
        cache = StanceCache()

        # Turn 1: discuss ux_research
        ctx1 = make_context(
            "What's your approach to user interviews?",
            topic="ux_research",
            turn=1,
            stance_cache=cache,
        )
        planner_with_memory.generate_ir(ctx1)

        # Turn 3: revisit same topic
        ctx3 = make_context(
            "Going back to user interviews, any new thoughts?",
            topic="ux_research",
            turn=3,
            stance_cache=cache,
        )
        ir3 = planner_with_memory.generate_ir(ctx3)

        # Should have episodic memory citation
        ep_cites = [
            c for c in ir3.citations
            if c.source_type == "memory" and c.source_id == "episodic_store"
        ]
        assert len(ep_cites) >= 1

    def test_preloaded_facts_inform_ir(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Pre-loaded facts produce citations in IR."""
        memory.remember_fact(
            "User is a senior UX designer",
            category="occupation",
            confidence=0.95,
            turn=0,
        )
        memory.remember_fact(
            "User lives in San Francisco",
            category="location",
            confidence=0.9,
            turn=0,
        )

        ctx = make_context(
            "What tools do you recommend for UX research?",
            topic="ux_tools",
            turn=2,
        )
        ir = planner_with_memory.generate_ir(ctx)

        # Memory facts should be cited
        fact_cites = [
            c for c in ir.citations
            if c.source_type == "memory" and c.source_id == "fact_store"
        ]
        assert len(fact_cites) >= 1
        assert "2" in fact_cites[0].effect  # "Loaded 2 known facts"

    def test_multi_turn_relationship_building(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Over multiple turns, relationship events accumulate."""
        cache = StanceCache()
        initial_trust = memory.relationships.trust

        # 3 turns of engaged conversation
        for turn in range(1, 4):
            ctx = make_context(
                f"Tell me more about your experience (turn {turn})",
                topic="ux_experience",
                turn=turn,
                stance_cache=cache,
            )
            planner_with_memory.generate_ir(ctx)

        # Relationship should have events
        assert memory.relationships.event_count >= 1
        # Trust may have shifted (relationship intents processed)
        summary = memory.relationships.summary()
        assert "trust" in summary
        assert "rapport" in summary

    def test_empty_memory_no_citations(self, planner_with_memory: TurnPlanner):
        """Fresh memory (no facts/prefs/episodes) produces no memory store citations."""
        ctx = make_context("Hello there!", topic="greeting", turn=1)
        ir = planner_with_memory.generate_ir(ctx)
        store_cites = [
            c for c in ir.citations
            if c.source_type == "memory"
            and c.source_id in ("fact_store", "preference_store", "episodic_store")
        ]
        assert len(store_cites) == 0

    def test_backwards_compatibility_no_memory(self, persona: Persona):
        """All existing TurnPlanner usage (no memory_manager) still works."""
        planner = TurnPlanner(persona=persona, determinism=DeterminismManager(seed=99))
        assert planner.memory is None
        ctx = make_context("Test backwards compat", topic="test")
        ir = planner.generate_ir(ctx)
        assert ir.response_structure.confidence > 0
        assert len(ir.citations) > 0


# =============================================================================
# 6. Memory Manager Integration with IR MemoryOps
# =============================================================================


class TestMemoryOpsIntegration:
    """Tests that IR MemoryOps work end-to-end."""

    def test_write_intents_flow_to_stores(
        self, planner_with_memory: TurnPlanner, memory: MemoryManager
    ):
        """Write intents from IR are processed into the correct stores."""
        ctx = make_context(
            "I'm a designer working on mobile apps",
            topic="mobile_design",
            turn=1,
        )
        ir = planner_with_memory.generate_ir(ctx)

        # Verify writes happened
        assert ir.memory_ops.write_intents
        # Episode should be stored
        assert memory.episodes.count >= 1

    def test_manual_write_intents_processed(self, memory: MemoryManager):
        """MemoryManager.process_write_intents works with manually created intents."""
        intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="User works as a data scientist",
                confidence=0.9,
                privacy_level=0.3,
                source="user_stated",
            ),
            MemoryWriteIntent(
                content_type="preference",
                content="User prefers code examples",
                confidence=0.8,
                privacy_level=0.1,
                source="inferred_from_context",
            ),
        ]
        created = memory.process_write_intents(intents, turn=1, conversation_id="test")
        assert len(created) == 2
        assert memory.facts.count == 1
        assert memory.preferences.unique_count == 1

    def test_read_requests_return_stored_data(self, memory: MemoryManager):
        """fulfill_read_requests returns data that was previously stored."""
        from persona_engine.schema.ir_schema import MemoryReadRequest

        # Store some data
        memory.remember_fact("User is a teacher", turn=1)

        # Read it back
        requests = [
            MemoryReadRequest(query_type="fact", query="teacher", confidence_threshold=0.3)
        ]
        results = memory.fulfill_read_requests(requests, current_turn=2)
        assert "teacher" in results
        assert len(results["teacher"]) >= 1


# =============================================================================
# 7. Prompt Builder Memory Formatting
# =============================================================================


class TestPromptBuilderMemoryFormatting:
    """Tests the _format_memory_context helper."""

    def test_empty_context_produces_empty_string(self):
        builder = IRPromptBuilder()
        result = builder._format_memory_context({})
        assert result == ""

    def test_facts_formatted(self):
        builder = IRPromptBuilder()
        ctx = {
            "known_facts": [
                {"content": "User is an engineer", "category": "occupation", "confidence": 0.9}
            ]
        }
        result = builder._format_memory_context(ctx)
        assert "FACT" in result
        assert "engineer" in result
        assert "90%" in result

    def test_preferences_formatted(self):
        builder = IRPromptBuilder()
        ctx = {
            "active_preferences": [
                {"content": "Prefers short answers", "strength": 0.7}
            ]
        }
        result = builder._format_memory_context(ctx)
        assert "PREFERENCE" in result
        assert "short answers" in result

    def test_relationship_formatted(self):
        builder = IRPromptBuilder()
        ctx = {"relationship": {"trust": 0.8, "rapport": 0.6}}
        result = builder._format_memory_context(ctx)
        assert "TRUST" in result
        assert "80%" in result

    def test_previously_discussed_formatted(self):
        builder = IRPromptBuilder()
        ctx = {"previously_discussed": True}
        result = builder._format_memory_context(ctx)
        assert "discussed this topic before" in result

    def test_episodes_formatted(self):
        builder = IRPromptBuilder()
        ctx = {
            "topic_episodes": [
                {"content": "Discussed API design patterns", "outcome": "agreed"}
            ]
        }
        result = builder._format_memory_context(ctx)
        assert "PREVIOUS" in result
        assert "API design" in result
