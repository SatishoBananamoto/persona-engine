# Memory — Claude Session Log

## Project: persona-engine
**Repo**: SatishoBananamoto/persona-engine
**Branch**: `claude/general-session-VUY6r`
**Date started**: 2026-03-13

---

## What This Project Is

A psychologically-grounded conversational persona engine that creates behaviorally coherent synthetic humans for testing, research, and simulation. The core idea: instead of prompting an LLM with "act like X", this system builds a structured **Intermediate Representation (IR)** from psychological models, then uses that IR to guide text generation.

**Stack**: Python 3.11+, Pydantic v2, PyYAML. Optional: Anthropic SDK, OpenAI SDK.
**Status**: MVP (Alpha) — 7 development phases completed.

---

## Architecture — The Full Pipeline

```
User Message
    ↓
PersonaEngine.chat(user_input)
    ↓
Input Validation (strip control chars, length ≤10k)
    ↓
TurnPlanner.generate_ir() — 5-stage pipeline:
    Stage 1: Foundation — trace context, deterministic seed, memory load
    Stage 2: Interpretation — topic relevance, bias simulation, state evolution, intent analysis, domain detection
    Stage 3: Behavioral Metrics — elasticity, stance, confidence, competence, tone, verbosity, formality, directness + cross-turn inertia smoothing (0.7 blend)
    Stage 4: Knowledge & Safety — disclosure level, uncertainty action, claim type, response patterns, invariant validation
    Stage 5: Finalization — memory write intents, IR assembly, snapshot storage
    ↓
PipelineValidator.validate() — 6 validators:
    1. IRCoherenceValidator — internal consistency
    2. PersonaComplianceValidator — traits/values alignment
    3. CrossTurnConsistencyValidator — drift detection
    4. IRValidator — schema-level checks
    5. StyleDriftDetector — style metric deviations
    6. KnowledgeBoundaryEnforcer — claim vs. proficiency
    ↓
ResponseGenerator.generate()
    Build system + generation prompts from IR
    Temperature = max(0.3, 1.0 - confidence * 0.5)
    Call LLM adapter → post-process (verbosity enforcement, constraint validation)
    ↓
ChatResult(text, ir, validation, response, turn_number)
```

---

## Schema — Persona Model (persona_schema.py)

**Persona** is the root model. Key components:

- **identity**: `Identity` — age (18-100), gender, location, education, occupation, background
- **psychology**: `PersonalityProfile` containing:
  - `BigFiveTraits` — openness, conscientiousness, extraversion, agreeableness, neuroticism (all 0.0-1.0)
  - `SchwartzValues` — 10 values: self_direction, stimulation, hedonism, achievement, power, security, conformity, tradition, benevolence, universalism
  - `CognitiveStyle` — analytical_intuitive, systematic_heuristic, risk_tolerance, need_for_closure, cognitive_complexity
  - `CommunicationPreferences` — verbosity, formality, directness, emotional_expressiveness
- **knowledge_domains**: list of `DomainKnowledge(domain, proficiency, subdomains)`
- **primary_goals / secondary_goals**: list of `Goal(goal, weight)`
- **social_roles**: dict of context → `SocialRole(formality, directness, emotional_expressiveness)`
- **uncertainty**: `UncertaintyPolicy` — admission_threshold, hedging_frequency, clarification_tendency, knowledge_boundary_strictness
- **claim_policy**: `ClaimPolicy` — allowed_claim_types, expert_threshold (default 0.7), lookup_behavior
- **invariants**: `PersonaInvariants` — identity_facts (immutable), cannot_claim, must_avoid
- **disclosure_policy**: `DisclosurePolicy` — base_openness, factors, bounds
- **topic_sensitivities**: list of `TopicSensitivity(topic, sensitivity)`
- **response_patterns**: list of `ResponsePattern(trigger, response, emotionality)` — triggers: disagreement, personal_question, expertise_request
- **biases**: list of `Bias(type, strength)` — confirmation_bias, availability_heuristic, social_desirability
- **initial_state**: `DynamicState` — mood_valence (-1 to 1), mood_arousal, fatigue, stress, engagement

---

## Schema — IR Model (ir_schema.py)

**IntermediateRepresentation** — the structured plan before text generation:

- **conversation_frame**: `ConversationFrame`
  - interaction_mode: casual_chat | interview | customer_support | survey | coaching | debate | small_talk | brainstorm
  - goal: gather_info | resolve_issue | build_rapport | persuade | educate | entertain | explore_ideas
  - success_criteria: list[str]
- **response_structure**: `ResponseStructure`
  - intent, stance, rationale (strings)
  - elasticity (openness to changing mind), confidence, competence (all 0-1 floats)
- **communication_style**: `CommunicationStyle`
  - tone: 19 enum values mapped to valence/arousal grid (warm_enthusiastic, frustrated_tense, neutral_calm, etc.)
  - verbosity: brief | medium | detailed
  - formality, directness (floats)
- **knowledge_disclosure**: `KnowledgeAndDisclosure`
  - disclosure_level (float), uncertainty_action (answer|hedge|ask_clarifying|refuse), knowledge_claim_type
- **citations**: list of `Citation` — delta-based traceability (source_type, source_id, effect, weight, target_field, operation, value_before, value_after, delta)
- **safety_plan**: `SafetyPlan` — active_constraints, blocked_topics, clamped_fields, pattern_blocks, cannot_claim, must_avoid
- **memory_ops**: `MemoryOps` — read_requests, write_intents, write_policy

---

## Memory System (persona_engine/memory/)

Four typed stores + stance cache + manager:

| Store | Key → Value | Purpose |
|---|---|---|
| `FactStore` | topic → (fact, confidence, source, turn) | Hard facts stated by user |
| `PreferenceStore` | topic → (preference, strength, source) | User preferences/likes |
| `RelationshipStore` | entity → (relation, trust, rapport, events[]) | People/org relationships |
| `EpisodicStore` | turn_range → (summary, topics, emotional_arc) | Compressed conversation summaries |
| `StanceCache` | topic → (stance, confidence, turn) | Rapid stance lookup for consistency |

**MemoryManager** orchestrates all stores. Methods: `load_context(query)`, `process_write_intents(ops)`, `stats()`.

Memory affects behavior via: context injection into IR, stance consistency enforcement, confidence decay over turns, preference reinforcement, trust/rapport tracking, episode summaries.

---

## Validation System (persona_engine/validation/)

6 validators run in `PipelineValidator`:

1. **IRCoherenceValidator** — internal IR consistency (e.g., high confidence + refuse = violation)
2. **PersonaComplianceValidator** — IR fields match persona traits/values
3. **CrossTurnConsistencyValidator** — detects drift across turns
4. **IRValidator** — schema-level range/type checks
5. **StyleDriftDetector** — flags sudden style metric changes
6. **KnowledgeBoundaryEnforcer** — ensures claims don't exceed proficiency

Results: `IRValidationResult(passed, violations[], checked_invariants[])`
Violations: `ValidationViolation(type, severity[error|warning], message, field_path, suggested_fix)`

---

## LLM Providers (persona_engine/generation/)

All implement `BaseLLMAdapter` interface: `generate(system_prompt, user_prompt, max_tokens, temperature, conversation_history) → str`

| Provider | Model | Notes |
|---|---|---|
| `AnthropicAdapter` | claude-sonnet-4-20250514 | Needs ANTHROPIC_API_KEY |
| `OpenAIAdapter` | gpt-4o-mini | Needs OPENAI_API_KEY |
| `MockLLMAdapter` | — | Template responses, tracks call_count, for testing |
| `TemplateAdapter` | — | Rule-based IR→text, fully deterministic, zero API cost |

Factory: `create_adapter(provider, api_key, model) → BaseLLMAdapter`

---

## PersonaBuilder (persona_engine/persona_builder.py)

Three ways to create personas without hand-crafting YAML:

1. **Fluent builder**: `PersonaBuilder("Name", "Occupation").age(30).traits("curious", "direct").archetype("expert").domain("AI", 0.9).build()`
2. **From description**: `PersonaBuilder.from_description("A 45-year-old chef named Marcus, passionate and direct")`
3. **Archetype shortcut**: `PersonaBuilder.archetype_persona("analyst", name="Dr. Lee", occupation="Physicist")`

Internals: baseline 0.5 for all traits → `TRAIT_MODIFIERS` dict maps adjectives to deltas → `ARCHETYPES` dict provides preset overrides → `OCCUPATION_DOMAINS` maps occupation to knowledge domains.

---

## Behavioral Layer (persona_engine/behavioral/)

- **StateManager** — evolves dynamic state (mood, fatigue, stress, engagement) across turns. Maps valence/arousal → Tone enum.
- **BiasSimulator** — applies cognitive biases (confirmation_bias, availability_heuristic, social_desirability) as modifiers to IR fields.
- **StanceGenerator** — generates stance text from persona + topic + context, checks invariants.
- **TopicRelevance** — computes topic_relevance score from persona domains + goals vs user input.
- **IntentAnalyzer** — analyzes user intent → inferred mode, goal, user_intent, needs_clarification.
- **DomainDetector** — keyword-based domain detection.
- **Cross-turn inertia** — CROSS_TURN_INERTIA ≈ 0.7 smoothing factor blends current metrics with prior snapshot.

---

## Planner (persona_engine/planner/)

**TurnPlanner** — orchestrates the 5-stage IR generation. Key constants:
- `CROSS_TURN_INERTIA = 0.7`
- expert_threshold from `claim_policy.expert_threshold` (default 0.7)

Maintains: `TurnSnapshot` (prior turn's metrics for inertia), `StanceCache`, conversation context.

---

## CLI (python -m persona_engine)

```
validate <file>           — Validate YAML persona (syntax + deep IR check)
info <file>               — Display persona details
plan <file> "<message>"   — Generate IR without LLM (--json flag)
chat <file> "<message>"   — Full chat (--provider mock|anthropic|openai|template)
list [directory]          — List YAML personas
```

---

## Public API Exports

```python
# Core
PersonaEngine, Conversation, PersonaBuilder, ChatResult

# Persona Schema
Persona, PersonalityProfile, BigFiveTraits, SchwartzValues, CognitiveStyle,
CommunicationPreferences, DomainKnowledge, Goal

# IR Schema
IntermediateRepresentation, ConversationFrame, ResponseStructure,
CommunicationStyle, KnowledgeAndDisclosure

# IR Enums
InteractionMode, ConversationGoal, Tone, Verbosity, UncertaintyAction
```

---

## Personas (personas/)

8 YAML personas: chef (Marcus, 41, Chicago), fitness_coach, lawyer, musician, physicist, ux_researcher, + 2 test-specific personas. Each follows the full Persona schema above.

---

## Tests (tests/)

**39 test files, ~290 test classes, ~1,698 test methods.** Cover:
- Schema validation, IR normalization, deterministic JSON
- TurnPlanner end-to-end, all 5 stages
- Each validator independently
- Memory stores (fact, preference, relationship, episodic)
- PersonaBuilder (fluent, from_description, archetypes)
- Conversation class (say, say_all, export)
- CLI commands
- Cross-turn consistency, style drift, knowledge boundaries
- Response generation, template adapter, style modulation
- PersonaEngine integration (chat, plan, save/load, reset)

---

## Key Files Quick Reference

| File | What it does |
|---|---|
| `persona_engine/engine.py` | PersonaEngine — main orchestrator, chat/plan/save/load/reset |
| `persona_engine/conversation.py` | Conversation — multi-turn wrapper, export to JSON/YAML/transcript |
| `persona_engine/persona_builder.py` | PersonaBuilder — fluent API, from_description, archetypes |
| `persona_engine/schema/persona_schema.py` | All Pydantic models for Persona |
| `persona_engine/schema/ir_schema.py` | All Pydantic models for IR, Citation, Safety, Memory ops |
| `persona_engine/planner/turn_planner.py` | TurnPlanner — 5-stage IR generation |
| `persona_engine/planner/topic_relevance.py` | Topic relevance scoring |
| `persona_engine/generation/response_generator.py` | ResponseGenerator — IR→text via LLM |
| `persona_engine/generation/adapters.py` | LLM adapters (Anthropic, OpenAI, Mock, Template) |
| `persona_engine/generation/ir_prompt_builder.py` | Builds system/generation prompts from IR |
| `persona_engine/generation/style_modulator.py` | Post-processing: verbosity enforcement, constraint validation |
| `persona_engine/validation/pipeline_validator.py` | PipelineValidator — runs all 6 validators |
| `persona_engine/memory/memory_manager.py` | MemoryManager — orchestrates all 4 stores |
| `persona_engine/behavioral/state_manager.py` | StateManager — dynamic state evolution |
| `persona_engine/behavioral/bias_simulator.py` | BiasSimulator — cognitive bias modifiers |

---

## Session Activity Log

- [x] Deep exploration of entire repo via 5 parallel agents
- [x] Created this memory.md with real understanding
- [ ] No feature work assigned yet — waiting for direction

---

## Notes for Future Me

- Branch `claude/general-session-VUY6r` is checked out and tracking remote
- Use `mock` or `template` provider for testing without API keys
- `PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")` is the fastest way to get a working engine
- Run `pytest` before pushing — there are ~1,698 tests
- The Citation system is delta-based: every IR field modification is traced back to its source (trait, value, state, memory, etc.)
- Cross-turn inertia (0.7) smooths behavioral metrics between turns to prevent sudden personality shifts
- `ir.normalize()` and `ir.to_json_deterministic()` exist for golden test comparison
