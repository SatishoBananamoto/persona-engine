# Building Universal Conversational Persona System - MVP

## MVP Goal
Build **Persona Engine + Turn Planner (IR) + Validator + Local SDK** with 10-15 excellent personas and deterministic test harness.

## Phase 1: Foundation & Schema
- [x] Finalize enhanced persona schema (goals, social roles, uncertainty, disclosure function, **claim policy, invariants**)
- [x] Define Intermediate Representation (IR) schema (**conversation frame, stance+rationale+elasticity**)
- [x] Set up Python package structure (`persona_engine/`)
- [x] Implement core Pydantic models with validation
- [x] **Define persona invariants (identity_facts, cannot_claim, must_avoid)**
- [x] Define safety boundaries (what system won't do)
- [x] Set up determinism controls (seeded randomness, strict mode)
- [x] **Create trait marker rubrics (concrete, measurable markers for Big 5)**
- [x] **Fix schema consistency: standardized tokens, Tone enum, Citation.source_type with 'goal'/'memory', removed runtime types**

## Phase 2: Psychological & Behavioral Core
- [x] Implement Big 5 trait → behavior mappings
- [x] Build Schwartz values system with conflict resolver
- [x] Create cognitive style interpreter
- [x] Implement dynamic state manager (mood, fatigue, stress, engagement)
- [x] Build behavioral rules engine (decision policies, social role modes)
- [x] **Fix architecture issues: determinism (sorted weighted_choice), stance cache, single-point uncertainty resolver, constraint safety**
- [x] Add subtle, bounded bias simulation

## Phase 3: Turn Planner (Critical - IR Layer)
- [x] Build core Turn Planner orchestration
- [x] Implement intent analyzer (template + heuristic logic)
- [x] Create stance generator (expert awareness + caching)
- [x] Build disclosure calculator (integrated clamp logic)
- [x] Generate IR with citations (full TraceContext integration)
- [x] Add uncertainty policy (driven by proficiency + challenge)

## Phase 4: Memory System (Typed Facts)
- [x] Design typed fact storage (facts, preferences, relationships, episodic)
- [x] Implement fact store (with confidence, privacy, recency)
- [x] Build preference store (learned patterns)
- [x] Create relationship store (trust levels)
- [x] Add episodic summary store (compressed, not verbatim)
- [x] Prevent persona drift from memory replay (confidence decay)
- [x] Wire memory reads into IR generation (facts influence confidence/stance)

## Phase 5: Response Generator (IR → Text)
- [x] Choose LLM provider (Anthropic Claude selected)
- [x] Build LLM adapter abstraction (AnthropicAdapter, OpenAIAdapter, MockLLMAdapter)
- [x] Implement IR → text renderer (IRPromptBuilder with constraint formatting)
- [x] Add template engine for strict mode (TemplateAdapter)
- [x] Create style modulator (applies verbosity, formality, directness from IR)
- [x] Implement strict mode (forces TemplateAdapter, deterministic output)
- [x] Add LLM error handling with typed exceptions and retry

## Phase 6: Validation & QA Suite
- [x] Build comprehensive Turn Planner test suite (determinism, contract, scenario tests)
  - [x] Determinism tests (same seed = identical IR)
  - [x] Contract tests: no naked writes, clamps recorded, non-expert guardrails, canonical bias IDs
  - [x] Bias trigger tests with persona fixtures (test_high_conformity, test_high_neuroticism)
  - [x] Scenario tests: domain detection, unknown domain, evidence strength, privacy
  - [x] Full citation dump (25 citations with complete traceability)
- [x] Behavioral coherence tests (psychological realism)
  - [x] Trait influence coherence (extraversion→disclosure, neuroticism→confidence, etc.)
  - [x] Social role adaptation (formality changes by context)
  - [x] Bias coherence (bounded ±0.15, canonical IDs, requires markers)
  - [x] State transition coherence (bounded values, stress affects tone)
  - [x] Knowledge boundary coherence (expert domain claims, unknown domain humility)
  - [x] Citation integrity (all behavioral floats cited, clamps recorded)
  - [x] Determinism coherence (same seed = identical IR)
- [x] Build IR validator (checks plan before generation)
- [x] Implement invariant checks (no contradictions, knowledge boundaries)
- [x] Create trait marker scorer (across domains)
- [x] Add style drift detection (verbosity/formality variance over turns)
- [x] Build knowledge boundary enforcer
- [x] Implement property-based testing framework
- [x] Add bias modifier application (biases affect IR fields with citations)

## Phase 7: Python SDK (Local Mode)
- [x] Create SDK package structure
- [x] Build high-level `Persona` class interface
- [x] Add `Conversation` class for multi-turn dialogue
- [x] Implement persona builder utilities
- [x] Create validator CLI tools
- [x] Add export formats (YAML, JSON)

## Phase 8: Persona Library & Testing
- [x] Design 12 diverse personas (chef, physicist, lawyer, musician, fitness coach, UX researcher, software engineer, social worker, entrepreneur, retired teacher + 2 test personas)
- [x] Create segment families with variants
- [x] Build counterfactual twins (5 Big Five trait pairs = 10 twin personas)
- [x] Generate benchmark conversations (casual chat, interview, customer support, survey)
- [x] Run full QA suite on all personas
- [x] Input sanitization consistency across all paths

## Phase 9: Documentation & Examples
- [x] Write SDK usage guide (`docs/sdk_guide.md`)
- [x] Create persona authoring guide (`docs/persona_authoring.md`)
- [x] Document IR structure and validation logic (`docs/ir_reference.md`)
- [x] Create "getting started" tutorial (`docs/tutorial.md`)
- [x] Add structured logging to all core modules
- [x] Add example scripts (`examples/`)

## Phase 10: Production Readiness (Current)
- [x] GitHub Actions CI/CD pipeline
- [x] Pre-commit hooks configuration
- [x] CHANGELOG tracking all phases
- [x] Comprehensive conftest.py with shared fixtures
- [ ] Distributional guarantees (trait markers within range)
- [ ] Deterministic failure reproduction (seeded)
- [ ] QA suite interpretation guide

## Future (Post-MVP)
- [ ] Optional: Reference FastAPI server (feature-flagged)
- [ ] Async/await LLM adapter support
- [ ] Expand persona library (breadth)
- [ ] Add multi-language support
- [ ] Build admin dashboard
- [ ] Production API service
- [ ] Event bus / hooks for extensibility
