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
- [ ] Design typed fact storage (facts, preferences, relationships, episodic)
- [ ] Implement fact store (with confidence, privacy, recency)
- [ ] Build preference store (learned patterns)
- [ ] Create relationship store (trust levels)
- [ ] Add episodic summary store (compressed, not verbatim)
- [ ] Prevent persona drift from memory replay

## Phase 5: Response Generator (IR → Text)
- [ ] Choose LLM provider (Claude vs GPT-4 - decide based on controllability)
- [ ] Build LLM adapter abstraction
- [ ] Implement IR → text renderer (LLM takes IR constraints)
- [ ] Add template engine for strict mode
- [ ] Create style modulator (applies verbosity, formality, directness from IR)
- [ ] Implement strict mode (reduces creativity, increases templates)

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
- [ ] Build IR validator (checks plan before generation)
- [ ] Implement invariant checks (no contradictions, knowledge boundaries)
- [ ] Create trait marker scorer (across domains)
- [ ] Add style drift detection (verbosity/formality variance over turns)
- [ ] Build knowledge boundary enforcer
- [ ] Implement property-based testing framework
- [ ] Add distributional guarantees (trait markers within range)
- [ ] Create deterministic failure reproduction (seeded)

## Phase 7: Python SDK (Local Mode)
- [ ] Create SDK package structure
- [ ] Build high-level `Persona` class interface
- [ ] Add `Conversation` class for multi-turn dialogue
- [ ] Implement persona builder utilities
- [ ] Create validator CLI tools
- [ ] Add export formats (YAML, JSON)

## Phase 8: Persona Library & Testing
- [ ] Design 10-15 diverse personas (depth over breadth)
- [ ] Create segment families with variants
- [ ] Build counterfactual twins (same persona, one trait differs)
- [ ] Generate benchmark conversations (casual, interview, support, survey)
- [ ] Run full QA suite on all personas
- [ ] Document coherence scores + validation reports

## Phase 9: Documentation & Examples
- [ ] Write SDK usage guide
- [ ] Create persona authoring guide
- [ ] Document IR structure and validation logic
- [ ] Add example conversations
- [ ] Write QA suite interpretation guide
- [ ] Create "getting started" tutorial

## Future (Post-MVP)
- [ ] Optional: Reference FastAPI server (feature-flagged)
- [ ] Expand persona library (breadth)
- [ ] Add multi-language support
- [ ] Build admin dashboard
- [ ] Production API service
