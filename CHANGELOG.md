# Changelog

All notable changes to the Persona Engine project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-03-14

### Added
- FastAPI reference server (`persona_engine/server.py`) with full REST API
- GitHub Actions CI/CD pipeline (pytest, mypy, ruff across Python 3.11/3.12)
- Pre-commit hooks configuration (ruff, mypy, trailing-whitespace, YAML/JSON checks)
- Distributional guarantee tests (67 tests verifying trait markers across personas)
- Edge case tests (45 tests for unicode, emoji, boundaries, concurrency, strict mode)
- Server tests (23 tests for all API endpoints)
- Example scripts in `examples/` directory (quick start, multi-turn, builder, twins, IR)
- Shared test fixtures in `tests/conftest.py`
- CHANGELOG.md tracking all project phases
- FastAPI optional dependency (`pip install persona-engine[server]`)

### Changed
- Updated ROADMAP.md with accurate completion status for all phases
- Updated pyproject.toml with server dependencies

## [0.3.0] - 2026-03-14

### Added
- Memory reads wired into IR generation for context-aware planning (Phase A)
- Working strict mode for reduced-creativity, template-driven responses (Phase A)
- Persona library with diverse pre-built personas (Phase C)
- Counterfactual twins for controlled trait-comparison testing (Phase C)
- Benchmark conversation scenarios (casual, interview, support, survey) (Phase C)
- Trait scorer with exports for quantitative persona evaluation (Phase C)
- Structured logging throughout the pipeline (Phase D)
- Comprehensive project documentation (Phase E)

### Changed
- Domain sanitization for more robust domain matching (Phase B)
- Disclosure bounds validation to enforce clamping constraints (Phase B)

### Fixed
- LLM error handling for resilient API interactions (Phase B)

## [0.2.0] - 2024-12-15

### Added
- Custom exception hierarchy for clear, actionable error messages (Phase 3)
- Input validation across all public API surfaces (Phase 3)
- Developer experience improvements and SDK polish (Phase 4)
- Architecture and maintainability refactoring with 22 new tests (Phase 5)
- Behavioral fidelity and validation infrastructure (Phase 6)
- Python SDK with `Conversation` class for multi-turn dialogue (Phase 7)
- CLI tool for persona validation and inspection (Phase 7)
- Expanded SDK exports for YAML and JSON formats (Phase 7)
- Implementation plan with 4-agent independent codebase review
- Phase 6 plan for behavioral fidelity and validation

### Changed
- Updated IMPLEMENTATION_PLAN.md to reflect all 5 phases complete

### Fixed
- All critical bugs: 7 fixes with 33 new tests (Phase 1)
- Structural issues: 6 fixes with 27 new tests (Phase 2)
- Audit gaps in Phase 3 and Phase 4 implementations
- 6 pre-existing test failures resolved

## [0.1.0] - 2024-11-01

### Added
- Initial Persona Engine with Turn Planner and IR generation
- Big 5 personality trait system with behavior mappings
- Schwartz values system with conflict resolver
- Cognitive style interpreter
- Dynamic state manager (mood, fatigue, stress, engagement)
- Behavioral rules engine with decision policies and social role modes
- Bounded bias simulation with safety constraints
- Turn Planner orchestration with intent analyzer, stance generator, and disclosure calculator
- IR schema with full citation traceability (TraceContext)
- Uncertainty policy driven by proficiency and challenge level
- Phase 4 Memory System: four typed stores (facts, preferences, relationships, episodic) with orchestrating MemoryManager
- MemoryManager wired into pipeline (read before plan, write after IR)
- Phase 5 Response Generation: IR-to-text with Anthropic Claude integration
- LLM adapter abstraction (AnthropicAdapter, OpenAIAdapter, MockLLMAdapter)
- IR prompt builder with constraint formatting
- Style modulator (verbosity, formality, directness from IR)
- Phase 6 Validation Layer: three-tier IR validation system
- Competence dimension separating knowledge depth from certainty
- Chef persona, food domain registry, and 10-turn full trace
- PersonaEngine SDK: unified entry point for the full pipeline
- Multi-turn context and persistence with expanded domains
- 4 new personas with cross-turn dynamics (trust, disclosure, memory, competence)
- PersonaBuilder for creating personas without hand-crafted YAML
- Subdomain inference and YAML export
- Full pipeline demo script for end-to-end conversations
- 875+ tests for behavioral coherence, scenario coverage, and module coverage (95%)
- ROADMAP.md with project status and phase tracking
- ARCHITECTURE.md as comprehensive design document

### Changed
- Consolidated response/ and generation/ modules into unified structure

### Fixed
- 12 bugs across two code review rounds (undefined enum, missing imports, type errors, indentation, duplicate code, None seed)
- All mypy errors resolved (41 to 0)
- Ruff autofix applied (1069 to 61 issues)
