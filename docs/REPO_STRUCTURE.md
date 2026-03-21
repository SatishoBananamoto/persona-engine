# Repository Structure

> Last updated: 2026-03-21

## Current Structure (post-reorganization)

```
persona-engine/
│
├── README.md                        # Project entry point
├── LICENSE                          # MIT license
├── pyproject.toml                   # Packaging + dependencies
├── requirements.txt                 # Pip dependencies
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contributor guide
├── CODE_OF_CONDUCT.md               # Community standards
├── ARCHITECTURE.md                  # Engine architecture overview
├── ARCHITECTURE-layer-zero.md       # Layer Zero architecture
├── GRAFT.md                         # Operation tracker (active reference)
│
├── persona_engine/                  # Core engine source
│   ├── __init__.py
│   ├── engine.py                    # PersonaEngine — main entry point
│   ├── persona_builder.py           # Fluent builder for Persona objects
│   ├── exceptions.py                # Custom exception hierarchy
│   ├── schema/
│   │   ├── persona_schema.py        # Persona data model (Pydantic)
│   │   └── ir_schema.py             # Intermediate Representation schema
│   ├── behavioral/                  # 16 behavioral interpreters
│   │   ├── trait_interpreter.py     # Big Five → behavioral parameters
│   │   ├── trait_interactions.py    # 9 emergent interaction patterns
│   │   ├── state_manager.py         # Dynamic state (mood, fatigue, stress)
│   │   ├── values_interpreter.py    # Schwartz values
│   │   ├── cognitive_interpreter.py # Cognitive style
│   │   ├── bias_simulator.py        # 8 cognitive biases
│   │   ├── social_cognition.py      # User modeling + adaptation
│   │   ├── emotional_appraisal.py   # Personality-dependent emotions
│   │   ├── linguistic_markers.py    # Interpolated character descriptions
│   │   ├── negation.py              # Negation handling
│   │   └── rules_engine.py          # Social roles + response patterns
│   ├── planner/                     # IR generation pipeline
│   │   ├── turn_planner.py          # Orchestrator (5-stage pipeline)
│   │   ├── stance_generator.py      # Compositional stance generation
│   │   ├── intent_analyzer.py       # Intent + mode detection
│   │   ├── domain_detection.py      # Domain + proficiency scoring
│   │   ├── domain_registry.py       # Domain keyword registry
│   │   ├── trace_context.py         # Citation trail
│   │   ├── engine_config.py         # Pipeline configuration
│   │   └── stages/                  # Mixin stage classes
│   │       ├── foundation.py        # Stage 1: trace setup, memory
│   │       ├── interpretation.py    # Stage 2: domain, intent, bias
│   │       ├── behavioral.py        # Stage 3: orchestrator
│   │       ├── behavioral_metrics.py # Stage 3a: confidence, elasticity
│   │       ├── behavioral_style.py  # Stage 3b: tone, verbosity, formality
│   │       ├── behavioral_guidance.py # Stage 3c: trait + cognitive guidance
│   │       ├── knowledge.py         # Stage 4: disclosure, claims, safety
│   │       ├── finalization.py      # Stage 5: IR assembly
│   │       └── stage_results.py     # Result dataclasses
│   ├── generation/                  # LLM text generation
│   │   ├── response_generator.py    # Response generation orchestrator
│   │   ├── prompt_builder.py        # IR → LLM prompt conversion
│   │   ├── llm_adapter.py           # 6 LLM provider adapters
│   │   └── style_modulator.py       # Post-generation enforcement
│   ├── memory/                      # Conversation memory
│   │   ├── memory_manager.py        # Memory orchestrator
│   │   ├── stance_cache.py          # Stance consistency cache
│   │   └── ...                      # Fact, preference, episode stores
│   ├── validation/                  # IR validation
│   │   └── cross_turn.py            # Cross-turn consistency checker
│   └── utils/
│       └── determinism.py           # Reproducible random state
│
├── layer_zero/                      # Persona Minting Machine
│   ├── __init__.py                  # mint(), from_description(), from_csv()
│   ├── models.py                    # MintRequest, MintedPersona
│   ├── sampler.py                   # Big Five sampling
│   ├── gap_filler.py                # Fill missing persona fields
│   ├── assembler.py                 # Build Persona objects
│   ├── validator.py                 # Validate coherence + diversity
│   ├── policy.py                    # Default policies
│   ├── diversity.py                 # Population diversity analysis
│   ├── evolution.py                 # Persona evolution
│   ├── export.py                    # Export formats
│   └── priors/                      # Demographic → psychology priors
│       ├── big_five.py              # Occupation → Big Five mapping
│       └── values.py                # Schwartz value generation
│
├── tests/                           # Test suite (2,649 tests)
├── personas/                        # 12 shipped persona YAML files
├── examples/                        # 11 usage examples
├── benchmarks/                      # Performance benchmarks
├── research/                        # Research reports
│   └── calibration_report.txt       # Psych literature calibration
│
├── eval/                            # Validation suites
│   ├── persona_eval.py              # 5 statistical suites (scipy)
│   ├── benchmark_profiles.py        # 8-profile direction checks
│   ├── dynamic_validation.py        # 15 multi-turn checks
│   ├── behavioral_text_validation.py # Real LLM text analysis
│   ├── end_to_end_conversation.py   # 3 E2E conversation tests
│   ├── correlation_analysis.py      # 212-persona correlation matrix
│   └── baseline_snapshot_*.json     # Performance baselines
│
└── docs/                            # Documentation
    ├── tutorial.md                  # Getting started
    ├── sdk_guide.md                 # Python SDK guide
    ├── persona_authoring.md         # Writing persona YAML files
    ├── layer_zero_guide.md          # Layer Zero usage
    ├── ir_reference.md              # IR field reference
    ├── PIPELINE_FLOWCHARTS.md       # Mermaid diagrams (pipeline + traits)
    ├── TRAIT_FLOW_ANALYSIS.md       # Per-field modifier chains
    ├── VALIDATION_SOURCES.md        # Papers + benchmark profiles
    ├── PSYCHOMETRIC_GROUNDING.md    # IPIP-NEO description mapping
    ├── REPO_STRUCTURE.md            # This file
    └── internal/                    # Historical planning/review docs
        ├── ROADMAP.md
        ├── IMPLEMENTATION_PLAN.md
        ├── IMPROVEMENT_PLAN.md
        ├── PSYCHOLOGICAL_REALISM_PLAN.md
        ├── AGENT_REVIEWS.md
        ├── MULTI_PERSPECTIVE_REVIEW.md
        ├── REVIEW.md
        └── ...
```

## Structure Changes Log

### Before merge (two branches)

**`main` (5 commits, stale since Feb 2026):**
- 26 root files including loose test files (test_*.py at root)
- Basic engine, no Layer Zero, no validation suites
- No docs/ directory

**`claude/analyze-test-coverage-d93F4` (147 commits):**
- Full engine with 16 interpreters, mixin architecture
- Layer Zero, FastAPI server, multi-provider adapters
- 27 root files — planning docs, review docs, debug scripts all at root

### After merge, before reorganization (148 commits)
- 27 root files — cluttered with development artifacts
- Planning/review docs mixed with project files
- Stale scripts (debug_test.py, semantic_enforcer.py) tracked
- Generated test results (test_results/) tracked

### After reorganization (current)
- **10 root files** — clean, stranger-friendly
- Planning/review docs → `docs/internal/` (13 files)
- Stale scripts removed (4 files)
- Generated artifacts removed from tracking (test_results/)
- Engineering docs organized in `docs/` (flowcharts, trait flow, validation, psychometric)

### What was removed from git tracking
These files were deleted from the repository (not moved):
- `debug_test.py` — one-off debug script
- `demo_full_pipeline.py` — superseded by examples/
- `demo_turn_planner.py` — superseded by examples/
- `semantic_enforcer.py` — stale, unused
- `test_results/*.txt` — generated artifacts (now gitignored)

### What was moved
| File | From | To |
|------|------|----|
| 13 planning/review/memory docs | root | docs/internal/ |

### What stayed at root
| File | Why |
|------|-----|
| README.md | Project entry point |
| LICENSE | Legal requirement |
| pyproject.toml | Packaging standard |
| requirements.txt | Dependency standard |
| CHANGELOG.md | Version history |
| CONTRIBUTING.md | Contributor guide |
| CODE_OF_CONDUCT.md | Community standard |
| ARCHITECTURE.md | Core architecture — important for understanding |
| ARCHITECTURE-layer-zero.md | Layer Zero architecture — paired with above |
| GRAFT.md | Active operation tracker — still referenced across sessions |
