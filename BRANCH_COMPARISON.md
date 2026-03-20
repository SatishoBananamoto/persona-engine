# Branch Comparison — persona-engine

> Generated 2026-03-20. Complete inventory of all branches and their contents.

## Branch Lineage

```
main (5 commits)
  └── claude-md (setup, +1 commit)
  └── explore-repo-KZTBj (+42 commits)
        ├── external-review (+49 commits)
        │     └── general-session-VUY6r (+8 commits — provider benchmarks, stage split)
        │     └── analyze-test-coverage-d93F4 (+17 commits — merge + review phases)
        └── review/tier1-bugfixes (+13 commits — THIS SESSION)
```

## Branch Summary

| Branch | Commits | Files | Python LOC | Last Updated | Status |
|--------|---------|-------|------------|--------------|--------|
| `main` | 5 | 55 | 11,683 | 2026-02-06 | Initial MVP, stale |
| `claude-md` | 3 | 47 | 9,688 | 2026-02-06 | CLAUDE.md only, stale |
| `explore-repo-KZTBj` | 47 | 130 | 40,145 | 2026-03-13 | Phase 1-7 complete |
| `external-review` | 91 | 222 | 55,130 | 2026-03-15 | Layer Zero + psych realism |
| `general-session-VUY6r` | 72 | 186 | 51,491 | 2026-03-17 | Multi-provider + stage split |
| `analyze-test-coverage-d93F4` | 108 | 242 | 57,761 | 2026-03-19 | **Most complete** |
| `review/tier1-bugfixes` | 60 | 132 | 40,283 | 2026-03-20 | This session's work |

---

## What Each Branch Has

### `main` — The Starting Point
- Initial persona engine with turn planner
- Basic response generator with Anthropic
- 6 bug fixes + behavioral coherence tests
- ROADMAP.md
- **Nothing else. Stale since Feb 6.**

### `claude-md` — Setup Only
- Just adds CLAUDE.md for AI assistant onboarding
- **Can be ignored.**

### `explore-repo-KZTBj` — Foundation (Phases 1-7)
Everything in main, plus:
- Phase 1: Foundation & schema (Pydantic v2 IR, persona schema)
- Phase 2: Bias simulator, domain detection, evidence strength
- Phase 3: Exception hierarchy, save/load (v2), input validation
- Phase 4: Repr methods, context manager, from_description
- Phase 5: Cross-turn dynamics, stance cache, inertia smoothing
- Phase 6: Behavioral fidelity, uncertainty resolver, Dunning-Kruger
- Phase 7: Python SDK — Conversation class, CLI tool
- 8 persona YAML files (chef, lawyer, musician, physicist, ux_researcher, etc.)
- ARCHITECTURE.md (1,209 lines)
- 35 enum values across 6 enums
- ~1,833 tests

### `external-review` — Psychological Realism + Layer Zero
Everything in explore-repo, plus:

**Psychological Realism (Phases R1-R7):**
- R1: Wired ALL orphaned interpreter methods into pipeline
- R2: Sigmoid activation, Dunning-Kruger curve, amplified effect sizes
- R3: Trait interactions engine — 9 emergent personality patterns
- R4: Emotional appraisal engine — personality-dependent emotions
- R5: LIWC markers, composable stance, stochastic expression
- R6: Social cognition, cognitive biases, self-schema protection
- R7: Psychometric validation test suite

**Production Hardening (Phases A-E, 10):**
- A: Wire memory reads into IR, strict mode
- B: LLM error handling, domain sanitization, disclosure bounds
- C: Persona library, counterfactual twins, benchmarks, trait scorer
- D+E: Structured logging, trait scorer exports, documentation
- Phase 10: CI/CD, FastAPI server, 135 new tests

**Layer Zero (Persona Minting Machine):**
- Architecture doc + implementation plan
- 12-phase implementation (models, parser, priors, sampler, circumplex, gap filler, policy, validator, assembler, API)
- Cognitive prior engine, occupation profiles
- Diversity analysis, population alignment, persona evolution
- v0.4.0, README rewritten
- ~292 tests

### `general-session-VUY6r` — Multi-Provider + Architecture
Everything in external-review, plus:
- Split TurnPlanner into 5 stage classes + orchestrator (mixin architecture)
- Typed stage method returns with dataclasses
- Text-level behavioral validation tests
- Negation handling, response/ removal, write_policy
- Multi-provider LLM adapters: Gemini, Mistral, Groq, Ollama, OpenAI-compatible
- End-to-end flow tests for multi-provider
- Provider benchmark: compare LLM providers on same persona conversations

### `analyze-test-coverage-d93F4` — **MOST COMPLETE**
Everything in general-session-VUY6r, plus:
- Merge of external-review (Layer Zero, diversity, evolution)
- 5-agent multi-perspective review of merged codebase
- Fix 5 critical issues from review
- Conversation tests, path traversal tests, modern type hints
- Phase A: Refactor into mixin modules (behavioral_metrics, behavioral_style, behavioral_guidance, knowledge)
- Phase B: Server hardening
- Phase C: Community readiness
- Phase D: Caricature prevention
- ~2,822 tests

### `review/tier1-bugfixes` — This Session (WRONG BASE)
Branched from explore-repo (NOT from analyze-test-coverage). Contains:
- 10 bug fixes (some unique: mood drift inversion, MemoryError shadow, state evolution timing)
- Structured logging + LLM exception handling
- Save/load round-trip fix (v3 with state/stance/snapshot)
- README rewrite + repo cleanup
- Stance generator rewrite (compositional, VALUE_TOPIC_TABLE)
- Calibration against psych literature (11 parameters, research report)
- Automated eval suite (5 statistical suites — scipy)
- Behavioral wiring audit

---

## Feature Comparison Matrix

| Feature | main | explore | external | general | analyze | review (ours) |
|---------|------|---------|----------|---------|---------|---------------|
| Basic IR pipeline | x | x | x | x | x | x |
| Bias simulator | | x | x | x | x | x |
| Cross-turn dynamics | | x | x | x | x | x |
| Save/load persistence | | x (v2) | x (v2) | x (v2) | x (v2) | x (**v3**) |
| Exception hierarchy | | x | x | x | x | x |
| Structured logging | | | x | x | x | x |
| LLM error handling | | | x | x | x | x |
| Dead methods wired | | | **x** (R1) | **x** | **x** | |
| Sigmoid activation / DK curve | | | **x** (R2) | **x** | **x** | |
| Trait interactions (9 patterns) | | | **x** (R3) | **x** | **x** | |
| Emotional appraisal | | | **x** (R4) | **x** | **x** | |
| LIWC markers / stochastic expr | | | **x** (R5) | **x** | **x** | |
| Social cognition / biases | | | **x** (R6) | **x** | **x** | |
| Psychometric validation | | | **x** (R7) | **x** | **x** | |
| Layer Zero (persona minting) | | | **x** | | **x** | |
| FastAPI server | | | **x** | | **x** | |
| CI/CD | | | **x** | | **x** | |
| Persona library / twins | | | **x** | | **x** | |
| Multi-provider adapters | | | | **x** | **x** | |
| Stage split (mixins) | | | | **x** | **x** | |
| Provider benchmarks | | | | **x** | **x** | |
| Caricature prevention | | | | | **x** | |
| Community readiness | | | | | **x** | |
| Psych literature calibration | | | | | | **x** (unique) |
| Calibration research report | | | | | | **x** (unique) |
| Compositional stance gen | | | | | | **x** (unique, conflicts with R5) |
| Automated eval suite (scipy) | | | | | | **x** (unique) |
| Save/load v3 (state+stance) | | | | | | **x** (unique) |
| Behavioral wiring audit | | | | | | **x** (unique) |
| Mood drift inversion fix | | | | | | **x** (check if bug exists on other branches) |

---

## Recommended Path Forward

**Base branch**: `analyze-test-coverage-d93F4` — it has everything.

**Cherry-pick from `review/tier1-bugfixes`** (our unique work):
1. Eval suite (new files, clean)
2. Calibration research report (new file)
3. Calibration code changes (check for conflicts with R2 sigmoid/DK values)
4. Save/load v3 (check engine.py differences)
5. Stance generator rewrite (CONFLICTS with R5 composable stance — manual merge)
6. Mood drift inversion fix (verify bug exists on target)
7. Wiring audit doc (update to reflect target branch state)

**Branches to archive/delete after merge:**
- `main` — superseded
- `claude-md` — superseded
- `explore-repo-KZTBj` — superseded by analyze-test-coverage
- `external-review` — merged into analyze-test-coverage
- `general-session-VUY6r` — merged into analyze-test-coverage
- `review/tier1-bugfixes` — cherry-picked into analyze-test-coverage

**Final state**: One clean branch with everything, ready for PyPI publish as `anima`.
