# persona-engine — Work Tracker

> Single source of truth for what's done, what's in progress, and what's pending.
> Updated as work happens. If it's not here, it didn't happen.

---

## Completed

### Tier 1 — Bug Fixes (10)
- [x] Fix NameError crash in style_modulator for ANECDOTAL claim type
- [x] Fix state evolution timing (evolve_state_post_turn moved to after IR finalization)
- [x] Fix mood_valence/arousal unbounded drift (add clamping after noise)
- [x] Fix TypeError crash when elasticity is None in cross-turn smoothing
- [x] Fix stance cache collision (derive topic signature from input content words)
- [x] Rename MemoryError to PersonaMemoryError (avoid shadowing Python builtin)
- [x] Fix MINIMAL verbosity always triggering warning (max_sentences=0)
- [x] Fix BehavioralRulesEngine permanently mutating persona's social_roles dict
- [x] Fix pre-existing test failure (test_high_stress_lowers_effective_confidence)
- [x] Fix "american" → "American" casing in persona_builder culture_map

### Tier 2 — Production Hardening
- [x] Add structured logging (engine.py, llm_adapter.py, response_generator.py, turn_planner.py)
- [x] Add LLM exception handling (Anthropic/OpenAI calls wrapped, typed exception mapping)
- [x] Fix save/load round-trip (serialize dynamic state, stance cache, prior snapshot — v3 format)

### Tier 3 — Packaging & Cleanup
- [x] Sync version to 0.2.0 (pyproject.toml ↔ __init__.py)
- [x] Fix placeholder URLs → SatishoBananamoto
- [x] Fix author name → Satish Patil
- [x] Fix requirements.txt (was listing beautifulsoup4/requests from wrong project)
- [x] Add hypothesis to dev dependencies
- [x] Remove --cov from pytest addopts (was crashing test runs)

### Path A — Ship It (Make It Usable)
- [x] Rewrite README ("pytest for AI personality", IR-first pitch)
- [x] Switch all examples from mock to template adapter
- [x] Promote from_description() as quickstart entry point
- [x] Move internal docs to docs/internal/ (IMPROVEMENT_PLAN, ROADMAP, etc.)
- [x] Move ARCHITECTURE.md to docs/
- [x] Clean repo root (only README, pyproject.toml, requirements.txt, .gitignore remain)

### Path C — Research-Level Improvements
- [x] Rewrite stance generator (compositional, topic-grounded, value-driven)
- [x] Calibrate behavioral parameters against psych literature (11 parameters, all applied)
  - [x] Openness elasticity 0.7→0.6
  - [x] Negative tone bias 0.7→0.5
  - [x] Extraversion added to verbosity
  - [x] Mood drift rate inverted (critical bug fix)
  - [x] Stress decay made trait-dependent
  - [x] Extraversion added to baseline valence
  - [x] Confidence modifier: N penalty 0.15→0.20, C boost 0.10→0.15
  - [x] Hedging: add N component (+N*0.2)
  - [x] Proactivity: floor/ceiling 0.2+E*0.6
  - [x] Self-disclosure: add N component (+N*0.1)
  - [x] Enthusiasm baseline: floor/ceiling 0.2+E*0.5
- [x] Build automated evaluation suite (5 statistical suites, all passing)

### Quick Fixes (from remaining review findings)
- [x] Add SMALL_TALK to intent analyzer mode_keywords
- [x] Fix input validation order (sanitize before length check)
- [x] Add thread safety documentation to PersonaEngine docstring
- [x] Remove dead response/ module (was never used by engine.py)
- [x] Update test_enum_coverage.py (removed dead response/ imports)

---

## In Progress

- [ ] Rename package to Anima for PyPI publish (parked — deciding scope of rename)

---

## Pending

### PyPI Publishing
- [ ] Decide rename scope: distribution name only vs full module rename
- [ ] Rename package to `anima`
- [ ] Publish v0.2.0 to PyPI
- [ ] Verify `pip install anima` works

### CI/CD
- [ ] Set up GitHub Actions (test on push, lint, mypy)
- [ ] Add eval suite to CI pipeline

### Remaining Technical Debt
- [ ] Add retry/backoff on LLM calls (needs design decision on retry policy)
- [ ] Fix save() accessing private store internals (add public export API to stores)

### Research-Level (Multi-Day)
- [ ] Expand bias simulator beyond 3 biases
- [ ] Topic relevance: token-overlap → semantic (needs embedding model)
- [ ] Memory consolidation/forgetting (reconstructive memory, Ebbinghaus curve)
- [ ] Personality-language output validation (verify LLM text matches IR constraints)

### Strategic
- [ ] Pick primary vertical (chatbot persona testing recommended by Product Strategist)
- [ ] Write case study / demo showing IR catching persona drift
- [ ] Launch on HN / r/LocalLLaMA

---

## Review Agents Summary

Five perspectives ran on 2026-03-20. Reports available in agent output logs.

| Agent | Key Verdict |
|-------|-------------|
| Enterprise Architect | 2/9 production criteria passed → now improved significantly |
| Adversarial QA | Found 4 critical + 7 high bugs → all critical/high fixed |
| AI Researcher | Borderline workshop accept → stance rewrite + calibration + eval suite improve this |
| Developer Adopter | First impression 7/10 → README rewrite + template examples improve onboarding |
| Product Strategist | Product clarity 4/10, Market need 7/10 → README + positioning addressed |

---

## Test Status

- **1822 tests passing, 0 failures** (as of last commit)
- **5/5 eval suites passing**
- Test count decreased from 2034 by removal of 212 tests for dead response/ module

---

## Branch

`review/tier1-bugfixes` — 11 commits, PR #1 open at:
https://github.com/SatishoBananamoto/persona-engine/pull/1
