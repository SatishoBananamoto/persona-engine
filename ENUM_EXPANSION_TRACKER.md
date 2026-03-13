# Enum Expansion Tracker

## Summary
- **Total new enum values**: 35 across 6 enums
- **Total files affected**: ~15 source files + ~6 test files
- **Branch**: `claude/explore-repo-KZTBj`

---

## Phase 0: Safety Net Tests
| # | File | Change | Status |
|---|------|--------|--------|
| 0.1 | `tests/test_enum_coverage.py` | NEW — exhaustive coverage tests for all consumer dicts | [x] |

---

## Phase 1: Tone (9) + Verbosity (1)
| # | File | Change | Status |
|---|------|--------|--------|
| 1.1 | `persona_engine/schema/ir_schema.py` | Add 9 Tone members + MINIMAL Verbosity | [x] |
| 1.2 | `persona_engine/behavioral/trait_interpreter.py` | Extend `get_tone_from_mood()` + `influences_verbosity()` | [x] |
| 1.3 | `persona_engine/generation/prompt_builder.py` | `_format_tone()` + `_format_verbosity()` dicts | [x] |
| 1.4 | `persona_engine/response/prompt_builder.py` | `TONE_PROMPTS` + `VERBOSITY_PROMPTS` dicts | [x] |
| 1.5 | `persona_engine/generation/style_modulator.py` | `VERBOSITY_TARGETS` + `enforce_verbosity()` for MINIMAL | [x] |
| 1.6 | `persona_engine/response/adapters.py` | `_OPENERS` for 9 tones + MINIMAL short-circuit | [x] |
| 1.7 | `persona_engine/planner/turn_planner.py` | MINIMAL guard in `_compute_verbosity()` | [x] |

---

## Phase 2: UncertaintyAction (5) + KnowledgeClaimType (5)
| # | File | Change | Status |
|---|------|--------|--------|
| 2.1 | `persona_engine/schema/ir_schema.py` | Add 5 UncertaintyAction + 5 KnowledgeClaimType members | [x] |
| 2.2 | `persona_engine/behavioral/uncertainty_resolver.py` | Extend `resolve_uncertainty_action()` + `infer_knowledge_claim_type()` | [x] |
| 2.3 | `persona_engine/generation/prompt_builder.py` | `_format_uncertainty()` + `_format_claim_type()` dicts | [x] |
| 2.4 | `persona_engine/response/prompt_builder.py` | `UNCERTAINTY_PROMPTS` + `CLAIM_TYPE_PROMPTS` dicts | [x] |
| 2.5 | `persona_engine/validation/ir_validator.py` | Coherence rules for new actions/claims | [x] |
| 2.6 | `persona_engine/validation/knowledge_boundary.py` | Boundary rules for new claim types | [x] |
| 2.7 | `persona_engine/generation/style_modulator.py` | Hedging checks for new claim types | [x] |
| 2.8 | `persona_engine/response/adapters.py` | Template text for 5 new uncertainty actions | [x] |
| 2.9 | `persona_engine/planner/turn_planner.py` | Wire new params to resolver + detection helpers | [x] |

---

## Phase 3: InteractionMode (7) + ConversationGoal (6)
| # | File | Change | Status |
|---|------|--------|--------|
| 3.1 | `persona_engine/schema/ir_schema.py` | Add 7 InteractionMode + 6 ConversationGoal members | [x] |
| 3.2 | `persona_engine/planner/intent_analyzer.py` | Keyword detection for 13 new values | [x] |
| 3.3 | `persona_engine/planner/turn_planner.py` | Mode overlays, tone/verbosity overrides | [x] |
| 3.4 | `persona_engine/generation/prompt_builder.py` | `_get_mode_instructions()` + AVOID_ENGAGE handling | [x] |
| 3.5 | `persona_engine/behavioral/rules_engine.py` | Social role mappings for 7 new modes | [x] |
| 3.6 | `persona_engine/response/adapters.py` | VENTING suppression of solution-seeking | [x] |

---

## Phase 4: Tests
| # | File | Change | Status |
|---|------|--------|--------|
| 4.1 | `tests/test_behavioral_coherence.py` | Expand hardcoded `negative_tones` sets | [ ] |
| 4.2 | `tests/test_trait_interpreter.py` | Fixed 5 tests for new tone/verbosity behavior | [x] |
| 4.3 | `tests/test_uncertainty_resolver.py` | Fixed 16 tests for new action/claim branches | [x] |
| 4.4 | `tests/test_intent_analyzer.py` | Keyword detection tests for 7 modes + 6 goals | [ ] |
| 4.5 | Full test suite | `pytest` green (369 tests passing) | [x] |

---

## Completion Log
| Phase | Date | Notes |
|-------|------|-------|
| Phase 0 | 2026-03-13 | Safety net tests created — 128 parametrized tests |
| Phase 1 | 2026-03-13 | 9 tones + MINIMAL verbosity across all consumers |
| Phase 2 | 2026-03-13 | 5 uncertainty actions + 5 claim types across all consumers |
| Phase 3 | 2026-03-13 | 7 interaction modes + 6 conversation goals |
| Phase 4 | 2026-03-13 | All existing tests fixed and passing (369 total) |
