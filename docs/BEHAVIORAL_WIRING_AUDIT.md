# Behavioral Interpreter Wiring Audit

> Conducted 2026-03-20. Traces every public method in TraitInterpreter, ValuesInterpreter, and CognitiveStyleInterpreter through `TurnPlanner.generate_ir()` to the final IR.

---

## Part 1: Three Method Traces (generate_ir → IR field)

### Trace A: `TraitInterpreter.get_elasticity(base_confidence)`

```
generate_ir()
  → _stage_behavioral_metrics()          [turn_planner.py:318]
    → _compute_elasticity(proficiency)   [turn_planner.py:775]
      → self.traits.get_elasticity(proficiency)  [turn_planner.py:786]
        Returns: float (0.1-0.9)
      → Blended with cognitive.get_elasticity_from_cognitive_style()  [turn_planner.py:795-808]
      → Modified by bias_simulator confirmation bias  [turn_planner.py:811-827]
      → Clamped to [ELASTICITY_MIN, ELASTICITY_MAX]  [turn_planner.py:831-839]
      Returns: float
    → Cross-turn inertia smoothed  [turn_planner.py:321-335]
    → Returned in metrics["elasticity"]  [turn_planner.py:431]
  → _stage_finalization()
    → IR.response_structure.elasticity  [turn_planner.py:624]
```

**Classification: LIVE** — reaches `IR.response_structure.elasticity`. Full citation trail.

---

### Trace B: `ValuesInterpreter.get_top_values(n=3)`

```
generate_ir()
  → _stage_behavioral_metrics()
    → _generate_stance()  [turn_planner.py:348]
      → generate_stance_safe()  [stance_generator.py:253]
        → values.get_top_values(n=3)  [stance_generator.py:256]
          Returns: [(value_name, weight), ...]
        → Used to select VALUE_TOPIC_TABLE entries  [stance_generator.py:258-263]
        → Primary value drives stance fragment  [stance_generator.py:258]
        → Secondary value drives nuance qualifier  [stance_generator.py:268-280]
        → Both appear in rationale  [stance_generator.py:285-289]
      Returns: (stance_str, rationale_str)
    → Returned in metrics["stance"], metrics["rationale"]  [turn_planner.py:432-433]
  → _stage_finalization()
    → IR.response_structure.stance  [turn_planner.py:621]
    → IR.response_structure.rationale  [turn_planner.py:622]
```

**Classification: LIVE** — reaches `IR.response_structure.stance` and `.rationale`.

Also called during `__init__`:
```
TurnPlanner.__init__()
  → self.values.get_value_priorities()  [turn_planner.py:149]
    → Passed to BiasSimulator constructor
```

---

### Trace C: `CognitiveStyleInterpreter.get_confidence_adjustment(base_confidence)`

```
generate_ir()
  → _stage_behavioral_metrics()
    → _compute_confidence(proficiency)  [turn_planner.py:358]
      → self.traits.get_confidence_modifier(confidence)  [turn_planner.py:863]
        Returns: float (trait-adjusted confidence)
      → self.cognitive.get_confidence_adjustment(confidence)  [turn_planner.py:877]
        Returns: float (cognitive-adjusted confidence)
        Citation recorded at turn_planner.py:878-888
      → Modified by authority bias  [turn_planner.py:891-907]
      → Clamped to [0, 1]  [turn_planner.py:910-915]
      Returns: float
    → Cross-turn inertia smoothed  [turn_planner.py:359-373]
    → Returned in metrics["confidence"]  [turn_planner.py:434]
  → _stage_finalization()
    → IR.response_structure.confidence  [turn_planner.py:625]
```

**Classification: LIVE** — reaches `IR.response_structure.confidence`.

---

## Part 2: IR Field → Interpreter Method Mapping (Backward Trace)

### IR.response_structure fields

| IR Field | Interpreter Methods Contributing | Pipeline Stage |
|----------|--------------------------------|----------------|
| `.intent` | None directly — assembled by `_generate_intent()` from `user_intent` + `conversation_goal` + `uncertainty_action` | Stage 5 |
| `.stance` | `ValuesInterpreter.get_top_values()`, `ValuesInterpreter.resolve_conflict_detailed()`, `CognitiveStyleInterpreter.get_reasoning_approach()` (via stance_generator) | Stage 3 |
| `.rationale` | `ValuesInterpreter.get_top_values()` (via stance_generator) | Stage 3 |
| `.elasticity` | `TraitInterpreter.get_elasticity()`, `CognitiveStyleInterpreter.get_elasticity_from_cognitive_style()` | Stage 3 |
| `.confidence` | `TraitInterpreter.get_confidence_modifier()`, `CognitiveStyleInterpreter.get_confidence_adjustment()` | Stage 3 |
| `.competence` | **RAW ACCESS**: `self.persona.psychology.big_five.openness` at turn_planner.py:976 — bypasses TraitInterpreter | Stage 3 |

### IR.communication_style fields

| IR Field | Interpreter Methods Contributing | Pipeline Stage |
|----------|--------------------------------|----------------|
| `.tone` | `TraitInterpreter.get_tone_from_mood()` | Stage 3 |
| `.verbosity` | `TraitInterpreter.influences_verbosity()` | Stage 3 |
| `.formality` | **No interpreter method** — base from `persona.psychology.communication.formality`, modified by `BehavioralRulesEngine.apply_social_role_adjustments()` | Stage 3 |
| `.directness` | `TraitInterpreter.influences_directness()` | Stage 3 |

### IR.knowledge_disclosure fields

| IR Field | Interpreter Methods Contributing | Pipeline Stage |
|----------|--------------------------------|----------------|
| `.disclosure_level` | `TraitInterpreter.get_self_disclosure_modifier()` | Stage 4 |
| `.uncertainty_action` | **RAW ACCESS**: `self.cognitive.style.risk_tolerance`, `self.cognitive.style.need_for_closure` at turn_planner.py:482-483 — bypasses CognitiveStyleInterpreter methods | Stage 4 |
| `.knowledge_claim_type` | None directly from interpreters — computed by `_infer_claim_type()` | Stage 4 |

### IR.conversation_frame fields

| IR Field | Interpreter Methods Contributing | Pipeline Stage |
|----------|--------------------------------|----------------|
| `.interaction_mode` | None — from `analyze_intent()` or user override | Stage 2 |
| `.goal` | None — from `analyze_intent()` or user override | Stage 2 |
| `.success_criteria` | None — from `_derive_success_criteria()` using persona goals | Stage 5 |

---

## Part 2b: Interpreter Methods That DON'T Contribute to Any IR Field

### TraitInterpreter — DEAD/UNUSED Methods

| Method | Classification | Evidence |
|--------|---------------|----------|
| `influences_abstract_reasoning()` | **DEAD** | Zero calls anywhere in turn_planner.py or pipeline. Grep confirms: only called in tests and `get_trait_markers_for_validation()`. |
| `get_novelty_seeking()` | **DEAD** | Zero calls in pipeline. Only referenced in tests. Returns raw `openness` — identity transform. |
| `get_planning_language_tendency()` | **DEAD** | Zero calls in pipeline. Returns raw `conscientiousness`. |
| `get_follow_through_likelihood()` | **DEAD** | Zero calls in pipeline. Returns raw `conscientiousness`. |
| `influences_proactivity()` | **DEAD** | Zero calls in pipeline. Not used by any stage. |
| `influences_response_length_social()` | **DEAD** | Zero calls in pipeline. Returns raw `extraversion`. |
| `get_enthusiasm_baseline()` | **DEAD** | Zero calls in pipeline. `_select_tone()` uses `get_tone_from_mood()` which accesses `self.traits.extraversion` directly inside its own logic. |
| `get_validation_tendency()` | **DEAD** | Zero calls in pipeline. Returns raw `agreeableness`. |
| `get_conflict_avoidance()` | **DEAD** | Zero calls in pipeline. Returns raw `agreeableness`. |
| `influences_hedging_frequency()` | **DEAD** | Zero calls in pipeline. Hedging is controlled by `uncertainty_action` not by this method. |
| `get_stress_sensitivity()` | **DEAD** | Zero calls in pipeline. StateManager accesses `traits.neuroticism` directly. |
| `influences_mood_stability()` | **DEAD** | Zero calls in pipeline. Returns `1.0 - neuroticism`. StateManager computes mood drift rate from raw traits. |
| `get_anxiety_baseline()` | **DEAD** | Zero calls in pipeline. Returns raw `neuroticism`. |
| `get_negative_tone_bias()` | **DEAD** | Zero calls in pipeline. `get_tone_from_mood()` uses `self.traits.neuroticism` directly for negative tone decisions. |
| `get_trait_markers_for_validation()` | **VALIDATION-ONLY** | Called by validators, not by generate_ir(). |

**Summary: 14 of 19 TraitInterpreter methods are DEAD in the pipeline. Only 5 are LIVE:**
- `get_elasticity()` → IR.response_structure.elasticity
- `get_confidence_modifier()` → IR.response_structure.confidence
- `get_tone_from_mood()` → IR.communication_style.tone
- `influences_verbosity()` → IR.communication_style.verbosity
- `influences_directness()` → IR.communication_style.directness
- `get_self_disclosure_modifier()` → IR.knowledge_disclosure.disclosure_level

(That's 6 actually — my count above was wrong. 13 dead, 6 live.)

### ValuesInterpreter — DEAD/UNUSED Methods

| Method | Classification | Evidence |
|--------|---------------|----------|
| `get_value_priorities()` | **LIVE** | Called in `TurnPlanner.__init__()` to initialize BiasSimulator. |
| `get_top_values()` | **LIVE** | Called by `generate_stance_safe()`. |
| `detect_value_conflicts()` | **DEAD** | Zero calls in pipeline. Only in `get_value_markers_for_validation()`. |
| `resolve_conflict()` | **DEAD** | Zero calls in pipeline. Wrapper for `resolve_conflict_detailed()`. |
| `resolve_conflict_detailed()` | **LIVE** | Called by `generate_stance_safe()` for nuance qualifier. |
| `get_value_influence_on_stance()` | **STUB** | Returns `{option: 0.5 for option in options}` — hardcoded placeholder. Comment says "to be enhanced with LLM or embedding matching." Zero calls in pipeline. |
| `get_rationale_influences()` | **DEAD** | Zero calls in pipeline. |
| `should_include_in_citation()` | **DEAD** | Zero calls in pipeline. |
| `get_value_markers_for_validation()` | **VALIDATION-ONLY** | Called by validators. |

**Summary: 3 LIVE, 4 DEAD, 1 STUB, 1 VALIDATION-ONLY.**

### CognitiveStyleInterpreter — DEAD/UNUSED Methods

| Method | Classification | Evidence |
|--------|---------------|----------|
| `get_reasoning_approach()` | **LIVE** | Called by `generate_stance_safe()` for competence frame selection. |
| `get_rationale_depth()` | **DEAD** | Zero calls in pipeline. The rationale string is built by stance_generator, not sized by this. |
| `prefers_systematic_processing()` | **DEAD** | Zero calls in pipeline. |
| `get_decision_time_modifier()` | **DEAD** | Zero calls in pipeline. |
| `get_risk_stance_modifier()` | **DEAD** | Zero calls in pipeline. Risk tolerance is accessed RAW at turn_planner.py:482. |
| `influences_uncertainty_action()` | **BYPASSED** | `resolve_uncertainty_action()` in `uncertainty_resolver.py` duplicates this exact logic with MORE parameters (stress, fatigue, competence). This method exists but is never called — the resolver does the same thing better. |
| `get_ambiguity_tolerance()` | **DEAD** | Zero calls in pipeline. |
| `prefers_definite_answers()` | **DEAD** | Zero calls in pipeline. |
| `get_nuance_capacity()` | **DEAD** | Zero calls in pipeline. `generate_stance_safe()` accesses `cognitive.style.cognitive_complexity` directly. |
| `should_acknowledge_tradeoffs()` | **DEAD** | Zero calls in pipeline. Stance generator does its own cognitive_complexity > 0.7 check. |
| `get_stance_complexity_level()` | **DEAD** | Zero calls in pipeline. |
| `get_elasticity_from_cognitive_style()` | **LIVE** | Called in `_compute_elasticity()`. |
| `get_confidence_adjustment()` | **LIVE** | Called in `_compute_confidence()`. |
| `get_cognitive_markers_for_validation()` | **VALIDATION-ONLY** | Called by validators. |

**Summary: 3 LIVE, 9 DEAD, 1 BYPASSED, 1 VALIDATION-ONLY.**

---

## Part 3: Raw Persona Field Access Bypassing Interpreters

### Confirmed Bypasses (interpreter method exists but raw field used instead)

| Location | Raw Access | Interpreter Method That Should Be Used | Severity |
|----------|-----------|---------------------------------------|----------|
| `turn_planner.py:976` | `self.persona.psychology.big_five.openness` (competence calculation) | `TraitInterpreter.get_novelty_seeking()` returns same value but provides abstraction | LOW — identity transform, no logic difference |
| `turn_planner.py:482-483` | `self.cognitive.style.risk_tolerance`, `self.cognitive.style.need_for_closure` | `CognitiveStyleInterpreter.get_risk_stance_modifier()` and `CognitiveStyleInterpreter.influences_uncertainty_action()` | **HIGH** — `influences_uncertainty_action()` encapsulates the same decision logic that `resolve_uncertainty_action()` duplicates. The resolver is more capable but bypasses the interpreter entirely. |
| `turn_planner.py:141-147` | All 5 Big Five traits extracted raw for BiasSimulator | No specific method — BiasSimulator needs all 5 as a dict. Interpreter wraps individual traits, not batches. | LOW — constructor initialization, not per-turn logic |
| `turn_planner.py:873` | `self.persona.psychology.big_five.conscientiousness/neuroticism` in citation reason string | Just for logging/citation text — `get_confidence_modifier()` is properly called on line 863 | NONE — display only |
| `turn_planner.py:1235/1237` | `self.persona.psychology.big_five.agreeableness` in citation strings | Just for logging — `influences_directness()` is properly called on line 1226 | NONE — display only |
| `turn_planner.py:1315` | `self.persona.psychology.big_five.extraversion` in citation string | Just for logging — `get_self_disclosure_modifier()` is properly called on line 1305 | NONE — display only |
| `stance_generator.py:274` | `cognitive.style.cognitive_complexity` accessed directly | `CognitiveStyleInterpreter.get_nuance_capacity()` or `.should_acknowledge_tradeoffs()` exist for this | MEDIUM — the interpreter's threshold logic (>0.7 = "high") is reimplemented inline |

---

## Part 4: Hardcoded/Identity/Stub Methods

| Method | Issue | Assessment |
|--------|-------|------------|
| `TraitInterpreter.get_novelty_seeking()` | Returns `self.traits.openness` unchanged | **Identity transform** — adds no value over raw access |
| `TraitInterpreter.get_planning_language_tendency()` | Returns `self.traits.conscientiousness` unchanged | **Identity transform** |
| `TraitInterpreter.get_follow_through_likelihood()` | Returns `self.traits.conscientiousness` unchanged | **Identity transform** |
| `TraitInterpreter.influences_proactivity()` | Returns `0.2 + self.traits.extraversion * 0.6` | **Calibrated transform** but never called. Proactivity is not an IR field. |
| `TraitInterpreter.influences_response_length_social()` | Returns `self.traits.extraversion` unchanged | **Identity transform** |
| `TraitInterpreter.get_enthusiasm_baseline()` | Returns `0.2 + self.traits.extraversion * 0.5` | **Calibrated transform** but never called. Enthusiasm is embedded in `get_tone_from_mood()`. |
| `TraitInterpreter.get_validation_tendency()` | Returns `self.traits.agreeableness` unchanged | **Identity transform** |
| `TraitInterpreter.get_conflict_avoidance()` | Returns `self.traits.agreeableness` unchanged | **Identity transform** |
| `TraitInterpreter.get_stress_sensitivity()` | Returns `self.traits.neuroticism` unchanged | **Identity transform** |
| `TraitInterpreter.get_anxiety_baseline()` | Returns `self.traits.neuroticism` unchanged | **Identity transform** |
| `CognitiveStyleInterpreter.get_ambiguity_tolerance()` | Returns `1.0 - self.style.need_for_closure` | **Simple inverse** — never called |
| `ValuesInterpreter.get_value_influence_on_stance()` | Returns `{option: 0.5 for option in options}` | **STUB** — hardcoded equal scores, comment says "to be enhanced" |

---

## Part 5: Interpreter Outputs With No IR Field / IR Fields That Skip Interpreters

### Interpreter Outputs With No Corresponding IR Field

| Interpreter Output | Concept | IR Gap |
|-------------------|---------|--------|
| `influences_proactivity()` | Whether persona initiates | No IR field for proactivity/initiative |
| `get_enthusiasm_baseline()` | Enthusiasm level | Subsumed into tone selection, no explicit field |
| `get_rationale_depth()` | Number of reasoning steps | IR.rationale is a string, no depth parameter |
| `get_decision_time_modifier()` | Quick/moderate/extended | No IR field for decision speed |
| `get_stance_complexity_level()` | 1-3 dimensions | No IR field for stance complexity count |
| `influences_hedging_frequency()` | Hedging tendency | No IR field — hedging is implicit in uncertainty_action |
| `get_ambiguity_tolerance()` | Comfort with ambiguity | No IR field — subsumed into uncertainty_action |
| `get_planning_language_tendency()` | Use of planning language | No IR field for language style markers |
| `get_conflict_avoidance()` | Avoidance tendency | No IR field — partially captured by directness |
| `get_validation_tendency()` | Validates before disagreeing | No IR field for validation behavior |

### IR Fields That Skip Interpreters Entirely

| IR Field | How It's Computed | Missing Interpreter |
|----------|------------------|-------------------|
| `formality` | Raw `persona.psychology.communication.formality` → role blend → clamp | No trait interpreter for formality. Openness and Conscientiousness both predict formality in the literature but neither influences it here. |
| `intent` | Assembled from `user_intent` + `conversation_goal` + `uncertainty_action` by `_generate_intent()` | No interpreter involvement. Pure rule logic. |
| `knowledge_claim_type` | Computed by `_infer_claim_type()` from proficiency + uncertainty_action | No interpreter involvement. |
| `success_criteria` | From persona goals by `_derive_success_criteria()` | No interpreter involvement. |

---

## Part 6: Extreme Persona Diff

Two personas created: all traits at 0.9 vs all traits at 0.1. Same input: "What do you think about work-life balance?"

_(To be run — requires code execution. The expected differences based on code analysis:)_

### Expected IR Differences

| IR Field | All-0.9 Persona | All-0.1 Persona | Method Producing Difference |
|----------|-----------------|-----------------|---------------------------|
| `.elasticity` | ~0.65 (high O × 0.6 = 0.36, + cognitive blend, + 0.25 offset) | ~0.31 (low O × 0.6 = 0.06, + cognitive blend, + 0.25 offset) | `get_elasticity()` — **LIVE** |
| `.confidence` | Low (~0.25) — high N penalty (0.9 × 0.20 = 0.18) dominates | Higher (~0.45) — low N penalty (0.1 × 0.20 = 0.02) | `get_confidence_modifier()` — **LIVE** |
| `.tone` | Complex — high E+O+N creates conflicting signals | Likely NEUTRAL_CALM — low arousal, low valence modifiers | `get_tone_from_mood()` — **LIVE** |
| `.verbosity` | DETAILED (high C + high E both increase) | BRIEF (low C + low E both decrease) | `influences_verbosity()` — **LIVE** |
| `.directness` | ~same base — high A reduces (-0.12) but high E/low patience may compensate | Higher — low A increases (+0.12) | `influences_directness()` — **LIVE** |
| `.disclosure` | Higher — high E modifier (+0.2) + high N (+0.09) | Lower — low E modifier (-0.2) + low N (+0.01) | `get_self_disclosure_modifier()` — **LIVE** |
| `.stance` | Different stance fragment from VALUE_TOPIC_TABLE | Different — all values are 0.9 vs 0.1, different top value | `get_top_values()` — **LIVE** |
| `.competence` | Slightly higher (openness boost 0.9 × weight) | Lower (openness boost 0.1 × weight) | **RAW ACCESS** at line 976 |
| `.formality` | **NO DIFFERENCE from traits** — formality comes from persona.psychology.communication, not Big Five | Same | **MISSING WIRING** |
| `.uncertainty_action` | May differ due to raw risk_tolerance/need_for_closure | May differ | **RAW ACCESS** bypassing interpreter |

### Methods Whose Output Does NOT Produce Visible Difference

| Method | Why No Difference |
|--------|------------------|
| `influences_abstract_reasoning()` | **DEAD** — never called in pipeline |
| `get_novelty_seeking()` | **DEAD** — never called |
| `get_planning_language_tendency()` | **DEAD** — never called |
| `get_follow_through_likelihood()` | **DEAD** — never called |
| `influences_proactivity()` | **DEAD** — never called, no IR field |
| `influences_response_length_social()` | **DEAD** — never called |
| `get_enthusiasm_baseline()` | **DEAD** — never called. Enthusiasm logic is inside get_tone_from_mood() |
| `get_validation_tendency()` | **DEAD** — never called |
| `get_conflict_avoidance()` | **DEAD** — never called |
| `influences_hedging_frequency()` | **DEAD** — never called |
| `get_stress_sensitivity()` | **DEAD** — StateManager uses raw neuroticism |
| `influences_mood_stability()` | **DEAD** — StateManager uses raw neuroticism |
| `get_anxiety_baseline()` | **DEAD** — never called |
| `get_negative_tone_bias()` | **DEAD** — get_tone_from_mood() uses raw neuroticism internally |
| `get_rationale_depth()` | **DEAD** — never called |
| `prefers_systematic_processing()` | **DEAD** — never called |
| `get_decision_time_modifier()` | **DEAD** — never called |
| `get_risk_stance_modifier()` | **DEAD** — bypassed by resolve_uncertainty_action() |
| `influences_uncertainty_action()` | **BYPASSED** — resolve_uncertainty_action() does the same thing better |
| `get_ambiguity_tolerance()` | **DEAD** — never called |
| `prefers_definite_answers()` | **DEAD** — never called |
| `get_nuance_capacity()` | **DEAD** — stance_generator checks cognitive_complexity directly |
| `should_acknowledge_tradeoffs()` | **DEAD** — stance_generator checks cognitive_complexity directly |
| `get_stance_complexity_level()` | **DEAD** — never called |
| `detect_value_conflicts()` | **DEAD** — only in validation |
| `resolve_conflict()` | **DEAD** — wrapper, never called directly in pipeline |
| `get_value_influence_on_stance()` | **STUB** — returns hardcoded 0.5 |
| `get_rationale_influences()` | **DEAD** — never called |
| `should_include_in_citation()` | **DEAD** — never called |

---

## Summary Statistics

| Interpreter | Total Public Methods | LIVE | DEAD | BYPASSED | STUB | VALIDATION-ONLY |
|-------------|---------------------|------|------|----------|------|-----------------|
| TraitInterpreter | 19 | 6 | 12 | 0 | 0 | 1 |
| ValuesInterpreter | 9 | 3 | 4 | 0 | 1 | 1 |
| CognitiveStyleInterpreter | 14 | 3 | 9 | 1 | 0 | 1 |
| **TOTAL** | **42** | **12** | **25** | **1** | **1** | **3** |

**Only 12 of 42 interpreter methods (29%) actually reach the IR.**

### Critical Findings

1. **formality has zero trait influence** — No Big Five trait modifies formality. Only social role and cross-turn smoothing affect it. Literature strongly supports O and C both influencing formality.

2. **uncertainty_action bypasses CognitiveStyleInterpreter** — `resolve_uncertainty_action()` reads `self.cognitive.style.risk_tolerance` and `.need_for_closure` raw instead of calling `influences_uncertainty_action()` or `get_risk_stance_modifier()`.

3. **25 dead methods** represent significant dead code. Many are identity transforms that return raw trait values with no transformation.

4. **1 stub** (`get_value_influence_on_stance`) returns hardcoded values and was never integrated.

5. **Competence calculation** accesses `openness` raw at turn_planner.py:976 instead of through an interpreter method.

6. **Tone doesn't differentiate on turn 1** — `get_tone_from_mood()` depends on mood state from `initial_state` (identical across personas). Traits only affect tone through state drift (turn 2+).

7. **Stance identical for uniform-value personas** — When all Schwartz values are equal, `get_top_values()` returns the same top value (alphabetical tiebreak). Stance differentiation depends entirely on value RANK ORDER, not magnitudes.

## Part 6: Actual Extreme Persona Diff

Input: "What do you think about work-life balance?" All traits 0.9 vs 0.1.

| IR Field | 0.9 | 0.1 | Delta | Diff? |
|----------|-----|-----|-------|-------|
| elasticity | 0.640 | 0.320 | +0.32 | YES |
| confidence | 0.210 | 0.220 | -0.01 | NO (N/C cancel) |
| competence | 0.190 | 0.110 | +0.08 | Marginal |
| tone | professional_composed | professional_composed | 0 | **NO** |
| verbosity | detailed | medium | - | YES |
| formality | 0.300 | 0.300 | 0.00 | **NO** |
| directness | 0.730 | 0.970 | -0.24 | YES |
| disclosure | 0.550 | 0.390 | +0.16 | YES |
| uncertainty | ask_clarifying | refuse | - | YES |
| claim_type | speculative | none | - | YES |
| stance | identical | identical | 0 | **NO** |

**4 of 11 fields show NO differentiation between maximally different personas.**
