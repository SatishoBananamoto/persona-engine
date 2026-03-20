# Trait Flow Analysis — How Big Five Reaches the IR

> Generated 2026-03-20 during The Graft session. Documents how personality traits
> flow through the pipeline, known issues, and validation strategy.

## Pipeline Overview

```
Stage 1: Foundation     — trace setup, memory context (not trait-dependent)
Stage 2: Interpretation — domain detection, bias modifiers, state evolution
Stage 3: Behavioral     — all numeric/enum IR fields (orchestrator + 3 mixins)
Stage 4: Knowledge      — disclosure, uncertainty, claim type
Stage 5: Finalization   — IR assembly, linguistic directives, stance cache
```

---

## Per-Field Modifier Chain

### CONFIDENCE (`response_structure.confidence`)

```
proficiency (domain match)
  → TraitInterpreter.get_confidence_modifier()     [DK curve + C boost + N penalty]
  → CognitiveStyle.get_confidence_adjustment()      [analytical penalty, closure boost]
  → BiasSimulator (authority +, DK bias +)           [additive]
  → Memory (known facts +, familiarity +)            [additive]
  → clamp [0.0, 1.0]
  → cross-turn inertia smoothing
  → schema effect (+0.05 if self-schema challenged)  [additive, NO re-clamp]
  → trait_interactions                                [additive, clamp 0.1-0.95]
```

**BUG: Double Dunning-Kruger.** `get_confidence_modifier()` applies the full DK curve
internally. `BiasSimulator._compute_dunning_kruger_bias()` adds a SECOND DK inflation
(up to +0.15) for low proficiency. Both are driven by the same proficiency value.

**Neuroticism hits confidence through 3 independent paths:**
1. `trait_interpreter` — N penalty via sigmoid (~-0.25 for high N)
2. `bias_simulator` — DK bias (if low proficiency, partially gated by knowledge_boundary_strictness)
3. `trait_interactions` — anxious_perfectionist -0.15, hostile_critic -0.10, vulnerable_ruminant -0.20

### ELASTICITY (`response_structure.elasticity`)

```
TraitInterpreter.get_elasticity()                   [O sigmoid + confidence penalty, clamp 0.1-0.9]
  → blend with CognitiveStyle.get_elasticity()       [average: (trait + cognitive) / 2]
  → BiasSimulator (confirmation -, anchoring -, status_quo -)  [additive, bounded ±0.30]
  → clamp [ELASTICITY_MIN, ELASTICITY_MAX]
  → cross-turn inertia smoothing
  → schema effect (-0.10)                            [additive, NO re-clamp]
  → trait_interactions                                [additive, clamp 0.1-0.9]
```

**Watch:** Schema modifier applied in unclamped gap between elasticity clamp and
interaction clamp. Interaction clamp catches overflow, but sequencing is inconsistent.

### DIRECTNESS (`communication_style.directness`)

```
persona.communication.directness (base)
  → social role blend (70/30)
  → debate mode (+0.15)
  → TraitInterpreter.influences_directness()          [A sigmoid modifier]
  → conflict_avoidance_boost (A * 0.15 if contentious) [subtract, conditional]
  → state: low patience → +DIRECTNESS_IMPATIENCE_BUMP  [additive]
  → clamp [0.0, 1.0]
  → trait_interactions                                  [additive, clamp 0.0-1.0]
  → cross-turn inertia smoothing
```

**Watch:** Agreeableness influences directness through 2 paths (trait modifier + conflict
avoidance). Combined effect for A=0.9 + contentious input: ~-0.335. By design (baseline
vs. situational), but aggressive.

### DISCLOSURE (`knowledge_disclosure.disclosure_level`)

```
persona.disclosure_policy.base_openness
  → TraitInterpreter.get_self_disclosure_modifier()    [E sigmoid + N co-factor]
  → StateManager.get_disclosure_modifier()             [mood/stress/fatigue]
  → trust modifier (from memory)                       [scaled by trust_factor]
  → privacy clamp (persona.privacy_sensitivity)        [upper bound]
  → topic sensitivity clamp                            [upper bound]
  → disclosure_policy.bounds                            [hard min/max]
  → cross-turn inertia smoothing
  → bias: empathy gap (low-A + low-N)                  [additive, clamp 0.0-1.0]
  → schema validation (+0.05)                           [additive, clamp 0.0-1.0]
  → social cognition: disclosure reciprocity            [additive, clamp 0.0-1.0]
```

**Clean.** No double-counting detected. Proper clamping at each stage.

### FORMALITY (`communication_style.formality`)

```
persona.communication.formality (base)
  → social role blend (70/30)
  → clamp [0.0, 1.0]
  → cross-turn inertia smoothing
  → social cognition: formality mirroring (A > 0.6)    [additive, clamp 0.0-1.0]
```

**Clean.** Traits influence only indirectly through social role blending.

### TONE (`communication_style.tone`)

```
StateManager.get_mood() → (valence, arousal)
  → enthusiasm_boost from trait_guidance (E * 0.2) → arousal  [additive]
  → bias modifiers (negativity, availability) → arousal       [additive]
  → TraitInterpreter.get_tone_from_mood(valence, arousal, stress)  [N, E, O thresholds → Tone enum]
```

**Clean.** Single-writer from `get_tone_from_mood()`. Emotional appraisal modifies mood
state BEFORE tone selection, not the tone itself.

### VERBOSITY (`communication_style.verbosity`)

```
persona.communication.verbosity + interaction_modifiers.verbosity_boost  [base]
  → TraitInterpreter.influences_verbosity()  [C + E co-factor → enum threshold]
  → state override: fatigue (-1) or engagement (+1)     [bump]
  → social cognition: depth_shift                       [bump]
```

**Clean.** No double-counting.

### COMPETENCE (`response_structure.competence`)

```
proficiency (direct/adjacency/unknown base)
  → openness * OPENNESS_COMPETENCE_WEIGHT              [additive]
  → memory: familiarity boost, known facts boost        [additive, capped]
  → clamp [0.0, 1.0]
  → cross-turn inertia smoothing
```

**Clean.**

### STANCE (`response_structure.stance`)

```
values (top Schwartz values + domain mapping) → template composition
  → cognitive complexity → nuance qualifier              [append]
  → invariants (cannot_claim caveat, must_avoid block)   [gate/prefix]
  → value conflicts → conflict expression text           [append]
  → _modulate_stance_by_personality (N, A, O)            [string prefix/suffix]
  → stance cache (can short-circuit entire pipeline)
```

**Clean.** String-based, not numeric.

---

## Trait Interaction Patterns (9)

These are emergent effects from trait COMBINATIONS, applied as additive modifiers
AFTER individual trait effects. Intentional by design.

| Pattern | Activation | Modifiers |
|---------|-----------|-----------|
| intellectual_combatant | O>0.65, A<0.35 | directness +0.20, elasticity +0.15, enthusiasm +0.15 |
| anxious_perfectionist | N>0.65, C>0.65 | confidence -0.15, verbosity +0.20, hedging +0.25 |
| warm_leader | E>0.65, A>0.65 | directness -0.10, enthusiasm +0.20, validation +0.15 |
| hostile_critic | N>0.65, A<0.35 | directness +0.15, neg_tone +0.20, confidence -0.10 |
| quiet_thinker | E<0.35, O>0.65 | verbosity -0.15, elasticity +0.10, novelty +0.10 |
| cautious_conservative | O<0.35, C>0.65 | elasticity -0.15, confidence +0.10, novelty -0.15 |
| impulsive_explorer | O>0.65, C<0.35 | elasticity +0.15, verbosity -0.10, novelty +0.20 |
| stoic_professional | N<0.35, E<0.35 | enthusiasm -0.15, confidence +0.10, hedging -0.10 |
| vulnerable_ruminant | N>0.65, E<0.35, C<0.35 | confidence -0.20, neg_tone +0.15, hedging +0.20, enthusiasm -0.15 |

---

## Known Issues

| ID | Severity | Field | Issue | Recommendation |
|----|----------|-------|-------|----------------|
| TF-001 | **FIXED** | confidence | Double Dunning-Kruger: DK curve in trait_interpreter + DK bias in bias_simulator | Fixed: DK bias disabled in bias_simulator. DK curve in trait_interpreter kept (more sophisticated). |
| TF-002 | **FIXED** | confidence | N hits confidence via 3 paths — extreme N (0.95) collapsed to floor | Fixed: N penalty 0.25→0.18, confidence floor 0.15 in trait_interpreter + 0.12 in behavioral_metrics. Validated: N=0.95 now produces 0.12 (not 0.10). |
| TF-003 | WATCH | directness | A double-influences via trait modifier + conflict_avoidance | Combined -0.335 for A=0.9 + contentious. Intentional but aggressive |
| TF-004 | MINOR | elasticity | Schema modifier applied in unclamped gap | Interaction clamp catches it. Low risk. |

---

## Files Referenced

| File | Role |
|------|------|
| `persona_engine/planner/stages/behavioral.py` | Orchestrator for stage 3 |
| `persona_engine/planner/stages/behavioral_metrics.py` | Elasticity, confidence, competence |
| `persona_engine/planner/stages/behavioral_style.py` | Tone, verbosity, formality, directness |
| `persona_engine/planner/stages/behavioral_guidance.py` | Trait + cognitive guidance, stance generation |
| `persona_engine/planner/stages/knowledge.py` | Disclosure, uncertainty, claim type |
| `persona_engine/planner/stages/finalization.py` | IR assembly, linguistic directives |
| `persona_engine/behavioral/trait_interpreter.py` | Big Five → behavioral parameters |
| `persona_engine/behavioral/trait_interactions.py` | 9 emergent patterns |
| `persona_engine/behavioral/cognitive_interpreter.py` | Cognitive style |
| `persona_engine/behavioral/values_interpreter.py` | Schwartz values |
| `persona_engine/behavioral/bias_simulator.py` | 8 cognitive biases |
| `persona_engine/behavioral/state_manager.py` | Dynamic state |
| `persona_engine/behavioral/social_cognition.py` | User modeling + adaptation |
| `persona_engine/behavioral/emotional_appraisal.py` | Mood mutation |
| `persona_engine/behavioral/linguistic_markers.py` | LIWC-based language directives |
| `persona_engine/planner/stance_generator.py` | Value-driven stance composition |
