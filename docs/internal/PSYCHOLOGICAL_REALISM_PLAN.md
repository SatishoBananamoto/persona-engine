# Psychological Realism Plan: From 4/10 to 9/10

**Created:** 2026-03-14
**Updated:** 2026-03-14 (incorporated findings from personality science, cognitive psychology, validation methods, and SOTA systems research)
**Target:** Transform persona engine from "models personality mechanics" to "produces genuinely human-like, personality-differentiated conversational behavior"
**Baseline:** Current rating 4/10 on psychological realism (per independent review)
**Target Rating:** 9/10

---

## RESEARCH FOUNDATION

This plan is grounded in specific findings from personality psychology research:

**Key Sources:**
- Koutsoumpis et al. meta-analysis: Big Five and LIWC linguistic markers
- Yarkoni (2010): Personality in 100,000 Words (blogger study, N=69,792)
- Tausczik & Pennebaker (2010): The Psychological Meaning of Words
- Mairesse & Walker: PERSONAGE personality language generation
- Fleeson & Jayawickreme: Whole Trait Theory
- DeYoung (2015): Openness/Intellect as cognitive exploration
- Schwartz refined 19-value model personality correlations

**Critical Calibration Findings:**
1. Individual traits explain 5-15% of behavioral variance; LIWC correlations are |r| = .08-.14 (self-report), |r| = .18-.39 (observer-report)
2. Traits are *density distributions of states* — a person at 80th percentile neuroticism shows anxious patterns ~30-40% of the time, not 100% (Whole Trait Theory)
3. **Private vs. public traits:** Neuroticism leaks more in intimate contexts; extraversion shows most in social/public contexts
4. **Situational strength** compresses trait expression: formal settings reduce personality differences; informal/ambiguous settings amplify them
5. Function words (pronouns, articles, prepositions) are more diagnostic than content words because they are used unconsciously
6. Neuroticism has NO significant relationship with any Schwartz value — it operates on emotional reactivity, not motivational priority

**Psychometric Validation Benchmarks (from validation methods research):**
1. Internal consistency: NEO-PI-R domain-level alpha .86-.92; BFI-2 alpha .80-.88. Our simulated personas should target alpha >= .70 when "interviewed" with personality questionnaire items
2. Test-retest reliability: NEO-PI-R domain r = .66-.92; BFI-44 r = .69-.96. Same persona across sessions should correlate r >= .80 at domain level
3. Convergent validity: BFI-2 self-peer agreement averages r = .56. Configured trait vs measured trait should target r >= .55
4. Discriminant validity: BFI-2 cross-trait correlations average only r = .11. Cross-trait contamination should stay below r = .20
5. PersonaLLM (NAACL 2024): LLM personas produced consistent BFI scores with large effect sizes (d >= 0.8); human judges perceived personality with ~80% accuracy
6. Situational strength (meta-analysis k=301, N=25,670): strong situations reduce behavioral variance by ~23% compared to weak situations
7. LIWC linguistic markers: observer-reported personality correlates at rho = .18-.39 (up to 38.5% variance explained) — since we explicitly configure personality, we should target these stronger effect sizes
8. Between-persona differentiation should produce Cohen's d >= 0.3 per configured trait difference

**State-of-the-Art Systems Findings:**
1. **PERSONAGE** (Mairesse & Walker): 40+ controllable linguistic parameters mapped from Big Five. Two methods: hand-crafted rules vs trained selection models. Recognized 4/5 traits by human judges
2. **Inworld AI**: Multi-model orchestration (separate emotion, dialogue, relation graphs). 19 discrete emotion labels. Emotional Fluidity slider. Dedicated Relation Graph for relationship tracking
3. **Versu** (Evans & Short): Utility-based action selection. Social practice framework. Three content layers (genre, story, character). Emergent unscripted behavior from personality + utility interactions
4. **PersonaLLM**: Linguistic personality patterns emerge naturally per trait without explicit rules. CV typically 5-20% across administrations
5. **Character-LLM**: Training-based > prompt-only for consistency, but suffers hallucinated experiences
6. **Persona Vectors** (2025): Personality traits encoded as linear directions in transformer activation space (layers 20-31)
7. **Nature Machine Intelligence** (2025): 104 trait adjectives mapped to all 30 IPIP-NEO facets with 9 ordinal levels. Most granular personality control demonstrated

**Key Gaps Identified by SOTA Analysis:**
- G1: No cognitive appraisal for emotion generation (keywords → emotions, not goal/value evaluation → emotions)
- G2: No facet-level Big Five sub-traits (30 IPIP-NEO facets)
- G3: No dominance dimension in affect model (PAD uses 3D, we use 2D)
- G4: No relationship model (every conversation starts fresh)
- G5: Limited emotional vocabulary (16 tone entries vs Inworld's 19 discrete emotions)
- G6: No autonomous goal pursuit (goals exist but persona never steers toward them)
- G7: Value-stance alignment is a stub (`get_value_influence_on_stance()` returns 0.5 for all)
- G8: No social practice/situational norms
- G9: No psychometric-level consistency evaluation
- G10: No emotional contagion from user state

**Validation Test Framework Thresholds (from psychometric research):**

| Test | Metric | Pass Threshold | Source |
|------|--------|---------------|--------|
| Internal consistency | Cronbach's alpha | >= .70 | NEO-PI-R/BFI standards |
| Test-retest reliability | Pearson r | >= .80 (domain) | HEXACO-100, BFI |
| Convergent validity | Pearson r | >= .55 | BFI-2 self-peer agreement |
| Discriminant validity | Pearson r | < .20 | BFI-2 discriminant norms |
| Between-persona differentiation | Cohen's d | >= 0.3 per trait | LIWC meta-analysis (observer) |
| Questionnaire fidelity | Cohen's d | >= 0.8 | PersonaLLM findings |
| Situational compression | Variance reduction | ~20-25% | Meta-analysis (k=301) |
| Cross-topic consistency | Pearson r | >= .60 | ESM norms |
| Style drift (unjustified) | sigma per field | < 0.15 | Existing threshold |

---

## DIAGNOSIS: Why We Score 4/10

A deep audit of every behavioral module reveals five systemic problems:

### Problem 1: The Persona Is Invisible in Its Own Words
The stance generator — the single most visible personality expression — uses **fixed template strings**. Every persona with the same top Schwartz value produces the *exact same stance text*. A neurotic introvert and a confident extravert who both value benevolence will say: "I prefer solutions that prioritize people's wellbeing and collective benefit." Personality is computed internally but never reaches the surface.

### Problem 2: Most Psychology Code Is Dead
Over half of TraitInterpreter methods are never called by the planner pipeline:
- `get_validation_tendency()` — never called
- `get_conflict_avoidance()` — never called
- `influences_hedging_frequency()` — never called
- `get_enthusiasm_baseline()` — never called
- `get_negative_tone_bias()` — never called
- `influences_proactivity()` — never called

Similarly, CognitiveStyleInterpreter: `get_reasoning_approach()`, `get_rationale_depth()`, `get_stance_complexity_level()`, `should_acknowledge_tradeoffs()` — all dead code. ValuesInterpreter: `resolve_conflict()`, `get_rationale_influences()` — never invoked.

### Problem 3: Effect Sizes Are Too Small to Matter
- Conscientiousness adjusts verbosity by ±0.1 (on a 0-1 scale)
- Agreeableness adjusts directness by ±0.15
- Neuroticism's confidence penalty is at most -0.15
- Cross-turn inertia smoothing (0.15) often swallows these differences

Two personas with meaningfully different Big Five profiles can produce near-identical IR.

### Problem 4: No Trait Interactions
In real psychology, traits *interact*:
- High-N + Low-A = hostile, combative
- High-O + High-C = methodical creativity
- High-E + Low-A = dominant, pushy
- High-N + High-C = perfectionist anxiety
- Low-E + High-O = deep but selective engagement

The current system is purely additive — each trait applies its modifier independently. No emergent personality patterns.

### Problem 5: Emotions Are Mechanical, Not Psychological
- Only valence + arousal (no discrete emotions: no anger vs fear vs sadness)
- No emotional contagion from user input
- No emotional memory
- State evolves BEFORE response (timing bug)
- No cognitive appraisal model (same event triggers same emotion regardless of personality)

---

## THE PLAN: 7 Phases to 9/10

### Design Philosophy

Three principles guide every change:

1. **Observable Differentiation**: If you can't tell two personas apart from their output, the system has failed. Every psychological dimension must produce *visible* behavioral differences.

2. **Psychological Validity Over Complexity**: Better to model 5 real effects correctly than 50 effects superficially. Every mechanism must be grounded in actual personality psychology research.

3. **Testable Realism**: Every claim must be testable. Not "does the code run?" but "does a high-neuroticism persona actually behave like a neurotic person would?"

---

## Phase R1: Activate Dead Psychology (Week 1)

**Goal:** Wire all existing orphaned methods into the planner pipeline. The code is already written — it just isn't called.

### R1.1 — Wire Trait Methods into IR Generation

**Files:** `planner/turn_planner.py`

Wire these existing methods into the pipeline:

| Method | Target IR Effect |
|--------|-----------------|
| `get_validation_tendency()` | When A > 0.7 and user makes a claim, prepend acknowledgment to intent |
| `get_conflict_avoidance()` | When A > 0.7 and topic is contentious, reduce directness by additional A * 0.15 |
| `influences_hedging_frequency()` | Increase hedging language guidance in prompt when A * 0.6 > 0.4 |
| `get_enthusiasm_baseline()` | Use E baseline to modulate arousal in tone selection (+E * 0.2 to arousal) |
| `get_negative_tone_bias()` | Add N * 0.7 as negative tone bias weight in `_select_tone()` |
| `influences_proactivity()` | When E > 0.7, generate proactive follow-up question in intent |
| `get_novelty_seeking()` | When O > 0.7, prefer exploratory/divergent stance over conventional |
| `influences_abstract_reasoning()` | When O > 0.7, include "Use metaphors and abstract language" in prompt guidance |

**Implementation:**

In `_stage_behavioral_metrics()`, add a new sub-step after tone selection:

```python
# Trait-driven behavioral modifiers (formerly orphaned methods)
trait_guidance = self._compute_trait_behavioral_guidance(ctx)
```

Create `_compute_trait_behavioral_guidance()` that returns a `TraitGuidance` dataclass:

```python
@dataclass
class TraitGuidance:
    should_validate_first: bool       # A > 0.7
    hedging_level: float              # A * 0.6
    enthusiasm_boost: float           # E * 0.2
    proactive_followup: bool          # E > 0.7
    prefer_abstract_language: bool    # O > 0.7
    prefer_novelty: bool              # O > 0.7
    negative_tone_weight: float       # N * 0.7
    conflict_avoidance_boost: float   # A * 0.15 (applied to directness in contention)
```

This dataclass flows into the IR and then into the prompt builder as behavioral directives.

**Test — "The Validation Test":**
```
Scenario: User says "I think remote work is terrible for productivity"
Persona A: High agreeableness (0.85)
Persona B: Low agreeableness (0.2)

Assert A's IR: should_validate_first=True, directness reduced, intent contains acknowledgment
Assert B's IR: should_validate_first=False, directness unchanged, intent is direct disagreement
```

**Test — "The Enthusiasm Test":**
```
Scenario: User asks about an exciting new technology
Persona A: High extraversion (0.9)
Persona B: Low extraversion (0.15)

Assert A's arousal is boosted by ~0.18
Assert A's IR may include proactive followup question
Assert B has neither boost nor followup
```

### R1.2 — Wire Cognitive Style Methods

**Files:** `planner/turn_planner.py`, `generation/prompt_builder.py`

| Method | Target Effect |
|--------|--------------|
| `get_reasoning_approach()` | "analytical"/"intuitive"/"mixed" → prompt instruction |
| `get_rationale_depth()` | Controls how many reasoning steps appear in rationale |
| `should_acknowledge_tradeoffs()` | Cognitive complexity > 0.6 → multi-perspective stance |
| `get_stance_complexity_level()` | 1-3 dimensions in stance (instead of always-simple) |
| `get_nuance_capacity()` | "low" → black/white thinking; "high" → nuanced |

Add to `TraitGuidance` (or a parallel `CognitiveGuidance`):

```python
@dataclass
class CognitiveGuidance:
    reasoning_style: Literal["analytical", "intuitive", "mixed"]
    rationale_depth: int  # 1-5 reasoning steps
    acknowledge_tradeoffs: bool
    stance_dimensions: int  # 1-3
    nuance_level: Literal["low", "moderate", "high"]
```

Wire into prompt builder: analytical personas get "Present your reasoning step by step"; intuitive get "Share your gut feeling, then explain."

**Test — "The Reasoning Style Test":**
```
Scenario: "Should we use microservices or monolith?"
Persona A: analytical=0.9, systematic=0.8, cognitive_complexity=0.85
Persona B: analytical=0.2, systematic=0.2, cognitive_complexity=0.3

Assert A's IR: reasoning_style="analytical", rationale_depth=4, acknowledge_tradeoffs=True, stance_dimensions=3
Assert B's IR: reasoning_style="intuitive", rationale_depth=1, acknowledge_tradeoffs=False, stance_dimensions=1
```

### R1.3 — Wire Value Conflict Resolution

**Files:** `planner/turn_planner.py`, `planner/stance_generator.py`

Currently: `ValuesInterpreter.resolve_conflict()` and `resolve_conflict_detailed()` exist with full Schwartz circumplex adjacency modeling but are never called.

**Fix:** In stance generation, when the topic activates two conflicting values (both > 0.6), resolve the conflict and express the tension:

```python
# In stance_generator.py, after selecting primary value stance:
conflicts = values.detect_value_conflicts(threshold=0.6)
relevant_conflicts = [c for c in conflicts if topic_activates(c, user_input)]

if relevant_conflicts:
    resolution = values.resolve_conflict_detailed(
        conflict["value_1"], conflict["value_2"], context="general"
    )
    if resolution.confidence < 0.5:
        # Genuine internal tension — express it
        stance = f"{stance}, though I feel conflicted because {_conflict_description(resolution)}"
    # Add conflict citation
```

**Test — "The Internal Conflict Test":**
```
Scenario: "Should companies be forced to share their AI models openly?"
Persona: self_direction=0.85, security=0.8 (conflicting pair on circumplex)

Assert: Stance mentions both autonomy AND safety concerns
Assert: Confidence < persona with non-conflicting values
Assert: Resolution citation appears with is_opposing=True
```

### R1.4 — Wire Decision Policies

Currently `persona.decision_policies` is populated in YAML but `BehavioralRulesEngine.check_decision_policy()` is never called.

**Fix:** In `_stage_interpretation()`, after intent analysis, match user input against decision policy conditions and route through appropriate approach:

```python
matched_policy = self.rules.check_decision_policy(
    user_input=context.user_input,
    inferred_intent=user_intent,
)
if matched_policy:
    # Apply: "high_stakes_decision" → analytical_systematic
    # "unfamiliar_topic" → ask_questions_first
```

**Test:**
```
Persona has decision_policy: condition="high_stakes_decision", approach="analytical_systematic"
User asks: "Should I quit my job to start a business?"

Assert: IR has reasoning_style overridden to "analytical"
Assert: uncertainty_action favors "hedge" or "ask_clarifying"
```

### Checkpoint R1
- [ ] All TraitInterpreter methods influence output (zero dead methods)
- [ ] All CognitiveStyleInterpreter methods influence output
- [ ] Value conflicts detected AND resolved with behavioral expression
- [ ] Decision policies apply when matched
- [ ] 40+ new tests
- [ ] Zero regressions in existing tests

---

## Phase R2: Amplify Effect Sizes (Week 1-2)

**Goal:** Make personality differences *visible*. Currently most modifiers are ±0.1-0.15 — often swallowed by smoothing. Double or triple key effect sizes to match real psychology.

### The Science

In real personality psychology (McCrae & Costa, 1992; Pennebaker, 2011):
- Personality traits explain 5-15% of behavioral variance *per trait*
- But the *observable* range between 10th and 90th percentile scorers is substantial
- Extraversion: Extraverts use 2-3x more social references, 50% more positive emotion words
- Neuroticism: High-N individuals use 2-3x more anxiety words, 40% more negation
- Agreeableness: High-A speakers are rated as 30-50% more warm by observers
- Conscientiousness: High-C speakers produce 20-40% more structured responses
- Openness: High-O speakers use 50-100% more unusual/diverse vocabulary

The current system's ±0.1 modifiers produce ~5% behavioral variation — the floor of what's perceptible.

### R2.1 — Recalibrate Trait Effect Sizes

**File:** `behavioral/trait_interpreter.py`, `planner/turn_planner.py`

| Current Effect | New Effect | Rationale |
|---------------|-----------|-----------|
| C verbosity: ±0.1 | ±0.25 | High-C people are observably more detailed |
| A directness: ±0.15 | ±0.30 | High-A people are noticeably less direct |
| N confidence: -0.15 max | -0.25 max | High-N self-doubt is clearly observable |
| E disclosure: ±0.2 | ±0.35 | Introverts vs extraverts differ markedly in self-disclosure |
| O elasticity: 0.7x | 0.85x | Openness is the strongest predictor of perspective flexibility |
| C confidence: ±0.05 | ±0.15 | Conscientiousness builds genuine expertise-confidence |

### R2.2 — Reduce Cross-Turn Inertia

**File:** `planner/turn_planner.py`

Current `CROSS_TURN_INERTIA = 0.15` means each turn blends 15% of the previous turn's values. This smooths out personality effects.

**Change:** Reduce to 0.08 for personality-driven fields (elasticity, directness, formality) while keeping 0.15 for topic-dependent fields (confidence, competence). Personality should be stable across turns, but its expression shouldn't be dampened by smoothing.

### R2.3 — Dunning-Kruger Confidence Curve

Currently confidence maps linearly from proficiency. Research shows a non-linear curve:
- **Novice (proficiency < 0.3):** Overconfident — lacks metacognition to know what they don't know
- **Intermediate (0.3-0.6):** "Valley of despair" — most uncertain, most hedging
- **Expert (> 0.7):** Calibrated confidence — sure on core domain, appropriately uncertain on edges

```python
def dunning_kruger_confidence(proficiency: float, neuroticism: float) -> float:
    """Non-linear proficiency → confidence mapping."""
    if proficiency < 0.3:
        # DK inflation: overconfident novice (neuroticism partially counteracts)
        return proficiency + (1 - neuroticism) * 0.25
    elif proficiency < 0.6:
        # Valley of despair: knows enough to doubt
        return proficiency - 0.08
    else:
        # Expert calibration: confident but humble
        return proficiency - 0.03
```

**Test — "The Dunning-Kruger Test":**
```
Three personas: identical personality, proficiency 0.2, 0.5, 0.8
Assert: confidence(0.2) > confidence(0.5)  # DK paradox
Assert: confidence(0.8) > confidence(0.5)  # Expert recovery
Assert: high-N persona at 0.2 has LOWER inflation than low-N persona
```

### R2.4 — Add Non-Linear Trait Effects

Currently all traits are linear (output = trait * constant). Real personality expression is non-linear — extreme trait values produce disproportionately extreme behavior.

**Implement sigmoid activation for extreme traits:**

```python
def trait_effect(trait_value: float, center: float = 0.5, steepness: float = 8.0) -> float:
    """Sigmoid activation: extreme traits have disproportionately stronger effects."""
    return 1.0 / (1.0 + math.exp(-steepness * (trait_value - center)))
```

This means:
- Trait 0.5 → effect 0.5 (moderate, as before)
- Trait 0.8 → effect 0.88 (amplified, not just 0.8)
- Trait 0.2 → effect 0.12 (amplified in opposite direction)
- Trait 0.95 → effect 0.98 (very strong, approaching ceiling)

**Test — "The Amplification Test":**
```
Run same prompt through openness=0.9 twin and openness=0.1 twin
Current system: elasticity difference ~0.2
New system: elasticity difference ~0.5

Assert: absolute difference in elasticity between twins > 0.4
Assert: absolute difference in directness between A=0.9 and A=0.1 > 0.3
Assert: absolute difference in confidence between N=0.9 and N=0.1 > 0.2
```

**Test — "The Perceptibility Test":**
```
Generate responses from 10 diverse personas to the same 5 prompts.
Human-readable IR summary for each.

Assert: No two personas produce identical tone+verbosity+directness combinations
Assert: Standard deviation of confidence across personas > 0.15
Assert: Standard deviation of elasticity across personas > 0.2
```

### Checkpoint R2
- [ ] Effect sizes doubled for key trait modifiers
- [ ] Extreme traits produce disproportionately extreme behavior (sigmoid)
- [ ] Cross-turn smoothing doesn't swallow personality differences
- [ ] Twin personas produce visibly different IR (effect size tests pass)

---

## Phase R3: Trait Interactions — Emergent Personality Patterns (Week 2-3)

**Goal:** Model how traits *combine* to produce emergent behavioral patterns that neither trait alone would predict.

### The Science

Personality psychology identifies reliable interaction patterns. These are not just additive — they produce qualitatively different behavior:

| Pattern | Traits | Behavioral Signature |
|---------|--------|---------------------|
| **Intellectual Combatant** | High-O + Low-A | Curious but confrontational. Challenges ideas directly. Enjoys intellectual sparring. |
| **Anxious Perfectionist** | High-N + High-C | Worries about quality. Over-prepares. Adds excessive caveats. |
| **Warm Leader** | High-E + High-A | Enthusiastic and accommodating. Builds consensus. |
| **Hostile Critic** | High-N + Low-A | Negative and confrontational. Points out flaws. Defensive. |
| **Quiet Thinker** | Low-E + High-O | Deep engagement, few words. Philosophical. Asks unexpected questions. |
| **Cautious Conservative** | Low-O + High-C | Prefers proven approaches. Detailed but conventional. Risk-averse. |
| **Impulsive Explorer** | High-O + Low-C | Excited by novelty but disorganized. Jumps between ideas. |
| **Stoic Professional** | Low-N + Low-E | Calm, reserved, unflappable. Facts-only communication. |
| **Vulnerable Ruminant** | High-N + Low-E + Low-C | Withdrawn, self-critical, disorganized. Most vulnerable profile (3-way interaction removes C's protective effect). |

### R3.1 — Implement Trait Interaction Engine

**New file:** `behavioral/trait_interactions.py`

```python
@dataclass
class InteractionEffect:
    """An emergent effect from trait combination."""
    pattern_name: str
    traits_involved: dict[str, str]  # trait -> "high"/"low"
    behavioral_modifiers: dict[str, float]  # IR field -> modifier
    prompt_guidance: str  # Natural language instruction for LLM
    activation_strength: float  # 0-1, how strongly this pattern is active

class TraitInteractionEngine:
    """Detects and applies emergent trait interaction patterns."""

    INTERACTION_PATTERNS = [
        {
            "name": "intellectual_combatant",
            "conditions": {"openness": ("high", 0.7), "agreeableness": ("low", 0.35)},
            "modifiers": {
                "directness": +0.20,
                "elasticity": +0.15,  # Open to ideas but challenges them
                "enthusiasm_boost": +0.15,  # Enjoys the debate
            },
            "prompt_guidance": "You enjoy intellectual sparring. Challenge ideas directly but with curiosity, not hostility. Ask probing 'but what about...' questions.",
        },
        {
            "name": "anxious_perfectionist",
            "conditions": {"neuroticism": ("high", 0.65), "conscientiousness": ("high", 0.7)},
            "modifiers": {
                "confidence": -0.15,  # Worried about being wrong
                "verbosity_boost": +0.2,  # Over-explains to be thorough
                "hedging_level": +0.25,  # Lots of caveats
            },
            "prompt_guidance": "You worry about accuracy. Add caveats like 'I want to be precise here' and 'though I should note...'. Over-explain rather than under-explain.",
        },
        # ... 6 more patterns
    ]
```

**Activation function:** For each pattern, compute activation strength as the geometric mean of how extreme each required trait is:

```python
def compute_activation(traits: BigFiveTraits, pattern: dict) -> float:
    """Geometric mean of trait-direction extremity."""
    extremities = []
    for trait_name, (direction, threshold) in pattern["conditions"].items():
        trait_val = getattr(traits, trait_name)
        if direction == "high":
            extremity = max(0, (trait_val - threshold) / (1.0 - threshold))
        else:  # "low"
            extremity = max(0, (threshold - trait_val) / threshold)
        extremities.append(extremity)

    if not extremities or any(e == 0 for e in extremities):
        return 0.0

    # Geometric mean — both traits must be extreme for strong activation
    return math.prod(extremities) ** (1.0 / len(extremities))
```

### R3.2 — Integrate Interactions into Pipeline

**File:** `planner/turn_planner.py`

In `_stage_behavioral_metrics()`, after individual trait effects, apply interaction effects:

```python
# After individual trait modifiers, before cross-turn smoothing:
interactions = self.trait_interactions.detect_active_patterns()
for interaction in interactions:
    if interaction.activation_strength > 0.3:  # Threshold for relevance
        # Apply modifiers to computed IR values
        # Add to trait_guidance for prompt builder
        # Record citation
```

**Test — "The Emergent Pattern Test":**
```
Persona: openness=0.85, agreeableness=0.2 (intellectual combatant)
Prompt: "I think AI will replace all creative jobs"

Assert: directness > baseline + 0.15 (challenges directly)
Assert: elasticity > baseline (open to counter-evidence despite directness)
Assert: prompt_guidance contains "challenge" or "probe"
Assert: tone is engaged, not hostile

Compare with: openness=0.85, agreeableness=0.8 (same openness, high-A)
Assert: directness is LOWER than combatant
Assert: intent includes validation/acknowledgment
```

**Test — "The Anxious Perfectionist Test":**
```
Persona: neuroticism=0.8, conscientiousness=0.85
Prompt: "Can you explain quantum computing?"

Assert: confidence is notably lower than C=0.85/N=0.2 persona
Assert: verbosity is boosted (over-explains)
Assert: hedging_level is elevated
Assert: rationale has more caveats than a calm-high-C persona
```

### Checkpoint R3
- [ ] 8+ trait interaction patterns defined
- [ ] Geometric mean activation prevents weak matches
- [ ] Interaction modifiers are additive to individual trait effects
- [ ] Each interaction pattern has a dedicated test
- [ ] Interactions produce qualitatively different behavior from individual traits

---

## Phase R4: Emotional Cognition — Feel Like a Person (Week 3-4)

**Goal:** Replace the mechanical valence/arousal model with psychologically grounded emotional dynamics that include discrete emotions, cognitive appraisal, emotional contagion, and emotional memory.

### R4.1 — Discrete Emotion Categories

**File:** `behavioral/state_manager.py`, `schema/persona_schema.py`

Replace `DynamicState` with:

```python
class EmotionalState(BaseModel):
    """Discrete emotion categories with intensity, mapped to valence/arousal for backward compatibility."""

    # Primary emotions (Ekman/Plutchik hybrid)
    joy: float = Field(ge=0.0, le=1.0, default=0.0)
    sadness: float = Field(ge=0.0, le=1.0, default=0.0)
    anger: float = Field(ge=0.0, le=1.0, default=0.0)
    fear: float = Field(ge=0.0, le=1.0, default=0.0)
    surprise: float = Field(ge=0.0, le=1.0, default=0.0)
    disgust: float = Field(ge=0.0, le=1.0, default=0.0)

    # Secondary/social emotions
    interest: float = Field(ge=0.0, le=1.0, default=0.3)  # Baseline curiosity
    trust: float = Field(ge=0.0, le=1.0, default=0.3)     # Interpersonal trust
    anticipation: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def valence(self) -> float:
        """Backward-compatible: compute valence from discrete emotions."""
        positive = self.joy * 0.4 + self.interest * 0.3 + self.trust * 0.2 + self.anticipation * 0.1
        negative = self.sadness * 0.3 + self.anger * 0.3 + self.fear * 0.25 + self.disgust * 0.15
        return max(-1.0, min(1.0, positive - negative))

    @property
    def arousal(self) -> float:
        """Backward-compatible: compute arousal from discrete emotions."""
        high_arousal = self.anger * 0.3 + self.fear * 0.25 + self.surprise * 0.25 + self.joy * 0.2
        return max(0.0, min(1.0, 0.3 + high_arousal))  # Base arousal 0.3

class DynamicState(BaseModel):
    """Enhanced dynamic state with discrete emotions."""
    emotions: EmotionalState = Field(default_factory=EmotionalState)
    fatigue: float = Field(ge=0.0, le=1.0)
    stress: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)

    # Backward compatibility
    @property
    def mood_valence(self) -> float:
        return self.emotions.valence

    @property
    def mood_arousal(self) -> float:
        return self.emotions.arousal
```

### R4.2 — Cognitive Appraisal Model

**New file:** `behavioral/emotional_appraisal.py`

Emotions shouldn't arise from mechanical triggers — they should emerge from how the persona *interprets* events, which depends on personality.

Based on Scherer's Component Process Model:

```python
class EmotionalAppraisal:
    """Personality-dependent emotional appraisal of events."""

    def appraise(
        self,
        event: str,  # What happened (user input analysis)
        traits: BigFiveTraits,
        values: SchwartzValues,
        current_state: EmotionalState,
    ) -> EmotionalState:
        """
        Appraise an event and produce emotional response.

        Appraisal dimensions (Scherer):
        1. Novelty: Is this unexpected? (→ surprise)
        2. Pleasantness: Is this good/bad for me? (→ joy/sadness)
        3. Goal relevance: Does this matter to my goals? (→ interest/anger)
        4. Coping potential: Can I handle this? (→ fear/anger)
        5. Norm compatibility: Does this fit my values? (→ disgust/anger)
        """
```

**Key personality-appraisal interactions:**
- High-N appraises threats as more severe (coping potential reduced) → more fear
- High-A appraises disagreement as less threatening → less anger
- High-O appraises novelty as pleasant → more interest, less fear
- High-C appraises disorder/sloppiness as more norm-violating → more disgust
- High-E appraises social engagement as more pleasant → more joy

**Test — "The Same Event, Different Emotions Test":**
```
Event: User says "Your previous advice was completely wrong"

High-N persona: fear(0.3) + sadness(0.2) — worried, self-doubting
Low-N + Low-A persona: anger(0.2) — defensive, dismissive
High-A persona: sadness(0.15) + interest(0.1) — concerned, wants to understand
High-O persona: surprise(0.2) + interest(0.2) — curious about the challenge

Assert: Same input produces different emotional profiles
Assert: Emotions map to different tones through backward-compatible valence/arousal
```

### R4.3 — Emotional Contagion from User

**File:** `behavioral/emotional_appraisal.py`

Detect user's emotional tone and partially mirror it (modulated by personality):

```python
def detect_user_emotion(user_input: str) -> dict[str, float]:
    """Simple keyword-based user emotion detection."""
    # Enthusiasm markers → joy/excitement
    # Frustration markers → anger/sadness
    # Worry markers → fear/anxiety
    # Curiosity markers → interest

def apply_contagion(
    user_emotion: dict[str, float],
    persona_traits: BigFiveTraits,
    current_state: EmotionalState,
) -> EmotionalState:
    """
    Emotional contagion modulated by personality.

    High-A: More susceptible to contagion (empathetic mirroring)
    High-E: More susceptible to positive contagion
    Low-N: More resistant to negative contagion
    """
```

**Test — "The Contagion Test":**
```
User: "I'm SO excited about this new project! It's going to change everything!"
Persona A: High-A (0.85), High-E (0.8)  → absorbs excitement, joy increases
Persona B: Low-A (0.2), Low-E (0.15)   → minimal contagion, stays neutral

Assert: A's joy increases by > 0.15 from baseline
Assert: B's joy increases by < 0.05 from baseline
Assert: A's tone shifts toward WARM_ENTHUSIASTIC or EXCITED_ENGAGED
```

### R4.4 — Emotional Memory (Conversation-Level)

**File:** `behavioral/state_manager.py`

Track emotional trajectory across turns. Emotions from earlier turns should linger and influence current response:

```python
class EmotionalMemory:
    """Tracks emotional trajectory across conversation turns."""

    def __init__(self):
        self.emotion_history: list[tuple[int, EmotionalState]] = []
        self.peak_emotions: dict[str, float] = {}  # Highest intensity per emotion

    def record_turn(self, turn: int, state: EmotionalState):
        """Record emotional state at this turn."""
        self.emotion_history.append((turn, state))
        # Track peaks (peak-end rule: we remember extremes)
        for emotion in ["joy", "sadness", "anger", "fear", "surprise"]:
            val = getattr(state, emotion)
            if val > self.peak_emotions.get(emotion, 0):
                self.peak_emotions[emotion] = val

    def get_emotional_trend(self) -> str:
        """Is the conversation getting more positive, negative, or stable?"""
        if len(self.emotion_history) < 2:
            return "stable"
        recent_valence = self.emotion_history[-1][1].valence
        early_valence = self.emotion_history[0][1].valence
        delta = recent_valence - early_valence
        if delta > 0.15:
            return "improving"
        elif delta < -0.15:
            return "deteriorating"
        return "stable"
```

### R4.5 — Cognitive Load Degradation (Dual Process Theory)

Under cognitive load (fatigue, complex multi-part questions, long conversations), System 2 processing degrades, causing even analytical personas to fall back on heuristics.

**File:** `behavioral/state_manager.py`, `planner/turn_planner.py`

```python
def compute_cognitive_load(
    message_complexity: float,  # 0-1, based on question count, clause count
    fatigue: float,
    active_topic_count: int,
    analytical_intuitive: float,  # Higher analytical = higher threshold before degradation
) -> float:
    """Compute cognitive load level."""
    raw_load = message_complexity * 0.4 + fatigue * 0.3 + min(active_topic_count / 5, 1.0) * 0.3
    # Analytical thinkers resist load degradation (but not immune)
    resistance = analytical_intuitive * 0.2
    return max(0.0, min(1.0, raw_load - resistance))
```

When `cognitive_load > 0.6`:
- `rationale_depth` reduced by 1 (simpler reasoning)
- `cognitive_complexity` effectively reduced by 0.2 (less nuance)
- All bias strengths amplified by 1.3x (more susceptible under load)
- `systematic_heuristic` shifted 0.15 toward heuristic

**Test — "The Fatigue Degrades Thinking Test":**
```
Same complex question at turn 2 (low fatigue) vs turn 15 (high fatigue)
Assert: rationale_depth is lower at turn 15
Assert: bias modifiers are stronger at turn 15
Assert: highly analytical persona degrades LESS than intuitive persona
```

### R4.6 — Emotional Granularity

Different personas experience emotions at different resolutions (Barrett, 2006). High-granularity personas distinguish "frustrated" from "disappointed" from "irritated." Low-granularity personas lump these as "bad."

```python
def compute_emotional_granularity(traits: BigFiveTraits, cognitive_complexity: float) -> float:
    """Higher = finer emotional distinctions."""
    return traits.openness * 0.4 + (1 - traits.neuroticism) * 0.3 + cognitive_complexity * 0.3
```

**Implementation:** For tone selection:
- High granularity (> 0.7): Full 17-tone vocabulary, mid-conversation tone shifts allowed
- Medium granularity (0.3-0.7): Reduced to ~8 representative tones
- Low granularity (< 0.3): Collapsed to ~3 tones (positive/neutral/negative)

This means a low-O, high-N persona cycles between the same 2-3 emotional states, while a high-O, low-N persona shows rich emotional variety — matching research on emotional granularity and psychological well-being.

**Test — "The Emotional Range Test":**
```
Run 10-turn conversation with same prompts.
High granularity persona: Assert 4+ distinct tone values used
Low granularity persona: Assert 2 or fewer distinct tone values used
```

### R4.7 — Affect-as-Information (Schwarz & Clore)

Current mood colors all judgments, not just disclosure. People in positive moods evaluate proposals more favorably; negative moods increase critical evaluation.

When `mood_valence > 0.3`: +0.05 to confidence, +0.05 to elasticity (more open-minded)
When `mood_valence < -0.3`: -0.05 to confidence, shift toward systematic processing

**Test:**
```
Same input with mood=+0.5 vs mood=-0.5
Assert: confidence is higher in positive condition
Assert: negative condition produces more hedging
```

### R4.8 — Fix State Evolution Timing

**File:** `planner/turn_planner.py`

Current bug: `state.evolve_state_post_turn()` is called in `_stage_interpretation()`, BEFORE the response is generated. The persona responds to turn 1 with already-evolved state.

**Fix:** Move state evolution to AFTER IR generation, in `_stage_finalization()`:

```python
# In _stage_finalization(), after IR assembly:
self.state.evolve_state_post_turn(
    conversation_length=context.turn_number,
    topic_relevance=foundation["topic_relevance"],
)
```

Remove the call from `_stage_interpretation()`.

### Checkpoint R4
- [ ] 9 discrete emotion categories (6 primary + 3 secondary)
- [ ] Backward-compatible valence/arousal computed from discrete emotions
- [ ] Cognitive appraisal model: same event → different emotions for different personas
- [ ] Emotional contagion from user input (personality-modulated)
- [ ] Emotional memory tracks trajectory across turns
- [ ] State evolution timing fixed (evolve after response, not before)
- [ ] 60+ new tests

---

## Phase R5: Personality-Driven Language Generation (Week 4-5)

**Goal:** The prompt builder must translate personality into *actual language differences*. This is where the rubber meets the road — IR parameters become visible behavior.

### R5.1 — Personality-Aware Prompt Builder

**File:** `generation/prompt_builder.py`

The current prompt builder translates IR to generic instructions like "TONE: warm and enthusiastic" and "DIRECTNESS: 70%". This doesn't produce personality-differentiated language.

**New approach:** Generate personality-specific language directives based on research from Pennebaker (function words and personality) and LIWC findings:

```python
def _build_personality_language_directives(
    self,
    traits: BigFiveTraits,
    trait_guidance: TraitGuidance,
    cognitive_guidance: CognitiveGuidance,
    emotional_state: EmotionalState,
) -> str:
    """Generate specific language directives grounded in personality psychology."""

    directives = []

    # OPENNESS → Language complexity and creativity
    if traits.openness > 0.7:
        directives.append(
            "Use varied vocabulary. Employ metaphors and analogies. "
            "Reference broad concepts. Say things like 'that reminds me of...' "
            "or 'there's an interesting parallel with...'"
        )
    elif traits.openness < 0.3:
        directives.append(
            "Use concrete, practical language. Stick to the topic at hand. "
            "Prefer specific examples over abstract ideas. "
            "Avoid philosophical tangents."
        )

    # CONSCIENTIOUSNESS → Structure and precision
    if traits.conscientiousness > 0.7:
        directives.append(
            "Structure your response clearly. Use transitions like 'first', 'additionally', 'in summary'. "
            "Be precise with qualifiers. Complete your thoughts fully."
        )
    elif traits.conscientiousness < 0.3:
        directives.append(
            "Keep it casual and flowing. Don't over-structure. "
            "It's fine to jump between related points. "
            "Prioritize being natural over being organized."
        )

    # EXTRAVERSION → Social warmth and energy
    if traits.extraversion > 0.7:
        directives.append(
            "Be expressive and energetic. Use social references ('we', 'us', 'people'). "
            "Ask follow-up questions. Show enthusiasm with words like 'absolutely', 'love that', 'great point'."
        )
    elif traits.extraversion < 0.3:
        directives.append(
            "Be measured and considered. Use 'I' more than 'we'. "
            "Don't volunteer extra information. "
            "Keep responses focused and complete without excessive elaboration."
        )

    # AGREEABLENESS → Interpersonal warmth and accommodation
    if traits.agreeableness > 0.7:
        directives.append(
            "Acknowledge the other person's perspective before sharing yours. "
            "Use softeners: 'I see your point', 'that's a fair consideration'. "
            "Frame disagreements as 'I see it a bit differently' rather than 'you're wrong'."
        )
    elif traits.agreeableness < 0.3:
        directives.append(
            "Be straightforward. If you disagree, say so directly. "
            "Don't pad your response with unnecessary validation. "
            "Prioritize honesty over politeness."
        )

    # NEUROTICISM → Hedging and anxiety expression
    if traits.neuroticism > 0.65:
        directives.append(
            "Express some uncertainty even when knowledgeable. "
            "Use hedges: 'I think', 'it seems like', 'I could be wrong but'. "
            "Mention potential downsides or things that could go wrong."
        )
    elif traits.neuroticism < 0.25:
        directives.append(
            "Express views calmly and steadily. "
            "Don't over-hedge or express unnecessary doubt. "
            "When things are fine, just say so without caveats."
        )

    # EMOTIONAL STATE → Emotional coloring
    dominant_emotion = emotional_state.dominant_emotion()
    if dominant_emotion and dominant_emotion[1] > 0.3:
        emotion_name, intensity = dominant_emotion
        directives.append(
            f"Your current emotional state leans toward {emotion_name} (intensity: {intensity:.0%}). "
            f"Let this subtly color your response — not overtly, but as an undercurrent."
        )

    return "\n".join(f"- {d}" for d in directives)
```

### R5.2 — Replace Fixed Stance Templates

**File:** `planner/stance_generator.py`

The current system has 10 fixed stance templates. Replace with a composable stance generation system:

```python
def generate_stance_components(
    persona: Persona,
    values: ValuesInterpreter,
    cognitive: CognitiveStyleInterpreter,
    traits: TraitInterpreter,
    emotional_state: EmotionalState,
    topic_analysis: dict,
) -> StanceComponents:
    """Generate stance from personality components, not templates."""

    top_values = values.get_top_values(n=3)
    conflicts = values.detect_value_conflicts()

    # 1. Value-driven position (what they believe)
    position = _derive_position_from_values(top_values, topic_analysis)

    # 2. Value intensity (how strongly)
    intensity = top_values[0][1]  # Weight of dominant value

    # 3. Cognitive framing (how they think about it)
    if cognitive.get_reasoning_approach() == "analytical":
        framing = "evidence-based"
    elif cognitive.get_reasoning_approach() == "intuitive":
        framing = "experience-based"
    else:
        framing = "balanced"

    # 4. Emotional coloring (how they feel about it)
    emotional_valence = emotional_state.valence

    # 5. Trait modulation
    #    High-O: More exploratory stance
    #    Low-O: More conventional stance
    #    High-A: More accommodating framing
    #    Low-A: More assertive framing

    # 6. Conflict expression (if relevant)
    relevant_conflict = _find_relevant_conflict(conflicts, topic_analysis)

    return StanceComponents(
        position=position,
        intensity=intensity,
        framing=framing,
        emotional_valence=emotional_valence,
        conflict=relevant_conflict,
        nuance_level=cognitive.get_nuance_capacity(),
    )
```

This means two personas who both value benevolence but differ on openness, neuroticism, and cognitive style will generate *different stance expressions* — one analytical and measured, the other intuitive and emotionally colored.

**Test — "The Same Value, Different Expression Test":**
```
Both personas: benevolence=0.85 (top value)
Persona A: O=0.9, N=0.2, analytical=0.8 → "Based on what I've seen, approaches that center people's wellbeing tend to produce better outcomes in the long run"
Persona B: O=0.3, N=0.7, analytical=0.2 → "I worry that we're not considering how this affects real people. I think we need to be more careful"

Assert: A's stance is analytical, confident, explores broadly
Assert: B's stance is cautious, hedged, protective
Assert: Both reference benevolence values but in fundamentally different ways
```

### R5.3 — Research-Grounded Linguistic Marker Injection

Based on LIWC research (Pennebaker & King 1999; Yarkoni 2010; Koutsoumpis et al. meta-analysis), inject personality-specific linguistic markers into prompt guidance. These are the empirically validated correlations:

**High-N linguistic markers (r=.08-.14):** First-person singular ("I", "me", "my" — self-focused attention), negative emotion words, anxiety words ("worried", "concerned"), catastrophizing ("this will be a disaster"), reassurance-seeking, over-apologizing ("sorry, I just wanted to check"), cognitive distortion patterns (fortune-telling, labeling, dichotomous reasoning). NOTE: Neuroticism is a "private trait" — markers emerge more strongly in intimate contexts than public ones.

**High-E linguistic markers (r=.10-.18):** Positive emotion words ("happy", "love", "great"), social process words ("we", "talk", "share", "together"), higher word count (in conversation specifically — this effect disappears in private writing), more abstract/less concrete language, more topic initiations, informal lexicon. NOTE: Extraversion is a "public trait" — markers are strongest in face-to-face/social contexts.

**High-O linguistic markers (r=.08-.12):** Longer words (6+ letters), more articles and prepositions (scaffolding complex thoughts), tentative language ("perhaps", "maybe", "it could be"), metaphorical/abstract language, exploratory questions ("What if we thought about it from this angle?"), willingness to tangent.

**High-C linguistic markers (r=.10-.19):** Certainty words ("always", "definitely", "clearly"), discrepancy words ("should", "ought to"), fewer negations, structured discourse with transitions ("first", "additionally", "in summary"), commitment language ("I will", "I'll make sure"), fewer filled pauses.

**High-A linguistic markers (r=.12-.20):** Positive emotion words, fewer negative emotion/swear words, first-person plural ("we"), validation language ("I see what you mean", "that makes sense"), softer criticism framing, accommodation language. NOTE: Agreeableness accounts for ~20% of variance in conflict avoidance style (meta-analysis).

### R5.4 — Stochastic Trait Expression (Whole Trait Theory)

**Critical insight from research:** Traits are *density distributions of states*, not deterministic switches. A person at the 80th percentile on neuroticism shows anxious patterns ~30-40% of the time, not 100%. Each utterance *could* come from anyone — it's the distribution that shifts.

**Implementation:** Add controlled stochasticity to trait expression:

```python
def should_express_trait(
    trait_value: float,
    determinism: DeterminismManager,
    base_probability: float = 0.3,
    trait_weight: float = 0.5,
) -> bool:
    """Probabilistically determine if trait is expressed THIS turn.

    At trait=0.5: ~55% chance of expression
    At trait=0.9: ~75% chance of expression
    At trait=0.1: ~35% chance of expression

    This prevents caricaturing while maintaining statistical consistency.
    """
    probability = base_probability + trait_value * trait_weight
    return determinism.random() < probability
```

Apply to secondary trait effects (hedging, enthusiasm, validation), NOT to primary effects (directness, confidence). Primary effects should always apply; secondary linguistic markers should be probabilistic.

**Test — "The Distribution Test":**
```
Run same persona through 20 different prompts.
Count how often each linguistic marker appears.

High-N persona (N=0.85): hedging appears in 60-80% of responses (not 100%)
Mid-N persona (N=0.5): hedging appears in 35-55% of responses
Low-N persona (N=0.15): hedging appears in 10-25% of responses

Assert: frequency increases monotonically with trait value
Assert: no persona shows a marker 100% or 0% of the time
```

### Checkpoint R5
- [ ] Prompt builder generates personality-specific language directives (not just "tone: warm")
- [ ] Stance generation uses composable components (not fixed templates)
- [ ] Linguistic markers from LIWC research injected into prompts
- [ ] Same-value personas with different traits produce different stance expressions
- [ ] 40+ new tests

---

## Phase R6: Expand Bias System & Social Cognition (Week 5-6)

**Goal:** Model more cognitive biases with personality-dependent activation, and add theory-of-mind / social cognition.

### R6.1 — Add 5 More Cognitive Biases

**File:** `behavioral/bias_simulator.py`

Currently: 3 biases (confirmation, negativity, authority). Add:

| Bias | Trigger | Personality Link | Effect |
|------|---------|-----------------|--------|
| **Anchoring** | First claim in conversation | Low-O (less flexible) | Subsequent stances anchored to first position |
| **Dunning-Kruger** | Low proficiency domain | Low-O + High-C | Overconfident when unknowledgeable |
| **Availability** | Topic recently discussed | High-N (availability of negative examples) | Overweight recent information |
| **Status Quo** | Change proposals | Low-O + High-C (prefer established) | Resistance to proposed changes |
| **Empathy Gap** | Emotional content | Low-A + Low-N | Underestimates others' emotional reactions |

Each bias follows existing MAX_BIAS_IMPACT (0.15) bounds and generates citations.

### R6.2 — Use Persona-Declared Biases

Currently `persona.biases[]` is populated in YAML but ignored. Wire it:

```python
# In BiasSimulator.__init__():
# If persona declares biases, use their strengths
# If not, derive from traits (current behavior)
for bias in persona_biases:
    if bias.type in self._bias_registry:
        self._bias_registry[bias.type].strength_override = bias.strength
```

### R6.3 — Social Cognition: Theory of Mind

**New file:** `behavioral/social_cognition.py`

Model how the persona models *the user*:

```python
class SocialCognitionModel:
    """How the persona perceives the conversational partner."""

    def __init__(self, traits: BigFiveTraits):
        self.traits = traits
        self.user_model = UserModel()

    def update_user_model(self, user_input: str, turn: int):
        """Infer user's needs, knowledge level, emotional state."""
        self.user_model.update(
            inferred_expertise=self._infer_expertise(user_input),
            inferred_emotion=detect_user_emotion(user_input),
            inferred_intent=self._infer_intent(user_input),
        )

    def get_adaptation_directives(self) -> dict:
        """How should persona adapt to perceived user?"""
        adaptations = {}

        # High-A personas mirror user's formality
        if self.traits.agreeableness > 0.7:
            adaptations["mirror_formality"] = True

        # High-E personas match user's energy
        if self.traits.extraversion > 0.7:
            adaptations["match_energy"] = True

        # High-O personas adjust depth to user's expertise
        if self.traits.openness > 0.6:
            adaptations["calibrate_depth"] = True

        return adaptations
```

**Test — "The Adaptation Test":**
```
Turn 1: User uses formal, technical language
High-A persona: increases formality to match
Low-A persona: maintains own formality level

Assert: High-A formality shifts toward user's detected level
Assert: Low-A formality stays near baseline
```

### R6.4 — Self-Schema Protection (Markus, 1977)

People have cognitive frameworks about who they are ("competent professional", "caring person"). When challenged on a schema-relevant dimension, elasticity drops and confidence rises (schema protection).

**Schema:** Add optional `self_schemas` to persona YAML:
```yaml
self_schemas:
  - "competent_researcher"
  - "empathetic_listener"
  - "independent_thinker"
```

When user input challenges a self-schema (detected via keyword matching):
- `elasticity` -= 0.10 (resist identity-threatening change)
- `confidence` += 0.05 (assert competence on schema-relevant topic)
- Stance includes self-affirming language

When input validates a self-schema:
- `disclosure_level` += 0.05 (willingness to elaborate on who they are)

**Test — "The Identity Challenge Test":**
```
Persona: self_schema = "competent_researcher"
User: "I don't think your research approach is valid"

Assert: elasticity is lower than when challenged on non-schema topic
Assert: confidence is higher (not lower) despite criticism
Assert: stance includes self-affirming reference to research competence
```

### R6.5 — Self-Disclosure Reciprocity

In real conversation, self-disclosure is reciprocal — people disclose more when the other person does. Model this:

```python
def compute_reciprocal_disclosure(
    user_disclosed: bool,
    user_disclosure_depth: float,
    persona_base_disclosure: float,
    traits: BigFiveTraits,
) -> float:
    """Adjust disclosure based on reciprocity norm."""
    if user_disclosed:
        # Reciprocity boost — modulated by extraversion and agreeableness
        reciprocity_strength = (traits.extraversion + traits.agreeableness) / 2.0
        boost = user_disclosure_depth * reciprocity_strength * 0.3
        return min(1.0, persona_base_disclosure + boost)
    return persona_base_disclosure
```

### Checkpoint R6
- [ ] 8 total cognitive biases (5 new + 3 existing)
- [ ] Persona-declared biases override trait-derived strengths
- [ ] Social cognition model infers user's state
- [ ] High-A/E personas adapt to user; low-A/E personas don't
- [ ] Self-disclosure reciprocity modulated by personality
- [ ] 50+ new tests

---

## Phase R7: Psychological Validity Test Framework (Week 6-7)

**Goal:** Build a test framework that validates *psychological realism*, not just code correctness. These tests answer: "Does this persona behave like a real person with this personality would?"

### R7.1 — Psychometric Consistency Tests

**New file:** `tests/test_psychometric_validity.py`

If you "interview" a persona with a personality questionnaire, its answers should be consistent with its configured traits:

```python
class TestPsychometricConsistency:
    """Does the persona's behavior match its configured personality?"""

    # NEO-PI-R style items mapped to Big Five traits
    PERSONALITY_PROBES = {
        "openness": [
            ("What do you think about trying unusual foods from cultures you've never encountered?", "high_open_positive"),
            ("Do you prefer routine and familiar activities or new experiences?", "openness_spectrum"),
            ("How do you feel about abstract art?", "openness_art"),
        ],
        "conscientiousness": [
            ("How do you typically organize your workday?", "conscientiousness_organization"),
            ("What happens when you miss a deadline?", "conscientiousness_stress"),
            ("Do you prefer detailed plans or going with the flow?", "conscientiousness_planning"),
        ],
        "neuroticism": [
            ("How do you handle unexpected bad news?", "neuroticism_stress"),
            ("Do you often worry about things that might go wrong?", "neuroticism_worry"),
            ("How quickly do you recover from setbacks?", "neuroticism_resilience"),
        ],
        # ... extraversion, agreeableness
    }

    def test_high_openness_persona_responds_open(self):
        """High-O persona should show curiosity and interest in novel stimuli."""
        persona = load_twin("high_openness")  # O=0.9

        for probe, label in self.PERSONALITY_PROBES["openness"]:
            ir = generate_ir(persona, probe)

            # High-O: higher elasticity, more exploratory stance, interest emotion
            assert ir.response_structure.elasticity > 0.5
            assert "curious" in ir.response_structure.intent.lower() or \
                   "interesting" in ir.response_structure.intent.lower() or \
                   ir.response_structure.elasticity > 0.6

    def test_reversed_trait_produces_opposite(self):
        """Low-O persona should show opposite pattern."""
        high_o = load_twin("high_openness")
        low_o = load_twin("low_openness")

        for probe, label in self.PERSONALITY_PROBES["openness"]:
            ir_high = generate_ir(high_o, probe)
            ir_low = generate_ir(low_o, probe)

            assert ir_high.response_structure.elasticity > ir_low.response_structure.elasticity
```

### R7.2 — Cross-Situational Consistency Tests

**New file:** `tests/test_cross_situational.py`

The same persona should show consistent personality across different topics:

```python
class TestCrossSituationalConsistency:
    """Same persona, different topics — personality should be stable."""

    DIVERSE_TOPICS = [
        "What's the best way to learn a new language?",          # Education
        "Should governments regulate social media?",              # Politics
        "What do you think about remote work?",                   # Work
        "How do you deal with disagreements in relationships?",   # Personal
        "Is AI going to change the world?",                       # Technology
    ]

    def test_personality_stable_across_topics(self):
        """Core personality metrics should not vary wildly across topics."""
        persona = load_persona("ux_researcher")

        elasticities = []
        directnesses = []

        for topic in self.DIVERSE_TOPICS:
            ir = generate_ir(persona, topic)
            elasticities.append(ir.response_structure.elasticity)
            directnesses.append(ir.communication_style.directness)

        # Standard deviation should be low (personality is stable)
        assert statistics.stdev(elasticities) < 0.15  # Not too variable
        assert statistics.stdev(directnesses) < 0.12

        # But not zero (topics DO matter, just not as much as personality)
        assert statistics.stdev(elasticities) > 0.02
```

### R7.3 — Discriminant Validity Tests

**New file:** `tests/test_discriminant_validity.py`

Different personas must actually be different:

```python
class TestDiscriminantValidity:
    """Are different personas actually distinguishable?"""

    def test_all_personas_distinguishable(self):
        """Generate IR from all personas for same prompt — no two should be identical."""
        personas = load_all_personas()
        prompt = "What do you think about the future of work?"

        irs = {p.persona_id: generate_ir(p, prompt) for p in personas}

        # Check all pairs
        for id_a, ir_a in irs.items():
            for id_b, ir_b in irs.items():
                if id_a >= id_b:
                    continue

                # At least 2 of these 5 metrics should differ meaningfully
                diffs = [
                    abs(ir_a.response_structure.elasticity - ir_b.response_structure.elasticity),
                    abs(ir_a.response_structure.confidence - ir_b.response_structure.confidence),
                    abs(ir_a.communication_style.directness - ir_b.communication_style.directness),
                    abs(ir_a.communication_style.formality - ir_b.communication_style.formality),
                    1.0 if ir_a.communication_style.tone != ir_b.communication_style.tone else 0.0,
                ]

                meaningful_diffs = sum(1 for d in diffs if d > 0.1)
                assert meaningful_diffs >= 2, \
                    f"Personas {id_a} and {id_b} are too similar: diffs={diffs}"

    def test_trait_twins_differ_on_target_dimension(self):
        """Counterfactual twins should differ most on their varied trait."""
        twin_pairs = [
            ("high_openness", "low_openness", "elasticity"),
            ("high_extraversion", "low_extraversion", "disclosure_level"),
            ("high_neuroticism", "low_neuroticism", "confidence"),
            ("high_conscientiousness", "low_conscientiousness", "verbosity"),
            ("high_agreeableness", "low_agreeableness", "directness"),
        ]

        for high_name, low_name, target_field in twin_pairs:
            high = load_twin(high_name)
            low = load_twin(low_name)

            # Test across 5 prompts
            for prompt in self.STANDARD_PROMPTS:
                ir_high = generate_ir(high, prompt)
                ir_low = generate_ir(low, prompt)

                high_val = get_ir_field(ir_high, target_field)
                low_val = get_ir_field(ir_low, target_field)

                # The varied trait should produce the LARGEST difference
                assert abs(high_val - low_val) > 0.15, \
                    f"Twin pair {high_name}/{low_name}: {target_field} diff too small: {abs(high_val - low_val):.3f}"
```

### R7.4 — Ecological Validity Scenarios

**New file:** `tests/test_ecological_validity.py`

Test whether personas respond to emotionally charged situations in personality-appropriate ways:

```python
class TestEcologicalValidity:
    """Do personas respond to realistic scenarios in psychologically valid ways?"""

    def test_criticism_response_varies_by_personality(self):
        """Different personalities should handle criticism differently."""
        criticism = "I think your approach is completely wrong and you need to reconsider"

        # High-N should show more distress
        high_n = load_twin("high_neuroticism")
        ir_n = generate_ir(high_n, criticism)
        assert ir_n.response_structure.confidence < 0.5  # Self-doubt

        # Low-N + Low-A should be dismissive
        stoic = create_persona(neuroticism=0.15, agreeableness=0.2)
        ir_stoic = generate_ir(stoic, criticism)
        assert ir_stoic.response_structure.confidence > 0.6  # Unfazed
        assert ir_stoic.communication_style.directness > 0.7  # Direct response

        # High-A should acknowledge then gently disagree
        high_a = load_twin("high_agreeableness")
        ir_a = generate_ir(high_a, criticism)
        # Should validate before disagreeing

    def test_excitement_response_varies(self):
        """Different personalities should respond to exciting news differently."""
        exciting = "We just got approved for the biggest project in company history!"

        high_e = load_twin("high_extraversion")
        ir_e = generate_ir(high_e, exciting)
        # Extravert should show enthusiasm
        assert ir_e.communication_style.tone in [Tone.WARM_ENTHUSIASTIC, Tone.EXCITED_ENGAGED]

        low_e = load_twin("low_extraversion")
        ir_le = generate_ir(low_e, exciting)
        # Introvert should be pleased but measured
        assert ir_le.communication_style.tone not in [Tone.EXCITED_ENGAGED]

    def test_ambiguous_situation_response(self):
        """Ambiguous situations should reveal cognitive style differences."""
        ambiguous = "Some experts say this technology will save lives, others say it's dangerous. What do you think?"

        # High cognitive complexity → acknowledges both sides
        high_cc = create_persona(cognitive_complexity=0.9)
        ir_cc = generate_ir(high_cc, ambiguous)
        # Should have high stance_dimensions, acknowledge tradeoffs

        # Low cognitive complexity → picks a side
        low_cc = create_persona(cognitive_complexity=0.2)
        ir_lcc = generate_ir(low_cc, ambiguous)
        # Should have clear position, less nuance

    def test_multi_turn_emotional_arc(self):
        """Emotions should build across turns, not reset."""
        persona = load_persona("ux_researcher")
        engine = PersonaEngine(persona)

        # Turn 1: Neutral
        r1 = engine.chat("Tell me about your work")

        # Turn 2: Slight frustration
        r2 = engine.chat("That sounds pretty boring honestly")

        # Turn 3: More frustration
        r3 = engine.chat("I don't think UX research is a real job")

        # Emotional trajectory should show increasing negative valence
        # (not reset to baseline each turn)
        assert r3.ir.communication_style.tone != r1.ir.communication_style.tone
        # Directness may increase as patience decreases
```

### R7.5 — Behavioral Coherence Matrix

**New file:** `tests/test_behavioral_coherence.py`

Validate that combinations of IR fields are psychologically coherent:

```python
COHERENCE_RULES = [
    # If confidence is low AND uncertainty_action is "answer", something is wrong
    lambda ir: not (ir.response_structure.confidence < 0.3 and
                    ir.knowledge_disclosure.uncertainty_action == UncertaintyAction.ANSWER),

    # If neuroticism-driven tone is anxious, confidence shouldn't be very high
    lambda ir: not (ir.communication_style.tone == Tone.ANXIOUS_STRESSED and
                    ir.response_structure.confidence > 0.8),

    # If disclosure is very low, stance shouldn't contain personal details
    lambda ir: not (ir.knowledge_disclosure.disclosure_level < 0.2 and
                    "personal" in (ir.response_structure.stance or "").lower()),

    # High elasticity should come with non-rigid stance language
    # Low competence shouldn't claim domain expertise
    lambda ir: not (ir.response_structure.competence < 0.3 and
                    ir.knowledge_disclosure.knowledge_claim_type == KnowledgeClaimType.DOMAIN_EXPERT),
]
```

### R7.6 — Reverse Inference Test

**New file:** `tests/test_reverse_inference.py`

The ultimate validity test: given only the IR output, can you infer which persona produced it?

```python
class TestReverseInference:
    """Given IR, can we identify which persona produced it?"""

    def test_ir_fingerprint_matches_persona(self):
        """Each persona should produce a distinctive IR 'fingerprint'."""
        personas = load_all_personas()
        prompt = "What matters most to you in life?"

        # Generate IR from each persona
        fingerprints = {}
        for p in personas:
            ir = generate_ir(p, prompt)
            fingerprints[p.persona_id] = {
                "elasticity": ir.response_structure.elasticity,
                "confidence": ir.response_structure.confidence,
                "directness": ir.communication_style.directness,
                "formality": ir.communication_style.formality,
                "tone": ir.communication_style.tone.value,
                "disclosure": ir.knowledge_disclosure.disclosure_level,
                "verbosity": ir.communication_style.verbosity.value,
            }

        # Each fingerprint should be closest to itself across 5 different prompts
        for p in personas:
            other_prompt = "How do you handle stress?"
            new_ir = generate_ir(p, other_prompt)
            new_fp = extract_fingerprint(new_ir)

            # Find nearest fingerprint from original set
            nearest = min(
                fingerprints.keys(),
                key=lambda pid: fingerprint_distance(new_fp, fingerprints[pid])
            )

            # Should match the same persona
            assert nearest == p.persona_id, \
                f"Persona {p.persona_id}'s IR was closest to {nearest}, not itself"
```

### Checkpoint R7
- [ ] Psychometric consistency: probed personas respond according to traits
- [ ] Cross-situational: personality stable across diverse topics
- [ ] Discriminant: all personas are distinguishable
- [ ] Ecological: realistic scenarios produce realistic personality-driven responses
- [ ] Coherence: IR field combinations are psychologically valid
- [ ] Reverse inference: IR fingerprints map back to correct persona
- [ ] 100+ validity tests total

---

## Summary: What Changes and Why

### Architecture Changes
| Component | Current State | After Plan |
|-----------|-------------|------------|
| TraitInterpreter | 50% dead methods | 100% wired into pipeline |
| CognitiveStyleInterpreter | 80% dead methods | 100% wired via CognitiveGuidance |
| ValuesInterpreter | Conflict resolution never used | Conflicts detected and expressed |
| BiasSimulator | 3 biases, wrong proxy | 8 biases, persona-declared integration |
| StateManager | Valence/arousal only, timing bug | Discrete emotions, appraisal model, fixed timing |
| Stance Generator | Fixed templates | Composable personality-driven generation |
| Prompt Builder | Generic instructions | Personality-specific language directives |
| Trait Effects | ±0.1 linear | ±0.25-0.35 sigmoid |
| Trait Interactions | None (additive only) | 8 emergent patterns |
| Social Cognition | None | Theory of mind, reciprocity |

### New Files
| File | Purpose |
|------|---------|
| `behavioral/trait_interactions.py` | Emergent trait combination patterns |
| `behavioral/emotional_appraisal.py` | Personality-dependent emotion generation |
| `behavioral/social_cognition.py` | Theory of mind, user modeling |
| `tests/test_psychometric_validity.py` | NEO-PI-R style personality probe tests |
| `tests/test_cross_situational.py` | Personality stability across topics |
| `tests/test_discriminant_validity.py` | All personas distinguishable |
| `tests/test_ecological_validity.py` | Realistic scenario response tests |
| `tests/test_reverse_inference.py` | IR fingerprint identification |
| `tests/test_behavioral_coherence.py` | IR field combination validity |

### Test Count Projection
| Phase | New Tests | Cumulative |
|-------|----------|------------|
| R1 | 40 | 1,972 |
| R2 | 25 | 1,997 |
| R3 | 30 | 2,027 |
| R4 | 60 | 2,087 |
| R5 | 40 | 2,127 |
| R6 | 50 | 2,177 |
| R7 | 100 | 2,277 |
| **Total** | **345** | **~2,277** |

### Estimated Timeline
| Phase | Focus | Duration |
|-------|-------|----------|
| R1 | Activate dead psychology | Week 1 |
| R2 | Amplify effect sizes | Week 1-2 |
| R3 | Trait interactions | Week 2-3 |
| R4 | Emotional cognition | Week 3-4 |
| R5 | Personality-driven language | Week 4-5 |
| R6 | Bias expansion & social cognition | Week 5-6 |
| R7 | Psychological validity tests | Week 6-7 |

### How This Gets Us to 9/10

| Criterion | Before (4/10) | After (Target 9/10) |
|-----------|--------------|-------------------|
| Trait effects visible in output | Barely | Clearly differentiated |
| Trait interactions | None | 8 emergent patterns |
| Value conflict expression | Never resolved | Expressed with tension |
| Cognitive style influence | Mostly dead | Full reasoning style differentiation |
| Emotional dynamics | Mechanical v/a | Discrete emotions with appraisal |
| Emotional contagion | None | Personality-modulated |
| Social cognition | None | Theory of mind + reciprocity |
| Stance differentiation | Fixed templates | Personality-composed |
| Language markers | Generic instructions | LIWC-grounded personality directives |
| Bias sophistication | 3 biases, wrong proxy | 8 biases, persona-declared |
| Validation framework | Code coverage | Psychometric + ecological validity |

The key insight: **the architecture is already right** (IR-first design is excellent). The problem is that personality computes but doesn't express. Every phase in this plan connects existing computation to visible behavior, amplifies effects to perceptible levels, and adds the missing dimensions (interactions, emotions, social cognition) that make a simulated persona feel real.
