# Pipeline Flowcharts — Persona Engine IR Generation

> Generated 2026-03-20. Mermaid diagrams for the full pipeline, per-field modifier
> chains, and trait fan-out. Render in GitHub or any Mermaid-compatible viewer.

---

## 1. Master Pipeline

What flows between stages during `TurnPlanner.generate_ir()`.

```mermaid
flowchart TD
    INPUT[/"ConversationContext<br/>user_input, turn_number,<br/>interaction_mode, goal,<br/>topic_signature, stance_cache"/]

    subgraph S1["Stage 1: Foundation"]
        F1[Create TraceContext]
        F2[Generate turn_seed]
        F3[Build memory_context<br/>known_facts, preferences,<br/>previously_discussed]
    end

    subgraph S2["Stage 2: Interpretation"]
        I1[compute_topic_relevance]
        I2[analyze_intent<br/>mutates mode + goal]
        I3[detect_domain + proficiency]
        I4[check_decision_policy]
        I5[bias_simulator.compute_modifiers<br/>stores on planner._current_bias_modifiers]
        I6[expert_allowed =<br/>is_domain_specific AND<br/>proficiency >= threshold]
    end

    subgraph S3["Stage 3: Behavioral Metrics"]
        direction TB
        subgraph S3a["3a: Metrics"]
            M1[compute_elasticity<br/>trait + cognitive + bias]
            M2[compute_confidence<br/>DK curve + trait + cognitive + bias + memory]
            M3[compute_competence<br/>domain match + openness + memory]
        end
        subgraph S3b["3b: Guidance"]
            G1[generate_stance<br/>cache check → stance_generator]
            G2[compute_trait_guidance<br/>hedging, enthusiasm, validation,<br/>conflict_avoidance, neg_tone]
            G3[compute_cognitive_guidance<br/>reasoning_style, rationale_depth,<br/>nuance_level]
        end
        subgraph S3c["3c: Style"]
            ST1[select_tone<br/>mood + stress + enthusiasm + bias → Tone enum]
            ST2[compute_verbosity<br/>base + C/E trait + state override]
            ST3[compute_communication_style<br/>formality + directness<br/>role blend + trait + state]
        end
        subgraph S3d["3d: Modulation"]
            EM[emotional_appraisal<br/>detect emotion → appraise → update mood]
            SC[social_cognition<br/>user model → adaptation + schema effect]
            TI[trait_interactions<br/>9 patterns → aggregate modifiers<br/>confidence, elasticity, directness,<br/>hedging, enthusiasm, neg_tone]
        end
    end

    subgraph S4["Stage 4: Knowledge & Safety"]
        K1[compute_disclosure<br/>base + trait + state + trust<br/>+ privacy + policy + bias + schema]
        K2[resolve_uncertainty_action<br/>proficiency + confidence +<br/>risk_tolerance + closure + stress]
        K3[infer_knowledge_claim_type<br/>proficiency + uncertainty +<br/>personal_experience + domain]
        K4[validate_stance_against_invariants]
    end

    subgraph S5["Stage 5: Finalization"]
        FN1[Build memory read/write ops]
        FN2[Propagate safety plan<br/>cannot_claim, must_avoid]
        FN3[Collect behavioral_directives<br/>trait + cognitive + adaptation + schema]
        FN4[build_personality_language_directives<br/>LIWC markers, emotional coloring]
        FN5[Assemble IR]
        FN6[Cache stance + store snapshot<br/>+ process memory writes]
    end

    POST[/"evolve_state_post_turn<br/>mood drift, fatigue, stress decay,<br/>engagement, noise"/]

    OUTPUT[/"IntermediateRepresentation<br/>conversation_frame, response_structure,<br/>communication_style, knowledge_disclosure,<br/>citations, safety_plan, memory_ops,<br/>behavioral_directives, personality_language"/]

    INPUT --> S1
    S1 -->|"TraceContext, turn_seed,<br/>MemoryOps, memory_context"| S2
    S2 -->|"InterpretationResult<br/>topic_relevance, domain, proficiency,<br/>expert_allowed, user_intent,<br/>bias_modifiers on planner"| S3
    S3 -->|"BehavioralMetricsResult<br/>elasticity, stance, confidence,<br/>competence, tone, verbosity,<br/>formality, directness,<br/>trait/cognitive guidance,<br/>adaptation, schema_effect"| S4
    S4 -->|"KnowledgeSafetyResult<br/>disclosure_level,<br/>uncertainty_action,<br/>claim_type"| S5
    S5 --> OUTPUT
    OUTPUT --> POST
```

---

## 2. Per-Field Modifier Chains

The 4 most complex IR fields. Each box shows source → operation → effect.

### 2a. Confidence

```mermaid
flowchart TD
    PROF[/"proficiency<br/>(domain match)"/]
    DK["dunning_kruger_confidence()<br/>5-segment piecewise curve<br/>+ N modulation"]
    TRAIT["TraitInterpreter<br/>C boost: (C-0.5)*0.3<br/>N penalty: sigmoid(N)*0.25"]
    COG["CognitiveStyle<br/>analytical penalty<br/>closure boost"]
    BIAS["BiasSimulator<br/>authority_bias (+)<br/>DK bias: DISABLED (TF-001)"]
    MEM["Memory<br/>known_facts: +0.05-0.10<br/>familiarity: +0.05-0.15"]
    CLAMP1["clamp [0.0, 1.0]"]
    INERTIA["cross-turn smoothing<br/>prev * alpha + new * (1-alpha)"]
    SCHEMA["schema_effect<br/>+0.05 if self-schema challenged"]
    INTERACT["trait_interactions<br/>anxious_perfectionist: -0.15<br/>hostile_critic: -0.10<br/>stoic_professional: +0.10<br/>cautious_conservative: +0.10<br/>vulnerable_ruminant: -0.20"]
    CLAMP2["clamp [0.1, 0.95]"]

    PROF -->|"set"| DK
    DK -->|"replace"| TRAIT
    TRAIT -->|"replace"| COG
    COG -->|"replace"| BIAS
    BIAS -->|"additive"| MEM
    MEM -->|"additive"| CLAMP1
    CLAMP1 --> INERTIA
    INERTIA --> SCHEMA
    SCHEMA -->|"additive, NO clamp"| INTERACT
    INTERACT --> CLAMP2

    style DK fill:#e8f5e9
    style SCHEMA fill:#fff3e0
    style INTERACT fill:#e3f2fd
```

### 2b. Elasticity

```mermaid
flowchart TD
    TRAIT["TraitInterpreter.get_elasticity()<br/>sigmoid(O) * 0.7 - confidence * 0.3<br/>+ 0.2 shift, clamp [0.1, 0.9]"]
    COG["CognitiveStyle.get_elasticity()<br/>complexity + closure"]
    BLEND["average<br/>(trait + cognitive) / 2"]
    BIAS["BiasSimulator<br/>confirmation: -<br/>anchoring: -<br/>status_quo: -<br/>bounded [-0.30, +0.30]"]
    CLAMP1["clamp [ELASTICITY_MIN, MAX]"]
    INERTIA["cross-turn smoothing"]
    SCHEMA["schema_effect: -0.10"]
    INTERACT["trait_interactions<br/>intellectual_combatant: +0.15<br/>quiet_thinker: +0.10<br/>cautious_conservative: -0.15<br/>impulsive_explorer: +0.15"]
    CLAMP2["clamp [0.1, 0.9]"]

    TRAIT --> BLEND
    COG --> BLEND
    BLEND --> BIAS
    BIAS -->|"additive"| CLAMP1
    CLAMP1 --> INERTIA
    INERTIA --> SCHEMA
    SCHEMA -->|"additive, NO clamp"| INTERACT
    INTERACT --> CLAMP2

    style SCHEMA fill:#fff3e0
    style INTERACT fill:#e3f2fd
```

### 2c. Directness

```mermaid
flowchart TD
    BASE[/"persona.communication.directness"/]
    ROLE["social_role blend<br/>70% persona / 30% role"]
    DEBATE["debate mode?<br/>+0.15"]
    TRAIT["TraitInterpreter.influences_directness()<br/>sigmoid(A) → modifier<br/>(0.5 - sigmoid(A)) * 0.5"]
    CONFLICT["conflict_avoidance_boost<br/>A * 0.15 if contentious input"]
    STATE["state: low patience?<br/>+DIRECTNESS_IMPATIENCE_BUMP"]
    CLAMP1["clamp [0.0, 1.0]"]
    INTERACT["trait_interactions<br/>intellectual_combatant: +0.20<br/>warm_leader: -0.10<br/>hostile_critic: +0.15"]
    CLAMP2["clamp [0.0, 1.0]"]
    INERTIA["cross-turn smoothing"]
    ADAPT["social_cognition<br/>adaptation.directness_shift"]

    BASE --> ROLE
    ROLE --> DEBATE
    DEBATE --> TRAIT
    TRAIT -->|"replace"| CONFLICT
    CONFLICT -->|"subtract"| STATE
    STATE -->|"additive"| CLAMP1
    CLAMP1 --> INTERACT
    INTERACT --> CLAMP2
    CLAMP2 --> INERTIA
    INERTIA --> ADAPT

    style CONFLICT fill:#fff3e0
    style INTERACT fill:#e3f2fd
```

### 2d. Disclosure

```mermaid
flowchart TD
    BASE[/"persona.disclosure_policy.base_openness"/]
    TRAIT["TraitInterpreter.get_self_disclosure_modifier()<br/>sigmoid(E)*0.45 + N*0.1"]
    STATE["StateManager.get_disclosure_modifier()<br/>mood*0.15 - stress*0.2 - fatigue*0.1"]
    TRUST["Memory trust level<br/>scaled by trust_factor"]
    PRIVACY["privacy filter clamp<br/>persona.privacy_sensitivity"]
    TOPIC["topic sensitivity clamp"]
    POLICY["disclosure_policy.bounds<br/>hard [min, max]"]
    INERTIA["cross-turn smoothing"]
    BIAS["BiasSimulator<br/>empathy_gap (low-A + low-N)"]
    SCHEMA["schema_effect<br/>+0.05 if validated"]
    RECIP["social_cognition<br/>disclosure_reciprocity"]
    CLAMP["clamp [0.0, 1.0]"]

    BASE --> TRAIT
    TRAIT -->|"additive"| STATE
    STATE -->|"additive"| TRUST
    TRUST -->|"additive"| PRIVACY
    PRIVACY -->|"upper bound"| TOPIC
    TOPIC -->|"upper bound"| POLICY
    POLICY -->|"hard clamp"| INERTIA
    INERTIA --> BIAS
    BIAS -->|"additive"| SCHEMA
    SCHEMA -->|"additive"| RECIP
    RECIP -->|"additive"| CLAMP

    style BIAS fill:#e3f2fd
    style SCHEMA fill:#fff3e0
```

---

## 3. Trait Fan-Out

Where each Big Five trait is consumed across the pipeline.

### 3a. Openness (O)

```mermaid
flowchart LR
    O(("O<br/>Openness"))
    O --> E1["get_elasticity()<br/>sigmoid(O)*0.7"]
    O --> E2["influences_abstract_reasoning()<br/>O > 0.7"]
    O --> E3["get_novelty_seeking()<br/>returns O"]
    O --> E4["compute_competence<br/>O * OPENNESS_WEIGHT"]
    O --> E5["_modulate_stance_by_personality<br/>O > 0.7: exploratory prefix"]
    O --> E6["trait_interactions<br/>intellectual_combatant (O>0.65, A<0.35)<br/>quiet_thinker (E<0.35, O>0.65)<br/>cautious_conservative (O<0.35, C>0.65)<br/>impulsive_explorer (O>0.65, C<0.35)"]
    O --> E7["bias_simulator<br/>DK susceptibility: (1-O)"]
    O --> E8["linguistic_markers<br/>abstract language directives"]

    style O fill:#a5d6a7
```

### 3b. Conscientiousness (C)

```mermaid
flowchart LR
    C(("C<br/>Conscientiousness"))
    C --> E1["influences_verbosity()<br/>(C-0.5)*0.5"]
    C --> E2["get_confidence_modifier()<br/>C boost: (C-0.5)*0.3"]
    C --> E3["get_planning_language_tendency()<br/>returns C"]
    C --> E4["get_follow_through_likelihood()<br/>returns C"]
    C --> E5["state_manager.increment_fatigue<br/>stamina: 1.0 - C*0.2"]
    C --> E6["trait_interactions<br/>anxious_perfectionist (N>0.65, C>0.65)<br/>cautious_conservative (O<0.35, C>0.65)<br/>impulsive_explorer (O>0.65, C<0.35)<br/>vulnerable_ruminant (N>0.65, E<0.35, C<0.35)"]
    C --> E7["bias_simulator<br/>DK susceptibility: C"]

    style C fill:#90caf9
```

### 3c. Extraversion (E)

```mermaid
flowchart LR
    E(("E<br/>Extraversion"))
    E --> E1["influences_proactivity()<br/>0.2 + E*0.6"]
    E --> E2["get_self_disclosure_modifier()<br/>sigmoid(E)*0.45"]
    E --> E3["get_enthusiasm_baseline()<br/>0.2 + E*0.5"]
    E --> E4["influences_response_length_social()<br/>returns E"]
    E --> E5["influences_verbosity()<br/>(E-0.5)*0.15 co-factor"]
    E --> E6["get_tone_from_mood()<br/>E > 0.7: enthusiasm bonus"]
    E --> E7["state_manager.apply_mood_drift<br/>baseline_valence: + E*0.15"]
    E --> E8["trait_interactions<br/>warm_leader (E>0.65, A>0.65)<br/>quiet_thinker (E<0.35, O>0.65)<br/>stoic_professional (N<0.35, E<0.35)<br/>vulnerable_ruminant (N>0.65, E<0.35, C<0.35)"]

    style E fill:#ffcc80
```

### 3d. Agreeableness (A)

```mermaid
flowchart LR
    A(("A<br/>Agreeableness"))
    A --> E1["influences_directness()<br/>(0.5 - sigmoid(A))*0.5"]
    A --> E2["get_validation_tendency()<br/>returns A"]
    A --> E3["get_conflict_avoidance()<br/>returns A"]
    A --> E4["influences_hedging_frequency()<br/>A*0.6 + N*0.2"]
    A --> E5["conflict_avoidance_boost<br/>A*0.15 if contentious"]
    A --> E6["_modulate_stance_by_personality<br/>A > 0.75: empathetic suffix<br/>A < 0.4 + low-N: assertive prefix"]
    A --> E7["social_cognition<br/>formality mirroring (A > 0.6)"]
    A --> E8["trait_interactions<br/>intellectual_combatant (O>0.65, A<0.35)<br/>warm_leader (E>0.65, A>0.65)<br/>hostile_critic (N>0.65, A<0.35)"]
    A --> E9["bias_simulator<br/>empathy_gap: low-A + low-N"]

    style A fill:#ef9a9a
```

### 3e. Neuroticism (N)

```mermaid
flowchart LR
    N(("N<br/>Neuroticism"))
    N --> E1["get_confidence_modifier()<br/>sigmoid(N)*0.25 penalty"]
    N --> E2["get_negative_tone_bias()<br/>N*0.5"]
    N --> E3["get_stress_sensitivity()<br/>returns N"]
    N --> E4["influences_mood_stability()<br/>1.0 - N"]
    N --> E5["get_anxiety_baseline()<br/>returns N"]
    N --> E6["influences_hedging_frequency()<br/>N*0.2 co-factor"]
    N --> E7["get_self_disclosure_modifier()<br/>N*0.1 co-factor"]
    N --> E8["get_tone_from_mood()<br/>N > 0.6 + stress: anxious tone"]
    N --> E9["state_manager<br/>mood_drift_rate: 0.12 - N*0.08<br/>stress_decay_rate: 0.08 + (1-N)*0.04<br/>baseline_valence: - N*0.2<br/>stress_sensitivity: 1 + N*0.5"]
    N --> E10["dunning_kruger_confidence()<br/>N modulates DK inflation"]
    N --> E11["_modulate_stance_by_personality<br/>N > 0.65: cautious prefix"]
    N --> E12["trait_interactions<br/>anxious_perfectionist (N>0.65, C>0.65)<br/>hostile_critic (N>0.65, A<0.35)<br/>stoic_professional (N<0.35, E<0.35)<br/>vulnerable_ruminant (N>0.65, E<0.35, C<0.35)"]
    N --> E13["bias_simulator<br/>empathy_gap: low-N component"]

    style N fill:#ce93d8
```

---

## Color Legend

- Green boxes: trait-based computation (individual trait effect)
- Blue boxes: interaction/bias modifiers (emergent or cognitive bias effects)
- Orange boxes: unclamped intermediate values (potential out-of-bounds, see TF-002/003/004)
- Purple/colored trait nodes: the Big Five trait being traced

---

## How to Use These Diagrams

1. **Debugging:** When an IR field has an unexpected value, follow its modifier chain top-to-bottom. Each box shows the formula and operation.
2. **Double-counting check:** If the same trait node appears in both a per-field chain AND an interaction pattern targeting the same field, verify the combined effect is intentional.
3. **Clamp analysis:** Orange boxes mark unclamped gaps. If a value goes out of expected range, check if it passes through an orange box.
4. **Adding new modifiers:** Before adding a new trait influence on any field, check the fan-out diagram to see what already touches that field.
