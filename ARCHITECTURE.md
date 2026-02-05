# Universal Conversational Persona System - Architecture

A psychologically-grounded persona engine that creates behaviorally coherent synthetic humans for testing, research, and simulation.

---

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        A[Client Application]
        C[Python SDK]
    end
    
    subgraph "Core Engine"
        D[Persona Engine]
        
        subgraph "Behavioral Interpreters"
            F[Trait Interpreter<br/>Big 5 → Behavior]
            G[Values Interpreter<br/>Schwartz Values]
            H[Cognitive Interpreter<br/>Decision Style]
            I[State Manager<br/>Mood/Fatigue/Stress]
            BS[Bias Simulator<br/>Confirmation/Negativity/Authority]
        end
        
        subgraph "Turn Planner"
            TP[Turn Planner]
            TC[TraceContext<br/>Citations + Safety]
            IA[Intent Analyzer]
            SG[Stance Generator]
            DD[Domain Detection]
        end
        
        subgraph "Memory"
            SC[Stance Cache]
            MS[Memory System]
        end
    end
    
    subgraph "Output"
        IR[Intermediate<br/>Representation]
        RG[Response Generator<br/>LLM Integration]
        VAL[Validator]
    end
    
    A --> D
    C --> D
    D --> TP
    TP --> F
    TP --> G
    TP --> H
    TP --> I
    TP --> BS
    TP --> TC
    TP --> IA
    TP --> SG
    TP --> DD
    TP --> SC
    TP --> IR
    IR --> RG
    IR --> VAL
    
    style D fill:#4A90E2
    style TP fill:#7ED321
    style IR fill:#F5A623
    style TC fill:#BD10E0
```

---

## Turn Planner Pipeline

The Turn Planner is the heart of the system. It orchestrates all behavioral interpreters to generate a complete IR with full citation trails.

```mermaid
flowchart LR
    subgraph "Input"
        UI[User Input]
        CTX[Conversation<br/>Context]
    end
    
    subgraph "Analysis Phase"
        DD[Domain Detection]
        IA[Intent Analysis]
        TR[Topic Relevance]
        ES[Evidence Strength]
    end
    
    subgraph "Bias Phase"
        BM[Bias Modifiers<br/>±0.15 max]
    end
    
    subgraph "Core Metrics"
        EL[Elasticity<br/>+ Confirmation Bias]
        CF[Confidence<br/>+ Authority Bias]
        TN[Tone<br/>+ Negativity Bias]
        VB[Verbosity]
        DS[Disclosure]
    end
    
    subgraph "Stance & Knowledge"
        ST[Stance Generation<br/>+ Cache Check]
        UA[Uncertainty Action]
        KC[Knowledge Claim Type]
    end
    
    subgraph "Output"
        IR[Intermediate<br/>Representation]
        CIT[Citations Trail]
        SP[Safety Plan]
    end
    
    UI --> DD
    UI --> IA
    UI --> TR
    UI --> ES
    CTX --> DD
    
    TR --> BM
    DD --> EL
    BM --> EL
    BM --> CF
    BM --> TN
    
    EL --> ST
    CF --> IR
    TN --> IR
    VB --> IR
    DS --> IR
    ST --> IR
    UA --> IR
    KC --> IR
    
    IR --> CIT
    IR --> SP
```

---

## Modifier Composition Sequence

All IR parameters follow a canonical modifier sequence to prevent double-counting:

```mermaid
flowchart TD
    subgraph "Canonical Sequence"
        B[1. Base Value<br/>From Persona Profile]
        R[2. Role Adjustment<br/>Social Role Mode]
        T[3. Trait Modifiers<br/>Big 5 Personality]
        S[4. State Modifiers<br/>Mood/Fatigue/Stress]
        BIAS[5. Bias Modifiers<br/>Confirmation/Negativity/Authority]
        C[6. Constraints Clamp<br/>Bounds + Safety]
    end
    
    B --> R --> T --> S --> BIAS --> C
    
    style B fill:#E8F5E9
    style C fill:#FFEBEE
    style BIAS fill:#FFF3E0
```

**Example: Elasticity Calculation**
```
elasticity = base(traits.elasticity)           # 0.60
          → blend(cognitive_complexity)         # 0.55
          → apply(confirmation_bias, -0.06)     # 0.49
          → clamp([0.1, 0.9])                   # 0.49 ✓
```

---

## Psychological Framework

```mermaid
graph TB
    subgraph "Personality (Big 5)"
        O[Openness<br/>Curiosity, creativity]
        C[Conscientiousness<br/>Organization, discipline]
        E[Extraversion<br/>Sociability, assertiveness]
        A[Agreeableness<br/>Compassion, cooperation]
        N[Neuroticism<br/>Emotional stability]
    end
    
    subgraph "Values (Schwartz)"
        SD[Self-Direction]
        ST[Stimulation]
        HE[Hedonism]
        AC[Achievement]
        PO[Power]
        SE[Security]
        CO[Conformity]
        TR[Tradition]
        BE[Benevolence]
        UN[Universalism]
    end
    
    subgraph "Cognitive Style"
        AN[Analytical vs Intuitive]
        RT[Risk Tolerance]
        NC[Need for Closure]
        CC[Cognitive Complexity]
    end
    
    subgraph "Dynamic State"
        MO[Mood<br/>Valence + Arousal]
        FA[Fatigue]
        STR[Stress]
        EN[Engagement]
    end
    
    subgraph "Behavioral Output"
        EL[Elasticity]
        CF[Confidence]
        TN[Tone]
        VB[Verbosity]
        DS[Disclosure]
        ST[Stance]
    end
    
    O --> EL
    C --> CF
    E --> VB
    A --> TN
    N --> STR
    
    CO --> CF
    TR --> CF
    SE --> CF
    
    N --> MO
    MO --> TN
    STR --> TN
    FA --> VB
    EN --> DS
    
    CC --> EL
    RT --> CF
    NC --> UA[Uncertainty Action]
```

---

## Bias Simulation (Phase 2)

Three bounded cognitive biases subtly influence persona behavior:

```mermaid
flowchart LR
    subgraph "Triggers"
        VA[Value Alignment > 0.6]
        NM[Negative Markers<br/>+ Neuroticism > 0.5]
        AM[Authority Markers<br/>+ Conformity > 0.5]
    end
    
    subgraph "Bias Types"
        CB[Confirmation Bias<br/>Reduces Elasticity]
        NB[Negativity Bias<br/>Increases Arousal]
        AB[Authority Bias<br/>Increases Confidence]
    end
    
    subgraph "Bounds"
        MAX[Max Impact: ±0.15]
    end
    
    subgraph "Counters"
        OP[High Openness<br/>Reduces CB]
    end
    
    VA --> CB
    NM --> NB
    AM --> AB
    
    CB --> MAX
    NB --> MAX
    AB --> MAX
    
    OP -.-> CB
    
    style CB fill:#FFCDD2
    style NB fill:#FFE0B2
    style AB fill:#C8E6C9
```

---

## Citation & Tracing System

Every decision is fully traceable via `TraceContext`:

```mermaid
flowchart TD
    subgraph "TraceContext"
        CIT[Citations List]
        SP[Safety Plan]
        CLAMP[Clamp Records]
    end
    
    subgraph "Citation Types"
        BASE[Base Value]
        TRAIT[Trait Modifier]
        STATE[State Modifier]
        RULE[Rule/Policy]
        VALUE[Value Influence]
        GOAL[Goal Influence]
        MEMORY[Memory Reference]
    end
    
    subgraph "Methods"
        NUM["ctx.num() - Numeric"]
        ENUM["ctx.enum() - Enum"]
        CL["ctx.clamp() - Bounds"]
        ADD["ctx.add_basic_citation()"]
    end
    
    NUM --> CIT
    ENUM --> CIT
    CL --> CLAMP
    CL --> CIT
    ADD --> CIT
    
    BASE --> CIT
    TRAIT --> CIT
    STATE --> CIT
    RULE --> CIT
    VALUE --> CIT
    GOAL --> CIT
    MEMORY --> CIT
    
    CLAMP --> SP
```

---

## Package Structure

```
persona_engine/
├── schema/
│   ├── persona_schema.py     # Pydantic persona models
│   └── ir_schema.py          # IR models + citations
├── behavioral/
│   ├── trait_interpreter.py  # Big 5 → behavior
│   ├── values_interpreter.py # Schwartz values
│   ├── cognitive_interpreter.py
│   ├── state_manager.py      # Dynamic state
│   ├── rules_engine.py       # Decision policies
│   ├── bias_simulator.py     # Cognitive biases ← NEW
│   ├── uncertainty_resolver.py
│   └── constraint_safety.py
├── planner/
│   ├── turn_planner.py       # Core orchestration
│   ├── trace_context.py      # Citation tracking
│   ├── intent_analyzer.py    # User intent
│   ├── stance_generator.py   # Stance + rationale
│   └── domain_detection.py   # Topic detection
├── memory/
│   ├── stance_cache.py       # Stance consistency
│   └── fact_store.py         # Typed facts (Phase 4)
└── utils/
    └── determinism.py        # Seeded randomness
```

---

## Data Flow Summary

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│                 Turn Planner                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Domain   │  │ Intent   │  │ Evidence │       │
│  │ Detection│  │ Analysis │  │ Strength │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │              │
│       ▼             ▼             ▼              │
│  ┌─────────────────────────────────────┐        │
│  │        Bias Modifiers (±0.15)       │        │
│  │  Confirmation │ Negativity │ Authority       │
│  └────────────────────┬────────────────┘        │
│                       │                          │
│       ┌───────────────┼───────────────┐         │
│       ▼               ▼               ▼         │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    │
│  │Elasticity│   │Confidence│    │  Tone   │    │
│  └─────────┘    └──────────┘    └─────────┘    │
│       │               │               │         │
│       └───────────────┴───────────────┘         │
│                       │                          │
│                       ▼                          │
│              ┌────────────────┐                  │
│              │ Assemble IR    │                  │
│              │ + Citations    │                  │
│              │ + Safety Plan  │                  │
│              └────────────────┘                  │
└─────────────────────────────────────────────────┘
                       │
                       ▼
            Intermediate Representation
                       │
                       ▼
              Response Generator (LLM)
                       │
                       ▼
                  Final Response
```

---

## Key Design Principles

1. **Single Source of Truth**: Each IR parameter computed by one authoritative process
2. **Canonical Modifier Sequence**: base → role → trait → state → bias → clamp
3. **Full Citation Trail**: Every decision traceable to source
4. **Bounded Biases**: Cognitive biases capped at ±0.15 impact
5. **Deterministic**: Seeded randomness for reproducible behavior
6. **Stance Consistency**: Cache prevents flip-flopping across turns

---

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Schema & Foundation | ✅ Complete |
| 2 | Behavioral Interpreters | ✅ Complete |
| 2 | Bias Simulation | ✅ Complete |
| 3 | Turn Planner | ✅ Complete |
| 3 | TraceContext & Citations | ✅ Complete |
| 4 | Memory System | 🔲 Planned |
| 5 | Response Generator | 🔲 Planned |
| 6 | Validation Suite | 🔲 Planned |
