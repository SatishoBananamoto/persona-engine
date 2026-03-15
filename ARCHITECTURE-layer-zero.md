# Layer Zero — Architecture Document

**Codename:** Layer Zero
**External name:** Persona Minting Machine
**Date:** 2026-03-15
**Status:** PROPOSED — Pre-code simulation. Review before implementation.

---

## What Layer Zero Does

Takes user inputs (text descriptions, structured fields, CSV segment data, or direct toggles) and mints psychologically coherent Persona objects that the persona-engine can run.

```
User Input (any format)
        |
        v
   Layer Zero
        |
        v
N × Persona objects → engine.chat() / engine.plan()
```

**What it does NOT do:**
- Replace the engine — it's a layer on top
- Use LLMs for psychological parameter generation — those come from research-backed mappings
- Produce one "average" persona — produces varied sets by default
- Require any specific input format — accepts text, structured fields, CSV, or direct specification

---

## Target Output Spec

Layer Zero must produce valid `Persona` objects matching the engine's schema. Full field inventory below.

### Critical Fields (50+ parameters across 15 models)

**Identity (6 fields):**
- persona_id (str, auto-generated), age (int, 18-100), gender (str|None), location (str), education (str), occupation (str)

**Big Five Traits (5 fields, all float 0.0-1.0):**
- openness, conscientiousness, extraversion, agreeableness, neuroticism
- ALL actively used by TurnPlanner. Most impactful fields in the entire system.

**Schwartz Values (10 fields, all float 0.0-1.0):**
- self_direction, stimulation, hedonism, achievement, power, security, conformity, tradition, benevolence, universalism
- Drive stance generation, topic relevance, rationale.

**Cognitive Style (5 fields, all float 0.0-1.0):**
- analytical_intuitive, systematic_heuristic, risk_tolerance, need_for_closure, cognitive_complexity
- Influence elasticity, reasoning depth, stance complexity.

**Communication Preferences (4 fields, all float 0.0-1.0):**
- verbosity, formality, directness, emotional_expressiveness

**Knowledge Domains (list of {domain: str, proficiency: float, subdomains: list[str]}):**
- Critical for confidence, competence, expert eligibility.

**Goals (primary + secondary, each with goal: str, weight: float):**
- Drive topic relevance computation.

**Social Roles (dict, must include 'default'):**
- Each role has formality, directness, emotional_expressiveness.

**Policies:**
- Uncertainty: admission_threshold, hedging_frequency, clarification_tendency, knowledge_boundary_strictness
- Claim: allowed_claim_types, lookup_behavior, expert_threshold
- Disclosure: base_openness, factors, bounds

**Safety:**
- Invariants: identity_facts (list[str]), cannot_claim (list[str]), must_avoid (list[str])

**State:**
- Initial: mood_valence (-1 to 1), mood_arousal (0-1), fatigue (0-1), stress (0-1), engagement (0-1)

**Behavioral Rules:**
- decision_policies, response_patterns, biases (all lists, optional)

**Unused/Deferred by engine (fill with defaults):**
- languages, cultural_knowledge, time_scarcity, self_schemas, topic_sensitivities

---

## Input Tiers

Layer Zero accepts 4 tiers of input, each activating different levels of inference.

### Tier 1: Text Description
```python
personas = layer_zero.from_description(
    "A 35-year-old product manager in fintech who values innovation",
    count=5
)
```
- Parsed via regex + heuristics (no heavy NLP)
- Extracts: age, occupation, industry, keywords
- Everything else inferred from demographic priors + occupation mappings

### Tier 2: Structured Fields
```python
personas = layer_zero.mint(
    age=35,
    occupation="product manager",
    industry="fintech",
    location="San Francisco, US",
    traits=["analytical", "risk-tolerant"],
    count=10
)
```
- Explicit fields used directly
- Trait adjectives mapped to Big Five adjustments
- Gaps filled by inference

### Tier 3: CSV Segment Data
```python
personas = layer_zero.from_csv("segments.csv", count_per_segment=20)
```
- Each row defines a segment (age_range, gender_dist, occupation, region, etc.)
- Generates N personas per segment
- Supports batch processing

### Tier 4: Direct Specification
```python
personas = layer_zero.mint(
    age=35,
    occupation="product manager",
    big_five={"openness": 0.82, "conscientiousness": 0.65, ...},
    values={"self_direction": 0.78, "security": 0.31, ...},
    count=1
)
```
- User provides exact psychological parameters
- Minimal inference — only gap-filling for unspecified fields
- Power-user mode

**Design principle:** Higher tiers override lower-tier inference. If the user specifies `openness=0.82`, that value is used directly — no demographic prior overrides it. Explicit always wins.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        LAYER ZERO                            │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐  │
│  │  Input   │──>│ Demographic│──>│ Sampler │──>│   Gap     │  │
│  │  Parser  │   │  Prior    │   │         │   │   Filler  │  │
│  └─────────┘   │  Engine   │   └─────────┘   └───────────┘  │
│                 └──────────┘                       │          │
│                                                    v          │
│                 ┌──────────┐   ┌─────────┐   ┌───────────┐  │
│                 │ Narrative │<──│ Persona │<──│Consistency│  │
│                 │ Enricher │   │Assembler│   │ Validator │  │
│                 │(optional)│   │         │   │           │  │
│                 └──────────┘   └─────────┘   └───────────┘  │
│                      │                                       │
│                      v                                       │
│               N × Persona objects                            │
└──────────────────────────────────────────────────────────────┘
```

### Stage 1: Input Parser

**Job:** Normalize any input format into a common `MintRequest` intermediate representation.

```python
@dataclass
class MintRequest:
    # Explicit inputs (None = not specified, infer later)
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    industry: str | None = None
    location: str | None = None
    education: str | None = None
    culture_region: str | None = None

    # Trait hints (adjectives like "analytical", "warm")
    trait_hints: list[str] = field(default_factory=list)

    # Direct overrides (user-specified psychological parameters)
    big_five_overrides: dict[str, float] = field(default_factory=dict)
    values_overrides: dict[str, float] = field(default_factory=dict)
    cognitive_overrides: dict[str, float] = field(default_factory=dict)
    communication_overrides: dict[str, float] = field(default_factory=dict)

    # Goals and domains
    goals: list[str] = field(default_factory=list)
    domains: list[dict] = field(default_factory=list)

    # Generation settings
    count: int = 1
    seed: int | None = None
```

**Parsing by tier:**
- Tier 1 (text): Regex extracts age, occupation, location, trait adjectives → populates MintRequest
- Tier 2 (structured): Direct field mapping → MintRequest
- Tier 3 (CSV): Each row → MintRequest, batch processed
- Tier 4 (direct): Overrides go into `*_overrides` dicts

### Stage 2: Demographic Prior Engine

**Job:** Map demographics to psychological trait distributions (mean + standard deviation for each trait).

This is where the research data lives. It does NOT produce point estimates — it produces distributions that the Sampler draws from.

**Mapping sources (all from published research):**

#### Big Five Priors

| Signal | Source | Strength | What It Gives |
|--------|--------|----------|---------------|
| Occupation → RIASEC → Big Five | Holland (1997), Barrick & Mount (1991) | r = 0.19-0.48 | Mean shifts per trait |
| Age → trait trajectories | Srivastava (2003), PMC7869960 | Moderate | Mean shifts (A,C up; E,O,N down with age) |
| Gender → trait differences | Schmitt (2008), Kajonius & Johnson (2018) | d = 0.1-0.65 | Mean shifts, culture-dependent |
| Culture region → trait norms | Cross-cultural Big Five studies | Moderate | Baseline means per region |

**Implementation:**
```python
def compute_big_five_prior(request: MintRequest) -> dict[str, tuple[float, float]]:
    """Returns {trait: (mean, std_dev)} for each Big Five trait."""
    # Start with population baseline: mean=0.5, sd=0.15
    priors = {t: (0.5, 0.15) for t in BIG_FIVE_TRAITS}

    # Apply occupation shift (strongest signal)
    if request.occupation:
        shifts = OCCUPATION_TRAIT_MAP.get(normalize_occupation(request.occupation), {})
        for trait, delta in shifts.items():
            priors[trait] = (priors[trait][0] + delta, priors[trait][1])

    # Apply age shift
    if request.age:
        age_shifts = compute_age_shifts(request.age)
        for trait, delta in age_shifts.items():
            priors[trait] = (priors[trait][0] + delta, priors[trait][1])

    # Apply user overrides (these REPLACE the prior, not shift it)
    for trait, value in request.big_five_overrides.items():
        priors[trait] = (value, 0.02)  # Tiny SD = nearly fixed

    return priors
```

#### Schwartz Value Priors

| Signal | Source | Strength |
|--------|--------|----------|
| Age → value trajectories | Schwartz (2011), PMC5549227 | Medium |
| Occupation → value profiles | Springer (2007) | Medium |
| Culture region → Inglehart-Welzel map | World Values Survey | Strong |

#### Cognitive Style Priors

No direct demographic correlations published. Inferred from:
- Big Five traits (Openness → analytical style, Conscientiousness → systematic)
- Occupation (engineer → systematic, artist → heuristic)

### Stage 3: Sampler

**Job:** Draw N trait vectors from the computed distributions, respecting inter-trait correlations.

**Method:** Multivariate normal sampling with the published Big Five correlation matrix.

**The correlation matrix (van der Linden et al., 2010 meta-analysis, K=212, N=144,117):**

```python
BIG_FIVE_CORRELATION_MATRIX = np.array([
    # O      C      E      A      N
    [1.00,  0.14,  0.43,  0.13, -0.12],  # Openness
    [0.14,  1.00,  0.18,  0.43, -0.29],  # Conscientiousness
    [0.43,  0.18,  1.00,  0.17, -0.25],  # Extraversion
    [0.13,  0.43,  0.17,  1.00, -0.22],  # Agreeableness
    [-0.12, -0.29, -0.25, -0.22, 1.00],  # Neuroticism
])
```

**Sampling process:**
1. Get prior means and SDs from Demographic Prior Engine
2. Convert correlation matrix to covariance matrix: `Cov = diag(SD) @ Corr @ diag(SD)`
3. Sample N vectors: `numpy.random.Generator.multivariate_normal(means, cov, size=N)`
4. Clamp to [0.0, 1.0] (truncated normal — resample if outside bounds)

**For Schwartz values:** Sample independently but apply circle constraints post-hoc (see Validator).

**For count=1 with full overrides:** No sampling needed. Use specified values directly.

### Stage 4: Gap Filler

**Job:** Compute all remaining fields that the Persona schema requires but the user didn't specify.

**What it fills:**

| Category | How Filled |
|----------|------------|
| Cognitive style | Derived from Big Five (O → analytical, C → systematic, etc.) |
| Communication prefs | Derived from Big Five (E → expressive, A → indirect, C → formal) |
| Knowledge domains | Inferred from occupation + industry |
| Goals | Inferred from occupation + values |
| Social roles | Template: default + at_work + friend, params derived from communication prefs |
| Uncertainty policy | Derived from C, directness, analytical style |
| Claim policy | Template with expert_threshold=0.7, lookup_behavior from cognitive style |
| Disclosure policy | Derived from A, O, privacy_sensitivity |
| Invariants | Auto-generated: identity_facts from demographics, cannot_claim from occupation, must_avoid defaults |
| Initial state | mood_valence=0.1, mood_arousal from E, stress from N, fatigue=0.3, engagement=0.6 |
| Biases | Confirmation from values alignment, authority from conformity, negativity from N |
| Decision policies | Template based on occupation + cognitive style |
| Response patterns | Template based on A, E, N |

**Key principle:** Gap Filler never overrides explicit user input. It only fills `None` fields.

### Stage 5: Consistency Validator

**Job:** Check that the assembled profile is psychologically coherent. Flag or fix contradictions.

**10 enforceable rules:**

#### Rule 1: Big Five Metatrait Coherence
The Stability metatrait (A + C + low-N) should show internal coherence.
- Flag if: A > 0.8 AND C > 0.8 AND N > 0.8 (all three high is statistically rare)
- The Plasticity metatrait (E + O) should correlate positively.
- Flag if: E > 0.8 AND O < 0.2 (or vice versa) — rare but not impossible

#### Rule 2: Cognitive Style Compatibility
- Flag if: `risk_tolerance > 0.7 AND need_for_closure > 0.7` (empirically incompatible)

#### Rule 3: Schwartz Adjacent Value Coherence
Adjacent values on the circle should not differ by more than ~0.5.
- Circle order: Self-Direction → Stimulation → Hedonism → Achievement → Power → Security → Conformity → Tradition → Benevolence → Universalism → (back)
- Flag if: any adjacent pair differs by > 0.5

#### Rule 4: Schwartz Opposing Value Conflict
- Flag if: both values in an opposing pair exceed 0.7
- Opposing pairs: Power↔Universalism, Achievement↔Benevolence, Self-Direction↔Conformity, Stimulation↔Tradition, etc.

#### Rule 5: Schwartz Sinusoidal Check
Plot values in circle order. Fit a sinusoid. Flag if R² < 0.3 (random noise rather than structured value system).

#### Rule 6: Big Five × Schwartz Cross-Check
Known correlation directions must hold:
- High Openness should correlate with high Self-Direction, Stimulation, Universalism — flag if O > 0.8 but all three < 0.3
- High Agreeableness should correlate with high Benevolence — flag if A > 0.8 but Benevolence < 0.3
- High Conscientiousness should correlate with high Security, Achievement — flag if C > 0.8 but both < 0.3

#### Rule 7: Domain-Expertise Consistency
- Flag if: a knowledge domain has proficiency > 0.7 but no plausible connection to occupation

#### Rule 8: Disclosure-Privacy Coherence
- Flag if: disclosure_policy.base_openness > 0.8 AND privacy_sensitivity > 0.8

#### Rule 9: Batch Diversity Check (when generating multiple personas)
- Compute pairwise trait distance across generated set
- Flag if: minimum pairwise distance < threshold (too similar = no diversity)

#### Rule 10: Cultural Confidence Scoring
- For non-WEIRD populations where Big Five validity is lower, attach a confidence score
- Flag with reduced confidence, don't block generation

**Validator modes:**
- `strict`: Contradictions raise errors, generation fails
- `warn` (default): Contradictions logged as warnings, generation continues
- `silent`: No checks, fastest path

### Stage 6: Persona Assembler

**Job:** Take the validated trait vectors + filled fields and construct engine-compatible `Persona` objects.

Straightforward assembly — map internal representation to Pydantic model. This is where we call `Persona(**fields)` and let Pydantic v2 do final schema validation.

Each assembled Persona gets:
- A unique `persona_id` (hash of name + occupation + seed)
- A generated `label` ("{name} - {occupation}, {location}")
- Auto-generated `identity_facts` from demographics
- All 50+ fields populated

### Stage 7: Narrative Enricher (Optional)

**Job:** Add human-readable background story and natural-sounding response patterns. Only if LLM is available.

**Rules:**
- Psychological parameters are LOCKED before this stage — enricher cannot change them
- Enricher adds: background narrative, expanded identity_facts, natural response_patterns
- Uses minimal prompting (census-style, not creative writing — per NeurIPS 2025 findings)
- Skipped entirely if no LLM provider configured

---

## Tech Stack

### Required Dependencies (mandatory, ~25MB total)

| Library | Purpose | Version | Size |
|---------|---------|---------|------|
| pydantic | Model validation, schema | >=2.6 | ~5MB |
| numpy | Multivariate normal sampling, correlation matrices | >=1.24 | ~20MB |

### Built-in (stdlib, zero overhead)

| Module | Purpose |
|--------|---------|
| csv | CSV segment ingestion |
| re | Text description parsing |
| hashlib | Persona ID generation |
| dataclasses | Internal data structures |

### Optional Dependencies

| Library | Purpose | When | Size |
|---------|---------|------|------|
| spacy + en_core_web_sm | Advanced text parsing | `pip install layer-zero[nlp]` | ~200MB |
| typer | CLI interface | `pip install layer-zero[cli]` | ~2MB |
| scipy | Truncated distributions | `pip install layer-zero[stats]` | ~40MB |
| mcp | MCP server mode | Deferred to post-v1 | ~5MB |

### Explicitly NOT Using

| Library | Why Not |
|---------|---------|
| pandas | Too heavy (~50MB) for reading segment CSVs. stdlib csv is sufficient. |
| transformers/torch | No GPU on Chromebook. No LLM inference in core pipeline. |
| LangChain | Over-engineering. Direct API calls if LLM needed. |

---

## API Design

### Primary Interface — Module-level functions

```python
import layer_zero

# Tier 1: From text
personas = layer_zero.from_description(
    "A 35-year-old product manager in fintech who values innovation",
    count=5,
    seed=42,
)

# Tier 2: From structured fields
personas = layer_zero.mint(
    age=35,
    occupation="product manager",
    industry="fintech",
    location="San Francisco, US",
    traits=["analytical", "risk-tolerant"],
    count=10,
)

# Tier 3: From CSV
personas = layer_zero.from_csv("segments.csv", count_per_segment=20)

# Tier 4: Direct specification
persona = layer_zero.mint(
    age=35,
    occupation="product manager",
    big_five={"openness": 0.82, "conscientiousness": 0.65,
              "extraversion": 0.55, "agreeableness": 0.48, "neuroticism": 0.30},
    count=1,
)

# Output: list of Persona objects (engine-compatible)
for p in personas:
    engine = PersonaEngine(p, llm_provider="template")
    result = engine.chat("Tell me about your approach to product decisions")
```

### Configuration

```python
# Validator strictness
personas = layer_zero.mint(occupation="nurse", validate="strict")   # errors on contradiction
personas = layer_zero.mint(occupation="nurse", validate="warn")     # warnings only (default)
personas = layer_zero.mint(occupation="nurse", validate="silent")   # no checks

# Reproducibility
personas = layer_zero.mint(occupation="nurse", count=10, seed=42)  # deterministic

# Export
layer_zero.to_yaml(personas, "output/")       # one YAML per persona
layer_zero.to_json(personas, "personas.json")  # single JSON array
```

### Why Factory Functions, Not Builder Pattern

- Python has keyword arguments — they solve the "many optional params" problem that builders address in Java
- Factory functions are idiomatic Python, composable, debuggable
- Type checkers work naturally with functions + Pydantic models
- Guido van Rossum explicitly discourages `return self` chaining in Python

---

## File Structure

```
layer_zero/
├── __init__.py              # Public API: mint(), from_description(), from_csv()
├── parser/
│   ├── __init__.py
│   ├── text_parser.py       # Regex-based description parsing → MintRequest
│   ├── csv_parser.py        # CSV segment ingestion → list[MintRequest]
│   └── models.py            # MintRequest dataclass
├── priors/
│   ├── __init__.py
│   ├── big_five.py          # Occupation/age/gender → Big Five distributions
│   ├── values.py            # Occupation/age/culture → Schwartz distributions
│   ├── cognitive.py         # Big Five + occupation → cognitive style
│   └── data/
│       ├── occupation_traits.json    # Occupation → RIASEC → Big Five mapping
│       ├── age_trajectories.json     # Age → trait shift curves
│       ├── culture_baselines.json    # Culture region → value baselines
│       └── correlation_matrix.json   # Big Five inter-correlation matrix
├── sampler.py               # Multivariate normal sampling with correlations
├── gap_filler.py            # Fill unspecified fields from traits + occupation
├── validator.py             # 10 consistency rules
├── assembler.py             # Construct engine Persona objects
├── enricher.py              # Optional LLM narrative enrichment
├── export.py                # YAML/JSON output
└── cli.py                   # Optional CLI (typer-based)
```

**~12 files.** Lean structure. No bloat.

---

## Ethical Guardrails

Based on research findings (NeurIPS 2025, arXiv:2602.03334):

### 1. Demographics Don't Determine Personality
Demographic priors shift the DISTRIBUTION, not the OUTCOME. A 25-year-old woman does not get assigned specific Big Five scores — she gets a distribution to sample from, and the sample may be anywhere within plausible bounds.

### 2. No Stereotype Amplification
- Occupation mappings are based on published RIASEC correlations (r = 0.19-0.48), not stereotypes
- Age/gender shifts are based on meta-analytic data, applied as small mean adjustments
- When generating sets, diversity constraints ensure no single demographic pattern dominates

### 3. Contradictions Are Surfaced, Not Resolved
"Introverted salesperson" is flagged as unusual, not auto-corrected. Real people have contradictions. The validator warns but does not override explicit user input.

### 4. Cultural Context Acknowledged
- Non-WEIRD populations get a reduced confidence flag
- Big Five validity is lower in non-Western contexts — the system acknowledges this
- Culture region affects value priors (via Inglehart-Welzel data) but not personality defaults

### 5. LLM Bias Mitigation
- LLMs are NEVER used for psychological parameter generation
- Narrative enricher operates AFTER parameters are locked
- No creative backstory generation — minimal census-style prompting

---

## Integration with Persona Engine

### Direct Python Integration
```python
from layer_zero import mint
from persona_engine import PersonaEngine

personas = mint(occupation="data scientist", count=5)
for persona in personas:
    engine = PersonaEngine(persona, llm_provider="template")
    result = engine.chat("How do you approach ambiguous data?")
    print(f"{persona.label}: {result.text}")
```

### YAML Export → Engine Load
```python
layer_zero.to_yaml(personas, "personas/generated/")
# Later...
engine = PersonaEngine.from_yaml("personas/generated/persona_001.yaml")
```

### Future: MCP Server Mode
```python
# Deferred to post-v1, but API is MCP-ready:
# - All functions return Pydantic models (JSON-serializable)
# - All functions are stateless
# - Type hints everywhere (FastMCP auto-generates schemas)
```

---

## What's NOT in v1

| Feature | Why Deferred |
|---------|-------------|
| MCP server mode | Core value is generation logic, not transport. ~20 lines to add later. |
| spaCy NLP | Regex covers 80% of text parsing. Optional extra if needed. |
| Behavioral data ingestion | Inferring personality from purchase/browsing data is ethically complex and accuracy-limited (r = 0.29-0.40). Phase 2. |
| Multi-language personas | Engine doesn't use the languages field yet. |
| Interactive persona refinement | "Make this persona more assertive" — useful but not MVP. |
| Web UI | CLI + Python API first. |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Demographic mappings produce stereotypical personas | High | High | Distributions not point estimates. Diversity constraints. Batch checks. |
| Generated personas fail engine schema validation | Medium | High | Assemble via Pydantic model — schema errors caught immediately. |
| Schwartz value profiles psychologically incoherent | Medium | Medium | Sinusoidal check + opposing value check in validator. |
| Occupation not in mapping table | High | Low | Graceful fallback to "general" with population defaults. |
| User provides contradictory input | Medium | Low | Validator warns but respects user intent. |
| Multivariate sampling produces out-of-bounds values | Certain | Low | Clamp to [0,1] with resample-if-extreme strategy. |

---

## Implementation Order

| Phase | What | Files | Tests |
|-------|------|-------|-------|
| 1 | MintRequest model + text parser + structured parser | parser/*.py | Input parsing tests |
| 2 | Big Five prior engine + occupation mapping data | priors/big_five.py, data/ | Prior computation tests |
| 3 | Sampler (multivariate normal + correlation matrix) | sampler.py | Sampling distribution tests |
| 4 | Schwartz value priors + age/culture mappings | priors/values.py | Value prior tests |
| 5 | Gap filler (cognitive, communication, policies, invariants) | gap_filler.py | Gap filling tests |
| 6 | Consistency validator (10 rules) | validator.py | Validation rule tests |
| 7 | Persona assembler + engine integration | assembler.py | End-to-end persona creation |
| 8 | CSV parser | parser/csv_parser.py | CSV ingestion tests |
| 9 | Export (YAML/JSON) | export.py | Export format tests |
| 10 | CLI (optional) | cli.py | CLI smoke tests |

---

## Success Criteria

Layer Zero v1 is done when:

1. `layer_zero.mint(occupation="nurse", count=10)` produces 10 distinct, valid Persona objects
2. All generated personas pass engine schema validation (`Persona(**fields)` succeeds)
3. Generated personas produce measurably different IR values when run through the engine
4. Consistency validator catches known contradictions (high risk-tolerance + high need-for-closure)
5. Twin test: two personas from same segment are different; two personas from different segments are more different
6. CSV ingestion works for a 10-row segment file
7. Text parsing handles: "35-year-old product manager in fintech"
8. Seed produces deterministic output: same seed + same input = same personas
9. Total install footprint < 30MB (pydantic + numpy)
10. All tests pass, 0 type errors
