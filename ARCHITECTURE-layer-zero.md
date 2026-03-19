# Layer Zero — Architecture Document (v2)

**Codename:** Layer Zero
**External name:** Persona Minting Machine
**Date:** 2026-03-15
**Revised:** 2026-03-15 (v2 — incorporates external review fixes)
**Status:** REVISED — Addresses 5 blocking issues from external review.

---

## Revision Summary

v2 incorporates fixes from external review. Key changes:

1. **Three-layer model** — Evidence, Persona, and Policy layers separated. Policy fields are system-governed defaults, not persona-derived.
2. **Logit-normal sampling** — Replaces multivariate normal + clamp. Preserves correlations at boundaries.
3. **Calibrated residual variance** — Derived fields get independent residual noise at every step, preventing cascade collapse.
4. **Provenance tracking** — Every field carries source type, mapping strength, and inferential depth.
5. **Separate SegmentRequest** — CSV segments model ranges/distributions, not point values.
6. **Decision tendencies** — Renamed from "biases" to avoid ethically loaded language.
7. **Domain proficiency defaults lowered** — Occupation gives familiarity (0.4-0.6), not expertise (0.7+).
8. **Cascade collapse test** — Added to success criteria.

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
- Weaken safety/policy guardrails based on personality — policy is system-governed

---

## The Three-Layer Model

Every generated persona is composed of three distinct layers with different governance:

```
┌─────────────────────────────────────────────────────────┐
│  EVIDENCE LAYER                                         │
│  Source: User input, parsed input, demographic priors   │
│  Contains: Explicit inputs, inferred demographics,      │
│            confidence scores, provenance metadata        │
│  Governance: Data-driven, transparent                    │
└─────────────────────────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│  PERSONA LAYER                                          │
│  Source: Sampled from priors + residual variance         │
│  Contains: Big Five, Schwartz values, cognitive style,  │
│            communication prefs, goals, identity          │
│  Governance: Research-backed distributions + variation   │
│  Changes per persona: YES                                │
└─────────────────────────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│  POLICY LAYER                                           │
│  Source: System defaults with persona-bounded style      │
│  Contains: claim_policy, disclosure_policy, uncertainty, │
│            invariants, safety constraints                 │
│  Governance: System-controlled safety floors             │
│  Changes per persona: PRESENTATION only, not permissions │
└─────────────────────────────────────────────────────────┘
```

**Critical design rule:** Policy fields are system-governed defaults with persona-bounded presentation, not persona-derived permissions. A high-openness persona gets a more exploratory communication *style*, but the same safety floor as a low-openness persona. Personality shapes how they talk, not whether they can fabricate expertise or weaken epistemic guardrails.

**What this means concretely:**
- `claim_policy.expert_threshold` = system default (0.7), NOT derived from personality
- `claim_policy.lookup_behavior` = system default ("hedge"), NOT derived from cognitive style
- `disclosure_policy.bounds` = system default ((0.1, 0.9)), NOT derived from openness
- `uncertainty.knowledge_boundary_strictness` = system default (0.7), NOT derived from conscientiousness
- `invariants.must_avoid` = system defaults + occupation-specific, NOT persona-variable

**What personality DOES influence (within policy bounds):**
- Communication style (formality, directness, verbosity, emotional expressiveness)
- Tone, stance, and rationale framing
- Initial state (mood, arousal, engagement)
- Decision tendencies (mild behavioral leanings, not permission changes)

---

## The Cascade Problem and Our Solution

### The Problem

The original architecture sampled Big Five traits once, then deterministically derived cognitive style, communication preferences, policies, and behavioral rules from those traits. This creates cascade collapse: stochasticity enters only at the trait-sampling stage, while many downstream fields are deterministic projections of those same traits. That makes downstream fields overly correlated, compresses conditional diversity, and causes personas from the same segment to converge on similar behavioral rules and policies.

### The Solution: Calibrated Residual Variance

Each derived field is computed from parent traits **plus calibrated residual variance**, with optional shared latent residuals for related fields.

```
Big Five: sampled from prior               (stochastic, SD from prior engine)
    ↓
Cognitive style: f(Big Five) + residual     (SD=0.08, shared residual for related fields)
    ↓
Communication: f(Big Five) + residual       (SD=0.08, shared residual for related fields)
    ↓
Goals: f(occupation, values) + residual     (SD=0.05)
    ↓
Initial state: f(Big Five) + residual       (SD=0.05)
```

**Why "calibrated residual" not "independent noise":** Fully independent noise makes the persona incoherent. Communication style, cognitive style, and social-role behavior should not all wobble independently. Related fields share latent residuals so they vary together but still differ from the parent traits.

**Implementation:** At each derivation step, sample a residual from N(0, σ²) where σ is calibrated per field. Related fields (e.g., formality and directness) share a latent factor that shifts them in correlated directions.

---

## Provenance Tracking

Every generated field carries provenance and confidence based on source type, mapping strength, and inferential depth.

```python
@dataclass(frozen=True)
class FieldProvenance:
    value: float | str | list | dict
    source: Literal["explicit", "sampled", "derived", "template", "default"]
    confidence: float          # 0.0-1.0, based on multiple factors
    mapping_strength: float    # strength of the empirical mapping used
    inferential_depth: int     # steps from direct evidence
    parent_fields: list[str]   # what this was derived from
    notes: str = ""            # e.g., "from RIASEC-Big Five mapping, r=0.48"
```

**Confidence computation:**
- `explicit` (user provided): confidence = 0.95
- `sampled` (drawn from researched prior): confidence = 0.6-0.8 (depends on prior strength)
- `derived` (computed from other fields): confidence = parent_confidence × mapping_strength × depth_decay
- `template` (occupation/role template): confidence = 0.4
- `default` (population baseline): confidence = 0.3

**Depth decay factor:** 0.85 per hop. A field derived 3 steps from evidence: base_confidence × 0.85³ ≈ 0.61 × base.

**Why this matters:**
- Debugging: "Why does this persona have high risk_tolerance?" → "Derived from openness=0.82, mapping_strength=0.35, depth=2, confidence=0.42"
- Ethical review: Fields with confidence < 0.3 are flagged as weakly grounded
- Transparency: Users can see which fields are evidence-backed vs. inferred guesswork

---

## Target Output Spec

Layer Zero must produce valid `Persona` objects matching the engine's schema. Full field inventory below.

### Persona Layer Fields (vary per persona)

**Identity (7 fields):**
- persona_id (str, auto-generated: content hash + batch index), name (str, generated or user-provided), age (int, 18-100), gender (str|None), location (str), education (str), occupation (str)

**Big Five Traits (5 fields, all float 0.0-1.0):**
- openness, conscientiousness, extraversion, agreeableness, neuroticism
- ALL actively used by TurnPlanner. Most impactful fields in the entire system.

**Schwartz Values (10 fields, all float 0.0-1.0):**
- self_direction, stimulation, hedonism, achievement, power, security, conformity, tradition, benevolence, universalism
- Generated from circumplex structure (angle + amplitude), not independent sampling.

**Cognitive Style (5 fields, all float 0.0-1.0):**
- analytical_intuitive, systematic_heuristic, risk_tolerance, need_for_closure, cognitive_complexity
- Derived from Big Five + occupation + calibrated residual variance.

**Communication Preferences (4 fields, all float 0.0-1.0):**
- verbosity, formality, directness, emotional_expressiveness
- Derived from Big Five + calibrated residual variance.

**Knowledge Domains (list of {domain: str, proficiency: float, subdomains: list[str]}):**
- Occupation gives familiarity (proficiency 0.4-0.6), NOT expertise (0.7+)
- Only user-declared expertise or explicit domain specification gets proficiency > 0.7

**Goals (primary + secondary, each with goal: str, weight: float):**
- Inferred from occupation + values + residual variance.

**Social Roles (dict, must include 'default'):**
- Each role has formality, directness, emotional_expressiveness.
- Template: default + at_work + friend, params from communication prefs + residual.

**Initial State:**
- mood_valence (-1 to 1), mood_arousal (0-1), fatigue (0-1), stress (0-1), engagement (0-1)
- Derived from Big Five + residual variance.

**Decision Tendencies** (renamed from "biases" — ethically safer):
- Mild behavioral leanings, not cognitive defects
- Derived from values + traits, low confidence scores, internal-only in v1

### Policy Layer Fields (system-governed, same safety floor for all personas)

**Uncertainty Policy:**
- admission_threshold, hedging_frequency, clarification_tendency, knowledge_boundary_strictness
- System defaults. Persona layer adjusts presentation style only.

**Claim Policy:**
- allowed_claim_types = ["personal_experience", "general_common_knowledge"] (system default)
- lookup_behavior = "hedge" (system default)
- expert_threshold = 0.7 (system default, NOT persona-variable)

**Disclosure Policy:**
- base_openness: persona-influenced (from A + O)
- bounds: system default (0.1, 0.9)
- factors: system default

**Invariants:**
- identity_facts: auto-generated from demographics
- cannot_claim: occupation-derived + system defaults
- must_avoid: system defaults + occupation-specific

### Metadata

**Provenance object:** Attached to every generated Persona, mapping field_name → FieldProvenance.

**Unused/Deferred by engine (fill with defaults):**
- languages, cultural_knowledge, time_scarcity, self_schemas, topic_sensitivities

---

## Input Tiers

Layer Zero accepts 4 tiers of input, each activating different levels of inference.

### Tier 1: Text Description
```python
personas = layer_zero.from_description(
    "A 35-year-old product manager in fintech who values innovation",
    count=5,
)
```
- Parsed via regex + heuristics (no heavy NLP)
- Extracts: age, occupation, industry, keywords
- Everything else inferred from demographic priors + occupation mappings
- Known limitation: regex covers ~80% of natural descriptions. Negation, mixed roles, sarcasm not handled. Documented.

### Tier 2: Structured Fields
```python
personas = layer_zero.mint(
    name="Alex",
    age=35,
    occupation="product manager",
    industry="fintech",
    location="San Francisco, US",
    traits=["analytical", "risk-tolerant"],
    count=10,
)
```
- Explicit fields used directly
- Trait adjectives mapped to Big Five adjustments
- Gaps filled by inference with residual variance

### Tier 3: CSV Segment Data
```python
personas = layer_zero.from_csv("segments.csv", count_per_segment=20)
```
- Uses `SegmentRequest` model (not `MintRequest`) — supports ranges and distributions:
  - `age_range: tuple[int, int]` (e.g., (30, 45))
  - `gender_distribution: dict[str, float]` (e.g., {"female": 0.6, "male": 0.35, "non-binary": 0.05})
  - `occupation: str | list[str]` (single or multiple)
- Each row defines a segment, generates N personas per segment
- Supports batch processing

### Tier 4: Direct Specification
```python
persona = layer_zero.mint(
    name="Alex",
    age=35,
    occupation="product manager",
    big_five={"openness": 0.82, "conscientiousness": 0.65,
              "extraversion": 0.55, "agreeableness": 0.48, "neuroticism": 0.30},
    values={"self_direction": 0.78, "security": 0.31},
    count=1,
)
```
- User provides exact psychological parameters
- Minimal inference — only gap-filling for unspecified fields
- Power-user mode

**Design principle:** Higher tiers override lower-tier inference. If the user specifies `openness=0.82`, that value is used directly — no demographic prior overrides it. Explicit always wins.

---

## Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          LAYER ZERO                            │
│                                                                │
│  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐  │
│  │  Input   │──>│ Demographic│──>│ Sampler  │──>│   Gap     │  │
│  │  Parser  │   │  Prior     │   │ (logit-  │   │  Filler   │  │
│  │          │   │  Engine    │   │  normal) │   │ +residuals│  │
│  └─────────┘   └───────────┘   └──────────┘   └───────────┘  │
│                                                     │          │
│                                                     v          │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐   │
│  │ Narrative  │<─│ Persona  │<─│Consistency│<─│  Policy  │   │
│  │ Enricher  │  │ Assembler│  │ Validator │  │  Applier │   │
│  │ (optional)│  │          │  │           │  │          │   │
│  └───────────┘  └──────────┘  └───────────┘  └──────────┘   │
│       │                                                       │
│       v                                                       │
│  N × Persona objects + Provenance metadata                    │
└────────────────────────────────────────────────────────────────┘
```

### Stage 1: Input Parser

**Job:** Normalize any input format into `MintRequest` (individual) or `SegmentRequest` (CSV segments).

```python
@dataclass
class MintRequest:
    # Explicit inputs (None = not specified, infer later)
    name: str | None = None
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


@dataclass
class SegmentRequest:
    """For CSV segments with ranges and distributions."""
    segment_name: str = ""
    age_range: tuple[int, int] = (25, 55)
    gender_distribution: dict[str, float] = field(default_factory=lambda: {"female": 0.5, "male": 0.5})
    occupations: list[str] = field(default_factory=list)
    location: str | None = None
    culture_region: str | None = None

    # Optional trait constraints (ranges, not point values)
    trait_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    count: int = 10
    seed: int | None = None
```

### Stage 2: Demographic Prior Engine

**Job:** Map demographics to psychological trait distributions (mean + standard deviation per trait).

Produces distributions, not point estimates. Same research data as v1.

| Signal | Source | Strength | What It Gives |
|--------|--------|----------|---------------|
| Occupation → RIASEC → Big Five | Holland, Barrick & Mount | r = 0.19-0.48 | Mean shifts per trait |
| Age → trait trajectories | Srivastava, PMC7869960 | Moderate | Mean shifts (A,C up; E,O,N down) |
| Gender → trait differences | Schmitt, Kajonius & Johnson | d = 0.1-0.65 | Mean shifts, culture-dependent |
| Culture region → trait norms | Cross-cultural studies | Moderate | Baseline means per region |
| Culture region → value norms | Inglehart-Welzel, ESS | Strong | Value baseline per region |

**Note on culture:** Culture region affects BOTH Big Five norms AND Schwartz value priors, via published cross-cultural data. For under-studied populations, confidence is reduced. This is consistent — not contradictory.

### Stage 3: Sampler

**Job:** Draw N trait vectors from computed distributions, respecting inter-trait correlations.

#### Big Five: Logit-Normal Sampling

**Why logit-normal:** Multivariate normal + clamp distorts correlations at boundaries. Logit-normal samples in unbounded space then applies sigmoid to [0,1], preserving the correlation structure.

**Process:**
1. Get prior means and SDs from Demographic Prior Engine
2. Transform means to logit space: `logit_mean = log(mean / (1 - mean))`
3. Build covariance matrix in logit space from published correlation matrix
4. Sample N vectors in logit space: `numpy.random.Generator.multivariate_normal(logit_means, logit_cov, size=N)`
5. Apply sigmoid: `traits = 1 / (1 + exp(-logit_samples))`
6. Result: bounded [0,1] values with preserved correlation structure

**The correlation matrix (van der Linden et al., 2010, K=212, N=144,117):**

```python
BIG_FIVE_CORRELATION = np.array([
    # O      C      E      A      N
    [1.00,  0.14,  0.43,  0.13, -0.12],  # Openness
    [0.14,  1.00,  0.18,  0.43, -0.29],  # Conscientiousness
    [0.43,  0.18,  1.00,  0.17, -0.25],  # Extraversion
    [0.13,  0.43,  0.17,  1.00, -0.22],  # Agreeableness
    [-0.12, -0.29, -0.25, -0.22, 1.00],  # Neuroticism
])
```

#### Schwartz Values: Circumplex-Structured Generation

**Why not independent sampling:** The 10 values follow a circular structure where adjacent values correlate positively and opposing values correlate negatively. Sampling independently then policing post-hoc is ad hoc.

**Process:**
1. Sample a peak angle θ (where on the circle is the persona's "North Star")
2. Sample an amplitude A (how strongly differentiated their values are)
3. Generate all 10 values from the sinusoidal pattern: `value_i = baseline + A × cos(θ - position_i)`
4. Apply demographic priors as shifts to the baseline
5. Add calibrated residual variance per value
6. Clamp to [0, 1]

This produces circumplex-structured profiles by construction, not post-hoc correction.

### Stage 4: Gap Filler (with Calibrated Residuals)

**Job:** Compute remaining fields from parent traits + calibrated residual variance.

Each derived field: `derived_value = f(parent_traits) + residual`

Where `residual ~ N(0, σ²_field)` with σ calibrated per field type.

| Category | Parent Traits | Residual SD | Shared Residuals |
|----------|--------------|-------------|-----------------|
| Cognitive style | Big Five + occupation | 0.08 | analytical ↔ systematic share a latent factor |
| Communication | Big Five | 0.08 | formality ↔ directness share a latent factor |
| Goals | Occupation + values | 0.05 | — |
| Social roles | Communication prefs | 0.05 | — |
| Initial state | Big Five | 0.05 | mood_valence ↔ mood_arousal share a latent factor |
| Decision tendencies | Values + traits | 0.04 | Low confidence, internal only |

**Key principle:** Gap Filler never overrides explicit user input. It only fills `None` fields. All residuals are seeded for reproducibility.

### Stage 5: Policy Applier (NEW — separated from Gap Filler)

**Job:** Apply system-governed policy defaults. These do NOT vary by personality.

```python
SYSTEM_POLICY_DEFAULTS = {
    "uncertainty": {
        "admission_threshold": 0.4,
        "hedging_frequency": 0.5,
        "clarification_tendency": 0.5,
        "knowledge_boundary_strictness": 0.7,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "general_common_knowledge"],
        "lookup_behavior": "hedge",
        "expert_threshold": 0.7,
    },
    "disclosure_policy": {
        "bounds": (0.1, 0.9),
    },
    "invariants": {
        "must_avoid": ["revealing private personal information"],
    },
}
```

**Persona-influenced presentation** (within policy bounds):
- `disclosure_policy.base_openness`: derived from A + O (persona layer), but clamped by system bounds
- `uncertainty.hedging_frequency`: mild shift from directness, but floor enforced
- `invariants.cannot_claim`: occupation-specific (nurse can't claim "licensed surgeon") + system defaults

### Stage 6: Consistency Validator

**Job:** Check that the assembled profile is psychologically coherent. Flag or adjust.

**10 rules — labeled as configurable heuristics, not empirical laws:**

All thresholds below are defaults that can be overridden by the user. They are informed by research but calibrated for practical use, not claimed as precise empirical cutoffs.

1. **Big Five Metatrait Coherence** — Flag if Stability traits (A, C, low-N) strongly contradict. Default: flag if A > 0.8 AND C > 0.8 AND N > 0.8.
2. **Cognitive Style Compatibility** — Flag if risk_tolerance and need_for_closure both high. Default: flag if both > 0.7.
3. **Schwartz Adjacent Value Coherence** — Flag if adjacent circle values differ excessively. Default: delta > 0.5.
4. **Schwartz Opposing Value Conflict** — Flag if opposing values both elevated. Default: both > 0.7.
5. **Schwartz Sinusoidal Fit** — Flag if value profile doesn't approximate circumplex structure. Default: R² < 0.3.
6. **Big Five × Schwartz Cross-Check** — Flag if known correlation directions violated (e.g., O > 0.8 but Self-Direction < 0.3).
7. **Domain-Expertise Consistency** — Flag if domain proficiency > 0.7 without plausible occupation link.
8. **Disclosure-Privacy Coherence** — Flag if disclosure_policy.base_openness > 0.8 AND privacy_sensitivity > 0.8.
9. **Batch Diversity Check** — Flag if minimum pairwise trait distance below threshold.
10. **Cultural Confidence Scoring** — Attach reduced confidence for non-WEIRD populations.
11. **Cascade Collapse Check (NEW)** — For a fixed segment or narrow Big Five band, downstream cognitive/communication fields must retain non-trivial entropy, while policy floors remain invariant across personas.

**Validator modes:**
- `strict`: Contradictions raise errors, generation fails
- `warn` (default): Contradictions logged as warnings, generation continues
- `silent`: No checks

### Stage 7: Persona Assembler

**Job:** Construct engine-compatible `Persona` objects + attach provenance metadata.

Each persona gets:
- A unique `persona_id`: `{content_hash}_{batch_index}` (content hash for reproducibility, batch index for uniqueness)
- A generated `label`: "{name} - {occupation}, {location}"
- Auto-generated `identity_facts` from demographics
- Provenance object mapping every field to its FieldProvenance
- All 50+ fields populated

### Stage 8: Narrative Enricher (Optional)

**Job:** Add human-readable background story and natural-sounding response patterns. Only if LLM available.

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
| numpy | Logit-normal sampling, correlation matrices | >=1.24 | ~20MB |
| pyyaml | YAML export (already a persona-engine dependency) | >=6.0 | ~0.5MB |

### Built-in (stdlib, zero overhead)

| Module | Purpose |
|--------|---------|
| csv | CSV segment ingestion |
| re | Text description parsing |
| hashlib | Persona ID generation (content hash) |
| dataclasses | Internal data structures |
| uuid | Batch-local unique IDs |

### Optional Dependencies

| Library | Purpose | When | Size |
|---------|---------|------|------|
| spacy + en_core_web_sm | Advanced text parsing | `pip install layer-zero[nlp]` | ~200MB |
| typer | CLI interface | `pip install layer-zero[cli]` | ~2MB |
| scipy | Advanced distributions | `pip install layer-zero[stats]` | ~40MB |
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
    name="Alex",
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
    name="Alex",
    age=35,
    occupation="product manager",
    big_five={"openness": 0.82, "conscientiousness": 0.65,
              "extraversion": 0.55, "agreeableness": 0.48, "neuroticism": 0.30},
    count=1,
)

# Output: list of Persona objects (engine-compatible) + provenance
for p in personas:
    engine = PersonaEngine(p.persona, llm_provider="template")
    result = engine.chat("Tell me about your approach to product decisions")
    # Access provenance:
    print(p.provenance["psychology.big_five.openness"])
    # → FieldProvenance(value=0.82, source="explicit", confidence=0.95, ...)
```

### Configuration

```python
# Validator strictness
personas = layer_zero.mint(occupation="nurse", validate="strict")
personas = layer_zero.mint(occupation="nurse", validate="warn")     # default
personas = layer_zero.mint(occupation="nurse", validate="silent")

# Custom validator thresholds
personas = layer_zero.mint(
    occupation="nurse",
    validator_config={"adjacent_value_max_delta": 0.6, "opposing_value_max": 0.75},
)

# Reproducibility
personas = layer_zero.mint(occupation="nurse", count=10, seed=42)

# Export
layer_zero.to_yaml(personas, "output/")
layer_zero.to_json(personas, "personas.json")
```

---

## File Structure

```
layer_zero/
├── __init__.py              # Public API: mint(), from_description(), from_csv()
├── models.py                # MintRequest, SegmentRequest, MintedPersona, FieldProvenance
├── parser/
│   ├── __init__.py
│   ├── text_parser.py       # Regex-based description parsing → MintRequest
│   └── csv_parser.py        # CSV segment ingestion → list[SegmentRequest]
├── priors/
│   ├── __init__.py
│   ├── big_five.py          # Occupation/age/gender/culture → Big Five distributions
│   ├── values.py            # Occupation/age/culture → Schwartz distributions
│   ├── cognitive.py         # Big Five + occupation → cognitive style (with residuals)
│   └── data/
│       ├── occupation_traits.json    # Occupation → RIASEC → Big Five mapping
│       ├── age_trajectories.json     # Age → trait shift curves
│       ├── culture_baselines.json    # Culture region → trait + value baselines
│       └── correlation_matrix.json   # Big Five inter-correlation matrix
├── sampler.py               # Logit-normal sampling + circumplex Schwartz generation
├── gap_filler.py            # Fill unspecified persona-layer fields with residuals
├── policy.py                # System-governed policy defaults (NEW — separated)
├── validator.py             # 11 consistency rules (configurable thresholds)
├── assembler.py             # Construct engine Persona objects + provenance
├── enricher.py              # Optional LLM narrative enrichment
├── export.py                # YAML/JSON output
└── cli.py                   # Optional CLI (typer-based)
```

**~14 files.** Lean structure. Two more than v1 (models.py extracted, policy.py separated).

---

## Ethical Guardrails

### 1. Demographics Don't Determine Personality
Demographic priors shift the DISTRIBUTION, not the OUTCOME. A 25-year-old woman does not get assigned specific Big Five scores — she gets a distribution to sample from, and residual variance ensures individual variation.

### 2. No Stereotype Amplification
- Occupation mappings based on published RIASEC correlations (r = 0.19-0.48), not stereotypes
- Residual variance at every derivation step prevents cascade collapse into uniform profiles
- Batch diversity checks ensure no single demographic pattern dominates
- Provenance tracking makes every inference visible and auditable

### 3. Contradictions Are Surfaced, Not Resolved
"Introverted salesperson" is flagged as unusual, not auto-corrected. Real people have contradictions. The validator warns but does not override explicit user input.

### 4. Cultural Context Acknowledged
- Non-WEIRD populations get reduced confidence on provenance metadata
- Big Five and Schwartz values validity varies by cultural context — the system acknowledges this via confidence scores
- Culture region affects both trait and value priors, via published cross-cultural data

### 5. Policy Separates Safety from Personality
- Personality shapes communication style, not epistemic permissions
- High-openness personas don't get weaker safety guardrails
- Claim policies, disclosure bounds, and knowledge boundaries are system-governed

### 6. Decision Tendencies, Not "Biases"
- Renamed from "biases" to avoid ethically loaded language
- Mild behavioral leanings, low confidence scores, internal-only in v1
- Never labeled as cognitive defects of the persona

### 7. LLM Bias Mitigation
- LLMs are NEVER used for psychological parameter generation
- Narrative enricher operates AFTER parameters are locked
- No creative backstory generation — minimal census-style prompting

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Demographic mappings produce stereotypical personas | High | High | Distributions + residual variance + diversity checks + provenance |
| Cascade collapse (downstream uniformity) | Medium | High | Calibrated residuals at every step + cascade collapse test |
| Generated personas fail engine schema validation | Medium | High | Assemble via Pydantic model — schema errors caught immediately |
| Safety weakened by personality inference | Low (fixed) | Critical | Policy layer is system-governed, not persona-derived |
| Schwartz profiles psychologically incoherent | Low (fixed) | Medium | Circumplex-structured generation by construction |
| Expert fabrication from occupation alone | Medium | Medium | Default proficiency 0.4-0.6 (familiarity), not 0.7+ (expertise) |
| Occupation not in mapping table | High | Low | Graceful fallback to "general" with population defaults |
| User provides contradictory input | Medium | Low | Validator warns but respects user intent |
| Logit-normal sampling edge cases | Low | Low | Tested with extreme inputs, fallback to clamped normal |

---

## Implementation Order

| Phase | What | Files |
|-------|------|-------|
| 1 | Core models: MintRequest, SegmentRequest, FieldProvenance, MintedPersona | models.py |
| 2 | Text parser + structured parser | parser/*.py |
| 3 | Big Five prior engine + occupation mapping data | priors/big_five.py, data/ |
| 4 | Logit-normal sampler + correlation matrix | sampler.py |
| 5 | Schwartz value circumplex generator | priors/values.py, sampler.py |
| 6 | Gap filler with calibrated residuals | gap_filler.py |
| 7 | Policy applier (system defaults) | policy.py |
| 8 | Consistency validator (11 rules) | validator.py |
| 9 | Persona assembler + provenance + engine integration | assembler.py |
| 10 | CSV parser + SegmentRequest handling | parser/csv_parser.py |
| 11 | Export (YAML/JSON) | export.py |
| 12 | CLI (optional) | cli.py |

---

## Success Criteria

### Technical Criteria

1. `layer_zero.mint(occupation="nurse", count=10)` produces 10 distinct, valid Persona objects
2. All generated personas pass engine schema validation (`Persona(**fields)` succeeds)
3. Seed produces deterministic output: same seed + same input = same personas
4. CSV ingestion works for a 10-row segment file with ranges and distributions
5. Text parsing handles: "35-year-old product manager in fintech"
6. Total install footprint < 30MB (pydantic + numpy + pyyaml)
7. All tests pass, 0 type errors

### Behavioral Criteria

8. Generated personas produce measurably different IR values when run through the engine
9. Twin test: two personas from same segment are different; two personas from different segments are more different
10. Consistency validator catches known contradictions (high risk-tolerance + high need-for-closure)

### Anti-Cascade Criteria (NEW)

11. **Cascade collapse test:** For a fixed segment (e.g., "nurse, age 30-35"), generated personas must retain non-trivial entropy in downstream cognitive and communication fields, while policy floors remain invariant across all personas
12. **No fabricated expertise:** Default domain proficiency from occupation alone never exceeds 0.6

### Validity Criteria (NEW)

13. **Provenance coverage:** Every generated field has provenance metadata (source, confidence, depth)
14. **Policy invariance:** claim_policy, disclosure bounds, and knowledge_boundary_strictness are identical across all personas in a batch (unless user explicitly overrides)
15. **Informal realism check:** Show 5 generated personas to a human reviewer — they should feel like plausible people, not demographic stereotypes
