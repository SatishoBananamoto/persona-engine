# Layer Zero: Persona Minting Guide

Layer Zero generates psychologically coherent persona objects from minimal input. It bridges the gap between "I need a 35-year-old nurse from Chicago" and a fully specified persona with 50+ psychological parameters.

## Quick Start

```python
from layer_zero import from_description, mint, from_csv

# From natural language (simplest)
personas = from_description("A 45-year-old French chef named Marcus, passionate and direct")
persona = personas[0].persona  # engine-compatible Persona object

# From structured demographics
personas = mint(
    name="Sarah",
    age=32,
    occupation="software engineer",
    location="San Francisco",
    culture_region="North America",
)

# From CSV segments (batch generation)
personas = from_csv("segments.csv", count_per_segment=10, seed=42)
```

## Three Entry Points

### `from_description(description, *, count=1, seed=42)`

Parses natural language into a persona. Extracts name, age, occupation, location, and personality adjectives via heuristic parsing (no LLM required).

```python
personas = from_description(
    "A 28-year-old introverted data scientist from Tokyo who values precision"
)
print(personas[0].persona.label)           # "Unnamed - data scientist, Tokyo"
print(personas[0].persona.psychology.big_five.openness)  # ~0.72
print(personas[0].provenance["big_five.openness"].source) # "sampled"
```

### `mint(*, name, age, occupation, ...)`

Structured input with explicit control over each field. Unspecified fields are sampled from empirically-calibrated priors.

```python
personas = mint(
    name="Marcus",
    age=45,
    occupation="chef",
    industry="hospitality",
    culture_region="Western Europe",
    big_five={"openness": 0.8, "extraversion": 0.7},  # partial override
    count=5,  # generate 5 variants
    seed=42,
)
```

### `from_csv(path, *, count_per_segment=10)`

Reads demographic segments from CSV and generates `count_per_segment` personas per row. Expected columns: `name`, `age`, `gender`, `occupation`, `location`.

## How It Works

The minting pipeline has six stages:

```
Input → Priors → Sampling → Gap Filling → Policy → Validation → Assembly
```

### 1. Priors

Empirically-calibrated base rates for psychological traits:

- **Occupation priors** (`priors/big_five.py`): Maps occupations to Big Five expectations via Holland's RIASEC model. A "scientist" maps to Investigative (high Openness, moderate Conscientiousness).

- **Age trajectories** (`priors/big_five.py`): Agreeableness and Conscientiousness increase with age; Neuroticism and Extraversion decrease. Based on Srivastava (2003) and Roberts (2006).

- **Culture baselines** (`priors/values.py`): Schwartz value shifts by culture region. "East Asia" shifts toward Conservation; "Western Europe" toward Openness to Change. Based on Schwartz (2006).

### 2. Sampling

The **logit-normal sampler** (`sampler.py`) generates Big Five traits that:
- Stay in [0, 1] bounds (logit transform prevents boundary violations)
- Respect inter-trait correlations (correlation matrix from empirical data)
- Center on prior means with calibrated variance

### 3. Gap Filling

The **gap filler** (`gap_filler.py`) derives downstream parameters from Big Five traits:

- **Cognitive style**: analytical vs. intuitive reasoning, risk tolerance, need for closure
- **Communication preferences**: formality, directness, verbosity, emotional expressiveness
- **Goals**: primary and secondary goals based on Schwartz values
- **Domain knowledge**: proficiency levels based on occupation mapping

### 4. Policy

System-level defaults that vary in presentation but not permissions:

- **Claim restrictions**: Nurses cannot claim to be doctors; teachers cannot claim legal credentials
- **Disclosure policy**: Base openness, privacy boundaries
- **Uncertainty behavior**: How the persona handles questions outside expertise
- **Invariants**: Hard constraints (cannot_claim, must_avoid)

### 5. Validation

Eleven rules check internal consistency:

| # | Rule | What it catches |
|---|------|-----------------|
| 1 | Big Five metatrait coherence | Alpha/Beta metatrait distributions too extreme |
| 2 | Cognitive risk-closure incompatibility | High risk tolerance + high need for closure |
| 3 | Schwartz adjacent value deltas | Adjacent values on circumplex differ too much |
| 4 | Schwartz opposing pair coherence | Opposing values both maximal |
| 5 | Circumplex sinusoidal fit | Values don't follow expected cosine structure |
| 6 | Big Five-values cross-check | Openness low but Self-Direction high |
| 7 | Domain proficiency without occupation | Expert in domain unrelated to occupation |
| 8 | Disclosure-privacy conflict | High disclosure + high privacy simultaneously |
| 9 | Batch diversity | Generated batch too homogeneous |
| 10 | (reserved) | — |
| 11 | Cascade collapse | Too many fields derived from single source |

Validation modes: `"strict"` (raise on failure), `"warn"` (log warnings), `"skip"` (no validation).

### 6. Assembly

The **assembler** (`assembler.py`) packs all filled fields into an engine-compatible `Persona` object ready for `PersonaEngine(persona)`.

## Provenance Tracking

Every field carries a `FieldProvenance` record:

```python
persona = personas[0]
prov = persona.provenance["big_five.openness"]
print(prov.source)             # "sampled" | "explicit" | "derived" | "default"
print(prov.confidence)         # 0.85
print(prov.mapping_strength)   # 0.7 (how strong the prior→value mapping is)
print(prov.parent_fields)      # ["occupation"] (what this was derived from)
print(prov.notes)              # "Sampled from occupation prior: scientist → O=0.72±0.12"
```

Sources:
- **explicit**: User provided this value directly
- **sampled**: Generated from calibrated priors (occupation, age, culture)
- **derived**: Computed from other fields (e.g., cognitive style from Big Five)
- **default**: System default (policy layer)

## Using with PersonaEngine

```python
from layer_zero import from_description
from persona_engine import PersonaEngine

# Mint a persona
minted = from_description("A 55-year-old retired teacher who loves gardening")
persona = minted[0].persona

# Use with the engine
engine = PersonaEngine(persona, llm_provider="template")
result = engine.chat("What do you enjoy most about retirement?")
print(result.text)
print(result.ir.response_structure.confidence)
```

## Batch Generation with Diversity

```python
from layer_zero import mint
from layer_zero.diversity import compute_diversity_metrics

personas = mint(
    occupation="nurse",
    culture_region="North America",
    count=20,
    seed=42,
)

metrics = compute_diversity_metrics([p.persona for p in personas])
print(f"Simpson index: {metrics.simpson_index:.3f}")
print(f"Mean pairwise distance: {metrics.mean_pairwise_distance:.3f}")
```
