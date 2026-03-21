# Layer Zero — Implementation Plan

**Date:** 2026-03-15
**Architecture:** v2 (post-external-review)
**Branch:** `claude/external-review`

---

## Overview

12 implementation phases, ordered by dependency. Each phase has clear inputs, outputs, files, tests, and a "done when" checkpoint. No phase starts until its dependencies are complete.

**Estimated total:** ~14 source files, ~800-1200 LOC of source, ~600-800 LOC of tests.

---

## Dependency Graph

```
Phase 1: Core Models
    ↓
Phase 2: Text Parser ─────────────────────┐
    ↓                                      │
Phase 3: Big Five Priors + Data            │
    ↓                                      │
Phase 4: Logit-Normal Sampler              │
    ↓                                      │
Phase 5: Schwartz Circumplex Generator     │
    ↓                                      │
Phase 6: Gap Filler + Residuals            │
    ↓                                      │
Phase 7: Policy Applier                    │
    ↓                                      │
Phase 8: Consistency Validator             │
    ↓                                      │
Phase 9: Persona Assembler + Provenance    │
    ↓                                      ↓
Phase 10: Public API (mint, from_description)
    ↓
Phase 11: CSV Parser + from_csv    (depends on Phase 1 models + Phase 10)
    ↓
Phase 12: Export (YAML/JSON)
```

---

## Phase 1: Core Models

**Goal:** Define the data structures everything else builds on.

**Files:**
- `layer_zero/models.py`

**What to build:**

```python
# Input models
MintRequest          # Single-persona request (Tier 1-4 normalized input)
SegmentRequest       # CSV segment request (ranges, distributions)

# Output models
FieldProvenance      # Per-field provenance: source, confidence, mapping_strength, depth
MintedPersona        # Wrapper: persona (engine Persona) + provenance (dict[str, FieldProvenance])

# Internal models
TraitPrior           # {trait: (mean, std_dev, source, mapping_strength)}
ValueCircumplexParams  # {peak_angle, amplitude, baseline}
ResidualConfig       # {field_name: (sd, shared_group)}
```

**Tests:**
- MintRequest accepts all valid field combinations
- MintRequest rejects invalid types
- SegmentRequest handles age_range, gender_distribution
- FieldProvenance computes confidence correctly (source_type × mapping_strength × depth_decay)
- MintedPersona.persona is a valid engine Persona type

**Done when:** All models instantiate, validate, and serialize. 10+ tests passing.

---

## Phase 2: Text Parser

**Goal:** Parse natural language descriptions into MintRequest.

**Files:**
- `layer_zero/parser/__init__.py`
- `layer_zero/parser/text_parser.py`

**What to build:**
- Regex patterns for age (`(\d{1,2})[-\s]?year[-\s]?old`)
- Occupation extraction (match against known occupation list)
- Location extraction (city/country patterns)
- Gender extraction (keyword match, optional)
- Trait adjective extraction ("analytical", "warm", "cautious" → trait_hints)
- Industry extraction (keyword match)

**Known limitations (documented, not fixed in v1):**
- No negation handling ("not very analytical" → incorrectly extracts "analytical")
- No compound roles ("product manager turned data scientist")
- No sarcasm detection

**Tests:**
- "35-year-old product manager in fintech" → age=35, occupation="product manager", industry="fintech"
- "A cautious nurse from Tokyo who values security" → age=None, occupation="nurse", location="Tokyo", traits=["cautious"], values hint
- "Senior software engineer, 42, San Francisco" → age=42, occupation="software engineer", location="San Francisco"
- Empty/garbage input → graceful failure with clear error
- 10+ test cases covering common description patterns

**Done when:** Parser extracts correct fields from 8/10 test descriptions. Failures are documented, not hidden.

---

## Phase 3: Big Five Priors + Mapping Data

**Goal:** Build the demographic-to-Big-Five distribution engine with occupation, age, gender, and culture mappings.

**Files:**
- `layer_zero/priors/__init__.py`
- `layer_zero/priors/big_five.py`
- `layer_zero/priors/data/occupation_traits.json`
- `layer_zero/priors/data/age_trajectories.json`
- `layer_zero/priors/data/culture_baselines.json`
- `layer_zero/priors/data/correlation_matrix.json`

**What to build:**
- `compute_big_five_prior(request: MintRequest) -> dict[str, TraitPrior]`
- Occupation → RIASEC → Big Five mapping table (20+ common occupations)
- Age → trait shift curves (based on Srivastava, PMC7869960)
- Gender → trait shift (based on Schmitt 2008, culture-dependent)
- Culture region → baseline norms (5-6 major regions: Western, East Asian, South Asian, Latin American, Middle Eastern, Sub-Saharan African)
- User overrides replace prior (tiny SD = nearly fixed)
- Unknown occupation → graceful fallback to population baseline

**Data format:**
```json
{
  "nurse": {
    "riasec": "Social",
    "big_five_shifts": {
      "agreeableness": 0.08,
      "conscientiousness": 0.05,
      "extraversion": 0.03,
      "neuroticism": -0.02,
      "openness": 0.0
    }
  }
}
```

**Tests:**
- "nurse" → A shifted up, C shifted up
- "software engineer" → O shifted up
- "entrepreneur" → N shifted down, E shifted up, A shifted down
- Age 25 vs age 65 → C higher for 65, N lower for 65
- Unknown occupation → population baseline (mean=0.5, sd=0.15)
- User override openness=0.9 → prior becomes (0.9, 0.02)
- 15+ tests

**Done when:** Priors shift in correct directions for 5+ occupation/age/gender combinations. Correlation matrix loads and validates as positive semi-definite.

---

## Phase 4: Logit-Normal Sampler

**Goal:** Sample Big Five trait vectors that respect inter-trait correlations and stay bounded [0,1].

**Files:**
- `layer_zero/sampler.py`

**What to build:**
- `sample_big_five(priors: dict[str, TraitPrior], correlation: np.ndarray, count: int, seed: int) -> np.ndarray`
- Logit transform: `logit(p) = log(p / (1-p))`
- Sigmoid inverse: `sigmoid(x) = 1 / (1 + exp(-x))`
- Process: transform means to logit space → build covariance in logit space → sample multivariate normal → sigmoid back to [0,1]
- Handle edge cases: means near 0 or 1 (clamp before logit to avoid inf)
- Seeded RNG for reproducibility

**Tests:**
- Same seed + same priors → identical output
- Different seeds → different output
- Output always in [0, 1]
- Correlation structure approximately preserved (check sample correlation matrix against input)
- 1000 samples from centered priors: mean ≈ 0.5, no pathological clustering at boundaries
- Override traits have near-zero variance in output
- 10+ tests

**Done when:** Sampler produces bounded, correlated trait vectors. Sample correlation matches input correlation within tolerance (±0.1 for N=1000).

---

## Phase 5: Schwartz Circumplex Generator

**Goal:** Generate Schwartz value profiles that follow the circumplex structure by construction.

**Files:**
- `layer_zero/priors/values.py`
- Updates to `layer_zero/sampler.py`

**What to build:**
- `generate_schwartz_values(priors: dict, count: int, seed: int) -> np.ndarray`
- Circle positions for 10 values (fixed, 36° apart)
- Process: sample peak angle θ → sample amplitude A → `value_i = baseline + A × cos(θ - position_i)` → add demographic prior shifts → add residual → clamp [0,1]
- Demographic priors from age, occupation, culture (see research doc)
- User overrides pin specific values

**Tests:**
- Adjacent values (e.g., Benevolence, Universalism) are positively correlated in batch
- Opposing values (e.g., Power, Universalism) are negatively correlated in batch
- Value profiles approximately sinusoidal (R² > 0.5 for unmodified profiles)
- User override pins that value while others follow circumplex
- 10+ tests

**Done when:** Generated value profiles follow circumplex structure. Adjacent correlation > 0, opposing correlation < 0.3 for N=100 batch.

---

## Phase 6: Gap Filler + Residuals

**Goal:** Derive cognitive style, communication prefs, goals, social roles, initial state, and decision tendencies from parent traits + calibrated residual variance.

**Files:**
- `layer_zero/gap_filler.py`

**What to build:**
- `fill_gaps(big_five, values, request, seed) -> dict` (all remaining persona-layer fields)
- Derivation functions for each field category with residual injection
- Shared latent residuals for related fields (e.g., formality ↔ directness)
- Cognitive style: O → analytical (r~0.35), C → systematic (r~0.30), etc. + residual SD=0.08
- Communication: E → expressive, A → indirect, C → formal + residual SD=0.08
- Goals: from occupation + top values + residual SD=0.05
- Social roles: template (default/at_work/friend) from communication prefs + residual SD=0.05
- Initial state: mood_valence=0.1, mood_arousal from E, stress from N, + residual SD=0.05
- Decision tendencies: from values + traits, low confidence, residual SD=0.04
- Name generation: random from a diverse name pool if not specified
- Knowledge domains: occupation → domain(s) with proficiency 0.4-0.6 (familiarity, NOT expertise)

**Tests:**
- Two personas with identical Big Five but different seeds → different cognitive styles (residual variance works)
- High O → analytical_intuitive trends higher (on average across 100 samples)
- Occupation "nurse" → domain "Healthcare" with proficiency ~0.5
- Name generated when not specified
- Explicit user values never overridden
- Cascade collapse test: for 50 nurses with similar Big Five, cognitive_style and communication fields show SD > 0.05
- 15+ tests

**Done when:** All persona-layer fields populated. Residual variance demonstrably prevents cascade collapse.

---

## Phase 7: Policy Applier

**Goal:** Apply system-governed policy defaults that do NOT vary by personality.

**Files:**
- `layer_zero/policy.py`

**What to build:**
- `apply_policy_defaults(persona_fields, request) -> dict` (all policy-layer fields)
- System defaults for: uncertainty, claim_policy, disclosure_policy, invariants
- `privacy_sensitivity` derived from persona traits but clamped by system bounds
- `identity_facts` auto-generated from demographics
- `cannot_claim` from occupation + system defaults
- `must_avoid` system defaults

**Tests:**
- High-openness persona gets same claim_policy.expert_threshold as low-openness persona
- High-risk-tolerance persona gets same claim_policy.lookup_behavior as low-risk persona
- identity_facts contain age, occupation, location
- cannot_claim for "nurse" includes occupations nurse cannot claim (e.g., "licensed physician")
- 8+ tests

**Done when:** Policy fields are identical across personas in a batch (unless user explicitly overrides). Safety floors invariant.

---

## Phase 8: Consistency Validator

**Goal:** Check assembled profiles for psychological coherence. Flag, don't block.

**Files:**
- `layer_zero/validator.py`

**What to build:**
- `validate(persona_fields, provenance, mode="warn") -> ValidationResult`
- 11 rules (all with configurable thresholds):
  1. Big Five metatrait coherence
  2. Cognitive style compatibility
  3. Schwartz adjacent value coherence
  4. Schwartz opposing value conflict
  5. Schwartz sinusoidal fit
  6. Big Five × Schwartz cross-check
  7. Domain-expertise consistency
  8. Disclosure-privacy coherence
  9. Batch diversity check
  10. Cultural confidence scoring
  11. Cascade collapse check
- Three modes: strict (raise), warn (log), silent (skip)
- Configurable thresholds via validator_config dict

**Tests:**
- risk_tolerance=0.9 + need_for_closure=0.9 → flagged
- Power=0.9 + Universalism=0.9 → flagged
- Adjacent values within 0.3 → no flag
- Batch of 10 identical personas → diversity flag
- Policy fields identical across batch → no flag (correct behavior)
- 12+ tests

**Done when:** All 11 rules fire correctly on synthetic test cases. Modes work (strict raises, warn logs, silent skips).

---

## Phase 9: Persona Assembler + Provenance

**Goal:** Construct engine-compatible Persona objects with full provenance metadata.

**Files:**
- `layer_zero/assembler.py`

**What to build:**
- `assemble(persona_fields, policy_fields, provenance_map) -> MintedPersona`
- Map internal field dict → engine `Persona(**fields)`
- Generate persona_id: `{sha256(name+occupation+seed)[:12]}_{batch_index}`
- Generate label: "{name} - {occupation}, {location}"
- Attach provenance: `MintedPersona(persona=Persona(...), provenance={...})`
- Validate via Pydantic (engine schema validation happens automatically)

**Tests:**
- Assembled persona passes `Persona(**fields)` without error
- persona_id is unique within batch
- Same seed produces same persona_id
- Provenance covers all fields (no field missing provenance)
- Provenance confidence values are plausible (explicit > sampled > derived > default)
- 8+ tests

**Done when:** Assembled persona loads into `PersonaEngine(persona)` without error. Provenance complete.

---

## Phase 10: Public API

**Goal:** Wire everything together into the user-facing `mint()` and `from_description()` functions.

**Files:**
- `layer_zero/__init__.py`

**What to build:**
- `mint(**kwargs) -> list[MintedPersona]` — Tier 2 + Tier 4 entry point
- `from_description(text, **kwargs) -> list[MintedPersona]` — Tier 1 entry point
- Pipeline: parse → priors → sample → fill → policy → validate → assemble
- Pass-through kwargs: count, seed, validate mode, validator_config

**Tests:**
- `mint(occupation="nurse", count=5)` → 5 valid MintedPersona objects
- `from_description("35-year-old product manager in fintech", count=3)` → 3 valid personas
- `mint(occupation="nurse", count=5, seed=42)` → deterministic
- `mint(big_five={"openness": 0.9}, count=1)` → openness is 0.9 in output
- Integration test: feed output to PersonaEngine, run `.plan()`, get valid IR
- 10+ tests

**Done when:** End-to-end pipeline works. Can mint personas and run them through the engine.

---

## Phase 11: CSV Parser + from_csv

**Goal:** Parse CSV segment files into SegmentRequests, generate personas per segment.

**Files:**
- `layer_zero/parser/csv_parser.py`
- Update `layer_zero/__init__.py` with `from_csv()`

**What to build:**
- `parse_csv(path) -> list[SegmentRequest]`
- CSV columns: segment_name, age_min, age_max, gender_dist, occupation, location, count
- `from_csv(path, count_per_segment=10) -> list[MintedPersona]`
- For each segment: sample age from range, sample gender from distribution, create MintRequest, run pipeline

**CSV format:**
```csv
segment_name,age_min,age_max,occupation,location,count
junior_nurses,22,30,nurse,Chicago,10
senior_engineers,35,50,software engineer,San Francisco,10
```

**Tests:**
- Valid 3-row CSV → 30 personas (10 per segment)
- Age distribution within specified range
- Gender distribution approximately matches specified proportions (for N≥20)
- Missing columns → clear error
- 6+ tests

**Done when:** CSV ingestion produces valid personas with demographics matching segment specs.

---

## Phase 12: Export

**Goal:** Export generated personas to YAML and JSON formats.

**Files:**
- `layer_zero/export.py`

**What to build:**
- `to_yaml(personas, output_dir)` — one YAML file per persona
- `to_json(personas, output_path)` — single JSON file with array
- Include provenance as optional section
- YAML format compatible with `PersonaEngine.from_yaml()`

**Tests:**
- Export → re-import via PersonaEngine.from_yaml() → same persona loads
- JSON round-trip: export → load → fields match
- Provenance included when requested
- 5+ tests

**Done when:** Exported personas load cleanly in PersonaEngine.

---

## Testing Strategy

### Test Organization

```
tests/
├── test_models.py          # Phase 1
├── test_text_parser.py     # Phase 2
├── test_big_five_priors.py # Phase 3
├── test_sampler.py         # Phase 4
├── test_schwartz.py        # Phase 5
├── test_gap_filler.py      # Phase 6
├── test_policy.py          # Phase 7
├── test_validator.py       # Phase 8
├── test_assembler.py       # Phase 9
├── test_api.py             # Phase 10
├── test_csv_parser.py      # Phase 11
├── test_export.py          # Phase 12
└── test_integration.py     # End-to-end: mint → engine → IR
```

### Test Counts by Phase

| Phase | Estimated Tests |
|-------|----------------|
| 1. Models | 10 |
| 2. Text Parser | 10 |
| 3. Big Five Priors | 15 |
| 4. Sampler | 10 |
| 5. Schwartz | 10 |
| 6. Gap Filler | 15 |
| 7. Policy | 8 |
| 8. Validator | 12 |
| 9. Assembler | 8 |
| 10. API | 10 |
| 11. CSV | 6 |
| 12. Export | 5 |
| Integration | 5 |
| **Total** | **~124** |

### Key Integration Tests

1. `mint(occupation="nurse", count=10)` → 10 distinct Persona objects → all load into PersonaEngine → all produce valid IR
2. `from_description("35-year-old product manager", count=5)` → same pipeline
3. `from_csv("test_segments.csv")` → personas match segment specs
4. Seed determinism: same inputs + same seed = identical output across runs
5. Cascade collapse: 50 nurses → downstream fields have SD > 0.05

---

## Definition of Done (v1)

Layer Zero v1 is shippable when ALL of these are true:

### Technical
- [ ] All 12 phases complete
- [ ] ~124 tests passing, 0 failures
- [ ] 0 mypy errors
- [ ] Install footprint < 30MB
- [ ] Seed determinism verified

### Behavioral
- [ ] Cascade collapse test passes (downstream entropy preserved)
- [ ] Policy invariance test passes (safety floors identical across batch)
- [ ] No fabricated expertise (default proficiency ≤ 0.6)
- [ ] Provenance coverage 100% (every field has metadata)
- [ ] Twin test passes (same segment → different; different segments → more different)

### Validation
- [ ] Informal realism check: 5 generated personas reviewed by Satish
- [ ] Engine integration: generated personas produce valid IR with measurable behavioral differences
