# Layer Zero — Session Log (2026-03-15)

**View this remotely:** This file is updated as work progresses on `claude/external-review` branch.

---

## What Was Built Tonight

### Layer Zero v1 — Persona Minting Machine
A library that mints psychologically coherent personas from user inputs (text, structured fields, CSV segments) for the persona-engine.

```python
import layer_zero

# From text
personas = layer_zero.from_description("35-year-old nurse from Chicago", count=10)

# From structured fields
personas = layer_zero.mint(occupation="software engineer", age=30, count=20)

# From CSV segments
personas = layer_zero.from_csv("segments.csv")

# Each persona loads directly into the engine
from persona_engine import PersonaEngine
engine = PersonaEngine(personas[0].persona, llm_provider="template")
result = engine.chat("What do you think about teamwork?")
```

---

## Architecture Highlights

- **Three-layer model:** Evidence (user input), Persona (sampled traits), Policy (system-governed safety floors)
- **Logit-normal sampling:** Big Five traits sampled in unbounded logit space, sigmoid back to [0,1] — preserves inter-trait correlations at boundaries
- **Schwartz circumplex generation:** Values generated from circle structure (angle + amplitude + cosine), not independent sampling
- **Calibrated residual variance:** Every derivation step adds controlled noise — prevents cascade collapse where 50 nurses all converge on identical downstream fields
- **Policy separation:** claim_policy, expert_threshold, disclosure bounds are system-governed, NOT derived from personality. High-openness personas don't get weaker safety.
- **Provenance per field:** Every generated field carries source type, confidence, inferential depth, parent fields

---

## Files Created

```
layer_zero/
├── __init__.py              # Public API: mint(), from_description(), from_csv()
├── models.py                # MintRequest, SegmentRequest, FieldProvenance, MintedPersona
├── parser/
│   ├── text_parser.py       # Regex-based description parsing
│   └── csv_parser.py        # CSV segment ingestion
├── priors/
│   ├── big_five.py          # Occupation/age/culture → Big Five distributions
│   ├── values.py            # Schwartz circumplex generator
│   └── data/
│       ├── occupation_traits.json    # 30 occupations with RIASEC mappings
│       ├── age_trajectories.json     # 6 age brackets
│       ├── culture_baselines.json    # 6 world regions
│       └── correlation_matrix.json   # Big Five inter-correlations (van der Linden 2010)
├── sampler.py               # Logit-normal multivariate sampling
├── gap_filler.py            # Derive cognitive/communication/goals + residuals
├── policy.py                # System-governed policy defaults
├── validator.py             # 11 consistency rules
├── assembler.py             # Build engine Persona objects + provenance
└── export.py                # YAML/JSON output

tests/
├── test_models.py           # 32 tests
├── test_text_parser.py      # 31 tests
├── test_big_five_priors.py  # 25 tests
├── test_sampler.py          # 18 tests
├── test_schwartz.py         # 17 tests
├── test_gap_filler.py       # 21 tests
├── test_policy.py           # 13 tests
├── test_validator.py        # 19 tests
├── test_assembler.py        # 12 tests
├── test_api.py              # 19 tests
├── test_csv_parser.py       # 16 tests
├── test_export.py           # 8 tests
└── test_integration.py      # 16 tests
                               ─────
                               247 tests total, 0 failures
```

---

## Review Cycles Completed

### Review 1 (Phases 1-4): Code quality
- 5 bugs fixed (Jacobian SD blowup, age regex, pronoun false positives, set ordering, substring culture matching)
- 2 statistical issues fixed (shift stacking cap, synonym deduplication)

### Review 2 (Phases 5-10): Architecture alignment
- 6 bugs fixed (shared latent variance, dead cascade validator, provenance parent fields, policy side-effect, scipy fallback, deterministic goals)
- 5 architecture mismatches resolved
- Provenance coverage expanded from ~50% to ~100% of fields

### Integration verification
All success criteria from ARCHITECTURE-layer-zero.md verified:
- ✅ Cascade collapse test (SD > 0.03 for 50 nurses)
- ✅ Policy invariance (identical across batch)
- ✅ No fabricated expertise (proficiency ≤ 0.6)
- ✅ Seed determinism
- ✅ Provenance on all major fields
- ✅ Engine integration (mint → plan → valid IR)
- ✅ YAML round-trip (export → PersonaEngine.from_yaml)

---

## What's on the Branch

All documents + code on `claude/external-review`:

| Document | Purpose |
|----------|---------|
| REVIEW.md | Verified code review of persona-engine |
| RESEARCH-persona-compiler.md | Research (11 topics, 40+ sources) |
| ARCHITECTURE-layer-zero.md (v2) | Architecture with external review fixes |
| IMPLEMENTATION-PLAN.md | 12-phase plan |
| MEMORY-project.md | Project state tracker |
| SESSION-LOG.md | This file |

---

## Current Status

**Layer Zero v1: COMPLETE**
- 14 source files, ~2800 LOC
- 13 test files, 247 tests, 0 failures
- All 12 implementation phases done
- 2 review cycles completed with fixes applied
- End-to-end integration verified with persona-engine

---

## Still TODO (future sessions)

- [ ] Gender-based trait priors (deferred — culturally sensitive, needs careful calibration)
- [ ] Narrative enricher (optional LLM-based background stories)
- [ ] CLI interface (typer-based)
- [ ] MCP server mode (post-v1)
- [ ] More occupation mappings (27 covered, ~30 in parser missing data)
- [ ] Human realism check (show generated personas to Satish)
