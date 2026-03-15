# Persona Engine — Project Memory

> This file is the single source of truth for project state across sessions.
> Every session must READ this at start and UPDATE it at end.
> Every entry must have a timestamp and a tag. No undated or untagged entries.

## Tag Registry

> Same tags used in main memory. Immutable once defined. New tags added here before first use.

| Tag | Meaning | When to use |
|-----|---------|-------------|
| `[LEARNING]` | Something I got wrong and corrected | After mistakes, failed approaches |
| `[REMEMBER]` | Fact or preference to retain | User requests, environment details |
| `[DECISION]` | Architecture/approach choice made | At decision points |
| `[PROCESS]` | How we work, methodology rules | When Satish gives process feedback |
| `[CONTEXT]` | Background info for future sessions | Project state, who/what/where |
| `[BLOCKER]` | Something preventing progress | When stuck |
| `[RESOLVED]` | A blocker that was fixed | When unblocked |

---

## Project Overview

- **Codename:** Layer Zero
- **External name:** Persona Minting Machine
- **Repo:** SatishoBananamoto/persona-engine
- **Working branch:** `claude/external-review` (our work — does not touch other branches)
- **Engine branch:** `claude/general-session-VUY6r` (separate Claude session refining the core engine)
- **Goal:** Build Layer Zero — a layer that takes user inputs (segment data, prompts, toggles) and mints psychologically coherent Persona objects for the engine

---

## Decisions

> Architecture choices, approach selections, trade-offs made. Each with reasoning and timestamp.

### `[DECISION]` [DEC-001] Work on separate branch
- **Date:** 2026-03-15
- **Decision:** All compiler work goes on `claude/external-review`, never touching `claude/general-session-VUY6r`
- **Reasoning:** Another Claude session is actively refining the engine. We don't interfere.

### `[DECISION]` [DEC-002] Research before building
- **Date:** 2026-03-15
- **Decision:** Conducted full research across 11 topic areas before writing any code
- **Reasoning:** Satish's principle — research first, understand the landscape, then build. Prevents building the wrong thing.

### `[DECISION]` [DEC-003] Compiler architecture — hybrid approach
- **Date:** 2026-03-15
- **Decision:** Use research-backed demographic-to-psychology mappings for core parameters, LLM only for narrative enrichment after parameters are locked
- **Reasoning:** NeurIPS 2025 research shows LLM-generated persona detail actually hurts accuracy. Data-grounded parameters + minimal LLM involvement = best results.
- **Source:** RESEARCH-persona-compiler.md, Section 7

### `[PROCESS]` [DEC-004] Tagging system for memory entries
- **Date:** 2026-03-15
- **Decision:** All memory entries (both project and main) must carry a tag from a fixed registry. Tags are immutable. New tags require registry update first.
- **Reasoning:** Satish identified that untagged entries make it hard to scan, filter, and understand entry purpose at a glance. Tags prevent drift by making intent explicit.

### `[DECISION]` [DEC-005] Project naming
- **Date:** 2026-03-15
- **Decision:** Internal codename is "Layer Zero" (`layer-zero` in code/refs). External name is "Persona Minting Machine" (for docs/README if shipped publicly).
- **Reasoning:** Layer Zero captures the architectural position (foundation layer before the engine). Persona Minting Machine is memorable and describes what it does — mints personas from raw inputs.

### `[DECISION]` [DEC-006] Three-layer model (from external review)
- **Date:** 2026-03-15
- **Decision:** Split persona generation into Evidence, Persona, and Policy layers. Policy fields (claim_policy, disclosure bounds, knowledge_boundary_strictness) are system-governed defaults, not derived from personality traits.
- **Reasoning:** External reviewer identified that persona-derived safety policies create risk — a high-openness persona could get weaker epistemic guardrails. Policy must be system-controlled.

### `[DECISION]` [DEC-007] Logit-normal sampling (from external review)
- **Date:** 2026-03-15
- **Decision:** Replace multivariate normal + clamp with logit-normal transform. Sample in unbounded logit space, apply sigmoid to get [0,1].
- **Reasoning:** Clamping distorts means, variances, and correlations at boundaries. Logit-normal preserves the correlation structure.

### `[DECISION]` [DEC-008] Calibrated residual variance at every derivation step
- **Date:** 2026-03-15
- **Decision:** Each derived field gets `f(parent_traits) + calibrated_residual` instead of pure deterministic derivation. Related fields share latent residuals.
- **Reasoning:** Prevents cascade collapse — stochasticity entering only at trait sampling causes downstream fields to become overly correlated and too uniform across personas from the same segment.

### `[DECISION]` [DEC-009] Provenance per field
- **Date:** 2026-03-15
- **Decision:** Every generated field carries FieldProvenance (source type, confidence, mapping strength, inferential depth, parent fields).
- **Reasoning:** Enables debugging, ethical review, and transparency. Confidence based on source type + mapping strength + depth decay, not just hop count.

### `[LEARNING]` [L-003] Cascade collapse in derived persona fields
- **Date:** 2026-03-15
- **Context:** External reviewer identified that deterministic derivation from sampled traits compresses conditional variance. 10 nurses from the same segment converge on identical cognitive styles, policies, and behavioral rules.
- **Lesson:** When building multi-stage inference pipelines, add controlled stochasticity at every stage, not just the first. Deterministic projections of random variables are not equivalent to independent random variables.
- **Also saved to:** main memory (feedback file)

---

## Progress

> What's been done, in chronological order.

### 2026-03-15 — Session 1
- [x] Cloned persona-engine, reviewed `claude/general-session-VUY6r` branch
- [x] Conducted code review (first pass had 4 errors, corrected in second pass)
- [x] Wrote verified REVIEW.md with file:line evidence
- [x] Created review methodology plan (plan.md)
- [x] Ran 2,376 tests — all passing
- [x] Pushed REVIEW.md to `claude/external-review` branch
- [x] Conducted parallel research on persona compiler (2 research agents, 11 topics)
- [x] Wrote RESEARCH-persona-compiler.md (569 lines, 40+ sources)
- [x] Pushed research doc to branch
- [x] Set up project memory system
- [x] Designed Layer Zero architecture (ARCHITECTURE-layer-zero.md)
- [x] Defined complete Persona field inventory (50+ fields mapped)
- [x] Researched tech stack (numpy + pydantic, ~25MB footprint)
- [x] Defined 10 consistency validation rules
- [x] Defined ethical guardrails
- [x] Defined 4-tier input system
- [x] Defined 7-stage pipeline
- [x] External review of architecture — 5 blocking issues identified and fixed
- [x] Architecture v2 written — three-layer model, logit-normal sampling, provenance, policy separation
- [x] Cascade collapse problem identified and solved (calibrated residual variance)
- [x] Implementation plan written (IMPLEMENTATION-PLAN.md, 12 phases)
- [x] **Phase 1: Core models** — MintRequest, SegmentRequest, FieldProvenance, MintedPersona, TraitPrior (32 tests)
- [x] **Phase 2: Text parser** — regex-based description parsing (31 tests)
- [x] **Phase 3: Big Five priors** — 30 occupations, 6 age brackets, 6 culture regions, correlation matrix (25 tests)
- [x] **Phase 4: Logit-normal sampler** — multivariate correlated sampling with sigmoid bounds (18 tests)
- [x] Code review round 1 — 5 bugs + 2 statistical issues found and fixed (106 tests passing)
- [x] Product alignment review — on track, 3 items to address before Phase 9
- [x] Code review round 1 bugs fixed (5 bugs + 2 statistical issues)
- [x] **Phase 5: Schwartz circumplex generator** — 17 tests
- [x] **Phase 6: Gap filler + residuals** — cascade collapse prevented, 21 tests
- [x] **Phase 7: Policy applier** — system-governed, invariant across personas, 13 tests
- [x] **Phase 8: Consistency validator** — 11 rules, 3 modes, 19 tests
- [x] **Phase 9: Persona assembler + provenance** — engine integration works, 12 tests
- [x] **Phase 10: Public API** — mint() + from_description() end-to-end, 19 tests
- [ ] **Phase 11: CSV parser** (deferred — not blocking)
- [ ] **Phase 12: Export** (deferred — not blocking)

**STATUS: Layer Zero v1 core is functional. 207 tests passing. End-to-end pipeline works.**

---

## Key Research Findings

> Condensed takeaways that directly inform architecture. Full details in RESEARCH-persona-compiler.md.

1. **Occupation is the strongest demographic signal** for personality inference (r = 0.48 for Artistic-Openness). Stronger than age or gender.
2. **Culture dominates values.** Schwartz/Inglehart data shows cultural context > age + gender for value priorities. Must be a first-class input.
3. **LLM detail hurts accuracy.** More backstory = more drift from real behavior (NeurIPS 2025). Lock psychology from data, use LLM only for narrative.
4. **Conscientiousness and Agreeableness are hardest** to both predict and simulate. Extra validation needed.
5. **Generate sets, not singletons.** Diversity gain maxes at ~40 personas per segment. Default to N personas per input.
6. **No one does this yet.** No existing tool maps inputs to validated psychological frameworks. This is the gap.
7. **Demographics explain ~1.5% of behavioral variance alone.** Adding Big Five + Schwartz values dramatically improves alignment.

---

## Blockers

> Things preventing progress. Each with status.

*None currently.*

---

## Learnings

> Things we discovered during this project that affect future work.

### `[LEARNING]` [L-001] Don't trust grep-based code analysis
- **Date:** 2026-03-15
- **Context:** First review fabricated 3 "dead code" claims because the exploration agent searched for imports instead of tracing data flow through the orchestrator (TurnPlanner)
- **Lesson:** Always trace the full consumption path. Data flows through intermediate objects (IR, metrics dicts, modifier lists), not just direct imports.
- **Also saved to:** main memory (`feedback_code_review_discipline.md`)

### `[LEARNING]` [L-002] Verify subagent output before presenting
- **Date:** 2026-03-15
- **Context:** Presented exploration agent's findings as facts without checking. Repo handler caught 4 errors.
- **Lesson:** Never present subagent findings without spot-checking the critical claims yourself. Read the actual code at the cited lines.
- **Also saved to:** main memory (`feedback_verify_subagent_output.md`)

---

## Next Steps

> Prioritized list of what to do next session.

1. Design the compiler architecture in detail (modules, interfaces, data flow)
2. Define the input schema (tiered: text → structured → CSV → direct)
3. Build the demographic-to-Big-Five mapping engine (from research data)
4. Build the Schwartz values inference layer
5. Build the gap-filler for cognitive style and communication prefs
6. Build the consistency validator
7. Build the persona assembler
8. Tests
9. Integration with persona-engine

---

## Session Protocol

> Rules for every session working on this project.

### At session start:
1. Read this file
2. Read RESEARCH-persona-compiler.md if needed for context
3. State current understanding and confirm direction with Satish

### At decision points:
1. Write the decision and reasoning HERE before acting
2. If uncertain, ask Satish — don't guess

### At session end:
1. Update Progress section with what was done
2. Update Next Steps with what's queued
3. Add any new Decisions or Learnings
4. Commit and push this file

### When deviating from plan:
1. STOP
2. Document why the deviation is needed
3. Get confirmation if it changes architecture or approach
