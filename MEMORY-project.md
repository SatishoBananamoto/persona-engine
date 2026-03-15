# Persona Engine — Project Memory

> This file is the single source of truth for project state across sessions.
> Every session must READ this at start and UPDATE it at end.
> Every entry must have a timestamp. No undated entries.

---

## Project Overview

- **Repo:** SatishoBananamoto/persona-engine
- **Working branch:** `claude/external-review` (our work — does not touch other branches)
- **Engine branch:** `claude/general-session-VUY6r` (separate Claude session refining the core engine)
- **Goal:** Build a persona compiler layer that takes user inputs (segment data, prompts, toggles) and compiles psychologically coherent Persona objects for the engine

---

## Decisions

> Architecture choices, approach selections, trade-offs made. Each with reasoning and timestamp.

### [DEC-001] Work on separate branch
- **Date:** 2026-03-15
- **Decision:** All compiler work goes on `claude/external-review`, never touching `claude/general-session-VUY6r`
- **Reasoning:** Another Claude session is actively refining the engine. We don't interfere.

### [DEC-002] Research before building
- **Date:** 2026-03-15
- **Decision:** Conducted full research across 11 topic areas before writing any code
- **Reasoning:** Satish's principle — research first, understand the landscape, then build. Prevents building the wrong thing.

### [DEC-003] Compiler architecture — hybrid approach
- **Date:** 2026-03-15
- **Decision:** Use research-backed demographic-to-psychology mappings for core parameters, LLM only for narrative enrichment after parameters are locked
- **Reasoning:** NeurIPS 2025 research shows LLM-generated persona detail actually hurts accuracy. Data-grounded parameters + minimal LLM involvement = best results.
- **Source:** RESEARCH-persona-compiler.md, Section 7

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
- [ ] Design compiler architecture (next)
- [ ] Build compiler (not started)

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

### [L-001] Don't trust grep-based code analysis
- **Date:** 2026-03-15
- **Context:** First review fabricated 3 "dead code" claims because the exploration agent searched for imports instead of tracing data flow through the orchestrator (TurnPlanner)
- **Lesson:** Always trace the full consumption path. Data flows through intermediate objects (IR, metrics dicts, modifier lists), not just direct imports.

### [L-002] Verify subagent output before presenting
- **Date:** 2026-03-15
- **Context:** Presented exploration agent's findings as facts without checking. Repo handler caught 4 errors.
- **Lesson:** Never present subagent findings without spot-checking the critical claims yourself. Read the actual code at the cited lines.

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
