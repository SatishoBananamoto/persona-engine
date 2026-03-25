# External Validation Study — Design Review

> Reviewed 2026-03-21. Two studies run by a separate Claude Code session comparing
> IR-driven (persona-engine) vs prompt-only (vanilla LLM). This document reviews
> the test design for flaws before accepting the results.

## Study 1: 20-Turn Conversation (eval_ir_vs_prompt.py)

**What it tests well:**
- Multi-turn behavioral consistency
- Domain awareness (competence drop out-of-domain)
- Adversarial resistance (character breaks under pressure)
- Token economics and latency comparison

**Design flaws:**

1. **Unfair word count comparison.** IR accumulates fatigue state → responses shrink.
   Prompt accumulates conversation history → context grows but LLM doesn't "get tired."
   Comparing word counts measures different phenomena, not which approach is "better."

2. **Prompt-only gets free parameter information.** The hand-written system prompt
   includes explicit Big Five scores ("conscientiousness: 0.95") and communication
   parameters ("formality: 0.85"). Claude can interpret these directly without needing
   the IR pipeline. The IR approach computes these — the prompt approach gets them for free.

3. **No state reset for IR.** The IR uses `engine.chat()` across all 20 turns.
   By turn 13, fatigue collapses responses to 25 words. This is the engine working
   as designed, but it means turns 13-20 can't express personality in text.

## Study 2: 10 Archetypes × 12 Scenarios (validation_study.py)

**What it tests well:**
- Construct validity — does the system produce Big Five-aligned behavior?
- Comparison of Layer Zero minting vs simple prompting
- Per-dimension breakdown (O, C, E, A, N)

**Design flaws:**

4. **Expected ratings are one person's judgment.** Lines 209-221 claim to be
   "research-grounded" and cite Costa & McCrae, John et al., Roberts et al.
   But the actual derivation (why A01 rates Q01 as 4) is not shown. These are
   interpretations, not empirical measurements.

5. **IR over-engineered for the task.** The persona engine fires 16 interpreters,
   generates stance, tone, verbosity, behavioral directives — all for a task that
   needs a single digit 1-5. The prompt-only approach sends "respond with a number"
   and gets a clean number. The engine isn't designed for rating scales.

6. **IR accumulates state across scenarios.** `engine.chat()` is called 12 times
   per archetype WITHOUT reset. Scenario 12 is influenced by scenarios 1-11
   (fatigue, mood, stance cache). Prompt-only is stateless per scenario.
   This systematically disadvantages IR on later scenarios.

7. **Rating extraction is fragile for IR.** The engine generates full-sentence
   responses; regex extracts a digit. The prompt-only approach requests only a number.
   Extraction failures or wrong digit captures bias against IR.

8. **Small sample.** 120 data points on a 1-5 ordinal scale. The r=0.68 vs r=0.58
   difference has wide confidence intervals and may not be statistically significant.

## Verdict

Study 1 results are **reliable for measurability, latency, domain awareness, and
adversarial resistance.** The word count / personality richness comparison is confounded
by fatigue state accumulation.

Study 2 results **systematically disadvantage IR** through state accumulation, over-
engineered response generation, and fragile rating extraction. The r=0.68 vs r=0.58
difference should NOT be interpreted as "prompt-only is better at personality" without
accounting for these confounds.

**What a fair Study 2 would look like:**
- Reset engine state between scenarios (new engine per archetype)
- Use `engine.plan()` instead of `engine.chat()` (IR without LLM generation)
- Compare IR parameters directly against expected trait directions
- Or: use the engine with a "rating" mode that bypasses stance/linguistic generation
