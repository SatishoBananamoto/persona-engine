# Product Value Evaluation

This harness tests whether Persona Engine is worth pursuing as an enterprise
research simulation product. It is intentionally decision-grade rather than a
vanity benchmark.

Current product wedge: enterprise rollout-risk simulation. The harness is not a
general market-prediction benchmark.

It compares:

- `prompt_only`: persona prompt without per-turn IR
- `schema_driven`: persona prompt plus compact schema reference plus correct IR values
- `schema_rich`: persona prompt plus richer behavioral contract reference, fuller IR JSON, and trace citations
- `engine_thin`: current `PersonaEngine.chat()` path
- `no_ir_ablation`: rich schema reference without per-turn IR
- `stale_ir`: rich schema reference with the previous turn's IR
- `contradictory_ir`: rich schema reference with competence/confidence inverted
- `shuffled_ir`: rich schema reference with IR from another persona/scenario

The important question is not whether outputs sound good. The important
question is whether correct IR beats missing, stale, shuffled, and contradictory
IR. If negative controls perform similarly to correct IR, the LLM prompt is doing
most of the work and the structured layer is not yet valuable enough.

The compact/rich split tests a specific product risk:

- If only `schema_rich` works, the value may be mostly a schema/prompt workflow.
- If `schema_driven` also works, compact IR values carry real behavioral signal.
- If `engine_thin` trails both, the current generation layer is losing value.
- If bad-IR controls perform close to correct IR, the IR is not causally useful yet.

Enterprise research turns now carry optional `ResearchIR` fields:

- `focus`
- `stakeholder_role`
- `workflow_exposure`
- `claim_basis`
- `likely_objections`
- `trust_conditions`
- `adoption_blockers`
- `workaround_risk`
- `evidence_boundary`

These fields are simulated stakeholder hypotheses. They are not measured market
evidence.

## Run

Offline smoke test:

```bash
python3 -m eval.product_value.run_eval --backend template --repeats 1
```

Decision-shaped run:

```bash
python3 -m eval.product_value.run_eval --backend template --repeats 3 --run-type decision
```

Live Anthropic run:

```bash
ANTHROPIC_API_KEY=... python3 -m eval.product_value.run_eval --backend anthropic --repeats 3
```

Agent-safe KV run:

```bash
kv run --capability api:anthropic --max-calls 400 -- \
  python3 -m eval.product_value.run_eval --backend anthropic --repeats 3
```

In KV capability mode, the eval process receives `KV_CAP_TOKEN`; the raw
Anthropic key stays with the local KV daemon.
The default decision-shaped Anthropic eval needs about 312 provider calls, so
the example uses a 400-call budget.

Outputs are written to `eval/product_value/results/`:

- `raw_results.json`
- `summary.json`
- `blind_review.jsonl`
- `report.md`

## Decision Gates

Proceed only if:

- The required method matrix is complete across all scenarios, turns, methods,
  and repeats. Missing baselines or negative controls fail closed.
- Correct compact or rich IR beats prompt-only on the weighted automatic score.
- Correct compact or rich IR beats `no_ir_ablation`.
- Correct compact or rich IR beats `stale_ir`, `shuffled_ir`, and `contradictory_ir`.
- Boundary violations are lower than prompt-only.
- Segment differentiation is stronger than prompt-only.
- IR trace correctness aligns with expected competence and claim boundaries.

The report emits separate proceed signals for `schema_driven` and `schema_rich`,
plus an overall `proceed_signal` if either clears the gates.
