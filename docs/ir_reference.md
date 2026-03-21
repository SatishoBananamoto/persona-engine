# Intermediate Representation (IR) Field Reference

The IR is the central data structure produced by the TurnPlanner. It captures every behavioral decision made for a single conversational turn, with full citation trails.

## Structure Overview

```
IntermediateRepresentation
├── conversation_frame
│   ├── interaction_mode      # casual_chat, interview, customer_support, ...
│   ├── goal                  # gather_info, resolve_issue, explore_ideas, ...
│   └── success_criteria
├── response_structure
│   ├── intent                # "inform", "clarify", "challenge", ...
│   ├── stance                # Persona's position on the topic
│   ├── rationale             # Why the persona holds this stance
│   ├── elasticity            # 0-1: willingness to change stance
│   ├── confidence            # 0-1: certainty in response
│   └── competence            # 0-1: perceived domain competence
├── communication_style
│   ├── tone                  # Enum: warm_enthusiastic, neutral_calm, ...
│   ├── verbosity             # brief | medium | detailed
│   ├── formality             # 0-1
│   └── directness            # 0-1
├── knowledge_disclosure
│   ├── disclosure_level      # 0-1: how much to share
│   ├── uncertainty_action    # answer | hedge | refuse | ask_clarifying
│   └── knowledge_claim_type  # domain_expert | personal_experience | ...
├── citations                 # Full audit trail of every decision
├── safety_plan               # Invariant constraints, must_avoid
├── memory_ops
│   ├── read_requests         # What was read from memory
│   └── write_intents         # What to write to memory
├── turn_id                   # Unique turn identifier
└── seed                      # Determinism seed for this turn
```

## Field Details

### response_structure.confidence (float, 0-1)

How certain the persona is in their response.

**Composition pipeline:**
1. Base from domain proficiency
2. Trait adjustment (conscientiousness ↑, neuroticism ↓)
3. Cognitive style adjustment (risk_tolerance, need_for_closure)
4. Authority bias modifier (if applicable)
5. Memory knowledge boost (prior facts about this topic)
6. Topic familiarity boost (previously discussed)
7. Cross-turn inertia smoothing
8. Bounds clamp [0, 1]

### response_structure.elasticity (float, 0-1)

Willingness to change stance when presented with new evidence.

**Composition pipeline:**
1. Trait-based elasticity (openness ↑, conscientiousness ↓)
2. Cognitive complexity blend
3. Confirmation bias modifier (if applicable)
4. Cross-turn inertia smoothing
5. Bounds clamp [0.1, 0.9]

### communication_style.tone (Tone enum)

Emotional coloring of the response. Selected from valence/arousal mapping.

**17 possible tones** organized by valence (positive/negative) and arousal (high/low):

| Positive, High Arousal | warm_enthusiastic, excited_engaged |
| Positive, Moderate | thoughtful_engaged, warm_confident, friendly_relaxed |
| Positive, Low Arousal | content_calm, satisfied_peaceful |
| Neutral | neutral_calm, professional_composed, matter_of_fact |
| Negative, High Arousal | frustrated_tense, anxious_stressed, defensive_agitated |
| Negative, Moderate | concerned_empathetic, disappointed_resigned |
| Negative, Low Arousal | sad_subdued, tired_withdrawn |

### knowledge_disclosure.disclosure_level (float, 0-1)

How much personal/professional information to share.

**Composition pipeline:**
1. Base from disclosure_policy.base_openness
2. Extraversion modifier
3. State modifier (mood, stress, fatigue)
4. Trust modifier (relationship store → higher trust = more open)
5. Privacy filter clamp
6. Topic sensitivity clamp
7. Bounds clamp

### knowledge_disclosure.uncertainty_action (enum)

What to do when the persona doesn't know the answer:

| Action | When Used |
|--------|-----------|
| `answer` | High proficiency + high confidence |
| `hedge` | Moderate proficiency, acknowledges limits |
| `refuse` | Low proficiency, persona can't speak to this |
| `ask_clarifying` | Needs more info before responding |

## Citations

Every IR field change is recorded as a `Citation`:

```python
Citation(
    source_type="trait",          # trait, state, memory, rule, cross_turn, value
    source_id="confidence_traits",
    target_field="response_structure.confidence",
    operation="add",              # set, add, multiply, blend, clamp
    effect="Traits adjust confidence (conscientiousness/neuroticism)",
    weight=0.8,
    reason="C=0.82, N=0.35"
)
```

### Reading Citations

```python
ir = planner.generate_ir(ctx)

# All citations
for c in ir.citations:
    print(f"[{c.source_type}:{c.source_id}] {c.effect}")

# Filter by field
confidence_cites = [c for c in ir.citations if "confidence" in (c.target_field or "")]
```

## Memory Operations

### Read Requests

Documents what was retrieved from memory during IR generation:

```python
for req in ir.memory_ops.read_requests:
    print(f"Read {req.query_type}: {req.query}")
```

### Write Intents

Memories to store after this turn:

```python
for intent in ir.memory_ops.write_intents:
    print(f"Write {intent.content_type}: {intent.content}")
```

## Safety Plan

Constraints that must never be violated:

```python
print(ir.safety_plan.cannot_claim)  # ["doctor", "lawyer"]
print(ir.safety_plan.must_avoid)    # ["revealing employer name"]
```

## Testing IR Fields

```python
# Assert behavioral expectations
assert ir.response_structure.confidence >= 0.7, "Expert should be confident"
assert ir.communication_style.tone.value.startswith("warm"), "Should be warm"
assert ir.knowledge_disclosure.disclosure_level < 0.5, "Should be reserved"

# Assert citation trail exists
assert any(c.source_id == "known_facts_boost" for c in ir.citations)
```
