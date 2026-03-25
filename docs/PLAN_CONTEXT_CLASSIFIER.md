# Plan: Context Classifier

> The IR pipeline treats every input as a knowledge query. 10 of 15 issues found
> in this session (66%) trace to this single root cause. This plan fixes it.

## The Problem

Ask a persona: "You're at a party where you know nobody. What do you do?"

Current pipeline output:
```
CONFIDENCE: 42%         ← based on domain proficiency. Parties aren't a domain.
KNOWLEDGE CLAIM: speculative  ← nobody "speculates" about socializing.
STANCE: "I value autonomy"    ← Schwartz values for a party question?
UNCERTAINTY: "let it show"    ← uncertain about... going to a party?
COMPETENCE: 16%         ← not a competence question.
```

The pipeline only has one mode: **assess topic expertise and constrain accordingly.**
It doesn't know the difference between a knowledge question, a social situation,
an emotional check-in, or an opinion request.

## The Fix: Context Classifier (core architecture)

A classifier at the top of Stage 2 (Interpretation) that routes processing.
This is a CORE architecture fix — not tied to any specific use case.

### Phase 1: Build Classifier (CC-1)

**Where:** `persona_engine/planner/stages/interpretation.py`, before domain detection.

**Categories:**

| Context Type | Examples | What drives the response |
|---|---|---|
| `knowledge` | "Explain quantum entanglement", "How does photosynthesis work?" | Domain proficiency, competence, claim type, uncertainty |
| `opinion` | "What do you think about X?", "Your view on Y?" | Values, personality, preferences |
| `social` | "You're at a party", "How would you introduce yourself?" | E/A traits, social cognition |
| `emotional` | "How are you feeling?", "Are you worried about this?" | N trait, mood state, disclosure |
| `personal` | "Tell me about your routine", "What do you enjoy?" | Memory, lifestyle, disclosure |
| `adversarial` | "You're wrong", "Defend your position" | Character stability, formality |

**Implementation:** Keyword-based first (deterministic, no cost). Can upgrade to
embedding or LLM classifier later.

```python
def classify_context(user_input: str) -> str:
    lower = user_input.lower()

    if any(w in lower for w in ["you're wrong", "defend", "prove it",
                                  "disagree", "challenge"]):
        return "adversarial"

    if any(w in lower for w in ["feeling", "worried", "happy", "stressed",
                                  "anxious", "excited", "how are you"]):
        return "emotional"

    if any(w in lower for w in ["your routine", "your morning", "your life",
                                  "about yourself", "what do you enjoy"]):
        return "personal"

    if any(w in lower for w in ["party", "dinner", "event", "meeting someone",
                                  "introduce", "stranger", "group of people"]):
        return "social"

    if any(w in lower for w in ["think about", "your view", "opinion on",
                                  "feel about", "what about", "how about",
                                  "would you", "do you prefer"]):
        return "opinion"

    return "knowledge"  # default: current behavior
```

Same limitation as other keyword systems — finite list, will miss edge cases.
Upgrade path: embedding similarity against category exemplars.

### Phase 2: Pipeline Routing (CC-2)

**What changes per context type:**

| Field | knowledge | opinion | social | emotional | personal | adversarial |
|---|---|---|---|---|---|---|
| Domain detection | YES | skip | skip | skip | skip | skip |
| Competence | compute | 0.5 (neutral) | 0.5 | 0.5 | 0.5 | hold current |
| Claim type | compute | personal_opinion | personal_experience | personal_experience | personal_experience | hold current |
| Uncertainty | compute | "answer" | "answer" | varies by N | "answer" | "answer" |
| Confidence source | proficiency + DK | personality (E, N) | E trait | mood + N | baseline | hold current |
| Stance source | values + domain | values + personality | personality | mood + values | memory | hold current |
| Verbosity modifier | standard | standard | E amplified | standard | standard | standard |
| Directness modifier | competence-modulated | personality-only | personality-only | N-modulated | personality-only | hold current |

**Key change:** For non-knowledge contexts, confidence comes from PERSONALITY
(high-E → confident opinions, low-N → confident in social settings), not from
domain proficiency. This eliminates the self-efficacy band-aid (CF-1) at the source.

### Phase 3: Thin IR Prompt (EV-2/EV-3)

**What:** Simplify what we send to the LLM. Keep full IR for measurement.

**Current:** ~3,000 characters, 30+ constraints, 8 generation rules.

**Proposed:** ~500 characters. Three sections:

```
CHARACTER: [interpolated personality description from _TRAIT_POLES]

SITUATION: [context-appropriate framing]
  knowledge: "You're being asked about [domain]. Your expertise: [level]."
  opinion: "Someone's asking your opinion."
  social: "You're in a social situation."
  emotional: "Someone's checking in on how you're feeling."

RESPONSE: [tone], [verbosity hint]. Respond naturally as this person.
```

The full IR (confidence=0.42, directness=0.67, formality=0.36...) is still
computed and stored for measurement/auditability. It's just not dumped into
the LLM prompt where it overwhelms natural expression.

---

## After the Core Fix: Use-Case Extensions

These are SEPARATE from the architecture fix. Each is a product decision.

| Extension | What it enables | Depends on |
|---|---|---|
| Segment-aware Layer Zero | Market research, consumer testing | Phase 1-3 + schema additions |
| Multi-persona panel | Focus group simulation | Layer Zero + orchestration |
| Purchase behavior mode | "Would you buy this?" with price sensitivity | Phase 2 + schema additions |
| Conversation replay | Re-run same conversation with different persona | Save/load v3 + determinism |
| Human evaluation framework | Blind comparison: persona vs real human | Phase 1-3 + evaluation protocol |

Each of these can be planned independently once the core architecture is fixed.

---

## Priority

```
NOW:     Phase 1 (classifier) → Phase 2 (routing) → Phase 3 (thin IR)
         One session. Fixes 66% of known issues.

THEN:    Re-run fair comparison (CC-5). Validate IR beats prompt-only
         on BOTH Claude and GPT-4o when context is correct.

LATER:   Use-case extensions based on product direction.
```

## Files to Change

| File | Change |
|------|--------|
| `interpretation.py` | Add `classify_context()`, call before domain detection, pass to downstream |
| `behavioral.py` | Route metrics computation based on context type |
| `behavioral_metrics.py` | Confidence from personality for non-knowledge contexts |
| `behavioral_style.py` | Verbosity: E-amplified for social contexts |
| `knowledge.py` | Skip claim type / uncertainty for non-knowledge contexts |
| `prompt_builder.py` | Thin IR: context-appropriate framing, fewer constraints |
| `stage_results.py` | Add `context_type` field to `InterpretationResult` |
