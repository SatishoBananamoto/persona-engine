# Plan: Context Classifier + Opinion Mode

> Use case: Coca-Cola wants to test how different customer segments react to products.
> "18-24 urban health-conscious" should respond DIFFERENTLY from "55-65 suburban traditional"
> — and the responses should be CLOSE to what real humans in those segments would say.

## What the engine currently does wrong for this use case

User asks a persona: "What do you think about our new zero-sugar cola?"

Current pipeline:
```
detect_domain("zero-sugar cola") → domain="food" or "general"
compute_competence(proficiency=0.3) → competence=0.15
knowledge_claim_type → "speculative"
uncertainty_handling → "let uncertainty show"
stance → "I value autonomy and the freedom to choose my own approach"
confidence → 0.42
```

This is absurd. A consumer has OPINIONS about cola — they don't "speculate" about it.
They don't need "domain proficiency" to know if they like a drink. Their response should
come from their VALUES, LIFESTYLE, and PERSONALITY — not from expertise assessment.

## What it should do

```
classify_context("What do you think about our new zero-sugar cola?")
→ context_type = "opinion_preference"

# Skip: domain proficiency, competence, knowledge claim, uncertainty
# Use: personality traits, values, lifestyle, age/generation, habits

Route to opinion mode:
  - Stance from VALUES + LIFESTYLE (not domain expertise)
    "I'm trying to cut sugar, so I'd definitely try it" (health-conscious)
    "I don't trust zero-sugar stuff, tastes artificial" (traditional)
  - Confidence from EXTRAVERSION + OPENNESS (not proficiency)
    High-E: strong opinion, shares freely
    Low-E: reserved, needs prompting
  - No "speculative" or "uncertain" — opinions aren't speculation
  - Verbosity from personality + engagement (not fatigue + domain)
```

## The Plan (6 phases)

### Phase 1: Context Classifier (CC-1/CC-2)

**What:** A classifier at the top of Stage 2 that categorizes user input.

**Where:** `persona_engine/planner/stages/interpretation.py`, before any domain detection.

**Categories:**

| Context Type | Examples | What should drive response |
|---|---|---|
| `knowledge` | "Explain quantum entanglement" | Domain proficiency, competence, claim type, uncertainty |
| `opinion` | "What do you think about Coke Zero?" | Values, lifestyle, personality, preferences |
| `social` | "You're at a party with strangers" | E/A traits, social cognition, mood |
| `emotional` | "How are you feeling about the layoffs?" | N trait, mood state, disclosure |
| `personal` | "Tell me about your morning routine" | Memory, lifestyle, disclosure, habits |
| `adversarial` | "You're wrong, defend yourself" | Character stability, formality hold |
| `purchase` | "Would you buy this? How much would you pay?" | Values, income sensitivity, risk tolerance, brand loyalty |

**Implementation:** Start with keyword-based (fast, deterministic, no LLM cost):

```python
def classify_context(user_input: str) -> str:
    lower = user_input.lower()

    # Purchase/preference signals
    if any(w in lower for w in ["buy", "purchase", "pay", "price", "worth",
                                  "try", "switch", "prefer", "choose"]):
        return "purchase"

    # Opinion signals
    if any(w in lower for w in ["think about", "your view", "opinion",
                                  "feel about", "what about", "how about"]):
        return "opinion"

    # Social signals
    if any(w in lower for w in ["party", "meeting", "dinner", "event",
                                  "introduce", "stranger", "group"]):
        return "social"

    # Emotional signals
    if any(w in lower for w in ["feeling", "worried", "happy", "stressed",
                                  "anxious", "excited", "upset"]):
        return "emotional"

    # Personal signals
    if any(w in lower for w in ["your routine", "your morning", "your day",
                                  "tell me about yourself", "your life"]):
        return "personal"

    # Adversarial signals
    if any(w in lower for w in ["you're wrong", "defend", "prove",
                                  "disagree", "challenge"]):
        return "adversarial"

    # Default: knowledge (current behavior)
    return "knowledge"
```

**Can upgrade later** to embedding-based or LLM-based classification.

---

### Phase 2: Pipeline Routing (CC-2)

**What:** Each context type routes to a different processing mode in the behavioral pipeline.

**Changes to `interpretation.py` and `behavioral.py`:**

```python
context_type = classify_context(user_input)

if context_type == "knowledge":
    # Current pipeline — unchanged
    proficiency = compute_proficiency(domain)
    competence = compute_competence(domain, proficiency)
    claim_type = infer_claim_type(proficiency)
    uncertainty = resolve_uncertainty(proficiency, confidence)

elif context_type in ("opinion", "purchase"):
    # Opinion mode — personality-driven, no expertise
    proficiency = None  # not relevant
    competence = 0.5    # everyone has opinions
    claim_type = "personal_opinion"  # new type
    uncertainty = "answer"  # opinions aren't uncertain
    # Confidence from personality, not proficiency:
    confidence = 0.4 + E * 0.3 + (1 - N) * 0.2  # range 0.4-0.9
    # Stance from values + lifestyle, not domain expertise

elif context_type == "social":
    # Social mode — E/A/N driven
    proficiency = None
    competence = 0.5
    claim_type = "personal_experience"
    uncertainty = "answer"
    confidence = 0.3 + E * 0.4  # extroverts confident socially
    # Verbosity: high-E → more verbose (opposite of current)
    # Directness: personality-only (no competence modulation)

elif context_type == "emotional":
    # Emotional mode — N/mood driven
    proficiency = None
    competence = 0.5
    claim_type = "personal_experience"
    # Disclosure amplified
    # N trait heavily influences response style
    # Mood state matters more than usual

elif context_type == "personal":
    # Personal mode — memory + disclosure
    # Pull from memory context
    # High disclosure
    # Lifestyle/habits drive response

elif context_type == "adversarial":
    # Adversarial mode — hold character
    # Maintain formality
    # Don't break character
    # Confidence and directness hold steady
```

---

### Phase 3: Thin IR for Text Generation (EV-2/EV-3)

**What:** Simplify the prompt sent to the LLM. Currently 3,000+ characters of constraints. Reduce to ~500 characters.

**Current prompt (30+ constraints):**
```
TONE: professional and composed
FORMALITY: 36%
DIRECTNESS: 67%
VERBOSITY: BRIEF
CONFIDENCE: 42%
COMPETENCE: 57%
STANCE: I value autonomy...
REASONING: self_direction value...
KNOWLEDGE CLAIM: speculative
UNCERTAINTY: let uncertainty show
CLAMPED LIMITS: 1
PERSONALITY BEHAVIOR: 2 directives
LANGUAGE STYLE: 2 paragraphs
GENERATION INSTRUCTIONS: 8 rules
```

**Thin IR prompt (~500 chars):**
```
CHARACTER: You're practical, organized, and reserved. You speak plainly
and don't waste words. You're fairly calm and not easily rattled.

CONTEXT: Someone's asking your opinion about a product.

TONE: neutral and straightforward
YOUR TAKE: [stance if relevant]

Respond naturally as this person would. 2-4 sentences.
```

The full IR is still computed for measurement/auditability. Only the PROMPT is thinned.

---

### Phase 4: Segment-Aware Layer Zero

**What:** Layer Zero should generate personas from SEGMENT descriptions, not just occupations.

**Current Layer Zero input:**
```python
layer_zero.mint(occupation="nurse", age=35, location="Chicago")
```

**Needed for market research:**
```python
layer_zero.mint(
    segment="18-24 urban health-conscious college student",
    # or structured:
    age_range=(18, 24),
    lifestyle=["health-conscious", "fitness-focused", "social-media-active"],
    income_level="low-medium",
    brand_preferences={"beverages": "kombucha, sparkling water"},
    values_emphasis={"health": 0.9, "social_image": 0.7, "price_sensitive": 0.8},
)
```

**New persona schema fields needed:**
- `lifestyle: list[str]` — lifestyle descriptors
- `consumption_habits: dict` — category → preferences
- `price_sensitivity: float` — 0-1
- `brand_loyalty: float` — 0-1 (high = sticks with known brands)
- `trend_sensitivity: float` — 0-1 (high = follows trends)
- `health_consciousness: float` — 0-1

These influence opinion/purchase mode responses directly.

---

### Phase 5: Validate Against Real Market Research

**What:** Compare engine output against real human segment responses.

**Method:**
1. Find published market research with known segment profiles + their responses
2. Create matching personas in persona-engine
3. Ask the same questions
4. Compare response themes, sentiment, and key phrases

**Sources for real data:**
- Published consumer surveys (Pew Research, Nielsen)
- Academic consumer behavior studies
- A/B test: Satish (or testers) answer as themselves, compare against their modeled persona

---

### Phase 6: Multi-Persona Panel

**What:** Run multiple personas through the same question simultaneously and aggregate.

```python
# Coke product test across segments
segments = [
    {"name": "Health Millennial", "age": 24, "lifestyle": ["health-conscious"]},
    {"name": "Traditional Boomer", "age": 58, "lifestyle": ["traditional", "brand-loyal"]},
    {"name": "Budget Parent", "age": 35, "lifestyle": ["budget-conscious", "family-first"]},
    {"name": "Trendy GenZ", "age": 19, "lifestyle": ["trend-following", "social-media"]},
]

question = "Coca-Cola is launching a new zero-sugar cola with added vitamins. Would you try it? Why or why not?"

for segment in segments:
    personas = layer_zero.mint(**segment, count=10)  # 10 per segment
    responses = [engine.chat(question) for engine in engines]
    # Aggregate: sentiment, key themes, purchase intent
```

---

## Priority Order

| Phase | Effort | Impact | Dependencies |
|-------|--------|--------|-------------|
| **Phase 1: Context classifier** | Small (1 file, keyword-based) | Fixes 66% of known issues | None |
| **Phase 2: Pipeline routing** | Medium (interpretation.py + behavioral.py) | Makes IR appropriate for non-knowledge Qs | Phase 1 |
| **Phase 3: Thin IR prompt** | Small (prompt_builder.py) | Improves text quality across all models | None (can parallelize) |
| Phase 4: Segment Layer Zero | Large (schema + Layer Zero + priors) | Enables market research use case | Phase 1+2 |
| Phase 5: Real data validation | Medium (research + eval scripts) | Proves accuracy | Phase 4 |
| Phase 6: Multi-persona panel | Medium (orchestration layer) | Production use case | Phase 4+5 |

**Phases 1-3 are immediate. Phases 4-6 are the product roadmap.**
