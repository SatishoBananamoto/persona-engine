# Persona Authoring Guide

## YAML Schema Overview

Every persona YAML file must include these sections:

```yaml
persona_id: "P_001_MY_PERSONA"   # Unique identifier
version: "1.0"
label: "Name - Role, Location"    # Human-readable label

identity:
  age: 34
  gender: "female"
  location: "London, UK"
  education: "Master's in Psychology"
  occupation: "UX Researcher"
  background: "Brief narrative background..."

psychology:
  big_five:          # Big Five personality traits (0.0 - 1.0)
    openness: 0.75
    conscientiousness: 0.82
    extraversion: 0.45
    agreeableness: 0.68
    neuroticism: 0.35

  values:            # Schwartz values (0.0 - 1.0)
    self_direction: 0.85
    stimulation: 0.55
    hedonism: 0.40
    achievement: 0.65
    power: 0.25
    security: 0.58
    conformity: 0.35
    tradition: 0.30
    benevolence: 0.78
    universalism: 0.72

  cognitive_style:
    analytical_intuitive: 0.70     # 0=intuitive, 1=analytical
    systematic_heuristic: 0.75     # 0=heuristic, 1=systematic
    risk_tolerance: 0.45           # 0=risk-averse, 1=risk-seeking
    need_for_closure: 0.40         # 0=comfortable with ambiguity, 1=needs closure
    cognitive_complexity: 0.80     # 0=simple thinking, 1=complex reasoning

  communication:
    verbosity: 0.60                # 0=terse, 1=verbose
    formality: 0.55                # 0=casual, 1=formal
    directness: 0.65               # 0=indirect, 1=blunt
    emotional_expressiveness: 0.50 # 0=stoic, 1=expressive

knowledge_domains:
  - domain: "Psychology"
    proficiency: 0.90              # 0=novice, 1=expert
    subdomains: ["UX research", "Cognitive psychology"]

claim_policy:
  allowed_claim_types:
    - "personal_experience"
    - "general_common_knowledge"
    - "domain_expert"
  citation_required_when:
    proficiency_below: 0.6
    factual_or_time_sensitive: true
  lookup_behavior: "hedge"         # "hedge", "refuse", or "answer"

invariants:
  identity_facts: ["Lives in London", "Age 34"]
  cannot_claim: ["doctor", "lawyer"]
  must_avoid: ["revealing employer name"]

disclosure_policy:
  base_openness: 0.5
  factors:
    topic_sensitivity: -0.3
    trust_level: 0.4
  bounds: [0.1, 0.9]              # [min, max] — min MUST be < max
```

## Big Five Traits → Behavioral Effects

| Trait | High (>0.7) | Low (<0.3) | IR Fields Affected |
|-------|-------------|------------|-------------------|
| **Openness** | Higher elasticity, creative language, explores alternatives | Lower elasticity, prefers proven approaches | `elasticity`, `tone` |
| **Conscientiousness** | Higher confidence, structured responses, detailed | Lower precision, casual, brief | `confidence`, `verbosity` |
| **Extraversion** | Higher disclosure, enthusiastic tone, self-sharing | Lower disclosure, reserved, private | `disclosure_level`, `tone` |
| **Agreeableness** | Less direct, warmer tone, validates others | More direct, blunt, challenges | `directness`, `tone` |
| **Neuroticism** | Lower confidence, more hedging, cautious | Higher confidence, calm, steady | `confidence`, `tone` |

## Schwartz Values → Goal Derivation

Values influence how the persona weighs competing priorities:

| Value | Effect on Behavior |
|-------|-------------------|
| **Self-direction** | Prefers independent thinking, resists conformity |
| **Achievement** | Competitive, goal-oriented responses |
| **Benevolence** | Supportive, helpful, prioritizes user wellbeing |
| **Security** | Risk-averse, cautious, prefers safe choices |
| **Conformity** | Susceptible to authority bias, follows norms |
| **Tradition** | Conservative, conventional approaches |

## Knowledge Domains

```yaml
knowledge_domains:
  - domain: "Psychology"
    proficiency: 0.90     # Expert: confident, authoritative claims
    subdomains: ["UX research", "Behavioral science"]
  - domain: "Technology"
    proficiency: 0.70     # Competent: speaks with experience
  - domain: "Finance"
    proficiency: 0.30     # Novice: hedges, admits uncertainty
```

**Proficiency thresholds:**
- `>= 0.8`: Expert — can make domain_expert claims
- `0.5 - 0.8`: Competent — speaks from experience, may hedge
- `< 0.5`: Novice — hedges or refuses, admits uncertainty

## Common Mistakes

1. **Bounds reversed**: `bounds: [0.9, 0.1]` will fail validation. Min must be < max.
2. **Missing invariants**: Always include at least one `identity_fact`.
3. **Unrealistic Big Five**: All traits at 0.9 creates an implausible personality.
4. **No knowledge domains**: Persona won't have domain expertise — all answers hedge.
5. **Missing claim_policy**: Required field. Controls how the persona frames knowledge.

## Testing Your Persona

Run the twin test pattern to validate trait influence:

```python
import yaml
from persona_engine.schema.persona_schema import Persona
from persona_engine.planner.turn_planner import TurnPlanner, ConversationContext
from persona_engine.utils.determinism import DeterminismManager

# Load and validate
with open("personas/my_persona.yaml") as f:
    persona = Persona(**yaml.safe_load(f))

# Generate IR
planner = TurnPlanner(persona=persona, determinism=DeterminismManager(seed=42))
# ... generate and inspect IR fields
```

Run the full benchmark suite to verify:

```bash
python -m pytest tests/test_benchmarks.py -v
```
