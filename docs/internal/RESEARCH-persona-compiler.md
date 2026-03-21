# Persona Compiler — Research Document

**Date:** 2026-03-15
**Purpose:** Inform the design of a persona compiler layer that takes user inputs (segment data, prompts, toggles) and compiles psychologically coherent `Persona` objects for the persona-engine.
**Method:** Parallel web research across 11 topic areas, synthesized into architectural implications.

---

## Table of Contents

1. [Existing Tools & Platforms](#1-existing-tools--platforms)
2. [Input Formats People Actually Have](#2-input-formats-people-actually-have)
3. [Demographics to Big Five Mapping](#3-demographics-to-big-five-mapping)
4. [Demographics to Schwartz Values Mapping](#4-demographics-to-schwartz-values-mapping)
5. [Psychographic Segmentation Methods](#5-psychographic-segmentation-methods)
6. [Behavioral Signals to Personality](#6-behavioral-signals-to-personality)
7. [LLM Persona Generation — What Works and What Doesn't](#7-llm-persona-generation--what-works-and-what-doesnt)
8. [Persona Specification Standards](#8-persona-specification-standards)
9. [Multi-Persona Generation from Segments](#9-multi-persona-generation-from-segments)
10. [Gaps and Open Problems](#10-gaps-and-open-problems)
11. [Architectural Implications](#11-architectural-implications)

---

## 1. Existing Tools & Platforms

### Commercial Tools

| Tool | Input Method | Output | Depth |
|------|-------------|--------|-------|
| [Delve AI](https://www.delve.ai/) | CRM data (HubSpot), analytics APIs (GA, Facebook, YouTube) | Visual persona cards | Most data-integrated; offers "synthetic users" for simulated surveys |
| [UXPressia](https://uxpressia.com/ai-persona-generator) | Manual form fields + AI from descriptions | Visual templates | Collaboration features, primarily template-driven |
| [HubSpot Make My Persona](https://www.hubspot.com/make-my-persona) | Plain-text description of ideal customer | Structured template (demographics, goals, pain points) | Free, shallow |
| [Juma](https://juma.ai/blog/customer-persona-generators) | Custom prompts (user controls structure/tone/depth) | Flexible LLM output | Most configurable of commercial tools |
| [UserPersona.dev](https://userpersona.dev/) | One-line product description | Ideal customer profile | Extremely shallow, no signup |
| [PersonaGen API](https://www.personagen.dev/) | API-first | Synthetic personas for LLM testing | Closest to what we'd build, but no psychological grounding |
| [Venngage](https://venngage.com/ai-tools/persona-generator) | Text description | Infographic-style persona cards | Visual focus |

### Research / Academic Systems

| System | Method | Key Innovation |
|--------|--------|----------------|
| [APG (QCRI)](https://persona.qcri.org/) | Non-negative matrix factorization on analytics data | Most rigorous data-driven approach. 15+ years of research. Can generate 5, 10, 50, or 100 personas from data. |
| [Survey2Personas](https://s2p.qcri.org/) | Multiple clustering algorithms on survey data | Extension of APG specifically for survey inputs |
| [PersonaCraft](https://www.sciencedirect.com/science/article/abs/pii/S1071581925000023) (2025) | LLM (GPT-4) + survey datasets | 5-stage pipeline. Evaluated with 127 general users + 21 UX professionals. Scored high on clarity, completeness, consistency. |
| [Persona Hub (Tencent)](https://arxiv.org/abs/2406.20094) | Web mining + LLM | 1 billion personas mined from web data. Text-to-persona and persona-to-persona expansion. |
| [DeepPersona](https://arxiv.org/abs/2511.07338) (NeurIPS 2025) | Taxonomy mining + progressive sampling | Hundreds of attributes per persona. Improved GPT-4.1-mini accuracy by 11.6%. |
| [PersonaCite](https://arxiv.org/abs/2601.22288) (CHI 2026) | RAG + Voice-of-Customer grounding | Personas explicitly abstain when evidence is missing instead of hallucinating. Source attribution per response. |

### The Gap We'd Fill

**No existing tool maps user inputs to validated psychological frameworks (Big Five, Schwartz values, cognitive styles).** Every tool produces either:
- (a) Shallow marketing personas (demographics + goals + pain points), or
- (b) Deep but unvalidated LLM-generated narratives

The persona compiler would be the first system that takes segment-level input and outputs psychologically validated, schema-compliant Persona objects that can be simulated by an engine.

---

## 2. Input Formats People Actually Have

### Common Data Sources

| Format | Source | Typical Fields |
|--------|--------|---------------|
| CSV/Excel | CRM exports (HubSpot, Salesforce), survey exports (SurveyMonkey, Qualtrics) | age, gender, location, income, job_title, industry, interests, pain_points |
| JSON | API pipelines, tech companies | Nested structures for complex attributes |
| Plain text | Marketing teams, stakeholder descriptions | "Our target customer is a 35-year-old product manager in fintech who..." |
| Survey responses | Questionnaire data | Likert scales, multiple choice, free text |
| Analytics summaries | Google Analytics, social media insights | Pre-aggregated segment demographics |

### Is There a Standard Schema?

**No.** No universal persona data standard exists anywhere. The closest:
- [Schema.org/Person](https://schema.org/Person) — SEO structured data, covers name/jobTitle/affiliation but nothing behavioral or psychological
- Adobe XDM — Enterprise customer profiles, covers demographics and behaviors but not psychology

This is an opportunity. The compiler's input schema could become a de facto standard.

### Minimum Viable Input

Research on "Minimum Viable Personas" suggests tiered input:

**Tier 1 — Bare minimum (3 fields):**
1. Age or age range
2. Occupation/role
3. One goal or pain point

**Tier 2 — Useful minimum (6-8 fields):**
1. Age
2. Gender
3. Location/culture
4. Occupation
5. Education level
6. 1-2 goals
7. 1-2 pain points

**Tier 3 — Rich input (for strong psychological inference):**
All of Tier 2, plus: interests, values (even informal), decision-making style, tech comfort, social context, frustrations, aspirations

**Tier 4 — Direct specification:**
Explicit Big Five scores, Schwartz values, cognitive style parameters. Power-user mode.

---

## 3. Demographics to Big Five Mapping

### Age Correlations

Well-replicated across multiple meta-analyses and longitudinal studies:

| Trait | Direction with Age | Strength | Notes |
|-------|-------------------|----------|-------|
| Agreeableness | Increases | Moderate | Fairly linear throughout adulthood |
| Conscientiousness | Increases then plateaus | Strong | Peaks between 50-70, most robust finding |
| Neuroticism | Decreases | Moderate | Non-linear; possible late-life uptick |
| Extraversion | Decreases | Weak-moderate | Gradual decline |
| Openness | Decreases | Weak-moderate | Declines particularly after middle age |

Personality change slows significantly after age 50.

**Sources:** Srivastava et al. (2003), Terracciano et al. (2005), coordinated analysis of 16 longitudinal samples ([PMC7869960](https://pmc.ncbi.nlm.nih.gov/articles/PMC7869960/)).

### Gender Correlations

| Trait | Direction | Effect Size (Cohen's d range across 105 countries) |
|-------|-----------|---------------------------------------------------|
| Neuroticism | Women higher | d = 0.03 to 0.57 |
| Agreeableness | Women higher | d = 0.03 to 0.69 |
| Extraversion | Women higher (warmth); men higher (assertiveness) | d = -0.23 to 0.86 |
| Conscientiousness | Women slightly higher | d = -0.37 to 0.75 |
| Openness | Mixed | d = -0.19 to 0.67 |

**Critical finding:** Gender differences are *larger* in more Western, industrialized, gender-egalitarian nations (the "gender equality paradox"). A single gender correction factor cannot be applied globally.

**Effect sizes:** Most fall in small-to-moderate range (d = 0.10 to 0.65). Not large enough to stereotype individuals, but meaningful at population level.

**Sources:** Schmitt et al. (2008) across 55 cultures; Kajonius & Johnson (2018) across 105 countries.

### Occupation Correlations

Big Five maps onto Holland's RIASEC vocational types:

| Vocational Type | Strongest Big Five Correlation | r value |
|----------------|-------------------------------|---------|
| Artistic | Openness | r = 0.48 (strongest) |
| Enterprising | Extraversion | r = 0.41 |
| Social | Extraversion | r = 0.31 |
| Investigative | Openness | r = 0.28 |
| Social | Agreeableness | r = 0.19 |

Occupation-specific findings:
- **Entrepreneurs:** Low Neuroticism, high Openness, high Conscientiousness, high Extraversion, lower Agreeableness than managers
- **Software/scientists:** High Openness
- **Healthcare:** High Agreeableness predicts job performance
- **Sales/management:** High Extraversion predicts success
- **Creative professions (actors, journalists):** Higher Neuroticism
- **Conscientiousness** predicts job performance at r = 0.26 across nearly all occupations (meta-analytic)
- Better personality-occupation fit correlates with higher income

**Sources:** Holland (1997), Barrick & Mount (1991), Zhao & Seibert (2006), [PMC10089283](https://pmc.ncbi.nlm.nih.gov/articles/PMC10089283/), [University of Edinburgh (2024)](https://www.ed.ac.uk/news/2024/personality-traits-that-typify-job-roles-revealed).

### Education / SES Correlations

**Weak.** Most r values below 0.10. However:
- Higher SES → lower Neuroticism, higher Extraversion, Openness, Conscientiousness
- Conscientiousness and Openness most relevant to educational achievement
- Higher education/occupation-related SES → positive development of Conscientiousness over time

**Practical implication:** SES/education should be a *weak prior* in the compiler, not a strong signal.

**Source:** [UC Press Collabra (2021)](https://online.ucpress.edu/collabra/article/7/1/24431/117346/The-Big-Five-Across-Socioeconomic-Status).

---

## 4. Demographics to Schwartz Values Mapping

### Age Patterns

| Value Cluster | Direction with Age | Strength |
|---------------|-------------------|----------|
| Conservation (tradition, conformity, security) | Increases | Medium |
| Openness to Change (stimulation, self-direction) | Decreases | Medium (stimulation strongest) |
| Self-transcendence (benevolence, universalism) | Increases | Weak-medium |
| Self-enhancement (power, achievement) | Decreases | Weak-medium |

Slight increase in self-direction around age 60 (retirement), but meaning shifts from novelty-seeking to autonomy.

### Gender Patterns

- Men: Higher on power, stimulation, hedonism, achievement, self-direction
- Women: Higher on benevolence, universalism
- **Gender explains less variance than age, and much less than culture**

### Cross-Cultural Variation — The Dominant Factor

Culture is the strongest source of variation in Schwartz values. The Inglehart-Welzel cultural map shows two axes:
- Traditional vs. Secular-rational
- Survival vs. Self-expression

**Critical for the compiler:** Nominally similar demographics (same age, same gender) in different countries diverge substantially on values because institutions imbue social categories with context-specific meanings. A 40-year-old Japanese woman and a 40-year-old American woman may share demographics but differ meaningfully on values.

### Occupation-Specific Value Profiles

| Finding | Source |
|---------|--------|
| Management → higher self-enhancement, lower self-transcendence | Schwartz research |
| Social professions (teaching, nursing) → higher benevolence/universalism, lower power/achievement | Cross-occupational studies |
| Achievement values predict compensation and job level | ESS data |
| Benevolence values predict work engagement | Organizational research |
| Security values negatively predict international mobility | Career research |
| Stimulation values positively predict job level | Career research |

**Sources:** Schwartz (2011), [PMC5549227](https://pmc.ncbi.nlm.nih.gov/articles/PMC5549227/), [Springer](https://link.springer.com/article/10.1007/BF03173223).

---

## 5. Psychographic Segmentation Methods

### Established Frameworks

| Framework | What It Does | Data Inputs | Limitations |
|-----------|-------------|-------------|-------------|
| **VALS** (SRI International) | 8 segments by resources + motivation (Ideals/Achievement/Self-Expression) | Questionnaire (values, attitudes, lifestyle) + demographics | Criticized as "very weak" at predicting purchases. US-specific. |
| **PRIZM** (Claritas) | 67 geodemographic lifestyle types | Geographic + demographic + psychographic from ~12 sources | US/Canada only. Updated 2025. |
| **AIO** | Activities, Interests, Opinions profiling | Survey/behavioral data | More behavioral than psychological |

### Statistical Methods

| Method | Use Case |
|--------|----------|
| Factor Analysis + Cluster Analysis | Find latent psychographic dimensions, group people |
| K-Means / Hierarchical Clustering | Identify natural groupings from behavioral/attitudinal data |
| Iterative Proportional Fitting (IPF) | Generate individual records matching macro-level distributions |
| Non-negative Matrix Factorization | APG's method for segmenting analytics data into persona clusters |

**Key takeaway:** These frameworks validate combining demographic inputs with psychographic inference, but accuracy is modest. The mapping from observable signals to psychological traits is inherently noisy. The compiler should treat outputs as probability distributions, not point estimates.

---

## 6. Behavioral Signals to Personality

### Digital Footprint Research

**Landmark study — Youyou, Kosinski & Stillwell (2015), PNAS.** N = 86,220.

Computer predictions from Facebook Likes (r = 0.56) outperformed personality judgments by:
- Work colleagues: 10 Likes needed to match
- Friends/cohabitants: 70 Likes
- Family members: 150 Likes
- Spouses: 300 Likes

### Meta-Analytic Accuracy (Azucar et al., 2018)

Correlations from digital footprints to Big Five:

| Trait | Correlation | Predictability |
|-------|------------|----------------|
| Extraversion | r = 0.40 | Best predicted |
| Openness | r = 0.40 | Best predicted |
| Conscientiousness | r = 0.33 | Moderate |
| Neuroticism | r = 0.33 | Moderate |
| Agreeableness | r = 0.29 | Hardest to predict |

Accuracy improves when combining demographics + multiple footprint types.

### Text/Language-Based Prediction

- Psycholinguistic features contribute ~70% of total importance in personality prediction models
- Park et al.'s Facebook language model: O r=0.46, C r=0.38, E r=0.41, A r=0.40, Emotional Stability r=0.39
- Deep learning (BERT/RoBERTa/XLNet) achieves 86-88% classification accuracy on binary personality labels

### Purchase Behavior — Weak Signal

Personality explains relatively little purchase variance: 5% of impulsive buying, 4% of compulsive behavior, 8% of panic buying. Reveals tendencies, not deterministic predictions.

### Ethical Considerations

- Personality inference from digital footprints raises "predictive privacy" concerns
- GDPR/HIPAA provide frameworks but regulation lags behind capabilities
- Consent, transparency, data minimization required
- The compiler should support behavioral inputs but must be transparent about the inference being made

**Sources:** [Youyou et al. (2015) PNAS](https://www.pnas.org/doi/10.1073/pnas.1418680112), [Azucar et al. (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0191886917307328).

---

## 7. LLM Persona Generation — What Works and What Doesn't

### The Critical Warning

**"LLM Generated Persona is a Promise with a Catch" (NeurIPS 2025, [arXiv:2503.16527](https://arxiv.org/abs/2503.16527)):**

- More detailed, "realistic" personas (rich backstories, personality quirks) **actually perform WORSE** than stripped-down census-style profiles
- Detailed personas drift further from real human behavior
- LLMs systematically exaggerate racial markers, flattening identity into predictable stereotypes
- Personas with more LLM-generated content skew left-leaning and less representative
- Default ChatGPT persona: dominant Agreeableness + Conscientiousness, low Neuroticism (reflects RLHF training)

### What Works

| Approach | Evidence |
|----------|----------|
| **Census-grounded demographic scaffolding** | Align attributes with Census categories BEFORE letting LLM fill narrative |
| **Minimal persona complexity** | TOP-2 or coarse attributes often suffice. Surplus detail degrades performance. |
| **Structured templates** | >50% of successful prompts use JSON output. Stick to objective categories. |
| **Adding sociopsychological traits** | Demographics alone explain ~1.5% of behavioral variance; Big Five + values yield higher alignment |
| **DeepPersona framework** | Enhanced GPT-4.1-mini accuracy by 11.6%, narrowed sim-to-real gap by 31.7%, reduced Big Five gap by 17% |

### What Doesn't Work

| Approach | Problem |
|----------|---------|
| Pure prompt-based roleplaying | "Persuasive but unverifiable" responses |
| Unconstrained LLM generation | Systematically biased — more optimistic, progressive, emotionally positive than real populations |
| Rich narrative backstories | More detail = more drift from real behavior |
| Ignoring cultural variance | LLM default voice is educated Western English-speaking |

### Trait-Specific Accuracy

LLMs accurately simulate Openness, Extraversion, and Neuroticism but produce inaccurate results for **Conscientiousness and Agreeableness**. These are the traits that need the most explicit constraint from the compiler.

### Consistency Failure

Within the same conversation, LLMs lose track of persona details over long interactions, shift tone, and overlook earlier instructions. This is exactly what persona-engine's IR solves — the personality is computed from math, not maintained by LLM memory.

**Sources:** [arXiv:2503.16527](https://arxiv.org/abs/2503.16527), [DeepPersona](https://arxiv.org/abs/2511.07338), [Polypersona](https://arxiv.org/abs/2512.14562), [The Prompt Makes the Person(a)](https://arxiv.org/abs/2507.16076).

---

## 8. Persona Specification Standards

### No Formal Standard Exists

No ISO, W3C, or industry standard for persona specification.

### Common Fields Across All Frameworks

Synthesized from Asana, Mural, Miro, Userpilot, Nielsen Norman Group:

**Universal (>90% of frameworks):**
1. Name
2. Age
3. Occupation / Role
4. Goals
5. Pain points / Frustrations
6. Photo/avatar

**Very Common (>70%):**
7. Location
8. Education
9. Brief biography
10. Behaviors / Habits
11. Motivations
12. Preferred channels / tools

**Common (>50%):**
13. Personality traits (informal: "analytical," "social")
14. Representative quote
15. Tech comfort level
16. Decision-making factors

**Rare but valuable:**
17. Formal values (never mapped to Schwartz in any commercial tool)
18. Cognitive style
19. Emotional patterns
20. Knowledge domains with proficiency levels

### What the Persona Engine Already Has

The existing `Persona` schema is far more sophisticated than any commercial or academic specification:
- Big Five traits (validated 0-1 scale)
- All 10 Schwartz values
- Cognitive style (5 dimensions)
- Communication preferences (4 dimensions)
- Uncertainty policy, claim policy, disclosure policy
- Dynamic state (mood, fatigue, stress, engagement)
- Self-schemas, biases, decision policies
- Knowledge domains with proficiency scores

**The compiler's job:** Translate from the shallow "common fields" world (Tiers 1-3) into this deep schema.

---

## 9. Multi-Persona Generation from Segments

### How Many Personas Per Segment?

**QCRI research ([persona.qcri.org](https://persona.qcri.org/blog/using-data-driven-personas-for-enhanced-user-segmentation/)):** Creating more personas per segment dramatically improves demographic representation. Relative diversity gain is maximal at approximately **40 personas** — implying organizations should create 4x more than the typical recommendation of 3-7 total.

### Variation Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Statistical sampling from trait distributions** | Compute mean + SD for each trait from demographic correlations, then sample. Each sample = a different persona. | Most principled. Grounded in research. |
| **Attribute-level variation within constraints** | Fix segment-defining attributes, vary psychological traits within plausible ranges. | When segment definition is rigid. |
| **Iterative Proportional Fitting (IPF)** | Standard in synthetic population generation. Takes macro-level distributions, generates individual records that match. | Census/epidemiology-grade realism. |
| **Persona-to-persona expansion** | Start with one persona, generate related but different personas via relationships (colleague, client, friend). | Expanding persona ecosystems. |
| **Factorial variation** | Identify 3-4 key dimensions, systematically vary high/low combinations → 2^n personas. | Maximum coverage with few personas. |

### Ensuring Diversity

- **Constraint-based generation:** Enforce minimum representation across sub-groups
- **Dissimilarity maximization:** Select final set to maximize pairwise distance in trait space
- **Coverage metrics:** Track which regions of trait space are covered, fill gaps
- **Census-derived priors:** Use real population statistics, not LLM defaults

**Sources:** [APG QCRI](https://persona.qcri.org/), [JASSS synthetic populations](https://www.jasss.org/25/2/6.html), [ACM representation study](https://dl.acm.org/doi/10.1145/3546155.3546654).

---

## 10. Gaps and Open Problems

### 1. No Published Demographics-to-Big-Five Lookup Table

The research exists in scattered papers. No single source maps (age, gender, occupation, culture) to (Big Five means and standard deviations). The compiler needs to synthesize this from multiple sources. Doable but requires careful calibration.

### 2. Schwartz Values Are Harder to Map from Demographics

Correlations are weaker and more culture-dependent than Big Five. Occupation and stated goals may be stronger signals than age/gender.

### 3. Internal Consistency Validation

No existing tool checks if generated traits are internally consistent. The Schwartz value circle provides theoretical constraints — adjacent values should correlate positively, opposing values negatively. Big Five traits also have known inter-correlations (e.g., high Agreeableness + high Neuroticism is unusual).

### 4. Cognitive Style Has No Demographic Correlations

Cognitive style and communication preferences need to be inferred from occupation and personality traits, not demographics directly. No published research maps demographics to risk tolerance, need for closure, or analytical-vs-intuitive style.

### 5. LLM Bias Correction

When using LLMs for narrative enrichment, systematic optimism/progressivism bias needs active mitigation. Census-derived constraints on psychological parameters are the main defense.

### 6. Ecosystem Validation

How do you validate that a *set* of personas from a segment is realistic, not just individual personas? No established methodology exists.

---

## 11. Architectural Implications

### Load-Bearing Takeaways

**1. Use demographic priors as probability distributions, not point estimates.**
Age-to-personality correlations are real but modest (r = 0.1-0.4). Gender effects vary by culture. SES effects are near-zero. The compiler should compute (mean, standard_deviation) for each trait given demographics, then sample.

**2. Occupation is the strongest signal.**
RIASEC-Big Five correlations reach r = 0.48. Schwartz values show clear occupation-specific profiles. Occupation should be the primary input for psychological inference, ahead of age and gender.

**3. Culture must be a first-class dimension.**
Schwartz/Inglehart data shows cultural context dominates over age and gender for values. A compiler that ignores cultural zone will produce American-default personas.

**4. Keep LLM involvement minimal and constrained.**
Rich narrative backstories from LLMs hurt accuracy. Census-style attribute scaffolding + structured output consistently outperforms creative descriptions. Use LLMs only for narrative enrichment (background story, response patterns) AFTER psychological parameters are locked by data.

**5. Conscientiousness and Agreeableness need extra care.**
Both are the hardest traits to predict from behavioral data (lowest r values) AND the hardest for LLMs to simulate accurately. Budget extra validation effort here.

**6. Layer psychological dimensions explicitly.**
Demographic-only personas explain ~1.5% of behavioral variance. Big Five + Schwartz values as separate tunable dimensions yield higher predictive alignment.

**7. Support multiple input tiers.**
Different users have different data available. The compiler must gracefully handle everything from "35-year-old product manager" (3 fields) to full survey datasets with psychometric scores.

**8. Generate sets, not singletons.**
Research shows diversity gain is maximal at ~40 personas per segment. The compiler should default to producing multiple varied personas per input, not one "average" persona.

### Proposed Pipeline

```
User Input (text / structured fields / CSV / toggles)
        |
        v
  +-----------------+
  |  Input Parser    |  Normalize to common intermediate format
  +-----------------+
        |
        v
  +-----------------+
  |  Demographic     |  Map age, gender, occupation, culture to
  |  Prior Engine    |  Big Five + Schwartz value distributions
  +-----------------+  (mean + SD for each trait)
        |
        v
  +-----------------+
  |  Sampler         |  Draw N trait vectors from distributions
  +-----------------+  (statistical sampling or factorial variation)
        |
        v
  +-----------------+
  |  Gap Filler      |  Infer cognitive style, comm prefs, policies
  +-----------------+  from traits + occupation + goals
        |
        v
  +-----------------+
  |  Consistency     |  Validate: Schwartz circle constraints,
  |  Validator       |  Big Five inter-correlations, trait-value
  +-----------------+  alignment
        |
        v
  +-----------------+
  |  Persona         |  Compile into engine-compatible Persona
  |  Assembler       |  objects with all ~50 parameters filled
  +-----------------+
        |
        v
  +-----------------+
  |  Narrative       |  Optional: LLM enrichment for background,
  |  Enricher        |  response patterns, quotes (parameters locked)
  +-----------------+
        |
        v
  N x Persona objects → ready for engine.chat()
```

### What the Compiler Does NOT Do

- Does NOT replace the engine — it's a layer on top
- Does NOT use LLMs for psychological parameter generation — those come from research-backed mappings
- Does NOT produce one "average" persona — produces varied sets
- Does NOT require any specific input format — accepts text, structured fields, CSV, or direct parameter specification

---

## Sources

### Personality-Demographics Research
- [Age Differences in Big Five (PMC2562318)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2562318/)
- [Big Five Trajectories: 16 Longitudinal Samples (PMC7869960)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7869960/)
- [Gender Differences Across 105 Countries (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0092656620301367)
- [Gender Differences in Personality (PMC3149680)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3149680/)
- [Big Five in the Workplace (PMC10089283)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10089283/)
- [Big Five Across SES (UC Press)](https://online.ucpress.edu/collabra/article/7/1/24431/117346/The-Big-Five-Across-Socioeconomic-Status)
- [SES Effects on Personality (PMC10256837)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256837/)
- [Personality and Occupation (University of Edinburgh 2024)](https://www.ed.ac.uk/news/2024/personality-traits-that-typify-job-roles-revealed)

### Values Research
- [Schwartz Values: Age and Gender (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0191886918305452)
- [Values and Adult Age: ESS (PMC5549227)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5549227/)
- [Values and Work Environment: 32 Occupations (Springer)](https://link.springer.com/article/10.1007/BF03173223)
- [Sex Differences in Value Priorities (Schwartz & Rubel)](https://www.researchgate.net/publication/7378572_Sex_Differences_in_Value_Priorities_Cross-Cultural_and_Multimethod_Studies)
- [Demographic Predictors of Values (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2590291122000183)
- [Inglehart-Welzel Cultural Map](https://en.wikipedia.org/wiki/Inglehart%E2%80%93Welzel_cultural_map_of_the_world)

### Behavioral Prediction
- [Computer-Based Personality Judgments (Youyou et al., PNAS)](https://www.pnas.org/doi/10.1073/pnas.1418680112)
- [Big Five from Digital Footprints: Meta-Analysis (Azucar et al.)](https://www.sciencedirect.com/science/article/abs/pii/S0191886917307328)

### LLM Persona Generation
- [LLM Persona: Promise with a Catch (NeurIPS 2025)](https://arxiv.org/abs/2503.16527)
- [DeepPersona (NeurIPS 2025)](https://arxiv.org/abs/2511.07338)
- [PersonaCite (CHI 2026)](https://arxiv.org/abs/2601.22288)
- [Polypersona](https://arxiv.org/abs/2512.14562)
- [The Prompt Makes the Person(a)](https://arxiv.org/abs/2507.16076)

### Tools & Platforms
- [Automatic Persona Generation (APG)](https://persona.qcri.org/)
- [Survey2Personas](https://s2p.qcri.org/)
- [PersonaCraft (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1071581925000023)
- [Persona Hub (Tencent)](https://arxiv.org/abs/2406.20094)
- [Delve AI](https://www.delve.ai/)
- [PersonaGen API](https://www.personagen.dev/)

### Segmentation & Synthetic Populations
- [VALS Framework](https://en.wikipedia.org/wiki/VALS)
- [Claritas PRIZM](https://www.claritas.com/prizm-premier/)
- [Psychographic Segmentation (Qualtrics)](https://www.qualtrics.com/articles/strategy-research/psychographic-segmentation/)
- [Synthetic Population Generation Review (JASSS)](https://www.jasss.org/25/2/6.html)
- [Creating More Personas Improves Representation (ACM)](https://dl.acm.org/doi/10.1145/3546155.3546654)
