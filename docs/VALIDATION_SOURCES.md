# Validation Sources — Personality to Behavior Mapping

> Compiled 2026-03-20. Papers, datasets, and frameworks for validating
> persona-engine's Big Five → behavioral parameter mapping.

## Tier 1: Correlation Tables (direct validation)

### Yarkoni (2010) — "Personality in 100,000 Words"
- **What:** N=694 bloggers, 66 LIWC categories × 5 traits, full correlation table
- **Key effects:** A vs anger (r=-.23), A vs swear (r=-.21), O vs articles (r=.20), N vs anxiety (r=.17), E vs social (r=.15)
- **Data:** Open access on PMC
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2885844/
- **Use:** Direction validation — verify our output moves right way per correlation signs

### Koutsoumpis et al. (2022) — "Kernel of Truth in Text-Based Personality Assessment"
- **What:** Meta-analysis, 31 samples, N=85,724. 52 LIWC × 5 traits
- **Key finding:** Strongest effects |rho| = .08-.14 for self-report. This is our magnitude ceiling.
- **Published:** Psychological Bulletin, Vol. 148(11-12), 843-868
- **Use:** Magnitude calibration — our effects should not dramatically exceed these

### Tackman et al. (2020) — "Personality in Its Natural Habitat Revisited"
- **What:** N=462, real-world behavioral observation via EAR (not just text)
- **Published:** European Journal of Personality, Vol. 34(5), 753-776
- **DOI:** 10.1002/per.2283
- **Use:** Behavioral (not just linguistic) validation

### Pennebaker & King (1999) — "Linguistic Styles"
- **What:** N=2,479, 1.9M words. Foundational LIWC-personality links
- **Published:** Journal of Personality and Social Psychology, Vol. 77(6)
- **Use:** Historical baseline, widely cited direction expectations

## Tier 2: Computational Precedent

### PERSONAGE (Mairesse & Walker 2007/2008)
- **What:** Maps Big Five to 67 generation parameters (verbosity, hedging, self-reference, etc.)
- **Why it matters:** Closest direct precedent to persona-engine. Same architecture (Big Five → parameters → text)
- **Key finding:** Both direct parameter setting and statistical selection "reliably generate utterances humans perceive as manifesting Big Five"
- **Links:** ACL 2007 + ACL 2008, Semantic Scholar
- **Use:** Compare our parameter mappings against their 67 parameters

### Google DeepMind Psychometric Framework (Serapio-Garcia et al. 2025)
- **What:** Administers IPIP-NEO (300 items) and BFI to 18 LLMs. Open-source code + data.
- **Published:** Nature Machine Intelligence
- **Code:** https://github.com/google-deepmind/personality_in_llms
- **Data:** https://storage.googleapis.com/personality_in_llms/index.html
- **Use:** Adapt their validation protocol for persona-engine

### PersonaLLM (Jiang et al. 2024)
- **What:** 320 personas, validated via 44-item BFI, effect sizes d=1.56-7.81
- **Key:** LIWC analysis showed 17/36 overlapping correlations with human data on Openness
- **arXiv:** 2305.02547
- **Use:** Replicate their protocol against persona-engine outputs

### Big5-Chat (2024)
- **What:** 100K dialogues grounded in real human personality expression (846K Facebook posts)
- **arXiv:** 2410.16491, published at ACL 2025
- **Use:** Ground truth dialogue corpus for personality expression

## Tier 3: Raw Data

### Kaggle Big Five Dataset
- **What:** 1,015,342 questionnaire responses (50 IPIP items)
- **Link:** https://www.kaggle.com/datasets/tunguz/big-five-personality-test
- **Use:** Population norms, percentile calibration

### SPADE Dataset
- **What:** Argumentative speech samples + continuous Big Five scores + 436 features
- **Link:** https://aclanthology.org/2022.lrec-1.688/
- **Use:** Speech-based personality validation

## Benchmark Profiles for Testing

Standard test inputs (0-1 scale). No canonical benchmark exists in literature;
these are derived from trait definitions and inter-trait correlations.

| Profile | O | C | E | A | N | Expected dominant behaviors |
|---------|---|---|---|---|---|---------------------------|
| Prototypical Extravert | .5 | .5 | .9 | .5 | .3 | High disclosure, enthusiasm, proactivity, social words |
| Prototypical Introvert | .6 | .5 | .1 | .5 | .5 | Low disclosure, brief, reactive, fewer social words |
| Prototypical Agreeable | .5 | .5 | .6 | .9 | .3 | High hedging, validation, low directness, positive emotion |
| Prototypical Antagonist | .4 | .5 | .5 | .1 | .5 | High directness, low hedging, anger words, blunt |
| Prototypical Neurotic | .5 | .4 | .3 | .5 | .9 | Low confidence, high hedging, anxiety words, negative tone |
| Emotionally Stable | .5 | .6 | .6 | .6 | .1 | High confidence, low hedging, calm tone, steady |
| Creative Intellectual | .9 | .4 | .5 | .5 | .5 | High elasticity, abstract reasoning, exploratory |
| Disciplined Achiever | .3 | .9 | .5 | .5 | .3 | High verbosity (detailed), planning language, low elasticity |

## Validation Strategy

1. **Direction check** — run 8 benchmark profiles through `engine.plan()`, verify each IR parameter moves the correct direction per Yarkoni Table 1
2. **Magnitude calibration** — effects should be bounded by meta-analytic range (|r|=.08-.14). If moderate trait differences produce wildly different output, we're over-expressing
3. **PERSONAGE comparison** — compare our parameter mappings against their 67 parameters for the same trait extremes
4. **Profile discrimination** — opposite profiles (extravert vs introvert, agreeable vs antagonist) should produce distinguishable output on the expected dimensions

## Key Warning

Literature consistently shows personality explains ~5% of behavioral variance.
These are SMALL effects. The risk is not wrong direction — our mappings are
well-grounded — but over-expression compared to real human variation.
