# Session Handoff — Cherry-Pick to analyze-test-coverage

> Written 2026-03-20. Read this BEFORE doing anything.

## Situation

Two divergent branches exist for persona-engine:

1. **`claude/analyze-test-coverage-d93F4`** — The advanced branch. 50+ commits with mixin refactoring, Layer Zero, multi-provider adapters, wired interpreter methods, emotional appraisal, LIWC markers, FastAPI server, CI/CD. This is the correct base.

2. **`review/tier1-bugfixes`** — Our session's branch. 13 commits of bug fixes, hardening, and research work. Branched from the WRONG base (`claude/explore-repo-KZTBj`).

**Goal**: Cherry-pick the UNIQUE work from `review/tier1-bugfixes` onto `claude/analyze-test-coverage-d93F4`. One commit at a time. Test after each.

## What's on review/tier1-bugfixes (13 commits)

```
3102005 Add behavioral interpreter wiring audit (6-part)
a593908 Add TRACKER.md
c92dad5 Quick fixes: SMALL_TALK, input validation, thread safety docs, remove dead response/
f33dbc4 Apply remaining calibration: confidence, hedging, proactivity, disclosure, enthusiasm
5b3e608 Add automated persona evaluation suite (5 statistical suites)
de21860 Calibrate behavioral parameters against personality psychology literature
03e3091 Rewrite stance generator: compositional, topic-grounded, value-driven
8cf9f6c Rewrite README, clean repo root, switch examples to template adapter
8177396 Fix save/load round-trip: serialize state, stance cache, and snapshot
9809104 Add structured logging and LLM exception handling
ad0d702 Fix previously-failing test + american casing bug
80f5ae5 Fix 8 bugs from multi-perspective review + packaging cleanup
```

## Cherry-Pick Decision Matrix

For each commit, assess: does the target branch already have this? Will it conflict?

### SKIP — Already done on target branch
- `80f5ae5` Bug fixes — PARTIALLY. Target branch has its own bug fixes (`cc080b5`). **Read both and diff.** Some fixes may be unique (mood drift inversion, MemoryError rename, state evolution timing). Cherry-pick ONLY the fixes not already present.
- `ad0d702` Test failure fix + american casing — CHECK if target branch has same test failure.
- `c92dad5` SMALL_TALK, remove response/ — Target branch already removed response/ in Phase A (`8340835`). SMALL_TALK might still be needed — check.
- `8cf9f6c` README rewrite + repo cleanup — Target branch already rewrote README (`0794294`). Skip the README. The repo cleanup (moving docs to docs/internal/) might still be useful.
- `3102005` Wiring audit — The audit found 28 dead methods. Target branch wired them in Phase R1 (`c4a30e1`). The audit doc itself is still valuable as documentation. Cherry-pick the doc only, update findings to note they've been addressed.

### LIKELY CHERRY-PICK — Unique work
- `9809104` Structured logging + LLM exception handling — CHECK if target branch has logging (`e946f88` "structured logging" is on target). May be duplicate. Read both.
- `8177396` Save/load round-trip fix — LIKELY unique. Target branch may not have v3 save format with state/stance/snapshot serialization. Verify.
- `03e3091` Stance generator rewrite — LIKELY conflicts. Target branch has its own stance work (Phase R5: "composable stance"). **This needs careful manual merge, not blind cherry-pick.**
- `de21860` + `f33dbc4` Calibration (2 commits) — LIKELY unique. Psych literature calibration with specific parameter changes + research report. Target branch may have different values from its own calibration. **Read target's trait_interpreter.py first.**
- `5b3e608` Eval suite — LIKELY unique. Statistical evaluation with 5 suites. This is new infrastructure. Check for dependency conflicts (file paths, imports).
- `a593908` TRACKER.md — Skip, rewrite for the new branch state.

### Process for each cherry-pick

1. Read the target branch's version of the affected files FIRST
2. Understand what's different
3. If clean cherry-pick possible: `git cherry-pick <hash>`, run tests
4. If conflicts: manually apply changes, understanding both sides
5. Run full test suite after EVERY commit
6. If tests fail, fix before moving to next

## Key Files That Will Conflict

These files are substantially different between branches:

- `persona_engine/planner/turn_planner.py` — Target has mixin refactoring (split into stages/)
- `persona_engine/planner/stance_generator.py` — Target has Phase R5 composable stance
- `persona_engine/behavioral/trait_interpreter.py` — Target may have different calibration
- `persona_engine/behavioral/state_manager.py` — Target may have different state logic
- `persona_engine/__init__.py` — Target is v0.4.0, ours is v0.2.0
- `README.md` — Both rewrote it differently
- `pyproject.toml` — Different versions, possibly different deps
- `persona_engine/response/` — Target already removed it

## Files That Should Cherry-Pick Cleanly

- `eval/persona_eval.py` — New file, doesn't exist on target
- `eval/__init__.py` — New file
- `research/calibration_report.txt` — New file
- `docs/BEHAVIORAL_WIRING_AUDIT.md` — New file (update content to reflect target branch state)

## Recommended Order

1. **Eval suite** (`5b3e608`) — new files, cleanest pick
2. **Calibration research report** (from `de21860`) — new file only
3. **Save/load fix** (`8177396`) — check target's engine.py first
4. **Calibration code changes** (`de21860` + `f33dbc4`) — read target's trait_interpreter first
5. **Stance generator** (`03e3091`) — most complex, do last, likely manual merge
6. **Bug fixes** (`80f5ae5`) — individual fixes, apply only what's missing
7. **Remaining quick fixes** — apply individually as needed

## Test Command

The target branch may have different test configuration. Check pyproject.toml first. If it has pytest-cov issues, use:
```bash
python3 -m pytest tests/ -p no:cacheprovider -p no:cov -p no:capture --override-ini="addopts=" --ignore=tests/test_property_based.py -v --tb=short
```

## PR

PR #1 is open against `claude/explore-repo-KZTBj`. It should be closed or retargeted once cherry-picks are done. The new PR should target `claude/analyze-test-coverage-d93F4` or `main`.

## Critical Warning

DO NOT blindly `git cherry-pick`. The codebases have DIVERGED significantly. The target branch has a different architecture (mixin-based stages vs monolithic TurnPlanner). Read the target branch's code first, understand it, then apply changes surgically.
