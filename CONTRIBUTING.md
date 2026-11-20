# Contributing

This is an early prototype. Bug reports, tests, and focused pull requests are welcome.

## Setup

```bash
git clone https://github.com/SatishoBananamoto/persona-engine.git
cd persona-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

Run a single module with `pytest tests/test_engine_sdk.py`. Use `llm_provider="mock"` or `llm_provider="template"` in tests so they never call a live API.

Please include tests for new behaviour. Keep changes focused, and do not add internal session notes, personal details, or coverage badges to the public tree.
