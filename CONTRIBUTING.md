# Contributing to VersatIL

Thanks for your interest in contributing! Bug reports, fixes, new components,
and documentation improvements are all welcome.

## Development Setup

Install VersatIL from source following the
[installation guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/installation/)
(Option B or C). `uv sync` installs the `dev` dependency group (pytest,
pytest-cov, ruff, pre-commit) by default, so the environment is ready for
development out of the box.

Then install the pre-commit hooks, which run Ruff formatting and linting on
every `git commit`:

```bash
pre-commit install
```

## Running Tests

```bash
# Run the default local suite: excludes slow, integration, GPU-only, and ExecuTorch-dependent tests
pytest

# Run all tests including integration tests
pytest -m ""

# Run specific test file
pytest tests/models/test_policy.py

# Run tests by marker
pytest -m "unit"                                      # Fast tests with mocked dependencies
pytest -m "integration"                               # Real component integration tests
pytest -m "requires_gpu"                              # GPU-required tests
pytest -m "not slow and not integration and not requires_gpu"  # Explicit default selection
```

Before writing or modifying tests, read [tests/AGENTS.md](tests/AGENTS.md) for
the mandatory testing guidelines (fixtures, factories, assertion conventions).

## Code Style

- **Docstrings**: Google-style, concise (avoid LLM patterns like numbered lists or excessive words)
- **Type hints**: Required for all function signatures
- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) (line length 88, lint target pinned to `py313` so annotation imports stay at runtime for OmegaConf)
- **No inline imports**: All imports at module top
- **Minimal comments**: Only for tensor shapes or non-obvious logic
- **Variables**: Use English words, avoid abbreviations
- **Function calls**: Use kwargs
- **Error handling**: Use `raise`, avoid assertions and try/catch blocks
- **Strings**: Use double quotes (`"foo"` not `'foo'`)
- **Constants**: Avoid hardcoded strings, use `Enum.MY_ENUM.value`
- **No wildcard imports**: Avoid `from module import *`
- Avoid `**kwargs` and `*args` signatures: Explicit is better than implicit

```bash
# Format code
ruff format src/ tests/

# Check formatting
ruff format --check src/ tests/

# Lint
ruff check src/ tests/

# Lint and auto-fix
ruff check --fix src/ tests/
```

The full working agreements (also used by AI coding agents) are in
[AGENTS.md](AGENTS.md).

## Pull Requests

- Keep PRs focused on a single change.
- Run the default test suite and Ruff before opening a PR.
- Every PR must pass the CI pipeline and receive at least one approval from a
  developer before it can be merged.
- New components need config dataclasses, ConfigStore registration, tests, and
  documentation updates (see the
  [architecture docs](https://lorenzo-mazza.github.io/VersatIL/architecture/overview/)
  for the component structure).
