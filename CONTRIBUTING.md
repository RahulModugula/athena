# Contributing to athena-verify

Thank you for your interest in contributing! This project is focused on one thing: **runtime verification of RAG answers against retrieved context**.

## Quick Start

```bash
git clone https://github.com/RahulModugula/athena.git
cd athena
pip install -e ".[dev,nli]"
pytest
```

## Development Setup

```bash
# Install with all dev dependencies
pip install -e ".[dev,nli]"

# Run tests
pytest

# Run linter
ruff check .

# Type check
mypy athena_verify/
```

## What We're Looking For

- **Benchmark results** — Run our benchmarks on new datasets and submit results
- **New integrations** — Haystack, Semantic Kernel, Llama.cpp, etc.
- **NLI model improvements** — Better models, faster inference, multilingual support
- **Bug fixes** — Especially in sentence splitting and overlap computation
- **Documentation** — Examples, tutorials, API clarifications

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for any new functionality
5. Run `pytest` and `ruff check .` to ensure everything passes
6. Submit a PR with a clear description

## Code Style

- Python 3.11+ with type hints
- `ruff` for linting (line length: 100)
- `mypy --strict` for type checking
- Google-style docstrings

## Project Structure

```
athena_verify/          # The library
├── core.py             # verify() and verify_async()
├── models.py           # Data models
├── nli.py              # NLI entailment scoring
├── overlap.py          # Lexical overlap computation
├── calibration.py      # Trust score calibration
├── llm_judge.py        # Optional LLM-as-judge
├── parser.py           # Sentence splitting
└── integrations/       # Framework integrations
tests/                  # Test suite
benchmarks/             # Benchmark runners and results
examples/               # Usage examples
```

## Reporting Issues

- **Bug reports**: Include Python version, OS, and a minimal reproduction
- **Feature requests**: Explain the use case and how it fits the verification layer scope
- **Benchmark issues**: Include the dataset, command, and output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
