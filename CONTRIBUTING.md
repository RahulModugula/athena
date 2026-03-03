# Contributing to Athena

Thanks for your interest. Contributions are welcome — bug fixes, new features, documentation improvements, and evaluation improvements all accepted.

---

## Development environment

### Requirements

- Python 3.12
- Docker and Docker Compose (for PostgreSQL)
- `uv` for dependency management

### Setup

```bash
git clone https://github.com/yourusername/athena.git
cd athena/backend

# Create a virtual environment and install all dependencies including dev extras
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy env template and fill in your ZhipuAI API key
cp .env.example .env
```

Start a local PostgreSQL with pgvector:

```bash
docker compose up postgres -d
```

Run database migrations:

```bash
alembic upgrade head
```

Start the development server with auto-reload:

```bash
uvicorn app.main:app --reload
```

---

## Running tests

```bash
# From backend/
pytest tests/ -v
```

Tests require a running PostgreSQL instance. The connection URL is read from the `ATHENA_DATABASE_URL` environment variable. The default in `.env.example` points to the Docker Compose service.

To run a specific test file:

```bash
pytest tests/test_retrieval.py -v
```

---

## Code style

The project uses **Ruff** for linting and formatting and **mypy** for static type checking.

```bash
# Lint and auto-fix
ruff check app/ tests/ --fix

# Format
ruff format app/ tests/

# Type check
mypy app/
```

All three checks run in CI and must pass before a pull request can be merged. Configure your editor to run Ruff on save if possible — this avoids CI surprises.

A few conventions to follow:

- All new functions and methods must have type annotations.
- Public functions must have a one-line docstring.
- Avoid mutable default arguments.
- Keep functions short and focused; prefer composition over long procedural blocks.

---

## Pull request process

1. Fork the repository and create a branch from `main`. Use a short, descriptive name: `fix/bm25-score-normalisation`, `feat/pdf-table-extraction`, `docs/chunking-strategies`.

2. Make your changes with tests. For bug fixes, add a test that reproduces the bug. For new features, add tests covering the happy path and at least one error case.

3. Run the full check suite locally before pushing:

   ```bash
   ruff check app/ tests/
   ruff format --check app/ tests/
   mypy app/
   pytest tests/ -v
   ```

4. Open a pull request against `main`. Fill in the description with:
   - What the change does and why
   - How to test it manually if relevant
   - Any design decisions worth discussing

5. A review pass will follow. Address comments and push updates to the same branch — do not open a new PR.

6. Once approved and CI is green, the PR will be merged with a merge commit.

---

## Reporting issues

Open a GitHub issue with a clear title and the following information:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Relevant log output or tracebacks
- Python version, OS, and Docker version if applicable
