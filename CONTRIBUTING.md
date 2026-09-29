# Contributing

Thank you for improving Enterprise Support Agent.

## Local setup

```bash
cp .env.example .env
docker compose up -d db redis
uv sync --dev
uv run alembic upgrade head
```

Use only synthetic or redistributable documents in tests and examples. Never commit `.env`, credentials, cookies, customer documents, local databases, Chroma data, or production evaluation results.

## Before opening a pull request

```bash
uv run pytest -q
uv run ruff check src main.py scripts eval
uv run mypy src main.py
docker compose --env-file .env.example config --quiet
bash scripts/audit_dependencies.sh
```

Keep migrations backward-aware, add tests for behavior changes, and describe any configuration or security impact in the pull request. By contributing, you agree that your contribution is distributed under the repository's MIT License.
