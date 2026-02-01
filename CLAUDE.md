# Kernelbot Development Guide

## Linting

Always run ruff before committing:

```bash
uv run ruff check . --exclude examples/ --line-length 120 --fix
```

To check without auto-fixing:

```bash
uv run ruff check . --exclude examples/ --line-length 120
```

## Testing

### Unit Tests

Run tests with pytest:

```bash
uv run pytest tests/ -v
```

Tests require a local PostgreSQL database. The test suite uses docker-compose to spin up a test database:

```bash
docker compose -f docker-compose.test.yml up -d
uv run pytest tests/ -v
docker compose -f docker-compose.test.yml down -v
```

### Test Requirements

When adding new functionality:

1. **Database methods** (`src/libkernelbot/leaderboard_db.py`):
   - Add tests to `tests/test_leaderboard_db.py`
   - Test empty results, basic functionality, edge cases, and pagination

2. **API endpoints** (`src/kernelbot/api/main.py`):
   - Add tests to `tests/test_admin_api.py` or create endpoint-specific test files
   - Test authentication, authorization (403), not found (404), and validation (400)

3. **Regression tests**: Use the popcorn-cli against a local instance to verify end-to-end functionality

### Local Instance for E2E Testing

Start a local development server:

```bash
# Start database and run migrations
docker compose up -d

# Run the API server
uv run uvicorn src.kernelbot.api.main:app --reload --port 8000
```

Then test with popcorn-cli:

```bash
export POPCORN_API_URL=http://localhost:8000
popcorn submissions list --leaderboard test-leaderboard
```

## Before Adding New Features

**Important:** Check for existing code before implementing:

1. **Database methods**: `src/libkernelbot/leaderboard_db.py`
2. **Discord commands**: `src/kernelbot/cogs/`
3. **API endpoints**: `src/kernelbot/api/main.py`

Reuse existing patterns:
- `validate_user_header` / `validate_cli_header` for authentication
- `get_submission_by_id()`, `delete_submission()` for submission operations
- `simple_rate_limit()` for rate limiting

## Local Development

See `SKILLS/test_bot.md` for local testing setup instructions.

## Architecture

### Problem Configuration

Problems are defined in the [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) repository. See that repo for examples of problem structure and `task.yml` format.

### Leaderboard Creation

- **Dev leaderboards** (via API): Created from a single problem directory. GPUs must be specified in the problem's `task.yml`. The leaderboard name is auto-derived as `{directory}-dev`.

- **Competition leaderboards** (via Discord admin_cog): Created from a competition YAML file that references multiple problems with their deadlines and GPU configurations.
