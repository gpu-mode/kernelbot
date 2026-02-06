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

### Test Requirements

When adding new functionality:

1. **Database methods** (`src/libkernelbot/leaderboard_db.py`):
   - Add tests to `tests/test_leaderboard_db.py`
   - Test empty results, basic functionality, edge cases, and pagination

2. **API endpoints** (`src/kernelbot/api/main.py`):
   - Add tests to `tests/test_admin_api.py` or create endpoint-specific test files
   - Test authentication, authorization (403), not found (404), and validation (400)

3. **Regression tests**: Use the popcorn-cli against a local instance to verify end-to-end functionality

### E2E Regression Testing with popcorn-cli

Full end-to-end testing requires running the API server locally and testing with popcorn-cli. This tests the complete flow from CLI through API to database.

#### Step 1: Start PostgreSQL

```bash
# macOS with Homebrew
brew services start postgresql@14

# Verify it's running
pg_isready
```

#### Step 2: Create Database and Run Migrations

```bash
# Create database (first time only)
createdb kernelbot

# Run migrations
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/kernelbot"
uv run yoyo apply --database "$DATABASE_URL" src/migrations/
```

#### Step 3: Create a Test User

The CLI requires a registered user. Create one in the database:

```bash
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/kernelbot"
psql "$DATABASE_URL" -c "INSERT INTO leaderboard.user_info (id, user_name, cli_id, cli_valid)
VALUES ('999999', 'testuser', 'test-cli-id-123', true)
ON CONFLICT (id) DO UPDATE SET cli_id = 'test-cli-id-123', cli_valid = true;"
```

#### Step 4: Start the API Server

```bash
cd src/kernelbot

# Set required environment variables
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/kernelbot"
export ADMIN_TOKEN="your-admin-token-here"  # Check .env for LOCAL_ADMIN_TOKEN

# Start API server (without Discord bot)
uv run python main.py --api-only

# Server runs on http://localhost:8000
```

#### Step 5: Sync Leaderboards

Leaderboards must be synced from reference-kernels before testing submissions:

```bash
curl -X POST "http://localhost:8000/admin/update-problems" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"problem_set": "pmpp_v2"}'
```

#### Step 6: Configure popcorn-cli for Local Testing

Temporarily update `~/.popcorn.yaml` to use test credentials:

```bash
# Backup existing config
cp ~/.popcorn.yaml ~/.popcorn.yaml.bak

# Set test CLI ID
echo "cli_id: test-cli-id-123" > ~/.popcorn.yaml
```

#### Step 7: Run CLI Commands

```bash
cd /path/to/popcorn-cli

# Set API URL to local server
export POPCORN_API_URL=http://localhost:8000

# Test various commands
cargo run --release -- submissions list --leaderboard vectoradd_v2
cargo run --release -- submissions show <ID>
cargo run --release -- submissions delete <ID>

# Test submission (runs on Modal H100, requires Modal account)
cargo run --release -- submit solution.py --gpu H100 --leaderboard vectoradd_v2 --mode test
```

#### Step 8: Restore Original Config

```bash
cp ~/.popcorn.yaml.bak ~/.popcorn.yaml
rm ~/.popcorn.yaml.bak
```

#### Quick Reference: Testing API Endpoints Directly

```bash
# List user submissions
curl -s "http://localhost:8000/user/submissions?leaderboard=vectoradd_v2" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123"

# Get submission details
curl -s "http://localhost:8000/user/submissions/1" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123"

# Delete submission
curl -s -X DELETE "http://localhost:8000/user/submissions/1" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123"

# Submit a file (multipart form)
curl -s -X POST "http://localhost:8000/vectoradd_v2/H100/test" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123" \
  -F "file=@solution.py"
```

#### Troubleshooting

- **401 Unauthorized**: CLI ID not in database or `cli_valid` is false
- **404 Not Found**: Leaderboards not synced - run update-problems first
- **500 Internal Error**: Check server logs in terminal, often a TypedDict vs object access issue
- **"Device not configured" from CLI**: Usually a TTY issue, try running with explicit env vars

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
