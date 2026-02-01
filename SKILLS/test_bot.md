# Local Testing Guide for Kernelbot + Popcorn CLI

This document describes how to set up and test the kernelbot API and popcorn-cli admin flow locally.

## Prerequisites

### 1. PostgreSQL Setup

Install and start PostgreSQL:

```bash
# Install PostgreSQL 14 via Homebrew
brew install postgresql@14

# Start the service
brew services start postgresql@14

# If brew services fails, start directly:
/opt/homebrew/opt/postgresql@14/bin/pg_ctl -D /opt/homebrew/var/postgresql@14 start
```

Create the kernelbot database:

```bash
# Replace YOUR_USERNAME with your system username
psql -U YOUR_USERNAME -d postgres -c "CREATE DATABASE kernelbot;"
```

### 2. Environment Variables

Create a `.env` file in the kernelbot root directory:

```bash
# Required for API startup
GITHUB_TOKEN=placeholder_github_token
GITHUB_REPO=owner/kernelbot

# Local PostgreSQL database (replace YOUR_USERNAME with your system username)
DATABASE_URL=postgresql://YOUR_USERNAME@localhost:5432/kernelbot
DISABLE_SSL=true

# Admin token for local testing
ADMIN_TOKEN=your_secure_token_here

# Problem directory (absolute path to examples folder)
PROBLEM_DEV_DIR=/path/to/kernelbot/examples
```

### 3. Database Migrations

Run yoyo migrations to set up the schema:

```bash
cd /path/to/kernelbot
# Replace YOUR_USERNAME with your system username
uv run yoyo apply --database "postgresql://YOUR_USERNAME@localhost:5432/kernelbot" src/migrations/ --batch
```

If migrations fail due to partial application, mark them as applied:

```bash
# Replace YOUR_USERNAME with your system username
uv run yoyo mark --all --database "postgresql://YOUR_USERNAME@localhost:5432/kernelbot" src/migrations/ --batch
```

## Running Tests

### Unit Tests (pytest)

Run the admin API tests:

```bash
cd /path/to/kernelbot
uv run pytest tests/test_admin_api.py -v
```

Run all tests:

```bash
uv run pytest tests/ -v
```

### Integration Tests with Docker

The test suite uses docker-compose for database setup:

```bash
uv run pytest tests/ -v
```

This automatically starts a test database container.

## Starting the API

### API-Only Mode (No Discord)

Start the API server without requiring Discord credentials:

```bash
cd /path/to/kernelbot/src/kernelbot
uv run python main.py --api-only
```

The API will be available at `http://localhost:8000`.

### Verify API is Running

```bash
curl http://localhost:8000/leaderboards
# Should return: []
```

## Testing Admin Commands

### Using curl

```bash
# Set your admin token
TOKEN="your_secure_token_here"

# Start accepting jobs
curl -X POST http://localhost:8000/admin/start -H "Authorization: Bearer $TOKEN"

# Get stats
curl http://localhost:8000/admin/stats -H "Authorization: Bearer $TOKEN"

# Stop accepting jobs
curl -X POST http://localhost:8000/admin/stop -H "Authorization: Bearer $TOKEN"
```

### Using Popcorn CLI

First, build the CLI:

```bash
cd /path/to/popcorn-cli
cargo build
```

Then run admin commands:

```bash
# Set environment variables
export POPCORN_API_URL=http://127.0.0.1:8000
export POPCORN_ADMIN_TOKEN=your_secure_token_here

# IMPORTANT: If you have HTTP proxy set, bypass it for local testing
unset HTTP_PROXY HTTPS_PROXY

# Admin commands
./target/debug/popcorn-cli admin start
./target/debug/popcorn-cli admin stats
./target/debug/popcorn-cli admin stop
./target/debug/popcorn-cli admin get-submission 123
./target/debug/popcorn-cli admin delete-submission 123
./target/debug/popcorn-cli admin create-leaderboard identity_py
./target/debug/popcorn-cli admin delete-leaderboard identity_py-dev

# Update problems from reference-kernels repo (mirrors Discord /admin update-problems)
./target/debug/popcorn-cli admin update-problems --problem-set nvidia
./target/debug/popcorn-cli admin update-problems --problem-set pmpp_v2 --force
./target/debug/popcorn-cli admin update-problems  # Updates all problem sets
```

## End-to-End CLI Testing

Full workflow to test the admin CLI:

```bash
# 1. Start the API server (in kernelbot repo)
cd /path/to/kernelbot/src/kernelbot
ADMIN_TOKEN=test_token PROBLEM_DEV_DIR=/path/to/kernelbot/examples uv run python main.py --api-only

# 2. In another terminal, set up CLI environment
cd /path/to/popcorn-cli
cargo build
unset HTTP_PROXY HTTPS_PROXY
export POPCORN_API_URL=http://127.0.0.1:8000
export POPCORN_ADMIN_TOKEN=test_token

# 3. Test admin commands
./target/debug/popcorn-cli admin start
./target/debug/popcorn-cli admin stats

# 4. Create and delete a leaderboard (name auto-derived as "{directory}-dev")
./target/debug/popcorn-cli admin create-leaderboard identity_py
curl -s http://127.0.0.1:8000/leaderboards | jq '.[].name'
./target/debug/popcorn-cli admin delete-leaderboard identity_py-dev
```

## Troubleshooting

### "Connection refused" errors

1. Check if API is running: `lsof -i :8000`
2. Make sure you're using `127.0.0.1` instead of `localhost` (IPv6 issues)
3. Check for HTTP proxy: `echo $HTTP_PROXY` - if set, unset it for local testing

### "folly::AsyncSocketException" errors

This indicates an HTTP proxy is intercepting requests. Fix:

```bash
unset HTTP_PROXY HTTPS_PROXY
export NO_PROXY=127.0.0.1,localhost
```

### Database connection errors

1. Check PostgreSQL is running: `brew services list | grep postgres`
2. Verify DATABASE_URL in .env matches your setup
3. Check database exists: `psql -U YOUR_USERNAME -d kernelbot -c "SELECT 1;"`

### "DISCORD_TOKEN not found" error

Make sure you're using `--api-only` flag when starting the server.

### Admin token errors

- 401 "Missing Authorization header": Token not being sent
- 401 "Invalid admin token": Token doesn't match ADMIN_TOKEN in .env
- 500 "ADMIN_TOKEN not configured": Set ADMIN_TOKEN in .env

## Architecture Notes

### API-Only Mode

The `--api-only` flag allows running the FastAPI server without Discord:
- Skips Discord token validation
- Creates backend without Discord bot initialization
- All admin endpoints work via HTTP API

### Admin Authentication

Admin endpoints require:
- Header: `Authorization: Bearer <ADMIN_TOKEN>`
- Token is read from ADMIN_TOKEN environment variable

### CLI Admin Commands

The popcorn-cli admin commands use:
- `POPCORN_API_URL`: API endpoint (default: production Heroku)
- `POPCORN_ADMIN_TOKEN`: Bearer token for admin endpoints

## Testing Against Production (Heroku)

### Prerequisites

1. Get the production admin token from Heroku config vars:
   ```bash
   heroku config:get ADMIN_TOKEN -a discord-cluster-manager
   ```

2. Or set it in Heroku if not already configured:
   ```bash
   heroku config:set ADMIN_TOKEN=your_secure_production_token -a discord-cluster-manager
   ```

3. If migrating from LOCAL_ADMIN_TOKEN:
   ```bash
   heroku config:set ADMIN_TOKEN=$(heroku config:get LOCAL_ADMIN_TOKEN -a discord-cluster-manager) -a discord-cluster-manager
   heroku config:unset LOCAL_ADMIN_TOKEN -a discord-cluster-manager
   ```

### Using CLI with Production

```bash
# Build CLI
cd /path/to/popcorn-cli
cargo build --release

# Use production API (default URL)
export POPCORN_ADMIN_TOKEN=<production_token_from_heroku>

# Admin commands will hit production
./target/release/popcorn-cli admin stats
./target/release/popcorn-cli admin start
./target/release/popcorn-cli admin stop
```

### Using curl with Production

```bash
PROD_URL="https://discord-cluster-manager-1f6c4782e60a.herokuapp.com"
PROD_TOKEN="<production_token_from_heroku>"

# Get stats
curl "$PROD_URL/admin/stats" -H "Authorization: Bearer $PROD_TOKEN"

# Start/stop job acceptance
curl -X POST "$PROD_URL/admin/start" -H "Authorization: Bearer $PROD_TOKEN"
curl -X POST "$PROD_URL/admin/stop" -H "Authorization: Bearer $PROD_TOKEN"
```

### Checking Heroku Logs

```bash
# View recent logs
heroku logs --tail -a discord-cluster-manager

# Filter for admin actions
heroku logs -a discord-cluster-manager | grep admin
```

### Production Environment Variables

Required Heroku config vars for full functionality:
- `DISCORD_TOKEN`: Discord bot token
- `GITHUB_TOKEN`: GitHub API token
- `GITHUB_REPO`: GitHub repository (e.g., `org/kernelbot`)
- `DATABASE_URL`: PostgreSQL connection string (auto-set by Heroku Postgres)
- `ADMIN_TOKEN`: Admin API authentication token
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`: Modal credentials for GPU runs

View all config vars:
```bash
heroku config -a discord-cluster-manager
```

### Safety Notes for Production

- Always test changes locally first before deploying to production
- Admin commands affect live users - use `admin stop` carefully
- Check stats before and after operations to verify expected behavior
- Monitor Heroku logs when testing production changes
