# Model Competition E2E Testing

How to test the model competition (vLLM fork benchmarking) end-to-end: API submission through Modal execution to DB result storage.

## Prerequisites

- PostgreSQL running locally with `kernelbot` database and migrations applied (see `test_bot.md`)
- Modal profile set to the workspace where `discord-bot-runner` is deployed
- Modal app deployed with model benchmark functions

## Modal Setup

### Check Active Profile

```bash
uv run modal profile list
```

The active profile must match the workspace where `discord-bot-runner` is deployed.

**Important:** If `.env` contains `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, those override the profile config. Make sure they point to the correct workspace, or override them when starting the API server.

### Deploy Modal Functions

```bash
uv run modal deploy src/runners/modal_runner_archs.py
```

This creates `run_model_benchmark_h100` and `run_model_benchmark_b200` functions in the `discord-bot-runner` app.

### Verify Deployment

```bash
uv run python -c "
import modal
fn = modal.Function.from_name('discord-bot-runner', 'run_model_benchmark_h100')
print('Function lookup succeeded')
"
```

## Running the E2E Test

### 1. Start API Server

```bash
# From repo root. Override Modal tokens if .env has wrong workspace tokens.
DATABASE_URL="postgresql://$(whoami)@localhost:5432/kernelbot" \
ADMIN_TOKEN="your_token" \
GITHUB_TOKEN="placeholder" \
GITHUB_REPO="owner/kernelbot" \
DISABLE_SSL=true \
PROBLEM_DEV_DIR="examples" \
MODAL_TOKEN_ID="<correct_workspace_token_id>" \
MODAL_TOKEN_SECRET="<correct_workspace_token_secret>" \
uv run python src/kernelbot/main.py --api-only
```

### 2. Create Test User (if not exists)

```bash
psql "postgresql://$(whoami)@localhost:5432/kernelbot" -c "
INSERT INTO leaderboard.user_info (id, user_name, cli_id, cli_valid)
VALUES ('999999', 'testuser', 'test-cli-id-123', true)
ON CONFLICT (id) DO UPDATE SET cli_id = 'test-cli-id-123', cli_valid = true;
"
```

### 3. Create Dev Leaderboard

```bash
curl -X POST "http://localhost:8000/admin/leaderboards" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"directory": "llama_8b_serving"}'
# Returns: {"status": "ok", "leaderboard": "llama_8b_serving-dev"}
```

### 4. Create Test Archive

```bash
python3 -c "
import io, tarfile
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tar:
    for d in ['vllm-fork', 'vllm-fork/vllm']:
        info = tarfile.TarInfo(name=d)
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    content = b'# Baseline - no modifications\n'
    info = tarfile.TarInfo(name='vllm-fork/vllm/_baseline_marker.py')
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))
with open('/tmp/test_submission.tar.gz', 'wb') as f:
    f.write(buf.getvalue())
print('Created /tmp/test_submission.tar.gz')
"
```

**Important:** Use `vllm-fork/vllm/` structure, not bare `vllm/`. A bare `vllm/` directory would overlay vLLM's own package files and break imports.

### 5. Submit via curl (async endpoint)

```bash
curl -X POST "http://localhost:8000/submission/llama_8b_serving-dev/H100/leaderboard" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123" \
  -F "file=@/tmp/test_submission.tar.gz"
# Returns: {"details": {"id": <sub_id>, "job_status_id": <job_id>}, "status": "accepted"}
```

Modes: `test` (perplexity only), `benchmark` (perplexity + benchmark), `leaderboard` (full scoring).

### 5b. Submit via popcorn-cli (streaming endpoint)

```bash
# Backup your config and set test CLI ID
cp ~/.popcorn.yaml ~/.popcorn.yaml.bak
echo "cli_id: test-cli-id-123" > ~/.popcorn.yaml

# Build popcorn-cli (from popcorn-cli/ dir)
cargo build --release

# Submit (--no-tui for non-interactive terminals)
POPCORN_API_URL=http://127.0.0.1:8000 \
  ./target/release/popcorn-cli submit /tmp/test_submission.tar.gz \
  --gpu H100 --leaderboard llama_8b_serving-dev --mode leaderboard --no-tui

# Restore your config
cp ~/.popcorn.yaml.bak ~/.popcorn.yaml && rm ~/.popcorn.yaml.bak
```

The CLI uses the streaming SSE endpoint (`POST /{leaderboard}/{gpu}/{mode}`) and prints status updates every 15s followed by the full result.

### 6. Poll for Completion (curl only — CLI streams automatically)

The Modal job runs 4 phases (~3-10 min on H100):
1. Install submission archive
2. Start vLLM server
3. Perplexity check (correctness gate)
4. Serving benchmark (1000 prompts)

```bash
# Check server logs for completion
# Or poll the admin endpoint:
curl -s "http://localhost:8000/admin/submissions/<sub_id>" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 7. Verify Results

```bash
# DB: check runs and scores
psql "postgresql://$(whoami)@localhost:5432/kernelbot" -c \
  "SELECT id, submission_id, mode, score, runner, passed FROM leaderboard.runs WHERE submission_id = <sub_id>;"

# API: check user submissions
curl -s "http://localhost:8000/user/submissions?leaderboard=llama_8b_serving-dev" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123"

# API: check leaderboard ranking
curl -s "http://localhost:8000/submissions/llama_8b_serving-dev/H100"
```

Expected DB runs for `leaderboard` mode:
- `test` run: perplexity check (score=null, passed=true)
- `benchmark` run: serving benchmark (score=null, passed=true)
- `leaderboard` run: same as benchmark but with score = `request_throughput` value

## How Correctness Is Defined

Model submissions are validated through a two-phase gate defined in `task.yml`:

### Phase 1: Perplexity Check (Correctness Gate)

```yaml
config:
  perplexity_baseline: 1.80    # expected perplexity of unmodified model
  perplexity_tolerance: 0.02   # max relative deviation (2%)
```

- Runs 10 fixed prompts against the vLLM server's `/v1/completions` endpoint
- Computes `measured_ppl = exp(-total_log_prob / total_tokens)`
- **Pass criteria:** `abs(measured - baseline) / baseline <= tolerance`
- For baseline 1.80 with tolerance 0.02: perplexity must be between 1.764 and 1.836
- If perplexity fails, the submission is rejected and no benchmark runs

### Phase 2: Serving Benchmark (Ranking)

```yaml
config:
  ranking_metric: "request_throughput"   # metric used for leaderboard ranking
  benchmark_shapes:
    - {num_prompts: 1000, input_len: 512, output_len: 128}
```

- Uses `vllm bench serve` with `--backend openai --endpoint /v1/completions --dataset-name random`
- Extracts metrics: `request_throughput`, `output_throughput`, latency percentiles
- **Pass criteria:** The `ranking_metric` key must exist in the benchmark results
- Score = value of `ranking_metric` (e.g., 42.30 req/s)

### Ranking

```yaml
ranking_by: "custom"       # use ranking_metric, not default benchmark mean
score_ascending: false      # higher request_throughput = better rank
```

The `compute_score()` function in `submission.py` extracts `request_throughput` from the leaderboard run results and stores it as the submission's score.

## Troubleshooting

- **`NotFoundError: Function not found`**: Modal tokens point to wrong workspace. Check `modal profile list` and compare with `.env` tokens.
- **`gpus` keyword argument error**: `task.yml` has `gpus:` field but `LeaderboardTask` doesn't accept it. Fixed by popping `gpus` before `from_dict()` in `task.py`.
- **`UnicodeDecodeError` on admin submission view**: Binary tar.gz archive can't be UTF-8 decoded. Fixed with `errors="replace"` in `leaderboard_db.py`.
- **Overlay breaks vLLM imports**: Test archive has bare `vllm/` dir that overwrites vLLM's package. Use `vllm-fork/vllm/` structure.
- **Benchmark 400 errors**: Using `openai-chat` backend with base model. Must use `--backend openai --endpoint /v1/completions`.
