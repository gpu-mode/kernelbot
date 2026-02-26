# Testing E2E Model Competitions

This guide walks through testing the model competition pipeline end-to-end, starting with Modal (easiest) and building up to the full API flow.

## Prerequisites

- Modal account with `modal` CLI authenticated (`modal setup`)
- Hugging Face account with access to gated models (e.g., Llama-3.1-8B)
  - Set `HF_TOKEN` env var or run `huggingface-cli login`
- The `speedrun` branch checked out

## Step 1: Build the Modal Image

The model image installs all vLLM dependencies, then uninstalls vllm itself (the user's fork replaces it at runtime). This takes a while the first time.

```bash
# Dry-run to verify the image definition parses
cd src/runners
modal run modal_runner.py
```

If the image build fails, check the vLLM install step — it pulls many transitive deps and can be sensitive to CUDA/PyTorch version mismatches.

## Step 2: Pre-download Model Weights

Model weights are stored in a persistent Modal volume so they don't need to be re-downloaded for every submission.

```bash
# Download Llama-3.1-8B (~14GB, takes a few minutes)
modal run src/runners/download_model.py --model meta-llama/Llama-3.1-8B
```

Verify the volume has the weights:

```bash
modal volume ls model-weights
# Should show: models--meta-llama--Llama-3.1-8B/
```

## Step 3: Test the Runner Directly on Modal

Create a test script that calls `run_model_benchmark` directly inside a Modal container, bypassing the API and launcher layers entirely. This validates the core pipeline: install → server start → perplexity check → benchmark → cleanup.

Create `src/runners/test_model_benchmark.py`:

```python
"""
Smoke test for model benchmark runner on Modal.

Usage:
    modal run src/runners/test_model_benchmark.py

This creates a stock vllm tarball, installs it, starts a server,
runs a small benchmark, and checks perplexity.
"""
import base64
import io
import json
import tarfile

import modal

app = modal.App("test-model-benchmark")

from modal_runner import model_image, model_weights, sccache_vol


@app.function(
    gpu="H100",
    image=model_image,
    volumes={"/models": model_weights, "/sccache": sccache_vol},
    timeout=3600,
)
def test_benchmark():
    from libkernelbot.run_eval import run_config

    # Create a minimal tarball that just installs stock vllm
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        setup_py = (
            b"from setuptools import setup\n"
            b"setup(name='vllm-test', version='0.1', install_requires=['vllm'])\n"
        )
        info = tarfile.TarInfo(name="vllm-test/setup.py")
        info.size = len(setup_py)
        tar.addfile(info, io.BytesIO(setup_py))

    archive_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    config = {
        "lang": "model",
        "mode": "leaderboard",
        "submission_archive": archive_b64,
        "model_config": {
            "model_name": "meta-llama/Llama-3.1-8B",
            "tensor_parallel": 1,
            "benchmark_shapes": [
                {"num_prompts": 10, "input_len": 128, "output_len": 32},
            ],
            "ranking_metric": "request_throughput",
            "perplexity_baseline": 6.14,
            "perplexity_tolerance": 0.05,  # 5% tolerance for smoke test
            "install_timeout": 600,
            "server_startup_timeout": 300,
            "benchmark_timeout": 300,
        },
    }

    result = run_config(config)

    # Print results
    print(f"\n{'='*60}")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"System: {result.system}")
    print(f"Runs: {list(result.runs.keys())}")

    for name, eval_result in result.runs.items():
        print(f"\n--- {name} ---")
        print(f"  success: {eval_result.run.success}")
        print(f"  passed: {eval_result.run.passed}")
        print(f"  duration: {eval_result.run.duration:.1f}s")
        if eval_result.run.result:
            for k, v in eval_result.run.result.items():
                print(f"  {k}: {v}")

    return result


@app.local_entrypoint()
def main():
    result = test_benchmark.remote()
    if not result.success:
        print(f"\nFAILED: {result.error}")
        raise SystemExit(1)
    print("\nPASSED")
```

Run it:

```bash
cd src/runners
modal run test_model_benchmark.py
```

### What to look for

- **Phase 1 (Install)**: `pip install` should complete within the timeout. If it fails, check that the base image has compatible PyTorch/CUDA versions.
- **Phase 2 (Server)**: vLLM server should start and the `/health` endpoint should respond. If it times out, check GPU memory — the model might not fit.
- **Phase 3 (Perplexity)**: Perplexity should be within tolerance of the baseline. If it fails, the baseline value in the task config may need recalibrating.
- **Phase 4 (Benchmark)**: `benchmark_serving.py` should run and produce metrics like `request_throughput`, `mean_ttft_ms`, etc.

### Test mode only (skip benchmark)

To test just the install + server + perplexity phases without the full benchmark:

```python
config["mode"] = "test"  # Only runs perplexity check, skips benchmark
```

## Step 4: Deploy the Full Runner

Once the smoke test passes, deploy the runner so the API can call it:

```bash
cd src/runners
modal deploy modal_runner.py
```

This registers `run_model_benchmark_h100` and `run_model_benchmark_b200` as callable Modal functions.

## Step 5: Test the Full API Flow

### Start the local API server

```bash
# Start postgres
brew services start postgresql@14  # macOS

# Create DB and run migrations
createdb kernelbot
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/kernelbot"
uv run yoyo apply --database "$DATABASE_URL" src/migrations/

# Create test user
psql "$DATABASE_URL" -c "
INSERT INTO leaderboard.user_info (id, user_name, cli_id, cli_valid)
VALUES ('999999', 'testuser', 'test-cli-id-123', true)
ON CONFLICT (id) DO UPDATE SET cli_id = 'test-cli-id-123', cli_valid = true;
"

# Start API (without Discord bot)
export ADMIN_TOKEN="test-token"
cd src/kernelbot
uv run python main.py --api-only
```

### Create a model leaderboard

The leaderboard needs to be created from a task directory. Use the example:

```bash
# Option 1: Via admin API
curl -X POST "http://localhost:8000/admin/create-leaderboard" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"directory": "examples/llama_8b_serving", "gpus": ["H100"]}'

# Option 2: Via problem sync (if using reference-kernels repo structure)
curl -X POST "http://localhost:8000/admin/update-problems" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"problem_set": "model_competitions"}'
```

### Submit a vLLM fork tarball

```bash
# Create a tarball from a vLLM fork directory
cd /path/to/your/vllm-fork
tar czf /tmp/vllm-fork.tar.gz .

# Submit via curl
curl -X POST "http://localhost:8000/llama_8b_serving-dev/H100/test" \
  -H "X-Popcorn-Cli-Id: test-cli-id-123" \
  -F "file=@/tmp/vllm-fork.tar.gz"

# Or submit via popcorn-cli
export POPCORN_API_URL=http://localhost:8000
cargo run --release -- submit /tmp/vllm-fork.tar.gz \
  --gpu H100 --leaderboard llama_8b_serving-dev --mode test
```

### What to verify in the full flow

1. **Upload accepted**: Server responds with a submission ID (not a 400/413 error)
2. **Binary storage**: The tarball is stored as bytes in `code_files`, not UTF-8 decoded
3. **Modal dispatch**: The launcher calls `run_model_benchmark_h100` on Modal
4. **Results returned**: SSE stream shows progress and final metrics
5. **Score computed**: For `mode=leaderboard`, the `request_throughput` metric is used as the score
6. **Leaderboard ranking**: Score is ranked descending (higher throughput = better)

## Step 6: Calibrate the Perplexity Baseline

The `perplexity_baseline` value in `task.yml` needs to match stock vLLM on the target hardware. To calibrate:

1. Run the smoke test (Step 3) with stock vLLM and a generous tolerance (e.g., `0.10`)
2. Note the computed perplexity from the results
3. Update `examples/llama_8b_serving/task.yml` with the measured value
4. Set tolerance to `0.01` (1%) for production

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `pip install` timeout | Large fork with CUDA extensions; increase `install_timeout` or pre-compile |
| Server never becomes healthy | Model too large for GPU memory; check `tensor_parallel` setting |
| Perplexity way off baseline | Wrong model revision or quantization applied; check vLLM server args |
| `benchmark_serving.py` not found | vLLM version doesn't include benchmarks; ensure fork is based on recent vLLM |
| 413 Request Entity Too Large | Tarball exceeds 50MB limit; strip unnecessary files from the fork |
| Modal function not found | Runner not deployed; run `modal deploy src/runners/modal_runner.py` |
| Score not appearing on leaderboard | Mode was `test` not `leaderboard`; resubmit with `--mode leaderboard` |
