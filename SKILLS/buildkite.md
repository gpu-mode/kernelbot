# Buildkite GPU Infrastructure Guide

This document describes how to set up and use the Buildkite infrastructure for GPU job isolation.

## Overview

Buildkite provides a parallel infrastructure for onboarding arbitrary GPU vendors with proper isolation. It runs alongside the existing GitHub Actions system, providing:

- Per-GPU job isolation via `NVIDIA_VISIBLE_DEVICES`
- Resource constraints (CPU, RAM, disk) via Docker cgroups
- Clear, reproducible Docker environment
- Automatic queue management

## Quick Start

1. Create queue in Buildkite UI: Agents → Default cluster → Queues → New Queue (select "Self hosted")
2. Run setup script on your GPU node:
   ```bash
   sudo BUILDKITE_AGENT_TOKEN=<token> GPU_TYPE=<queue-name> ./deployment/buildkite/setup-node-simple.sh
   ```
3. Test with `pipeline-test-docker.yml`

## Current Status

**Working**: Full GPU isolation with auto-resource detection. Tested on 2x NVIDIA L40S node with:
- Each agent gets 1 GPU, 8 CPUs, 144GB RAM (auto-calculated from 16 CPUs / 2 GPUs, 289GB / 2 GPUs)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VENDOR 8-GPU NODE                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐     ┌─────────────┐           │
│  │ Agent GPU-0 │ │ Agent GPU-1 │ ... │ Agent GPU-7 │           │
│  │ NVIDIA_VIS  │ │ NVIDIA_VIS  │     │ NVIDIA_VIS  │           │
│  │ IBLE_DEV=0  │ │ IBLE_DEV=1  │     │ IBLE_DEV=7  │           │
│  └──────┬──────┘ └──────┬──────┘     └──────┬──────┘           │
│         └───────────────┴───────────────────┘                   │
│                         │                                       │
│            ┌────────────▼────────────┐                         │
│            │   queue = "nvidia-b200" │  ← All agents same queue│
│            └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    BUILDKITE CLOUD    │
              │  Routes to idle agent │
              └───────────────────────┘
```

## Prerequisites

### Buildkite Account Setup

1. Create/access Buildkite organization at https://buildkite.com
2. Create a pipeline named `kernelbot`
3. Generate two tokens:
   - **Agent Token**: For nodes to connect (Agents → Agent Tokens)
   - **API Token**: For submitting jobs (Personal Settings → API Access Tokens)

### API Token Permissions

The API token needs these scopes:
- `read_builds` - Poll build status
- `write_builds` - Create/trigger builds
- `read_artifacts` - Download result.json artifact
- `read_agents` (optional) - Check queue status

## Vendor Node Setup

### Prerequisites (Do This First in Buildkite UI)

Before running the setup script on your node:

1. **Create Buildkite account** at https://buildkite.com
2. **Create pipeline** named `kernelbot`
3. **Generate Agent Token**: Go to Agents → Agent Tokens → New Token
4. **Create Queue**: Go to Agents → Default cluster → Queues → New Queue
   - Enter your GPU type as the key (e.g., `test`, `b200`, `h100`)
   - Select **Self hosted**
   - Click Create Queue

### Run Setup Script

On your GPU node:

```bash
git clone https://github.com/gpu-mode/kernelbot.git
cd kernelbot

sudo BUILDKITE_AGENT_TOKEN=<your-token> GPU_TYPE=<queue-name> ./deployment/buildkite/setup-node-simple.sh
```

The script will:
- Install Buildkite agent (if not present)
- Create one agent per GPU with proper isolation
- Configure git to use HTTPS (avoids SSH key issues)
- Create environment hook that sets `NVIDIA_VISIBLE_DEVICES` per job
- Start all agents as systemd services

### Verify Setup

1. Check agents appear in Buildkite: https://buildkite.com/organizations/YOUR_ORG/agents
2. Run a test build with this pipeline:

```yaml
steps:
  - label: "GPU Test"
    command: "echo NVIDIA_VISIBLE_DEVICES=$$NVIDIA_VISIBLE_DEVICES && nvidia-smi -L"
    agents:
      queue: "your-queue-name"
```

### Environment Variables

The setup script sets these automatically:
- `GPU_TYPE` (required): Queue name matching what you created in Buildkite
- `BUILDKITE_AGENT_TOKEN` (required): Agent token from Buildkite
- `NODE_NAME` (optional): Defaults to hostname
- `GPU_COUNT` (optional): Auto-detected from nvidia-smi

## Pipeline Configuration

### Create Pipeline in Buildkite

1. Go to Pipelines → New Pipeline
2. Name: `kernelbot`
3. Repository: `https://github.com/gpu-mode/kernelbot`
4. Steps: Either upload from repo or paste directly

### Pipeline YAML

The pipeline is at `deployment/buildkite/pipeline.yml`:

```yaml
steps:
  - label: ":rocket: Kernel Evaluation"
    command: "python /app/src/runners/buildkite-runner.py"
    agents:
      queue: "${KERNELBOT_QUEUE}"
    plugins:
      - docker#v5.11.0:
          image: "${KERNELBOT_IMAGE:-ghcr.io/gpu-mode/kernelbot:latest}"
          runtime: nvidia
          propagate-environment: true
          environment:
            - NVIDIA_VISIBLE_DEVICES
            - CUDA_VISIBLE_DEVICES
            - KERNELBOT_PAYLOAD
            - KERNELBOT_RUN_ID
    timeout_in_minutes: 15
```

## Testing

### Working Docker Pipeline

Use this tested pipeline configuration for GPU jobs:

```yaml
steps:
  - label: ":whale: Docker GPU Test"
    agents:
      queue: "test"  # Must match your queue name
    plugins:
      - docker#v5.11.0:
          image: "nvidia/cuda:12.4.0-runtime-ubuntu22.04"
          always-pull: true
          gpus: "all"  # Use gpus instead of runtime: nvidia
          propagate-environment: true
          environment:
            - NVIDIA_VISIBLE_DEVICES
            - CUDA_VISIBLE_DEVICES
          cpus: "${KERNELBOT_CPUS:-8}"
          memory: "${KERNELBOT_MEMORY:-64g}"
    command: |
      echo "=== Resource Isolation ==="
      echo "NVIDIA_VISIBLE_DEVICES=$$NVIDIA_VISIBLE_DEVICES"
      nvidia-smi
      nproc
      free -h
    timeout_in_minutes: 5
```

**Key points**:
- Use `gpus: "all"` instead of `runtime: nvidia` (more reliable)
- Use `$$NVIDIA_VISIBLE_DEVICES` (double dollar) in YAML to prevent variable stripping
- The environment hook auto-sets KERNELBOT_CPUS, KERNELBOT_MEMORY based on machine resources

### End-to-End Test

Run from your local machine:

```bash
cd kernelbot
BUILDKITE_API_TOKEN=<api-token> uv run python tests/e2e_buildkite_test.py --queue test
```

Options:
- `--queue <name>`: Target queue (default: test)
- `--org <slug>`: Buildkite org (default: mark-saroufim)
- `--pipeline <slug>`: Pipeline name (default: kernelbot)
- `--dry-run`: Print config without submitting

### Check Queue Status

```bash
BUILDKITE_API_TOKEN=<api-token> uv run python -c "
import asyncio
from libkernelbot.launchers.buildkite import BuildkiteLauncher, BuildkiteConfig

async def main():
    launcher = BuildkiteLauncher(BuildkiteConfig(api_token='<api-token>'))
    status = await launcher.get_queue_status('test')
    print(f'Queue: {status[\"queue\"]}')
    print(f'Total agents: {status[\"total\"]}')
    print(f'Idle agents: {status[\"idle\"]}')
    for agent in status['agents']:
        print(f'  - {agent[\"name\"]}: busy={agent[\"busy\"]}')

asyncio.run(main())
"
```

## GPU Types

Buildkite-managed GPUs are registered with `_BK` suffix:

| GPU Type | Queue | SM Arch |
|----------|-------|---------|
| `B200_BK` | `b200` | 100 |
| `H100_BK` | `h100` | 90a |
| `MI300_BK` | `mi300` | (AMD) |
| `L40S_BK` | `test` | 89 (Ada Lovelace) |

## Environment Variables

### On Heroku/Backend (where the app runs)

These are set in Heroku config vars or your `.env` file:

| Variable | Required | Description |
|----------|----------|-------------|
| `BUILDKITE_API_TOKEN` | Yes | API token for submitting jobs and downloading artifacts. Get from Buildkite → Personal Settings → API Access Tokens |
| `BUILDKITE_ORG` | No | Organization slug (default: `mark-saroufim`) |
| `BUILDKITE_PIPELINE` | No | Pipeline slug (default: `kernelbot`) |

**API Token Permissions Required:**
- `read_builds` - Poll build status
- `write_builds` - Create/trigger builds
- `read_artifacts` - Download result.json artifact
- `read_agents` (optional) - Check queue status

### On GPU Runner Nodes

These are set during node setup:

| Variable | Set By | Description |
|----------|--------|-------------|
| `BUILDKITE_AGENT_TOKEN` | Admin (setup script) | Agent token for connecting to Buildkite |
| `NVIDIA_VISIBLE_DEVICES` | Environment hook | GPU index for isolation (auto-set per job) |
| `CUDA_VISIBLE_DEVICES` | Environment hook | Same as above |
| `KERNELBOT_GPU_INDEX` | Environment hook | GPU index (0, 1, 2, ...) |
| `KERNELBOT_CPUSET` | Environment hook | CPU cores for this agent |
| `KERNELBOT_MEMORY` | Environment hook | Memory limit for Docker |

### Passed to Jobs (via Buildkite API)

These are set automatically by the launcher:

| Variable | Description |
|----------|-------------|
| `KERNELBOT_RUN_ID` | Unique run identifier |
| `KERNELBOT_PAYLOAD` | Base64+zlib compressed job config |
| `KERNELBOT_QUEUE` | Target queue name |
| `KERNELBOT_IMAGE` | Docker image to use |

## Troubleshooting

### Agent not appearing in dashboard

1. Check agent is running: `sudo systemctl status buildkite-agent`
2. Check logs: `sudo journalctl -u buildkite-agent -f`
3. Verify token is correct in `/etc/buildkite-agent/buildkite-agent.cfg`

### Job stuck in queue

1. Check agents are idle: Buildkite dashboard → Agents
2. Verify queue name matches agent tags
3. Check agent logs for errors

### Docker permission denied

```bash
sudo usermod -aG docker buildkite-agent
sudo systemctl restart buildkite-agent
```

### GPU not visible in container

1. Verify nvidia-container-toolkit: `nvidia-ctk --version`
2. Configure docker runtime: `sudo nvidia-ctk runtime configure --runtime=docker`
3. Restart docker: `sudo systemctl restart docker`
4. Test: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

### Package dependency conflicts (nvidia-container-toolkit)

If you see version conflicts:
```bash
sudo apt-get install -y nvidia-container-toolkit=1.18.1-1 nvidia-container-toolkit-base=1.18.1-1
```

### Agent fails with "Missing build-path"

The config file needs `build-path` set:

```bash
sudo nano /etc/buildkite-agent/buildkite-agent.cfg
```

Add this line:
```
build-path="/var/lib/buildkite-agent/builds"
```

Then:
```bash
sudo mkdir -p /var/lib/buildkite-agent/builds
sudo chown buildkite-agent:buildkite-agent /var/lib/buildkite-agent/builds
sudo systemctl restart buildkite-agent
```

### Agent not appearing - "Could not find queue"

You must create the queue in Buildkite web UI:
1. Go to **Agents** tab → **Default cluster** → **Queues**
2. Click **New Queue**
3. Enter queue name (e.g., `test`)
4. Select **Self hosted**
5. Click **Create Queue**

### Jobs run on hosted agents instead of self-hosted

Make sure your pipeline steps include the queue:

```yaml
steps:
  - label: ":rocket: Test Job"
    command: "nvidia-smi"
    agents:
      queue: "test"  # This is required!
```

Without `agents: queue:`, Buildkite uses hosted runners by default.

### Docker runtime crashes / "nvidia-container-runtime: no such file"

Use `gpus: "all"` in the Docker plugin instead of `runtime: nvidia`:

```yaml
plugins:
  - docker#v5.11.0:
      gpus: "all"  # ✓ Use this
      # runtime: nvidia  # ✗ Avoid - can cause crashes
```

If issues persist, reinstall nvidia-container-toolkit:
```bash
sudo apt-get remove --purge nvidia-container-toolkit nvidia-container-toolkit-base
sudo apt-get install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Environment hook not running

Make sure the hook has a shebang line:
```bash
#!/bin/bash
# Rest of hook script...
```

### Git clone fails with "Permission denied (publickey)"

The buildkite-agent user doesn't have SSH keys for GitHub. Fix by using HTTPS:

```bash
cd /tmp && sudo -u buildkite-agent git config --global url."https://github.com/".insteadOf "git@github.com:"
```

## Resource Isolation

| Resource | Method | Enforcement |
|----------|--------|-------------|
| GPU | `NVIDIA_VISIBLE_DEVICES` | Per-agent env var |
| CPU | `--cpuset-cpus` | Docker cgroups |
| Memory | `--memory` | Docker cgroups |
| Disk | Separate build paths | Filesystem |
| Network | Docker bridge | Container isolation |

## Files Reference

| File | Purpose |
|------|---------|
| `deployment/buildkite/setup-node-simple.sh` | Vendor node setup script (recommended) |
| `deployment/buildkite/pipeline.yml` | Buildkite pipeline config |
| `deployment/buildkite/pipeline-test-docker.yml` | Docker test pipeline |
| `deployment/buildkite/Dockerfile` | Docker image for jobs |
| `src/libkernelbot/launchers/buildkite.py` | BuildkiteLauncher class |
| `src/runners/buildkite-runner.py` | Job execution script |
| `tests/e2e_buildkite_test.py` | E2E test script |

## Auto-Resource Detection

The environment hook automatically detects and divides machine resources:

```
Machine: 16 CPUs, 289GB RAM, 2 GPUs
   ↓
Per-GPU allocation:
  - GPU 0: CPUs 0-7, 144GB RAM
  - GPU 1: CPUs 8-15, 144GB RAM
```

This is calculated in the environment hook as:
- `CPUS_PER_GPU = TOTAL_CPUS / GPU_COUNT`
- `RAM_PER_GPU = TOTAL_RAM_GB / GPU_COUNT`
- `KERNELBOT_CPUSET = (GPU_INDEX * CPUS_PER_GPU) to ((GPU_INDEX + 1) * CPUS_PER_GPU - 1)`

## Summary of Key Decisions

1. **Use `gpus: "all"` not `runtime: nvidia`** - More reliable with nvidia-container-toolkit
2. **Environment hook for isolation** - Sets `NVIDIA_VISIBLE_DEVICES`, `KERNELBOT_*` vars before each job
3. **Auto-detect resources** - No hardcoded CPU/RAM values; divides machine resources by GPU count
4. **One agent per GPU** - Each agent has its own build path and GPU assignment
5. **HTTPS for git** - Avoids SSH key issues on buildkite-agent user
6. **Queue must exist first** - Create queue in Buildkite UI before agents can connect
7. **Follow S3 redirects for artifacts** - Buildkite returns 302 to S3; must fetch without auth header

## E2E Workflow (Verified Working)

The complete end-to-end flow for submitting jobs and retrieving results:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Backend   │────▶│    Buildkite    │────▶│  GPU Runner     │
│                 │     │     Cloud       │     │  (Self-hosted)  │
│ BuildkiteLauncher     │                 │     │                 │
│ ._launch()      │     │  Routes to      │     │  Runs Docker    │
│                 │     │  idle agent     │     │  container      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │  1. POST /builds      │                       │
        │  (payload encoded)    │                       │
        │──────────────────────▶│                       │
        │                       │  2. Dispatch job      │
        │                       │──────────────────────▶│
        │                       │                       │
        │                       │                       │  3. Run evaluation
        │                       │                       │  Write result.json
        │                       │                       │
        │                       │  4. Upload artifact   │
        │                       │◀──────────────────────│
        │                       │                       │
        │  5. Poll status       │                       │
        │◀─────────────────────▶│                       │
        │                       │                       │
        │  6. Download artifact │                       │
        │  (via S3 redirect)    │                       │
        │◀──────────────────────│                       │
        │                       │                       │
        ▼                       │                       │
   Return result                │                       │
```

### Verified Test Output

```
=== Buildkite E2E Test ===
Organization: mark-saroufim
Pipeline: kernelbot
Queue: test
Mode: artifact

Submitting test job...
[UPDATE] Build created: [28]
[UPDATE] Build completed: [28]

=== Result ===
Success: True
Build URL: https://buildkite.com/mark-saroufim/kernelbot/builds/28
Downloaded artifact:
{
  "success": true,
  "error": "",
  "runs": {},
  "system": {
    "gpu_name": "test",
    "cuda_version": "12.4",
    "python_version": "N/A"
  }
}

=== Queue Status ===
Queue: test
Total agents: 0
Idle agents: 0
```

### How It Works

1. **BuildkiteLauncher._launch()** encodes config as base64+zlib compressed payload
2. **POST to Buildkite API** creates a build with env vars (KERNELBOT_PAYLOAD, KERNELBOT_RUN_ID)
3. **Buildkite routes** the job to an idle agent in the specified queue
4. **Agent runs Docker container** with GPU isolation (NVIDIA_VISIBLE_DEVICES set by environment hook)
5. **Container writes result.json** to working directory
6. **Buildkite uploads artifact** to S3
7. **BuildkiteLauncher polls** until build completes
8. **Downloads result.json** by following S3 redirect (without auth header)
9. **Returns parsed result** to caller

### Running the E2E Test

```bash
BUILDKITE_API_TOKEN=<your-token> uv run python tests/e2e_buildkite_test.py \
  --org <your-org> \
  --queue test
```

## Real Evaluation Jobs

### Submit a Real Kernel Evaluation

```bash
BUILDKITE_API_TOKEN=<your-token> uv run python scripts/submit_buildkite_job.py --eval vectoradd_py
```

This runs the full evaluation pipeline on actual GPU hardware and returns real benchmark results:

```
=== Result ===
Success: True
System: SystemInfo(gpu='NVIDIA L40S', device_count=1, cpu='AMD EPYC 9254 24-Core Processor', runtime='CUDA', platform='Linux-5.15.0-164-generic-x86_64-with-glibc2.35', torch='2.6.0+cu124', hostname='...')

test:
  Passed: True
  Duration: 3.18s
  Result: {'test-count': '5', 'test.0.status': 'pass', 'test.1.status': 'pass', ...}
```

### Integration Tests

Run the full integration test suite:

```bash
BUILDKITE_API_TOKEN=<your-token> uv run pytest tests/test_buildkite.py -v -m integration
```

Tests include:
- `test_buildkite_launcher_python_script` - Real evaluation with vectoradd_py
- `test_buildkite_launcher_failing_script` - Verifies cheating scripts correctly fail
- `test_buildkite_queue_status` - Tests agent queue API

### Available Examples

Any example in the `examples/` directory works:

```bash
# List available examples
ls examples/

# Run a specific example
BUILDKITE_API_TOKEN=xxx uv run python scripts/submit_buildkite_job.py --eval identity_py
```

## Operational Model

### Option 1: No Pre-Built Image (Current Default)

The pipeline installs dependencies at runtime. Each job:

1. Uses base `nvidia/cuda:12.4.0-devel-ubuntu22.04` image
2. Installs dependencies at runtime (~30-40 seconds):
   - `uv` for Python package management
   - Clones kernelbot repo
   - Runs `uv sync` and `uv pip install torch triton numpy`
3. Runs the evaluation

**Advantages:**
- No Dockerfile to maintain or rebuild
- No image registry to manage
- Always uses latest code from repo
- **No admin action needed** after code updates

**Trade-off:**
- Slower cold starts (~40 seconds)

### Option 2: Pre-Built Image (Fast Cold Starts)

For faster cold starts (~5 seconds), build the Docker image on each node:

```bash
# During initial setup:
sudo BUILDKITE_AGENT_TOKEN=xxx GPU_TYPE=test BUILD_IMAGE=true ./deployment/buildkite/setup-node-simple.sh

# Or build separately:
./deployment/buildkite/build-image.sh
```

Then update the Buildkite pipeline config to use the local image:
```yaml
image: "kernelbot:latest"
```

**When to rebuild the image:**
- When dependencies change (new PyTorch version, new packages)
- When you want the latest kernelbot code baked in
- NOT needed for problem/task changes (those come via config)

**Rebuild command:**
```bash
./deployment/buildkite/build-image.sh
```

### When Admin Action Is Needed

| Scenario | Action Required |
|----------|-----------------|
| Code changes (no deps) | None - pipeline clones fresh code |
| Dependency changes | Rebuild image: `./build-image.sh` |
| Initial node setup | Run `setup-node-simple.sh` once |
| NVIDIA driver updates | May need to rebuild image |
| Buildkite agent updates | Rare - Buildkite handles this |

### Shared Evaluation Logic

All runners (GitHub, Modal, Buildkite) use the exact same evaluation engine:

```python
# src/runners/buildkite-runner.py:49
from libkernelbot.run_eval import run_config
result = run_config(config)
```

This means:
- Any problem that works on GitHub/Modal works on Buildkite
- Same result format (`FullResult`)
- Same test/benchmark logic
- Same correctness checking

## Current Branch

The Buildkite infrastructure is on the `buildkite-infrastructure` branch. The pipeline clones from this branch:

```yaml
git clone --depth 1 --branch buildkite-infrastructure https://github.com/gpu-mode/kernelbot.git
```

Once merged to `main`, update the pipeline config to use `main` branch.

## E2E Testing with Database

A comprehensive end-to-end test script is available that:
1. Creates a test leaderboard in the database
2. Submits a real kernel evaluation to Buildkite
3. Stores results in PostgreSQL
4. Verifies data integrity

### Running E2E Tests

```bash
# Test mode (correctness only)
BUILDKITE_API_TOKEN=xxx uv run python scripts/e2e_buildkite_with_db.py \
  --org mark-saroufim --queue test

# Leaderboard mode (with benchmarks and scoring)
BUILDKITE_API_TOKEN=xxx uv run python scripts/e2e_buildkite_with_db.py \
  --org mark-saroufim --queue test --mode leaderboard

# With cleanup (delete test leaderboard after)
BUILDKITE_API_TOKEN=xxx uv run python scripts/e2e_buildkite_with_db.py \
  --org mark-saroufim --queue test --mode leaderboard --cleanup
```

### Verified Working (2026-02-04)

| Mode | Status | Details |
|------|--------|---------|
| Test | ✅ | 5 tests passed, ~3.4s duration |
| Benchmark | ✅ | 30 runs, 4.07ms mean |
| Leaderboard | ✅ | Score computed and stored |
| Database | ✅ | All runs stored with system info |

---

## Known Limitations & Review Notes

This section documents known limitations and tradeoffs for code reviewers.

### 1. Cold Start Overhead

**Problem**: Each job incurs significant startup overhead:

| Phase | Time | Notes |
|-------|------|-------|
| Docker pull | 10-30s | First run only if image not cached |
| Container start | 2-5s | Includes cgroup setup |
| Python imports | 5-10s | PyTorch, Triton, etc. |
| Code clone | 3-5s | If using runtime install |
| **Total cold start** | **20-50s** | Varies by image caching |

**Current Approach**: We use a pre-built Docker image (`ghcr.io/gpu-mode/kernelbot:latest`) with dependencies baked in. This reduces cold start to ~10-15s after first pull.

### 2. Dependency Installation Tradeoffs

There are two operational models with different tradeoffs:

#### Option A: Pre-Built Image (Current Default)
```yaml
image: "ghcr.io/gpu-mode/kernelbot:latest"
```
- **Pros**: Fast cold starts (~5-10s), consistent environment
- **Cons**: Must rebuild image for dependency changes, requires image registry
- **When to rebuild**: PyTorch version change, new packages, security updates

#### Option B: Runtime Installation
```yaml
image: "nvidia/cuda:12.4.0-devel-ubuntu22.04"
command: |
  pip install torch triton numpy
  python eval.py
```
- **Pros**: Always latest dependencies, no image maintenance
- **Cons**: Slow cold starts (~40-60s), network dependency, version drift
- **Use when**: Testing new dependencies, development

#### Option C: Cached Dependencies on Host
```yaml
volumes:
  - "/var/lib/buildkite-agent/cache/pip:/root/.cache/pip:rw"
```
- **Pros**: Fast after first run, no image rebuild needed
- **Cons**: Cache invalidation complexity, disk usage, per-node setup
- **Use when**: Frequent dependency changes, limited registry access

**Recommendation**: Use Option A (pre-built image) for production. Use Option B for development/testing new dependencies.

### 3. GPU Isolation Limitations

**Current Isolation Model**:
- GPU isolation via `NVIDIA_VISIBLE_DEVICES` environment variable
- CPU isolation via Docker `--cpuset-cpus`
- Memory isolation via Docker `--memory`

**Known Gaps**:

| Resource | Isolation Level | Notes |
|----------|-----------------|-------|
| GPU Compute | ✅ Strong | Only assigned GPU visible |
| GPU Memory | ⚠️ Partial | Other processes could exhaust VRAM if running |
| PCIe Bandwidth | ❌ None | Shared across all GPUs |
| NVLink | ❌ None | If present, shared |
| CPU Cache | ⚠️ Partial | L3 cache shared across cores |
| Network | ⚠️ Partial | Docker bridge, but shared bandwidth |
| Disk I/O | ❌ None | Shared unless using separate volumes |

**Potential Issues**:
- **Noisy neighbor**: One job could impact another via shared resources
- **VRAM exhaustion**: If host processes use GPU memory
- **Timing variability**: Benchmark results may vary due to shared resources

**Mitigations**:
- Run one agent per GPU (current approach)
- Use dedicated benchmark nodes for competition scoring
- Monitor for outlier results

### 4. Artifact Handling

**Current Flow**:
1. Job writes `result.json` to working directory
2. Buildkite agent uploads to S3
3. Backend downloads via Buildkite API (302 redirect to S3)

**Limitations**:
- **Size limit**: ~100MB per artifact (Buildkite limit)
- **Retention**: 6 months by default
- **Download latency**: 1-2s for small files, more for large profiles

### 5. Queue Management

**Current Model**: One queue per GPU type (e.g., `b200`, `h100`, `mi300`)

**Limitations**:
- No priority queuing (FIFO only)
- No job preemption
- No fair-share scheduling between users
- Queue depth visibility requires API calls

**Potential Improvements**:
- Implement priority via build metadata
- Add rate limiting per user
- Create admin queue for verification runs

### 6. Error Handling

**Automatic Retries**:
```yaml
retry:
  automatic:
    - exit_status: -1   # Infrastructure failure
      limit: 2
    - exit_status: 255  # Agent disconnect
      limit: 1
```

**Not Automatically Retried**:
- Compilation errors (user code issue)
- Test failures (user code issue)
- Timeout (15 min default)
- OOM errors

### 7. Security Considerations

**Sandboxing**:
- Jobs run in Docker containers
- No host network access
- Limited volume mounts

**Risks**:
- User code has full GPU access (could mine crypto briefly)
- User code could attempt network attacks (mitigated by Docker networking)
- Large submissions could exhaust disk space

**Mitigations**:
- Timeout limits (15 min)
- Disk quotas (via Docker)
- Network isolation (Docker bridge)
- Result validation before storing

---

## Future Improvements

- [ ] Add MIG (Multi-Instance GPU) support for H100/A100
- [ ] Implement job priority queuing
- [ ] Add per-user rate limiting
- [ ] Support multi-GPU jobs for distributed problems
- [ ] Add warm pool of pre-started containers
- [ ] Implement result caching for identical submissions

