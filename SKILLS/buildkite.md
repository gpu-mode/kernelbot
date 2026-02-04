# Buildkite GPU Infrastructure Guide

This document describes how to set up and use the Buildkite infrastructure for GPU job isolation.

## Overview

Buildkite provides a parallel infrastructure for onboarding arbitrary GPU vendors with proper isolation. It runs alongside the existing GitHub Actions system, providing:

- Per-GPU job isolation via `NVIDIA_VISIBLE_DEVICES`
- Resource constraints (CPU, RAM, disk) via Docker cgroups
- Clear, reproducible Docker environment
- Automatic queue management

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
- `read_builds`
- `write_builds`
- `read_agents` (optional, for queue status)

## Vendor Node Setup

### Automated Setup (Full)

For a fresh Ubuntu node with NVIDIA GPUs:

```bash
git clone https://github.com/gpu-mode/kernelbot.git
cd kernelbot
git checkout buildkite-infrastructure

sudo BUILDKITE_AGENT_TOKEN=<agent-token> GPU_TYPE=<gpu-type> ./deployment/buildkite/setup-node.sh
```

Environment variables:
- `BUILDKITE_AGENT_TOKEN` (required): Agent token from Buildkite
- `GPU_TYPE` (required): Queue name, e.g., `b200`, `h100`, `mi300`, `test`
- `GPU_COUNT` (optional): Number of GPUs (auto-detected)
- `CPUS_PER_GPU` (optional): CPUs per agent (default: 8)
- `RAM_PER_GPU` (optional): RAM per agent (default: 64g)
- `NODE_NAME` (optional): Node identifier (default: hostname)

### Manual Setup (Existing Docker/NVIDIA)

If Docker and nvidia-container-toolkit are already installed:

```bash
# Install Buildkite agent
sudo apt-get install -y apt-transport-https gnupg
curl -fsSL https://keys.openpgp.org/vks/v1/by-fingerprint/32A37959C2FA5C3C99EFBC32A79206696452D198 | \
    sudo gpg --dearmor -o /usr/share/keyrings/buildkite-agent-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/buildkite-agent-archive-keyring.gpg] https://apt.buildkite.com/buildkite-agent stable main" | \
    sudo tee /etc/apt/sources.list.d/buildkite-agent.list
sudo apt-get update
sudo apt-get install -y buildkite-agent

# Configure agent
export BUILDKITE_TOKEN="<your-agent-token>"
export GPU_TYPE="test"
export NODE_NAME=$(hostname)

echo "token=\"${BUILDKITE_TOKEN}\"
name=\"${NODE_NAME}-gpu0\"
tags=\"queue=${GPU_TYPE},gpu=${GPU_TYPE},gpu-index=0,node=${NODE_NAME}\"" | \
    sudo tee /etc/buildkite-agent/buildkite-agent.cfg

# Add to docker group and start
sudo usermod -aG docker buildkite-agent
sudo systemctl enable buildkite-agent
sudo systemctl start buildkite-agent
```

### Verify Agent Connection

Check the Buildkite dashboard:
```
https://buildkite.com/organizations/<org>/agents
```

Or via API:
```bash
curl -H "Authorization: Bearer <api-token>" \
    https://api.buildkite.com/v2/organizations/<org>/agents
```

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

### End-to-End Test

Run from your local machine:

```bash
cd kernelbot
BUILDKITE_API_TOKEN=<api-token> uv run python tests/e2e_buildkite_test.py --queue test
```

Options:
- `--queue <name>`: Target queue (default: test)
- `--org <slug>`: Buildkite org (default: gpu-mode)
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

## Environment Variables

### For Kernelbot API/Backend

- `BUILDKITE_API_TOKEN`: API token for submitting jobs

### For Buildkite Agents (set by setup script)

- `NVIDIA_VISIBLE_DEVICES`: GPU index for isolation
- `CUDA_VISIBLE_DEVICES`: Same as above
- `KERNELBOT_GPU_INDEX`: GPU index (0, 1, 2, ...)
- `KERNELBOT_CPUSET`: CPU cores for this agent
- `KERNELBOT_MEMORY`: Memory limit

### For Jobs (passed via pipeline)

- `KERNELBOT_RUN_ID`: Unique run identifier
- `KERNELBOT_PAYLOAD`: Base64+zlib compressed job config
- `KERNELBOT_QUEUE`: Target queue name

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
| `deployment/buildkite/setup-node.sh` | Vendor node setup script |
| `deployment/buildkite/pipeline.yml` | Buildkite pipeline config |
| `deployment/buildkite/Dockerfile` | Docker image for jobs |
| `src/libkernelbot/launchers/buildkite.py` | BuildkiteLauncher class |
| `src/runners/buildkite-runner.py` | Job execution script |
| `tests/e2e_buildkite_test.py` | E2E test script |
