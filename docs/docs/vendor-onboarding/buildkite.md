---
sidebar_position: 1
---

# Buildkite Vendor Onboarding

This guide explains how to set up Buildkite agents on your hardware to run GPU kernel competitions for Kernelbot.

## Overview

Kernelbot uses Buildkite to run GPU kernel submissions on vendor-donated hardware. Each GPU on your machine runs as an isolated Buildkite agent with:

- **GPU Isolation**: Single GPU per agent via `CUDA_VISIBLE_DEVICES`
- **CPU/RAM Limits**: Resource constraints via systemd cgroups
- **Queue Routing**: Jobs routed to specific GPUs via queue tags

## Prerequisites

Before setting up agents, ensure you have:

1. **Linux server** with NVIDIA GPUs (Ubuntu 22.04+ recommended)
2. **Docker** installed with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. **Buildkite agent** installed ([installation guide](https://buildkite.com/docs/agent/v3/installation))
4. **Buildkite organization token** from the Kernelbot team

### Verify Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker with GPU support
docker run --rm --gpus all nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi

# Check Buildkite agent
buildkite-agent --version
```

## Queue Naming Convention

Queues follow the pattern: `{vendor}-{gpu_type}-{index}`

Examples:
- `nvidia-h100-0` - NVIDIA-donated H100, GPU index 0
- `nvidia-h100-1` - NVIDIA-donated H100, GPU index 1
- `amd-mi300-0` - AMD-donated MI300
- `google-tpu-0` - Google-donated TPU
- `nebius-h100-0` - Nebius-donated H100

Contact the Kernelbot team to register your queue names.

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/gpu-mode/kernelbot.git
cd kernelbot
```

### Step 2: Pull the Runner Image

```bash
docker pull ghcr.io/gpu-mode/kernelbot-runner:latest
```

### Step 3: Configure Agents

Run the setup script for each GPU. For an 8-GPU node:

```bash
# Set your Buildkite token
echo "BUILDKITE_AGENT_TOKEN=your-token-here" | sudo tee /etc/buildkite-agent/token

# Set up each GPU (adjust queue names for your vendor)
sudo ./scripts/buildkite/setup-agent.sh 0 nvidia-h100-0 32G 16
sudo ./scripts/buildkite/setup-agent.sh 1 nvidia-h100-1 32G 16
sudo ./scripts/buildkite/setup-agent.sh 2 nvidia-h100-2 32G 16
# ... repeat for all GPUs
```

Arguments:
- `GPU_INDEX`: GPU device index (0, 1, 2, ...)
- `QUEUE_NAME`: Queue name following convention above
- `MEMORY_LIMIT`: RAM limit per agent (default: 32G)
- `CPU_CORES`: CPU cores per agent (default: 16)

### Step 4: Start Agents

```bash
# Start all GPU agents
sudo systemctl start buildkite-agent-gpu0
sudo systemctl start buildkite-agent-gpu1
# ... etc

# Or start all at once
sudo systemctl start 'buildkite-agent-gpu*'
```

### Step 5: Verify Setup

```bash
# Check agent status
sudo systemctl status buildkite-agent-gpu0

# View logs
sudo journalctl -u buildkite-agent-gpu0 -f

# Verify agent appears in Buildkite dashboard
# https://buildkite.com/organizations/<org>/agents
```

## Testing Your Setup

### Local Test (Without Buildkite)

Test the runner image directly:

```bash
# Create a test payload
TEST_PAYLOAD=$(python3 -c "
import json, zlib, base64
config = {
    'lang': 'py',
    'mode': 'test',
    'files': {'main.py': 'print(\"Hello GPU!\")'},
    'tests': [],
    'benchmarks': []
}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
")

# Run in container (single GPU)
docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="$TEST_PAYLOAD" \
  ghcr.io/gpu-mode/kernelbot-runner:latest

# Check if result.json would be created
ls -la result.json
```

### Integration Test (Via Buildkite)

Trigger a test build:

```bash
curl -X POST "https://api.buildkite.com/v2/organizations/gpu-mode/pipelines/kernelbot-runner/builds" \
  -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "commit": "HEAD",
    "branch": "main",
    "message": "Test submission",
    "env": {
      "GPU_QUEUE": "your-queue-name",
      "SUBMISSION_PAYLOAD": "'"$TEST_PAYLOAD"'"
    }
  }'
```

Check the Buildkite dashboard for job results.

### Isolation Verification

Verify GPU and resource isolation:

```bash
# Inside agent container, verify only 1 GPU visible
docker run --rm --gpus '"device=0"' nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi
# Should show only GPU 0

# Verify cgroup limits
cat /sys/fs/cgroup/buildkite-gpu0.slice/memory.max
cat /sys/fs/cgroup/buildkite-gpu0.slice/cpu.max
```

## Updating the Runner Image

When notified of a new image release:

```bash
sudo ./scripts/buildkite/update-image.sh
```

This pulls the latest image and optionally restarts agents.

### Automatic Updates (Optional)

Set up a cron job for automatic updates:

```bash
# Check for updates daily at 3 AM
echo "0 3 * * * root /path/to/kernelbot/scripts/buildkite/update-image.sh --auto" | sudo tee /etc/cron.d/kernelbot-update
```

## Troubleshooting

### Agent Not Picking Up Jobs

1. Check agent is running: `systemctl status buildkite-agent-gpu0`
2. Verify queue tag matches: check `/etc/buildkite-agent/agent-0/buildkite-agent.cfg`
3. Ensure agent appears in Buildkite dashboard

### GPU Not Visible in Container

1. Check NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi`
2. Verify CUDA_VISIBLE_DEVICES is set correctly in systemd unit
3. Check Docker runtime config: `docker info | grep -i runtime`

### Jobs Timing Out

1. Check resource limits aren't too restrictive
2. Review job logs in Buildkite dashboard
3. Test image locally first

### Memory/CPU Limits Not Working

1. Verify cgroup v2 is enabled: `mount | grep cgroup2`
2. Check slice file exists: `cat /etc/systemd/system/buildkite-gpu0.slice`
3. Reload systemd: `systemctl daemon-reload`

## Support

- **Slack**: #kernelbot-infra in GPU Mode Discord
- **Issues**: https://github.com/gpu-mode/kernelbot/issues
- **Email**: infra@gpu-mode.org

## Hardware Requirements

Per GPU agent:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| CPU Cores | 8 | 16 |
| Disk | 50 GB | 100 GB |
| Network | 100 Mbps | 1 Gbps |

For an 8-GPU node, plan for:
- 256 GB RAM (32 GB per GPU)
- 128 CPU cores (16 per GPU)
- 800 GB disk
- Fast network for image pulls
