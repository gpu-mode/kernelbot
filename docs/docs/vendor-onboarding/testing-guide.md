---
sidebar_position: 2
---

# Buildkite Testing Guide

This guide covers how to test the Buildkite integration at various levels: local development, vendor validation, and end-to-end integration.

## Testing Levels

| Level | Who | Purpose |
|-------|-----|---------|
| Unit Tests | Kernelbot developers | Test launcher logic with mocked API |
| Local Container | Vendors | Verify runner image works with GPU |
| Agent Integration | Vendors | Verify agent picks up and runs jobs |
| End-to-End | Both | Full submission flow through Discord/API |

---

## 1. Unit Tests (Kernelbot Developers)

### Test BuildkiteLauncher with Mocked API

```python
# tests/test_buildkite_launcher.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import base64
import zlib

from libkernelbot.launchers.buildkite import BuildkiteLauncher
from libkernelbot.consts import BuildkiteGPU, GPU


@pytest.fixture
def launcher():
    return BuildkiteLauncher(
        org="test-org",
        pipeline="test-pipeline",
        token="test-token"
    )


@pytest.fixture
def mock_config():
    return {
        "lang": "py",
        "mode": "test",
        "files": {"main.py": "print('hello')"},
        "tests": [],
        "benchmarks": [],
        "test_timeout": 180,
    }


@pytest.fixture
def gpu_type():
    return GPU(name="NVIDIA_H100", value="nvidia-h100", runner="Buildkite")


class TestBuildkiteLauncher:
    def test_init(self, launcher):
        assert launcher.name == "Buildkite"
        assert launcher.org == "test-org"
        assert launcher.pipeline == "test-pipeline"
        assert launcher.gpus == BuildkiteGPU

    @pytest.mark.asyncio
    async def test_run_submission_creates_build(self, launcher, mock_config, gpu_type):
        """Test that run_submission creates a Buildkite build."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "number": 123,
            "web_url": "https://buildkite.com/test/builds/123",
            "state": "scheduled",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch.object(launcher, "_wait_for_completion", new_callable=AsyncMock):
                with patch.object(launcher, "_download_and_parse_result", new_callable=AsyncMock) as mock_download:
                    mock_download.return_value = MagicMock(success=True)

                    mock_status = AsyncMock()
                    result = await launcher.run_submission(mock_config, gpu_type, mock_status)

                    # Verify API was called
                    mock_post.assert_called_once()
                    call_args = mock_post.call_args

                    # Check URL
                    assert "test-org" in call_args[0][0]
                    assert "test-pipeline" in call_args[0][0]

                    # Check payload was compressed
                    body = call_args[1]["json"]
                    assert "SUBMISSION_PAYLOAD" in body["env"]
                    assert body["env"]["GPU_QUEUE"] == "nvidia-h100"

    @pytest.mark.asyncio
    async def test_payload_compression(self, launcher, mock_config, gpu_type):
        """Test that config is properly compressed."""
        captured_payload = None

        def capture_post(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs["json"]["env"]["SUBMISSION_PAYLOAD"]
            response = MagicMock()
            response.json.return_value = {"number": 1, "web_url": "http://test", "state": "scheduled"}
            response.raise_for_status = MagicMock()
            return response

        with patch("requests.post", side_effect=capture_post):
            with patch.object(launcher, "_wait_for_completion", new_callable=AsyncMock):
                with patch.object(launcher, "_download_and_parse_result", new_callable=AsyncMock):
                    mock_status = AsyncMock()
                    await launcher.run_submission(mock_config, gpu_type, mock_status)

        # Decompress and verify
        decompressed = zlib.decompress(base64.b64decode(captured_payload)).decode()
        parsed = json.loads(decompressed)
        assert parsed["lang"] == "py"
        assert parsed["mode"] == "test"
```

### Run Unit Tests

```bash
pytest tests/test_buildkite_launcher.py -v
```

---

## 2. Local Container Tests (Vendors)

### 2.1 Basic Image Test

Verify the image runs and has correct dependencies:

```bash
# Pull the image
docker pull ghcr.io/gpu-mode/kernelbot-runner:latest

# Check Python and dependencies
docker run --rm ghcr.io/gpu-mode/kernelbot-runner:latest python --version
docker run --rm ghcr.io/gpu-mode/kernelbot-runner:latest pip list | grep torch

# Check GPU access
docker run --rm --gpus all ghcr.io/gpu-mode/kernelbot-runner:latest nvidia-smi
```

### 2.2 Single GPU Isolation Test

Verify only the specified GPU is visible:

```bash
# Should only show GPU 0
docker run --rm --gpus '"device=0"' ghcr.io/gpu-mode/kernelbot-runner:latest nvidia-smi

# Should only show GPU 1
docker run --rm --gpus '"device=1"' ghcr.io/gpu-mode/kernelbot-runner:latest nvidia-smi
```

### 2.3 Runner Script Test

Test the runner with a simple payload:

```bash
# Create test payload
create_test_payload() {
    python3 -c "
import json, zlib, base64
config = {
    'lang': 'py',
    'mode': 'test',
    'files': {
        'main.py': '''
import torch
print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")
    print(f\"GPU count: {torch.cuda.device_count()}\")
'''
    },
    'tests': [],
    'benchmarks': [],
    'test_timeout': 60,
    'benchmark_timeout': 60,
    'ranked_timeout': 60,
}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
"
}

# Run with payload
docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="$(create_test_payload)" \
  -v "$(pwd)/test-output:/workdir" \
  -w /workdir \
  ghcr.io/gpu-mode/kernelbot-runner:latest

# Check output
cat test-output/result.json | jq .
```

### 2.4 CUDA Kernel Test

Test a simple CUDA kernel submission:

```bash
create_cuda_payload() {
    python3 -c "
import json, zlib, base64
config = {
    'lang': 'py',
    'mode': 'test',
    'files': {
        'main.py': '''
import torch
import torch.nn as nn

# Simple GPU operation
x = torch.randn(1000, 1000, device=\"cuda\")
y = torch.randn(1000, 1000, device=\"cuda\")
z = torch.matmul(x, y)
print(f\"Matrix multiply result shape: {z.shape}\")
print(f\"Result sum: {z.sum().item():.2f}\")
'''
    },
    'tests': [],
    'benchmarks': [],
    'test_timeout': 60,
    'benchmark_timeout': 60,
    'ranked_timeout': 60,
}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
"
}

docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="$(create_cuda_payload)" \
  -v "$(pwd)/test-output:/workdir" \
  -w /workdir \
  ghcr.io/gpu-mode/kernelbot-runner:latest
```

### 2.5 Resource Limit Test

Test memory limits are enforced:

```bash
# Run with memory limit
docker run --rm --gpus '"device=0"' \
  --memory=4g \
  -e SUBMISSION_PAYLOAD="$(create_test_payload)" \
  ghcr.io/gpu-mode/kernelbot-runner:latest

# Check container saw the limit
docker run --rm --memory=4g ghcr.io/gpu-mode/kernelbot-runner:latest \
  cat /sys/fs/cgroup/memory.max
```

---

## 3. Agent Integration Tests (Vendors)

### 3.1 Agent Health Check

After setting up agents, verify they're healthy:

```bash
# Check systemd service status
sudo systemctl status buildkite-agent-gpu0
sudo systemctl status buildkite-agent-gpu1

# Check agent logs
sudo journalctl -u buildkite-agent-gpu0 --since "5 minutes ago"

# Verify agent appears in Buildkite dashboard
curl -s -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  "https://api.buildkite.com/v2/organizations/gpu-mode/agents" | jq '.[] | {name, connection_state, metadata}'
```

### 3.2 Cgroup Isolation Verification

Verify resource isolation is working:

```bash
# Check memory limit
cat /sys/fs/cgroup/buildkite-gpu0.slice/memory.max
# Should show your configured limit (e.g., 34359738368 for 32G)

# Check CPU quota
cat /sys/fs/cgroup/buildkite-gpu0.slice/cpu.max
# Should show something like "1600000 100000" for 16 cores

# Verify agent is in the slice
systemctl status buildkite-agent-gpu0 | grep "CGroup"
```

### 3.3 GPU Binding Verification

Verify each agent only sees its assigned GPU:

```bash
# Check what GPU agent 0 sees
sudo -u buildkite CUDA_VISIBLE_DEVICES=0 nvidia-smi -L
# Should show only GPU 0

# Check what GPU agent 1 sees
sudo -u buildkite CUDA_VISIBLE_DEVICES=1 nvidia-smi -L
# Should show only GPU 1
```

### 3.4 Trigger Test Build

Trigger a test build and verify it runs on correct agent:

```bash
# Create a test payload
TEST_PAYLOAD=$(python3 -c "
import json, zlib, base64
config = {
    'lang': 'py',
    'mode': 'test',
    'files': {'main.py': 'import torch; print(torch.cuda.get_device_name(0))'},
    'tests': [],
    'benchmarks': [],
    'test_timeout': 60,
    'benchmark_timeout': 60,
    'ranked_timeout': 60,
}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
")

# Trigger build on specific queue
curl -X POST "https://api.buildkite.com/v2/organizations/gpu-mode/pipelines/kernelbot-runner/builds" \
  -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "commit": "HEAD",
    "branch": "main",
    "message": "Agent integration test",
    "env": {
      "GPU_QUEUE": "nvidia-h100-0",
      "SUBMISSION_PAYLOAD": "'"$TEST_PAYLOAD"'"
    }
  }' | jq '{number, web_url, state}'

# Watch the build
# Check Buildkite dashboard or poll API
```

### 3.5 Concurrent Job Test

Verify jobs don't interfere with each other:

```bash
# Trigger jobs on different GPUs simultaneously
for i in 0 1 2 3; do
  curl -X POST "https://api.buildkite.com/v2/organizations/gpu-mode/pipelines/kernelbot-runner/builds" \
    -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "commit": "HEAD",
      "branch": "main",
      "message": "Concurrent test GPU '"$i"'",
      "env": {
        "GPU_QUEUE": "nvidia-h100-'"$i"'",
        "SUBMISSION_PAYLOAD": "'"$TEST_PAYLOAD"'"
      }
    }' &
done
wait

# All 4 should run in parallel on different agents
```

---

## 4. End-to-End Tests (Full System)

### 4.1 API Submission Test

Test the full flow through Kernelbot's API:

```bash
# This requires the full Kernelbot stack running
# Submit via API endpoint
curl -X POST "http://localhost:8000/leaderboard/test-leaderboard/nvidia-h100/test" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch; print(torch.cuda.get_device_name(0))",
    "user_id": "test-user",
    "user_name": "Test User"
  }'
```

### 4.2 Discord Bot Test

Test submission via Discord (manual):

1. Go to the Discord server with Kernelbot
2. Use `/leaderboard submit test` command
3. Select a Buildkite GPU type (e.g., `nvidia-h100`)
4. Upload a test script
5. Verify the submission runs and returns results

### 4.3 Benchmark Accuracy Test

Compare results between launchers:

```bash
# Run same benchmark on Modal and Buildkite
# Results should be within acceptable variance (< 5% for microbenchmarks)

# This requires a benchmark that runs on both
# Compare the timing results in the database
```

---

## 5. Troubleshooting Tests

### 5.1 Timeout Behavior

Test that timeouts work correctly:

```bash
# Create a payload that times out
TIMEOUT_PAYLOAD=$(python3 -c "
import json, zlib, base64
config = {
    'lang': 'py',
    'mode': 'test',
    'files': {'main.py': 'import time; time.sleep(300)'},  # 5 minutes
    'tests': [],
    'benchmarks': [],
    'test_timeout': 10,  # 10 second timeout
    'benchmark_timeout': 10,
    'ranked_timeout': 10,
}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
")

# Should timeout after ~10 seconds
docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="$TIMEOUT_PAYLOAD" \
  ghcr.io/gpu-mode/kernelbot-runner:latest
```

### 5.2 Error Handling

Test error cases:

```bash
# Missing GPU
docker run --rm \
  -e SUBMISSION_PAYLOAD="$TEST_PAYLOAD" \
  ghcr.io/gpu-mode/kernelbot-runner:latest
# Should fail gracefully with error in result.json

# Invalid payload
docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="not-valid-base64" \
  ghcr.io/gpu-mode/kernelbot-runner:latest
# Should fail with clear error message

# Missing payload
docker run --rm --gpus '"device=0"' \
  ghcr.io/gpu-mode/kernelbot-runner:latest
# Should fail with "SUBMISSION_PAYLOAD not set" error
```

### 5.3 Agent Recovery

Test agent recovers from failures:

```bash
# Kill the agent process
sudo systemctl kill -s SIGKILL buildkite-agent-gpu0

# Check it restarts automatically
sleep 5
sudo systemctl status buildkite-agent-gpu0
# Should show "active (running)"
```

---

## Test Checklist

Use this checklist before going live:

### Vendor Checklist

- [ ] Image pulls successfully
- [ ] Image runs with GPU access
- [ ] Single GPU isolation works
- [ ] Runner script executes test payload
- [ ] CUDA operations work in container
- [ ] All agents show as connected in Buildkite
- [ ] Cgroup limits are enforced
- [ ] Test build completes successfully
- [ ] Artifacts are uploaded correctly
- [ ] Agent restarts after failure

### Developer Checklist

- [ ] Unit tests pass
- [ ] BuildkiteLauncher creates builds
- [ ] Polling works correctly
- [ ] Artifacts are downloaded and parsed
- [ ] Timeouts are handled
- [ ] Errors return proper FullResult
- [ ] GPU enum is registered correctly
- [ ] Launcher is registered in main.py
