# Buildkite Integration POC

## Executive Summary

This document describes a proof-of-concept implementation of Buildkite as a new scheduler for Kernelbot GPU kernel competitions. Buildkite solves critical isolation problems that make microbenchmarking on vendor-donated hardware unreliable.

**Status**: Implementation complete, unit tests passing, ready for integration testing with real Buildkite agents.

---

## Problem Statement

When vendors donate GPU compute for kernel competitions, we face these challenges:

| Problem | Impact |
|---------|--------|
| Multiple kernels on same GPU | Measurements become unreliable |
| No CPU/RAM isolation | Neighbor jobs affect benchmarks |
| Complex runner setup | Vendors spend weeks configuring isolation |
| No standardized onboarding | Each vendor does it differently |

### Current State

- **Modal**: Good isolation but cloud-only, can't use donated on-prem hardware
- **GitHub Actions**: Runners see all GPUs, no resource limits, complex setup

---

## Solution: Buildkite

Buildkite provides the primitives we need for proper isolation:

| Requirement | Buildkite Solution |
|-------------|-------------------|
| 1 GPU per job | 1 agent per GPU, bound via `CUDA_VISIBLE_DEVICES` |
| CPU/RAM limits | Agent runs in systemd cgroup slice |
| No interference | Agent processes 1 job at a time (default) |
| Queue routing | Agent tags route jobs to specific GPUs |
| Easy onboarding | Bootstrap script + Dockerfile in our repo |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Vendor Node (8x H100, 256GB RAM, 128 cores)                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐     ┌──────────────┐        │
│  │ Agent 0      │ │ Agent 1      │ ... │ Agent 7      │        │
│  │ GPU 0 only   │ │ GPU 1 only   │     │ GPU 7 only   │        │
│  │ 32GB RAM     │ │ 32GB RAM     │     │ 32GB RAM     │        │
│  │ 16 CPU cores │ │ 16 CPU cores │     │ 16 CPU cores │        │
│  │ queue=       │ │ queue=       │     │ queue=       │        │
│  │ nvidia-h100-0│ │ nvidia-h100-1│     │ nvidia-h100-7│        │
│  └──────────────┘ └──────────────┘     └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Queue Naming Convention

Format: `{vendor}-{gpu_type}-{index}`

Examples:
- `nvidia-h100-0` - NVIDIA-donated H100, first GPU
- `amd-mi300-0` - AMD-donated MI300
- `google-tpu-0` - Google-donated TPU
- `nebius-h100-0` - Nebius-donated H100

This supports concurrent competitions where different vendors donate the same GPU type.

---

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `src/libkernelbot/launchers/buildkite.py` | BuildkiteLauncher class |
| `src/runners/buildkite-runner.py` | Runner script for agents |
| `docker/kernelbot-runner/Dockerfile` | Container image (source of truth) |
| `docker/kernelbot-runner/requirements-runner.txt` | Python dependencies |
| `.buildkite/pipeline.yml` | Buildkite pipeline config |
| `scripts/buildkite/setup-agent.sh` | Agent bootstrap script |
| `scripts/buildkite/update-image.sh` | Image update script |
| `.github/workflows/build-runner-image.yml` | Auto-build on changes |
| `docs/docs/vendor-onboarding/buildkite.md` | Vendor setup guide |
| `docs/docs/vendor-onboarding/testing-guide.md` | Testing instructions |
| `tests/test_buildkite.py` | Unit tests |

### Files Modified

| File | Changes |
|------|---------|
| `src/libkernelbot/consts.py` | Added `BuildkiteGPU` enum, `BUILDKITE` scheduler |
| `src/libkernelbot/launchers/__init__.py` | Export `BuildkiteLauncher` |
| `src/kernelbot/env.py` | Buildkite env vars |
| `src/kernelbot/main.py` | Register launcher if token set |

### Key Code

**BuildkiteLauncher** (`src/libkernelbot/launchers/buildkite.py`):
```python
class BuildkiteLauncher(Launcher):
    def __init__(self, org: str, pipeline: str, token: str):
        super().__init__(name="Buildkite", gpus=BuildkiteGPU)
        # ...

    async def run_submission(self, config, gpu_type, status) -> FullResult:
        # 1. Compress config (zlib + base64)
        # 2. Create build via Buildkite API
        # 3. Poll for completion
        # 4. Download artifacts
        # 5. Parse result.json -> FullResult
```

**Agent Setup** (`scripts/buildkite/setup-agent.sh`):
```bash
# Creates per-GPU systemd service with:
Environment="CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
Environment="BUILDKITE_AGENT_TAGS=queue=${QUEUE_NAME}"
Slice=buildkite-gpu${GPU_INDEX}.slice  # cgroup isolation
```

---

## Testing

### Unit Tests (All Passing)

```bash
uv run pytest tests/test_buildkite.py -v
```

```
tests/test_buildkite.py::TestBuildkiteGPU::test_enum_values PASSED
tests/test_buildkite.py::TestBuildkiteGPU::test_scheduler_type_exists PASSED
tests/test_buildkite.py::TestBuildkiteGPU::test_gpu_lookup PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_init PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_headers PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_payload_compression PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_run_submission_creates_build PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_run_submission_handles_api_error PASSED
tests/test_buildkite.py::TestBuildkiteLauncher::test_status_updates PASSED
tests/test_buildkite.py::TestBuildkiteRunner::test_runner_script_syntax PASSED

============================== 10 passed ==============================
```

### Import/Integration Tests

```bash
# Verify imports work
uv run python -c "
from libkernelbot.launchers import BuildkiteLauncher
from libkernelbot.consts import BuildkiteGPU, get_gpu_by_name

launcher = BuildkiteLauncher(org='test', pipeline='test', token='fake')
print(f'Launcher: {launcher.name}')
print(f'GPUs: {[g.value for g in BuildkiteGPU]}')
"
```

Output:
```
Launcher: Buildkite
GPUs: ['nvidia-h100', 'nvidia-b200', 'nvidia-a100', 'amd-mi300', 'amd-mi250', 'google-tpu', 'nebius-h100']
```

### Local Container Test (For Vendors)

```bash
# Build image
docker build -t kernelbot-runner:test -f docker/kernelbot-runner/Dockerfile .

# Test with single GPU
docker run --rm --gpus '"device=0"' \
  -e SUBMISSION_PAYLOAD="$(python3 -c '
import json, zlib, base64
config = {"lang": "py", "mode": "test", "files": {"main.py": "import torch; print(torch.cuda.get_device_name(0))"}, "tests": [], "benchmarks": [], "test_timeout": 60, "benchmark_timeout": 60, "ranked_timeout": 60}
print(base64.b64encode(zlib.compress(json.dumps(config).encode())).decode())
')" \
  kernelbot-runner:test
```

---

## Vendor Onboarding Flow

### For Vendors

1. **Get Buildkite token** from Kernelbot team
2. **Clone repo**: `git clone https://github.com/gpu-mode/kernelbot.git`
3. **Pull image**: `docker pull ghcr.io/gpu-mode/kernelbot-runner:latest`
4. **Run setup script** for each GPU:
   ```bash
   sudo ./scripts/buildkite/setup-agent.sh 0 nvidia-h100-0 32G 16
   sudo ./scripts/buildkite/setup-agent.sh 1 nvidia-h100-1 32G 16
   # ... for all GPUs
   ```
5. **Set token**: Edit `/etc/buildkite-agent/token`
6. **Start agents**: `sudo systemctl start 'buildkite-agent-gpu*'`
7. **Verify**: Check Buildkite dashboard for connected agents

### For Kernelbot Team

1. Set env vars:
   ```bash
   BUILDKITE_API_TOKEN=bkua_xxxxx
   BUILDKITE_ORG=gpu-mode
   BUILDKITE_PIPELINE=kernelbot-runner
   ```
2. Launcher auto-registers if token is set
3. Add GPU types to leaderboard configs

---

## Next Steps

### Immediate (For Integration Testing)

- [ ] Create Buildkite organization and pipeline
- [ ] Set up 1 test agent on a GPU machine
- [ ] Run end-to-end test via API
- [ ] Compare benchmark results with Modal

### Before Production

- [ ] Set up GitHub Container Registry for image
- [ ] Configure Slack webhook for vendor notifications
- [ ] Test with multiple concurrent jobs
- [ ] Document SLA expectations for vendors

### Future Enhancements

- [ ] Webhook-based completion (instead of polling)
- [ ] Agent health monitoring dashboard
- [ ] Automatic image version checking
- [ ] Support for non-NVIDIA GPUs (TPU, AMD)

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BUILDKITE_API_TOKEN` | Yes | - | Buildkite API token |
| `BUILDKITE_ORG` | No | `gpu-mode` | Buildkite org slug |
| `BUILDKITE_PIPELINE` | No | `kernelbot-runner` | Pipeline slug |

### GPU Types (BuildkiteGPU Enum)

| Enum Name | Queue Value | Description |
|-----------|-------------|-------------|
| `NVIDIA_H100` | `nvidia-h100` | NVIDIA H100 |
| `NVIDIA_B200` | `nvidia-b200` | NVIDIA B200 |
| `NVIDIA_A100` | `nvidia-a100` | NVIDIA A100 |
| `AMD_MI300` | `amd-mi300` | AMD MI300 |
| `AMD_MI250` | `amd-mi250` | AMD MI250 |
| `GOOGLE_TPU` | `google-tpu` | Google TPU |
| `NEBIUS_H100` | `nebius-h100` | Nebius H100 |

---

## Files Reference

```
kernelbot/
├── .buildkite/
│   └── pipeline.yml                    # Buildkite pipeline
├── .github/workflows/
│   └── build-runner-image.yml          # Auto-build Docker image
├── docker/kernelbot-runner/
│   ├── Dockerfile                      # Runner container
│   └── requirements-runner.txt         # Python deps
├── docs/docs/vendor-onboarding/
│   ├── buildkite.md                    # Vendor setup guide
│   └── testing-guide.md                # Testing instructions
├── scripts/buildkite/
│   ├── setup-agent.sh                  # Agent bootstrap
│   └── update-image.sh                 # Image updater
├── src/
│   ├── kernelbot/
│   │   ├── env.py                      # +Buildkite env vars
│   │   └── main.py                     # +Register launcher
│   ├── libkernelbot/
│   │   ├── consts.py                   # +BuildkiteGPU enum
│   │   └── launchers/
│   │       ├── __init__.py             # +Export
│   │       └── buildkite.py            # BuildkiteLauncher
│   └── runners/
│       └── buildkite-runner.py         # Runner script
└── tests/
    └── test_buildkite.py               # Unit tests
```

---

## Contact

- **Implementation**: [Your name]
- **Questions**: #kernelbot-infra on Discord
- **Issues**: https://github.com/gpu-mode/kernelbot/issues
