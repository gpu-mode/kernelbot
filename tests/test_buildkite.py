"""Tests for BuildkiteLauncher."""

import base64
import json
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libkernelbot.consts import BuildkiteGPU, GPU, SchedulerType, get_gpu_by_name
from libkernelbot.launchers import BuildkiteLauncher
from libkernelbot.report import RunProgressReporter


class MockProgressReporter(RunProgressReporter):
    """Test progress reporter that captures messages."""

    def __init__(self, title: str = "Test Buildkite Run"):
        super().__init__(title)
        self.messages = []
        self.updates = []

    async def push(self, message: str):
        self.messages.append(message)

    async def update(self, message: str):
        self.updates.append(message)


class TestBuildkiteGPU:
    """Tests for BuildkiteGPU enum."""

    def test_enum_values(self):
        """Test that BuildkiteGPU has expected values."""
        assert BuildkiteGPU.NVIDIA_H100.value == "nvidia-h100"
        assert BuildkiteGPU.NVIDIA_B200.value == "nvidia-b200"
        assert BuildkiteGPU.AMD_MI300.value == "amd-mi300"
        assert BuildkiteGPU.GOOGLE_TPU.value == "google-tpu"

    def test_scheduler_type_exists(self):
        """Test that BUILDKITE scheduler type exists."""
        assert SchedulerType.BUILDKITE.value == "buildkite"

    def test_gpu_lookup(self):
        """Test that Buildkite GPUs are in the lookup table."""
        gpu = get_gpu_by_name("nvidia_h100")
        assert gpu is not None
        assert gpu.value == "nvidia-h100"
        assert gpu.runner == "Buildkite"

        gpu = get_gpu_by_name("amd_mi300")
        assert gpu is not None
        assert gpu.value == "amd-mi300"
        assert gpu.runner == "Buildkite"


class TestBuildkiteLauncher:
    """Tests for BuildkiteLauncher class."""

    @pytest.fixture
    def launcher(self):
        return BuildkiteLauncher(
            org="test-org",
            pipeline="test-pipeline",
            token="test-token",
        )

    @pytest.fixture
    def mock_config(self):
        return {
            "lang": "py",
            "mode": "test",
            "files": {"main.py": "print('hello')"},
            "tests": [],
            "benchmarks": [],
            "test_timeout": 180,
            "benchmark_timeout": 180,
            "ranked_timeout": 180,
        }

    @pytest.fixture
    def gpu_type(self):
        return GPU(name="NVIDIA_H100", value="nvidia-h100", runner="Buildkite")

    def test_init(self, launcher):
        """Test launcher initialization."""
        assert launcher.name == "Buildkite"
        assert launcher.org == "test-org"
        assert launcher.pipeline == "test-pipeline"
        assert launcher.gpus == BuildkiteGPU

    def test_headers(self, launcher):
        """Test API headers are set correctly."""
        assert "Authorization" in launcher._headers
        assert launcher._headers["Authorization"] == "Bearer test-token"
        assert launcher._headers["Content-Type"] == "application/json"

    def test_payload_compression(self, mock_config):
        """Test that payload compression/decompression works."""
        # Compress (same logic as launcher)
        payload = base64.b64encode(
            zlib.compress(json.dumps(mock_config).encode("utf-8"))
        ).decode("utf-8")

        # Decompress (same logic as runner)
        decompressed = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
        restored = json.loads(decompressed)

        assert restored == mock_config

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

        with patch("libkernelbot.launchers.buildkite.requests.post", return_value=mock_response) as mock_post:
            with patch.object(launcher, "_wait_for_completion", new_callable=AsyncMock):
                with patch.object(launcher, "_download_and_parse_result", new_callable=AsyncMock) as mock_download:
                    mock_download.return_value = MagicMock(success=True)

                    reporter = MockProgressReporter()
                    result = await launcher.run_submission(mock_config, gpu_type, reporter)

                    # Verify API was called
                    mock_post.assert_called_once()
                    call_args = mock_post.call_args

                    # Check URL contains org and pipeline
                    url = call_args[0][0]
                    assert "test-org" in url
                    assert "test-pipeline" in url

                    # Check payload was compressed and queue set
                    body = call_args[1]["json"]
                    assert "SUBMISSION_PAYLOAD" in body["env"]
                    assert body["env"]["GPU_QUEUE"] == "nvidia-h100"

    @pytest.mark.asyncio
    async def test_run_submission_handles_api_error(self, launcher, mock_config, gpu_type):
        """Test that API errors are handled gracefully."""
        import requests

        with patch("libkernelbot.launchers.buildkite.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("API Error")

            reporter = MockProgressReporter()
            result = await launcher.run_submission(mock_config, gpu_type, reporter)

            assert result.success is False
            assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_status_updates(self, launcher, mock_config, gpu_type):
        """Test that status updates are sent correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "number": 456,
            "web_url": "https://buildkite.com/test/builds/456",
            "state": "scheduled",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("libkernelbot.launchers.buildkite.requests.post", return_value=mock_response):
            with patch.object(launcher, "_wait_for_completion", new_callable=AsyncMock):
                with patch.object(launcher, "_download_and_parse_result", new_callable=AsyncMock) as mock_download:
                    mock_download.return_value = MagicMock(success=True)

                    reporter = MockProgressReporter()
                    await launcher.run_submission(mock_config, gpu_type, reporter)

                    # Check status messages were sent
                    assert any("456" in msg for msg in reporter.messages)
                    assert any("completed" in msg.lower() for msg in reporter.updates)


class TestBuildkiteRunner:
    """Tests for buildkite-runner.py script."""

    def test_runner_script_syntax(self):
        """Test that runner script has valid Python syntax."""
        import py_compile
        from pathlib import Path

        runner_path = Path(__file__).parent.parent / "src" / "runners" / "buildkite-runner.py"
        # This will raise SyntaxError if invalid
        py_compile.compile(str(runner_path), doraise=True)
