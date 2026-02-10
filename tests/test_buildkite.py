"""Integration tests for Buildkite launcher.

Usage:
    BUILDKITE_API_TOKEN=xxx pytest tests/test_buildkite.py -v -m integration

These tests require:
1. A Buildkite account with a 'kernelbot' pipeline
2. A self-hosted runner in the 'test' queue
3. The pipeline configured with deployment/buildkite/pipeline-eval.yml
"""

import os
from pathlib import Path

import pytest

from libkernelbot.consts import BuildkiteGPU, SubmissionMode
from libkernelbot.launchers.buildkite import BuildkiteConfig, BuildkiteLauncher
from libkernelbot.report import RunProgressReporter
from libkernelbot.task import build_task_config, make_task_definition


class MockProgressReporter(RunProgressReporter):
    """Test progress reporter that captures messages."""

    def __init__(self, title: str = "Test Buildkite Run"):
        super().__init__(title)
        self.messages = []
        self.updates = []

    async def push(self, message: str):
        self.messages.append(message)
        print(f"[STATUS] {message}")

    async def update(self, message: str):
        self.updates.append(message)
        print(f"[UPDATE] {message}")


@pytest.fixture(scope="session")
def buildkite_config():
    """Get Buildkite configuration from environment."""
    token = os.getenv("BUILDKITE_API_TOKEN")
    if not token:
        pytest.skip("Buildkite integration tests require BUILDKITE_API_TOKEN environment variable")

    org = os.getenv("BUILDKITE_ORG", "mark-saroufim")
    pipeline = os.getenv("BUILDKITE_PIPELINE", "kernelbot")

    return BuildkiteConfig(
        org_slug=org,
        pipeline_slug=pipeline,
        api_token=token,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_type", [BuildkiteGPU.L40S_BK])
async def test_buildkite_launcher_python_script(
    project_root: Path, buildkite_config: BuildkiteConfig, gpu_type: BuildkiteGPU
):
    """
    Test BuildkiteLauncher with a real Python script.
    Uses the vectoradd_py example to verify end-to-end evaluation.
    """
    launcher = BuildkiteLauncher(buildkite_config)
    reporter = MockProgressReporter("Buildkite Integration Test")

    # Load the vectoradd_py task
    task_path = project_root / "examples" / "vectoradd_py"
    if not task_path.exists():
        pytest.skip("examples/vectoradd_py not found - skipping Buildkite integration test")

    task_definition = make_task_definition(task_path)
    submission_content = (task_path / "submission_triton.py").read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=0,  # L40S uses Ada Lovelace architecture
        mode=SubmissionMode.TEST,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info
    assert "L40S" in result.system.gpu or "NVIDIA" in result.system.gpu
    assert "Linux" in result.system.platform

    # Test run structure
    assert "test" in result.runs
    test_run = result.runs["test"]

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is True
    assert test_run.run.exit_code == 0
    assert test_run.run.duration > 0

    # Test results
    assert test_run.run.result["check"] == "pass"
    test_count = int(test_run.run.result["test-count"])
    assert test_count >= 1

    # Sanity check for timings
    assert test_run.start < test_run.end

    # Check reporter messages
    assert any("Buildkite" in msg or "queue" in msg for msg in reporter.messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_buildkite_launcher_failing_script(
    project_root: Path, buildkite_config: BuildkiteConfig
):
    """
    Test BuildkiteLauncher with a script designed to fail.
    Ensures we don't pass incorrect submissions.
    """
    launcher = BuildkiteLauncher(buildkite_config)
    reporter = MockProgressReporter("Buildkite Failing Test")
    gpu_type = BuildkiteGPU.L40S_BK

    # Load the identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping Buildkite integration test")

    task_definition = make_task_definition(task_path)
    # Use a cheating script that should fail
    submission_content = (task_path / "cheat-rng.py").read_text()

    task_definition.task.seed = 653212
    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=0,
        mode=SubmissionMode.LEADERBOARD,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # The workflow should run successfully
    assert result.success, f"Expected successful workflow run, got: {result.error}"
    assert result.error == ""

    # But the actual test or benchmark should fail
    test_passed = result.runs.get("test", {}).run.passed if "test" in result.runs else True
    benchmark_passed = result.runs.get("benchmark", {}).run.passed if "benchmark" in result.runs else True

    assert not (test_passed and benchmark_passed), "Expected at least one run to fail for cheating script"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_buildkite_queue_status(buildkite_config: BuildkiteConfig):
    """Test that we can query queue status."""
    launcher = BuildkiteLauncher(buildkite_config)

    status = await launcher.get_queue_status("test")

    assert "queue" in status
    assert status["queue"] == "test"
    assert "total" in status
    assert "idle" in status
    assert "agents" in status
    assert isinstance(status["agents"], list)
