import os
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest
from dotenv import load_dotenv

from libkernelbot.consts import GitHubGPU, SubmissionMode
from libkernelbot.launchers import GitHubLauncher
from libkernelbot.report import RunProgressReporter
from libkernelbot.task import build_task_config, make_task_definition
from libkernelbot.utils import get_github_branch_name

# Named tuple for better readability
GitHubConfig = namedtuple('GitHubConfig', ['token', 'repo', 'branch'])


class MockProgressReporter(RunProgressReporter):
    """Test progress reporter that captures messages."""

    def __init__(self, title: str = "Test GitHub Run"):
        super().__init__(title)
        self.messages = []
        self.updates = []

    async def push(self, message: str):
        self.messages.append(message)

    async def update(self, message: str):
        self.updates.append(message)


def get_github_repo():
    """Get GitHub repository from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        remote_url = result.stdout.strip()

        # Parse GitHub repo from remote URL
        # Handle both SSH and HTTPS formats
        if remote_url.startswith("git@github.com:"):
            repo = remote_url.replace("git@github.com:", "").replace(".git", "")
        elif remote_url.startswith("https://github.com/"):
            repo = remote_url.replace("https://github.com/", "").replace(".git", "")
        else:
            return None

        return repo
    except subprocess.CalledProcessError:
        return None


@pytest.fixture(scope="session")
def github_config():
    """
    Get GitHub test configuration from environment or git.
    Skips tests if required configuration is missing.
    """
    # Load .env file if it exists
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO") or get_github_repo()
    branch = os.getenv("GITHUB_BRANCH") or get_github_branch_name()

    if not token:
        pytest.skip("GitHub integration tests require GITHUB_TOKEN environment variable")

    if not repo:
        pytest.skip(
            "GitHub integration tests require GITHUB_REPO environment variable "
            "or a valid git remote origin"
        )

    return GitHubConfig(token=token, repo=repo, branch=branch)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_type", [GitHubGPU.NVIDIA])
async def test_github_launcher_python_script(project_root: Path, github_config: GitHubConfig, gpu_type: GitHubGPU):
    """
    Test GitHubLauncher with a real Python script using real GitHub Actions.
    Tests all GPU types to verify runners are working.
    """
    launcher = GitHubLauncher(repo=github_config.repo, token=github_config.token, branch=github_config.branch)
    reporter = MockProgressReporter("GitHub Integration Test")

    # Load the real identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping GitHub integration test")

    task_definition = make_task_definition(task_path)
    submission_content = (task_path / "submission.py").read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=0,  # Not used for GitHub launcher
        mode=SubmissionMode.TEST,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info - test actual expected values based on GPU type
    if gpu_type == GitHubGPU.NVIDIA:
        assert "NVIDIA" in result.system.gpu or "GeForce" in result.system.gpu or "RTX" in result.system.gpu
    else:  # AMD GPUs
        assert "MI" in result.system.gpu or "AMD" in result.system.gpu

    assert "Linux" in result.system.platform

    # Test run structure
    assert "test" in result.runs
    test_run = result.runs["test"]

    # For Python runs, compilation is None
    assert test_run.compilation is None

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is True
    assert test_run.run.exit_code == 0
    assert test_run.run.duration > 0

    # Test results need to succeed
    assert test_run.run.result["check"] == "pass"
    test_count = int(test_run.run.result["test-count"])
    assert test_count == 5
    for i in range(test_count):
        assert test_run.run.result[f"test.{i}.status"] == "pass"
        assert "size:" in test_run.run.result[f"test.{i}.spec"]
        assert "seed:" in test_run.run.result[f"test.{i}.spec"]

    # Sanity check for timings
    assert test_run.start < test_run.end

    # Check reporter messages
    assert any("Waiting for workflow" in msg for msg in reporter.messages)
    assert any("artifacts" in msg.lower() for msg in reporter.messages)
    assert any("completed" in update for update in reporter.updates)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_github_launcher_failing_script(project_root: Path, github_config: GitHubConfig):
    """
    Test GitHubLauncher with a script designed to fail.
    Simple test to ensure we don't pass wrong submissions.
    """
    launcher = GitHubLauncher(repo=github_config.repo, token=github_config.token, branch=github_config.branch)
    reporter = MockProgressReporter("GitHub Failing Test")
    gpu_type = GitHubGPU.NVIDIA  # Use NVIDIA for simplicity

    # Load the real identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping GitHub integration test")

    task_definition = make_task_definition(task_path)
    # Use one of the cheating scripts
    submission_content = (task_path / "cheat-rng.py").read_text()

    # Set a specific seed for reproducible results
    task_definition.task.seed = 653212
    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=0,
        mode=SubmissionMode.PUBLIC,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure should still be successful (the workflow ran)
    assert result.success, f"Expected successful workflow run, got: {result.error}"
    assert result.error == ""

    # But the actual test or benchmark should fail
    test_passed = result.runs.get("test", {}).run.passed if "test" in result.runs else True
    private_passed = result.runs.get("private", {}).run.passed if "private" in result.runs else True

    assert not (test_passed and private_passed), "Expected at least one run to fail for cheating script"




@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="MI300x8 machines are no longer available")
@pytest.mark.parametrize("gpu_type", [GitHubGPU.MI300x8])
async def test_github_launcher_multi_gpu(project_root: Path, github_config: GitHubConfig, gpu_type: GitHubGPU):
    """
    Test GitHubLauncher with a real Python script using real GitHub Actions.
    Tests all GPU types to verify runners are working.

    Note: This test is skipped because MI300x8 machines are no longer available.
    """
    launcher = GitHubLauncher(repo=github_config.repo, token=github_config.token, branch=github_config.branch)
    reporter = MockProgressReporter("GitHub Integration Test")

    # Load the real identity_py task
    task_path = project_root / "examples" / "gather"
    if not task_path.exists():
        pytest.skip("examples/gather not found - skipping GitHub integration test")

    task_definition = make_task_definition(task_path)
    submission_content = (task_path / "submission.py").read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=0,  # Not used for GitHub launcher
        mode=SubmissionMode.TEST,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    assert result.system.device_count == 8

    # Test run structure
    assert "test" in result.runs
    test_run = result.runs["test"]

    # For Python runs, compilation is None
    assert test_run.compilation is None

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is True
    assert test_run.run.exit_code == 0
    assert test_run.run.duration > 0

    # Test results need to succeed
    assert test_run.run.result["check"] == "pass"
    test_count = int(test_run.run.result["test-count"])
    assert test_count == 1
    for i in range(test_count):
        assert test_run.run.result[f"test.{i}.status"] == "pass"
        assert "size:" in test_run.run.result[f"test.{i}.spec"]
        assert "seed:" in test_run.run.result[f"test.{i}.spec"]

    # Sanity check for timings
    assert test_run.start < test_run.end

    # Check reporter messages
    assert any("Waiting for workflow" in msg for msg in reporter.messages)
    assert any("artifacts" in msg.lower() for msg in reporter.messages)
    assert any("completed" in update for update in reporter.updates)
