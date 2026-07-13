import os
import pprint
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libkernelbot.consts import GPU_TO_SM, ModalGPU, SubmissionMode, get_gpu_by_name
from libkernelbot.launchers import ModalLauncher
from libkernelbot.report import RunProgressReporter
from libkernelbot.task import build_task_config, make_task_definition


class MockProgressReporter(RunProgressReporter):
    """Test progress reporter that captures messages."""

    def __init__(self, title: str = "Test Modal Run"):
        super().__init__(title)
        self.messages = []
        self.updates = []

    async def push(self, message: str):
        self.messages.append(message)

    async def update(self, message: str):
        self.updates.append(message)


@pytest.mark.asyncio
async def test_modal_submission_uses_native_async_api():
    launcher = ModalLauncher(add_include_dirs=["/extra/include"])
    reporter = MockProgressReporter()
    function = MagicMock()
    expected = object()
    function.remote.aio = AsyncMock(return_value=expected)
    config = {"lang": "cu"}

    with patch("libkernelbot.launchers.modal.modal.Function.from_name", return_value=function):
        result = await launcher.run_submission(config, get_gpu_by_name("B200"), reporter)

    assert result is expected
    assert config["include_dirs"] == ["/extra/include"]
    function.remote.aio.assert_awaited_once_with(config=config)
    function.remote.assert_not_called()
    assert reporter.messages == ["⏳ Waiting for Modal run to finish..."]
    assert reporter.updates == ["✅ Waiting for modal run to finish... Done"]


@pytest.mark.asyncio
async def test_modal_queue_status_uses_function_stats():
    launcher = ModalLauncher(add_include_dirs=[])
    function = MagicMock()
    function.get_current_stats.return_value = SimpleNamespace(
        backlog=5,
        num_total_runners=2,
    )

    with patch("libkernelbot.launchers.modal.modal.Function.from_name", return_value=function):
        status = await launcher.get_queue_status(get_gpu_by_name("B200"), {"lang": "cu"})

    assert status.runner == "Modal"
    assert status.gpu == "B200"
    assert status.queued_jobs == 5
    assert status.available_runners == 2


@pytest.mark.asyncio
async def test_modal_queue_status_reports_unavailable_on_error():
    launcher = ModalLauncher(add_include_dirs=[])

    with patch(
        "libkernelbot.launchers.modal.modal.Function.from_name",
        side_effect=RuntimeError("modal unavailable"),
    ):
        status = await launcher.get_queue_status(get_gpu_by_name("B200"), {"lang": "cu"})

    assert status.runner == "Modal"
    assert status.queued_jobs is None
    assert status.status == "unavailable"
    assert status.error == "modal unavailable"


@pytest.fixture(scope="session")
def modal_deployment(project_root: Path):
    """
    Fixture that ensures Modal is deployed before running tests.
    Runs once per test session and deploys to the specified Modal environment.
    """
    # Determine Modal environment (default to 'test' if not specified)
    modal_env = os.getenv("PYTEST_MODAL_ENV", "pytest")

    print(f"🚀 Deploying to Modal environment: {modal_env}")

    # Deploy to Modal with specific environment
    try:
        result = subprocess.run(
            ["modal", "deploy", "--env", modal_env, "modal_runner_archs.py"],
            cwd=project_root / "src" / "runners",
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout in case image needs to be built (can be very slow)
        )

        if result.returncode != 0:
            # if it fails simply because the environment does not exist, we can fix  that
            if "No such environment" in result.stderr or "not found" in result.stderr:
                result = subprocess.run(
                    ["modal", "environment", "create", modal_env],
                    cwd=project_root / "src" / "runners",
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    pytest.fail(
                        f"Modal environment `{modal_env}` not available, "
                        f"and failed to create: {result.stderr}"
                    )
                else:
                    # try again, now that the env exists.
                    result = subprocess.run(
                        ["modal", "deploy", "--env", modal_env, "modal_runner_archs.py"],
                        cwd=project_root / "src" / "runners",
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode != 0:
                        pytest.fail(
                            f"Modal deploy failed:\n"
                            f"STDOUT:\n{result.stdout}\n"
                            f"STDERR:\n{result.stderr}"
                        )
            else:
                pytest.fail(
                    f"Modal deploy failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

        print(f"✅ Modal deployment to '{modal_env}' completed successfully")
        print(f"Deploy output: {result.stdout}")

        # Set the Modal environment for the session
        original_env = os.environ.get("MODAL_ENVIRONMENT")
        os.environ["MODAL_ENVIRONMENT"] = modal_env

        yield modal_env

        # Restore original environment
        if original_env is not None:
            os.environ["MODAL_ENVIRONMENT"] = original_env
        elif "MODAL_ENVIRONMENT" in os.environ:
            del os.environ["MODAL_ENVIRONMENT"]

    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"Modal deploy timed out after 5 minutes:\nstdout: {e.stdout}, stderr:{e.stderr}"
        )
    except Exception as e:
        pytest.fail(f"Modal deploy failed with exception: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu_type", [ModalGPU.T4, ModalGPU.H100]  # Reduced from 5 GPUs to 2 for faster CI
)
@pytest.mark.parametrize(
    "task",
    [
        ("vectoradd_py", "submission_cuda_inline.py"),
        ("vectoradd_py", "submission_triton.py"),
    ],
)
async def test_modal_launcher_python_script(
    modal_deployment, project_root: Path, gpu_type: ModalGPU, task: Tuple[str, str]
):
    """
    Test ModalLauncher with a real Python script using examples/identity_py.
    """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")

    # Load the real identity_py task
    task_path = project_root / "examples" / task[0]
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping Modal integration test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / task[1]).read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[gpu_type.name],
        mode=SubmissionMode.TEST,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info - test actual expected values
    assert gpu_type.name in result.system.gpu
    assert "Linux" in result.system.platform

    # Test run structure
    assert "test" in result.runs
    test_run = result.runs["test"]

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is True
    assert test_run.run.exit_code == 0
    assert test_run.run.duration > 0

    # Test need to succeed
    assert test_run.run.result["check"] == "pass"
    test_count = int(test_run.run.result["test-count"])
    assert test_count == 5
    for i in range(test_count):
        assert test_run.run.result[f"test.{i}.status"] == "pass"
        assert "size:" in test_run.run.result[f"test.{i}.spec"]
        assert "seed:" in test_run.run.result[f"test.{i}.spec"]

    # sanity check for timings
    assert test_run.start < test_run.end

    # check messages
    assert reporter.messages == ["⏳ Waiting for Modal run to finish..."]
    assert reporter.updates == ["✅ Waiting for modal run to finish... Done"]


@pytest.mark.skip(reason="Multi-GPU L4x4 NCCL issues on Modal infrastructure")
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("script, good", [("submission.py", True), ("wrong.py", False)])
async def test_modal_multi_gpu(modal_deployment, project_root: Path, script: str, good: bool):
    """
    This isn't really a modal test, but instead a test using modal to check
    that multi-gpu submission testing works (on modal...).
    """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")

    # Load the real identity_py task
    task_path = project_root / "examples" / "gather"
    if not task_path.exists():
        pytest.skip("examples/gather not found - skipping Modal multi-gpu test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / script).read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[ModalGPU.L4x4.name],
        mode=SubmissionMode.TEST,
    )

    result = await launcher.run_submission(config, ModalGPU.L4x4, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info - test actual expected values
    pprint.pprint(result)
    assert result.system.device_count == 4

    # Test run structure
    assert "test" in result.runs
    test_run = result.runs["test"]

    # For Python runs, compilation is None
    assert test_run.compilation is None

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is good


@pytest.mark.skip(reason="Multi-GPU L4x4 NCCL issues on Modal infrastructure")
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("script, good", [("submission.py", True), ("wrong.py", False)])
async def test_modal_multi_gpu_benchmark(
    modal_deployment, project_root: Path, script: str, good: bool
):
    """
    This isn't really a modal test, but instead a test using modal
    to check that multi-gpu submission testing works (on modal...).
    """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")

    # Load the real identity_py task
    task_path = project_root / "examples" / "gather"
    if not task_path.exists():
        pytest.skip("examples/gather not found - skipping Modal multi-gpu test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / script).read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[ModalGPU.L4x4.name],
        mode=SubmissionMode.BENCHMARK,
    )

    result = await launcher.run_submission(config, ModalGPU.L4x4, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info - test actual expected values
    pprint.pprint(result)
    assert result.system.device_count == 4

    # Test run structure
    assert "benchmark" in result.runs
    bench_run = result.runs["benchmark"]

    # For Python runs, compilation is None
    assert bench_run.compilation is None

    # Run needs to succeed
    assert bench_run.run.success is True
    assert bench_run.run.passed is good


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["cheat-fd.py", "cheat-input.py", "cheat-rng.py"])
async def test_modal_launcher_failing_script(modal_deployment, project_root: Path, script: str):
    """
    Test ModalLauncher with a real Python scripts that are designed to be wrong.
    """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")
    gpu_type = ModalGPU.T4

    # Load the real identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping Modal integration test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / script).read_text()
    task_definition.task.seed = 653212
    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[gpu_type.name],
        mode=SubmissionMode.LEADERBOARD,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert result.runs["test"].run.passed is False or result.runs["benchmark"].run.passed is False
