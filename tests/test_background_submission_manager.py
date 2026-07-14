import asyncio
import datetime
from unittest import mock

import pytest
from test_report import create_eval_result, sample_system_info

from libkernelbot.background_submission_manager import BackgroundSubmissionManager
from libkernelbot.consts import SubmissionMode
from libkernelbot.kernelguard import KernelGuardRejected
from libkernelbot.run_eval import FullResult
from libkernelbot.submission import ProcessedSubmissionRequest
from libkernelbot.task import make_task_definition


@pytest.fixture
def mock_backend():
    backend = mock.Mock()
    backend.accepts_jobs = True

    # Mock database context manager
    db_context = mock.Mock()
    backend.db = db_context
    db_context.__enter__ = mock.Mock(return_value=db_context)
    db_context.__exit__ = mock.Mock(return_value=None)

    # Default mock responses
    mock_task = mock.Mock()
    db_context.get_leaderboard.return_value = {
        "task": mock_task,
        "secret_seed": 12345,
        "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
        "name": "test_board",
    }
    db_context.get_leaderboard_gpu_types.return_value = ["A100", "V100"]

    return backend


def get_req(i: int) -> ProcessedSubmissionRequest:
    return ProcessedSubmissionRequest(
        leaderboard="lb",
        task="dummy_task",
        secret_seed=12345,
        task_gpus=["A100"],
        file_name=f"f{i}.py",
        code="print('hi')",
        user_id=1,
        user_name="tester",
        gpus=None,
    )


def _full_result(*modes: str, failing_mode: str | None = None) -> FullResult:
    runs = {}
    for mode in modes:
        eval_mode = "test" if mode == "test" else "benchmark"
        eval_result = create_eval_result(eval_mode)
        if mode == failing_mode:
            eval_result.run.passed = False
        runs[mode] = eval_result
    return FullResult(success=True, error="", system=sample_system_info(), runs=runs)


@pytest.mark.asyncio
async def test_enqueue_and_run_job(mock_backend):
    # mock upsert/update
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    submit_calls = []

    # mock submit_full
    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        submit_calls.append((sub_id, skip_precheck))
        await asyncio.sleep(0.01)  # simulate a long-running job
        return None, None

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=2, idle_seconds=0.1
    )
    await manager.start()

    # create a fake submission request
    job_id, sub_id = await manager.enqueue(get_req(1), SubmissionMode.TEST, sub_id=42)
    assert job_id == 42

    # wait for the queue is clear
    await manager.queue.join()
    await asyncio.sleep(0.05)

    # check db status
    assert (
        mock.call(42, status="pending", last_heartbeat=mock.ANY)
        in db_context.upsert_submission_job_status.call_args_list
    )
    assert (
        mock.call(42, status="running", last_heartbeat=mock.ANY)
        in db_context.upsert_submission_job_status.call_args_list
    )
    assert (
        mock.call(42, status="succeeded", last_heartbeat=mock.ANY)
        in db_context.upsert_submission_job_status.call_args_list
    )
    assert submit_calls == [(42, False)]

    await manager.stop()


@pytest.mark.asyncio
async def test_accepted_job_survives_request_task_cancellation(mock_backend):
    """Disconnecting an accepted request must not cancel its background job."""
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(side_effect=lambda *args, **kwargs: args[0])
    db_context.update_heartbeat_if_active = mock.Mock()
    started = asyncio.Event()
    release = asyncio.Event()
    backend_cancelled = False

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        nonlocal backend_cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            backend_cancelled = True
            raise
        return None, None

    mock_backend.submit_full = fake_submit_full
    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()
    request_accepted = asyncio.Event()

    async def simulated_request():
        await manager.enqueue(get_req(1), SubmissionMode.BENCHMARK, sub_id=42)
        request_accepted.set()
        await asyncio.Event().wait()

    request_task = asyncio.create_task(simulated_request())
    try:
        await asyncio.wait_for(request_accepted.wait(), timeout=1)
        await asyncio.wait_for(started.wait(), timeout=1)

        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        assert not backend_cancelled
        release.set()
        await asyncio.wait_for(manager.queue.join(), timeout=1)

        assert not backend_cancelled
        assert (
            mock.call(42, status="succeeded", last_heartbeat=mock.ANY)
            in db_context.upsert_submission_job_status.call_args_list
        )
    finally:
        release.set()
        if not request_task.done():
            request_task.cancel()
        await manager.stop()


@pytest.mark.asyncio
async def test_leaderboard_secret_failure_marks_job_failed(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    public_result = _full_result("test", "benchmark", "leaderboard")
    secret_result = _full_result("test", "benchmark", failing_mode="benchmark")

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        return sub_id, [public_result, secret_result]

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()

    req = get_req(1)
    req.gpus = ["A100"]
    await manager.enqueue(req, SubmissionMode.LEADERBOARD, sub_id=42)
    await manager.queue.join()

    assert (
        mock.call(
            42,
            status="failed",
            last_heartbeat=mock.ANY,
            error="secret validation failed on A100; submission will not appear on the leaderboard",
        )
        in db_context.upsert_submission_job_status.call_args_list
    )

    await manager.stop()


@pytest.mark.asyncio
async def test_stop_rejects_new_jobs(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(return_value=1)
    db_context.update_heartbeat_if_active = mock.Mock()
    mock_backend.submit_full = mock.AsyncMock()

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()
    await manager.stop()

    req = get_req(1)
    with pytest.raises(RuntimeError):
        await manager.enqueue(req, SubmissionMode.TEST, 99)


@pytest.mark.asyncio
async def test_stop_marks_running_job_failed(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(side_effect=lambda *args, **kwargs: args[0])
    db_context.update_heartbeat_if_active = mock.Mock()
    db_context.fail_submission_job_if_active = mock.Mock(return_value=True)
    started = asyncio.Event()

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        started.set()
        await asyncio.Event().wait()

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()
    await manager.enqueue(get_req(1), SubmissionMode.TEST, sub_id=99)
    await asyncio.wait_for(started.wait(), timeout=1)

    await manager.stop()

    db_context.fail_submission_job_if_active.assert_called_once_with(
        99,
        "job interrupted while kernelbot was shutting down; please resubmit",
        mock.ANY,
    )


@pytest.mark.asyncio
async def test_stop_marks_real_database_job_failed(database, task_directory):
    definition = make_task_definition(task_directory / "task.yml")
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    with database as db:
        db.create_leaderboard(
            name="shutdown-test",
            deadline=now + datetime.timedelta(days=1),
            definition=definition,
            creator_id=1,
            forum_id=1,
            gpu_types=["A100"],
        )
        sub_id = db.create_submission(
            "shutdown-test",
            "submission.py",
            1,
            "print('hello')",
            now,
            user_name="shutdown-test-user",
        )

    backend = mock.Mock()
    backend.db = database
    started = asyncio.Event()

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        started.set()
        await asyncio.Event().wait()

    backend.submit_full = fake_submit_full
    manager = BackgroundSubmissionManager(
        backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()
    await manager.enqueue(get_req(1), SubmissionMode.TEST, sub_id=sub_id)
    await asyncio.wait_for(started.wait(), timeout=1)

    await manager.stop()

    with database as db:
        submission = db.get_submission_by_id(sub_id)
    assert submission["job_status"] == "failed"
    assert submission["job_error"] == (
        "job interrupted while kernelbot was shutting down; please resubmit"
    )


@pytest.mark.asyncio
async def test_scale_up_and_down(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        await asyncio.sleep(0.05)
        return None, None

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=3, idle_seconds=0.2
    )
    await manager.start()

    # send multiple request to scale up
    for i in range(6):
        await manager.enqueue(
            get_req(i),
            SubmissionMode.TEST,
            sub_id=i + 1,
        )

    await manager.queue.join()

    # idle timeout
    await asyncio.sleep(manager.idle_seconds + 0.1)

    async with manager._state_lock:
        assert len(manager._workers) == manager.min_workers
    await manager.stop()


@pytest.mark.asyncio
async def test_enqueue_prunes_dead_workers_before_autoscaling(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        return None, None

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=0, max_workers=1, idle_seconds=0.1
    )
    await manager.start()

    async def dead_worker():
        raise RuntimeError("dead worker")

    dead_task = asyncio.create_task(dead_worker(), name="dead-bg-worker")
    await asyncio.sleep(0)
    async with manager._state_lock:
        manager._workers.append(dead_task)

    await manager.enqueue(get_req(1), SubmissionMode.TEST, sub_id=99)
    await manager.queue.join()

    assert (
        mock.call(99, status="succeeded", last_heartbeat=mock.ANY)
        in db_context.upsert_submission_job_status.call_args_list
    )

    await manager.stop()


@pytest.mark.asyncio
async def test_run_one_initial_status_failure_marks_failed(mock_backend):
    db_context = mock_backend.db
    statuses = []

    def fake_upsert(sub_id, status=None, error=None, last_heartbeat=None):
        statuses.append((status, error))
        if status == "running":
            raise RuntimeError("database unavailable")
        return sub_id

    db_context.upsert_submission_job_status = mock.Mock(side_effect=fake_upsert)
    db_context.update_heartbeat_if_active = mock.Mock()
    mock_backend.submit_full = mock.AsyncMock()

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()

    await manager.enqueue(get_req(1), SubmissionMode.TEST, sub_id=123)
    await manager.queue.join()

    assert ("pending", None) in statuses
    assert ("running", None) in statuses
    assert any(status == "failed" and error == "database unavailable" for status, error in statuses)
    mock_backend.submit_full.assert_not_called()

    async with manager._state_lock:
        assert len(manager._workers) == 1

    await manager.stop()


@pytest.mark.asyncio
async def test_hacked_submission_sets_hacked_status(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    async def fake_submit_full(req, mode, reporter, sub_id, skip_precheck=False):
        raise KernelGuardRejected("blocked by kernelguard", result={})

    mock_backend.submit_full = fake_submit_full

    manager = BackgroundSubmissionManager(
        mock_backend, min_workers=1, max_workers=1, idle_seconds=0.1
    )
    await manager.start()

    await manager.enqueue(get_req(1), SubmissionMode.BENCHMARK, sub_id=42)
    await manager.queue.join()
    await asyncio.sleep(0.05)

    assert (
        mock.call(42, status="hacked", last_heartbeat=mock.ANY, error="blocked by kernelguard")
        in db_context.upsert_submission_job_status.call_args_list
    )

    await manager.stop()
