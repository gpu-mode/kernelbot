import asyncio
import datetime
from unittest import mock

import pytest

from libkernelbot.background_submission_manager import BackgroundSubmissionManager
from libkernelbot.consts import SubmissionMode
from libkernelbot.submission import ProcessedSubmissionRequest


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


@pytest.mark.asyncio
async def test_enqueue_and_run_job(mock_backend):
    # mock upsert/update
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    # mock submit_full
    async def fake_submit_full(req, mode, reporter, sub_id):
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
async def test_scale_up_and_down(mock_backend):
    db_context = mock_backend.db
    db_context.upsert_submission_job_status = mock.Mock(
        side_effect=lambda *a, **k: a[0]
    )
    db_context.update_heartbeat_if_active = mock.Mock()

    async def fake_submit_full(req, mode, reporter, sub_id):
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
