"""Tests for admin API endpoints."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from libkernelbot.launchers import RunnerQueueStatus


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock()
    backend.accepts_jobs = False
    backend.db = MagicMock()
    return backend


@pytest.fixture
def mock_background_manager():
    manager = MagicMock()
    manager.enqueue = AsyncMock()
    return manager


@pytest.fixture
def test_client(mock_backend, mock_background_manager):
    """Create a test client with mocked backend."""
    # Patch env before importing the app
    with patch.dict('os.environ', {'ADMIN_TOKEN': 'test_token'}):
        from kernelbot.api.main import app, init_api, init_background_submission_manager
        init_api(mock_backend)
        init_background_submission_manager(mock_background_manager)
        yield TestClient(app)


class TestAdminAuth:
    """Test admin authentication."""

    def test_admin_requires_auth_header(self, test_client):
        """Admin endpoints require Authorization header."""
        response = test_client.post("/admin/start")
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing Authorization header"

    def test_admin_rejects_invalid_token(self, test_client):
        """Admin endpoints reject invalid tokens."""
        response = test_client.post(
            "/admin/start",
            headers={"Authorization": "Bearer wrong_token"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid admin token"

    def test_admin_accepts_valid_token(self, test_client, mock_backend):
        """Admin endpoints accept valid tokens."""
        response = test_client.post(
            "/admin/start",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert mock_backend.accepts_jobs is True


class TestAdminStartStop:
    """Test admin start/stop endpoints."""

    def test_admin_start(self, test_client, mock_backend):
        """POST /admin/start enables job acceptance."""
        mock_backend.accepts_jobs = False
        response = test_client.post(
            "/admin/start",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "accepts_jobs": True}
        assert mock_backend.accepts_jobs is True

    def test_admin_stop(self, test_client, mock_backend):
        """POST /admin/stop disables job acceptance."""
        mock_backend.accepts_jobs = True
        response = test_client.post(
            "/admin/stop",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "accepts_jobs": False}
        assert mock_backend.accepts_jobs is False


class TestRunnerQueue:
    def test_get_runner_queue(self, test_client, mock_backend):
        """GET /runner_queue/{gpu_type} returns runner backlog."""
        mock_backend.get_runner_queue_status = AsyncMock(
            return_value=RunnerQueueStatus(
                runner="Modal",
                gpu="B200",
                queued_jobs=6,
                available_runners=1,
            )
        )

        response = test_client.get("/runner_queue/B200")

        assert response.status_code == 200
        assert response.json() == {
            "runner": "Modal",
            "gpu": "B200",
            "queued_jobs": 6,
            "running_jobs": None,
            "available_runners": 1,
            "status": "available",
            "error": None,
        }


class TestUserSubmissions:
    def test_submission_details_require_authentication(self, test_client):
        response = test_client.get("/user/submissions/42")

        assert response.status_code == 400

    def test_submission_details_return_not_found(self, test_client, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.validate_identity = MagicMock(
            return_value={"user_id": "123", "user_name": "test-user"}
        )
        mock_backend.db.get_submission_by_id = MagicMock(return_value=None)

        response = test_client.get(
            "/user/submissions/42",
            headers={"X-Popcorn-Cli-Id": "cli-token"},
        )

        assert response.status_code == 404

    def test_submission_details_reject_other_owner(self, test_client, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.validate_identity = MagicMock(
            return_value={"user_id": "123", "user_name": "test-user"}
        )
        mock_backend.db.get_submission_by_id = MagicMock(return_value={"user_id": "456"})

        response = test_client.get(
            "/user/submissions/42",
            headers={"X-Popcorn-Cli-Id": "cli-token"},
        )

        assert response.status_code == 403

    def test_submission_details_include_stored_run_results(self, test_client, mock_backend):
        """Polling clients receive the per-row data stored for completed runs."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.validate_identity = MagicMock(
            return_value={"user_id": "123", "user_name": "test-user"}
        )
        benchmark_result = {
            "benchmark-count": "2",
            "benchmark.0.status": "pass",
            "benchmark.0.spec": "shape-0",
            "benchmark.0.mean": "1.5",
            "benchmark.1.status": "pass",
            "benchmark.1.spec": "shape-1",
            "benchmark.1.mean": "2.5",
        }
        test_result = {
            "test-count": "2",
            "test.0.status": "pass",
            "test.0.spec": "case-0",
            "test.0.message": "correct",
            "test.1.status": "fail",
            "test.1.spec": "case-1",
            "test.1.error": "mismatch",
        }
        mock_backend.db.get_submission_by_id = MagicMock(
            return_value={
                "submission_id": 42,
                "leaderboard_id": 7,
                "leaderboard_name": "test-leaderboard",
                "file_name": "submission.py",
                "user_id": "123",
                "submission_time": "2026-07-06T12:00:00Z",
                "done": True,
                "code": "print('ok')",
                "runs": [
                    {
                        "start_time": "2026-07-06T12:00:00Z",
                        "end_time": "2026-07-06T12:01:00Z",
                        "mode": "test",
                        "secret": False,
                        "runner": "B200",
                        "score": None,
                        "passed": False,
                        "result": test_result,
                    },
                    {
                        "start_time": "2026-07-06T12:00:00Z",
                        "end_time": "2026-07-06T12:01:00Z",
                        "mode": "benchmark",
                        "secret": False,
                        "runner": "B200",
                        "score": None,
                        "passed": True,
                        "result": benchmark_result,
                    },
                    {
                        "start_time": "2026-07-06T12:00:00Z",
                        "end_time": "2026-07-06T12:01:00Z",
                        "mode": "benchmark",
                        "secret": True,
                        "runner": "B200",
                        "score": None,
                        "passed": True,
                        "result": {"secret-result": "must not be exposed"},
                    },
                    {
                        "start_time": "2026-07-06T12:00:00Z",
                        "end_time": "2026-07-06T12:01:00Z",
                        "mode": "leaderboard",
                        "secret": False,
                        "runner": "B200",
                        "score": 1.5,
                        "passed": True,
                        "result": {"leaderboard-result": "not part of this contract"},
                    },
                    {
                        "start_time": "2026-07-06T12:00:00Z",
                        "end_time": "2026-07-06T12:01:00Z",
                        "mode": "profile0",
                        "secret": False,
                        "runner": "B200",
                        "score": None,
                        "passed": True,
                        "result": {"profile-report": "PROFILE_SENTINEL"},
                    },
                ],
                "job_status": "succeeded",
                "job_error": None,
                "job_last_heartbeat": "2026-07-06T12:01:00Z",
            }
        )
        mock_backend.get_runner_queue_status = AsyncMock(
            return_value=RunnerQueueStatus(
                runner="Modal",
                gpu="B200",
                queued_jobs=0,
                available_runners=1,
            )
        )

        response = test_client.get(
            "/user/submissions/42",
            headers={"X-Popcorn-Cli-Id": "cli-token"},
        )

        assert response.status_code == 200
        runs = response.json()["runs"]
        assert runs[0]["result"] == test_result
        assert runs[1]["result"] == benchmark_result
        assert "result" not in runs[2]
        assert runs[2]["secret"] is True
        assert runs[2]["passed"] is True
        assert "result" not in runs[3]
        assert "result" not in runs[4]
        assert "must not be exposed" not in response.text
        assert "PROFILE_SENTINEL" not in response.text


class TestAdminStats:
    """Test admin stats endpoint."""

    def test_admin_stats(self, test_client, mock_backend):
        """GET /admin/stats returns statistics."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.generate_stats = MagicMock(return_value={
            "num_submissions": 10,
            "num_users": 5,
        })

        response = test_client.get(
            "/admin/stats",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["stats"]["num_submissions"] == 10

    def test_admin_stats_last_day_only(self, test_client, mock_backend):
        """GET /admin/stats with last_day_only parameter."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.generate_stats = MagicMock(return_value={
            "num_submissions": 3,
            "num_users": 2,
        })

        response = test_client.get(
            "/admin/stats?last_day_only=true",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        mock_backend.db.generate_stats.assert_called_once()
        args, kwargs = mock_backend.db.generate_stats.call_args
        assert args[0] is True  # last_day_only

    def test_admin_stats_with_leaderboard_name(self, test_client, mock_backend):
        """GET /admin/stats with leaderboard_name parameter."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.generate_stats = MagicMock(return_value={
            "num_submissions": 5,
            "num_users": 3,
        })

        response = test_client.get(
            "/admin/stats?leaderboard_name=my-leaderboard",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        mock_backend.db.generate_stats.assert_called_once()
        args, kwargs = mock_backend.db.generate_stats.call_args
        assert args[1] == "my-leaderboard"  # leaderboard_name


class TestAdminSubmissions:
    """Test admin submission endpoints."""

    def test_admin_submission_allows_after_deadline(
        self, test_client, mock_backend, mock_background_manager
    ):
        """POST /admin/submission queues an authenticated user submission after deadline."""
        mock_backend.accepts_jobs = True
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.validate_identity = MagicMock(return_value={
            "user_id": "123",
            "user_name": "admin_user",
            "id_type": "cli",
        })
        mock_task = MagicMock()
        mock_backend.db.get_leaderboard = MagicMock(return_value={
            "task": mock_task,
            "secret_seed": 12345,
            "deadline": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1),
            "name": "expired-lb",
            "gpu_types": ["B200"],
            "visibility": "public",
        })
        mock_backend.db.get_leaderboard_gpu_types = MagicMock(return_value=["B200"])
        mock_backend.db.is_user_banned = MagicMock(return_value=False)
        mock_backend.db.check_rate_limit = MagicMock(return_value=None)
        mock_backend.db.create_submission = MagicMock(return_value=123)
        mock_backend.db.upsert_submission_job_status = MagicMock(return_value=456)
        mock_backend.get_runner_queue_status = AsyncMock(
            return_value=RunnerQueueStatus(
                runner="Modal",
                gpu="B200",
                queued_jobs=0,
                available_runners=1,
            )
        )

        response = test_client.post(
            "/admin/submission/expired-lb/B200/leaderboard",
            headers={
                "Authorization": "Bearer test_token",
                "X-Popcorn-Cli-Id": "cli-token",
            },
            files={"file": ("submission.py", b"print('ok')", "text/x-python")},
        )

        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        assert response.json()["details"] == {"id": 123, "job_status_id": 456}
        mock_background_manager.enqueue.assert_awaited_once()
        queued_req, _, sub_id = mock_background_manager.enqueue.await_args.args
        assert queued_req.leaderboard == "expired-lb"
        assert sub_id == 123

    def test_list_leaderboard_submissions(self, test_client, mock_backend):
        """GET /admin/leaderboards/{name}/submissions returns submission IDs."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.get_leaderboard_submission_ids = MagicMock(return_value=[123, 122])

        response = test_client.get(
            "/admin/leaderboards/test-lb/submissions?limit=50&offset=10",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "leaderboard": "test-lb",
            "limit": 50,
            "offset": 10,
            "submission_ids": [123, 122],
        }
        mock_backend.db.get_leaderboard_submission_ids.assert_called_once_with("test-lb", 50, 10)

    def test_get_submission(self, test_client, mock_backend):
        """GET /admin/submissions/{id} returns submission."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.get_submission_by_id = MagicMock(return_value={
            "id": 123,
            "code": "test code",
        })

        response = test_client.get(
            "/admin/submissions/123",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["submission"]["id"] == 123

    def test_get_submission_not_found(self, test_client, mock_backend):
        """GET /admin/submissions/{id} returns 404 for missing submission."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.get_submission_by_id = MagicMock(return_value=None)

        response = test_client.get(
            "/admin/submissions/999",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404

    def test_delete_submission(self, test_client, mock_backend):
        """DELETE /admin/submissions/{id} deletes submission."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_submission = MagicMock()

        response = test_client.delete(
            "/admin/submissions/123",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        mock_backend.db.delete_submission.assert_called_once_with(123)

    def test_delete_submissions_for_user(self, test_client, mock_backend):
        """DELETE /admin/submissions deletes by leaderboard ID and username."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_submissions_for_user = MagicMock(return_value={
            "deleted_job_status": 2,
            "deleted_runs": 5,
            "deleted_submissions": 3,
        })

        response = test_client.delete(
            "/admin/submissions?leaderboard_id=765&user_name=Borui%20Xu",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "leaderboard_id": 765,
            "user_name": "Borui Xu",
            "deleted_job_status": 2,
            "deleted_runs": 5,
            "deleted_submissions": 3,
        }
        mock_backend.db.delete_submissions_for_user.assert_called_once_with(765, "Borui Xu")


class TestAdminLeaderboards:
    """Test admin leaderboard endpoints."""

    def test_create_leaderboard_missing_directory(self, test_client, mock_backend):
        """POST /admin/leaderboards returns 400 for missing directory."""
        response = test_client.post(
            "/admin/leaderboards",
            headers={"Authorization": "Bearer test_token"},
            json={}  # missing directory
        )
        assert response.status_code == 400
        assert "Missing required field: directory" in response.json()["detail"]

    def test_create_leaderboard_invalid_directory(self, test_client, mock_backend):
        """POST /admin/leaderboards returns 400 for invalid directory."""
        response = test_client.post(
            "/admin/leaderboards",
            headers={"Authorization": "Bearer test_token"},
            json={
                "directory": "../../../etc/passwd",  # path traversal attempt
            }
        )
        assert response.status_code == 400

    def test_create_leaderboard_with_gpu_list(self, test_client, mock_backend):
        """POST /admin/leaderboards reads GPUs from task definition."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_leaderboard = MagicMock()
        mock_backend.db.create_leaderboard = MagicMock()

        # Mock a definition with gpus
        mock_definition = MagicMock()
        mock_definition.gpus = ["H100", "A100"]

        with patch('kernelbot.api.main.resolve_problem_directory', return_value="/valid/path"):
            with patch('kernelbot.api.main.make_task_definition', return_value=mock_definition):
                response = test_client.post(
                    "/admin/leaderboards",
                    headers={"Authorization": "Bearer test_token"},
                    json={"directory": "identity_py"}
                )
                assert response.status_code == 200
                assert response.json()["leaderboard"] == "identity_py-dev"
                # Verify gpu_types was passed from definition.gpus
                call_kwargs = mock_backend.db.create_leaderboard.call_args[1]
                assert call_kwargs["gpu_types"] == ["H100", "A100"]

    def test_create_leaderboard_without_gpu(self, test_client, mock_backend):
        """POST /admin/leaderboards returns 400 when no GPUs in task.yml."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        # Mock a definition without gpus
        mock_definition = MagicMock()
        mock_definition.gpus = []

        with patch('kernelbot.api.main.resolve_problem_directory', return_value="/valid/path"):
            with patch('kernelbot.api.main.make_task_definition', return_value=mock_definition):
                response = test_client.post(
                    "/admin/leaderboards",
                    headers={"Authorization": "Bearer test_token"},
                    json={"directory": "identity_py"}
                )
                assert response.status_code == 400
                assert "No gpus specified in task.yml" in response.json()["detail"]

    def test_delete_leaderboard(self, test_client, mock_backend):
        """DELETE /admin/leaderboards/{name} deletes leaderboard."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_leaderboard = MagicMock()

        response = test_client.delete(
            "/admin/leaderboards/test-leaderboard",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json()["leaderboard"] == "test-leaderboard"
        mock_backend.db.delete_leaderboard.assert_called_once_with("test-leaderboard", force=False)

    def test_delete_leaderboard_force(self, test_client, mock_backend):
        """DELETE /admin/leaderboards/{name}?force=true force deletes."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_leaderboard = MagicMock()

        response = test_client.delete(
            "/admin/leaderboards/test-leaderboard?force=true",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json()["force"] is True
        mock_backend.db.delete_leaderboard.assert_called_once_with("test-leaderboard", force=True)


class TestAdminUpdateProblems:
    """Test admin update-problems endpoint."""

    def test_update_problems_requires_auth(self, test_client):
        """POST /admin/update-problems requires authorization."""
        response = test_client.post("/admin/update-problems", json={})
        assert response.status_code == 401

    def test_update_problems_success(self, test_client, mock_backend):
        """POST /admin/update-problems returns sync results."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.created = ["problem1", "problem2"]
        mock_result.updated = ["problem3"]
        mock_result.skipped = [{"name": "problem4", "reason": "no changes"}]
        mock_result.errors = []

        with patch('kernelbot.api.main.sync_problems', return_value=mock_result) as mock_sync:
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["created"] == ["problem1", "problem2"]
            assert data["updated"] == ["problem3"]
            assert data["skipped"] == [{"name": "problem4", "reason": "no changes"}]
            assert data["errors"] == []

            # Verify default parameters
            mock_sync.assert_called_once()
            call_kwargs = mock_sync.call_args[1]
            assert call_kwargs["repository"] == "gpu-mode/reference-kernels"
            assert call_kwargs["branch"] == "main"
            assert call_kwargs["force"] is False
            assert call_kwargs["problem_set"] is None

    def test_update_problems_with_problem_set(self, test_client, mock_backend):
        """POST /admin/update-problems with specific problem_set."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.created = ["nvidia-problem"]
        mock_result.updated = []
        mock_result.skipped = []
        mock_result.errors = []

        with patch('kernelbot.api.main.sync_problems', return_value=mock_result) as mock_sync:
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={"problem_set": "nvidia"}
            )
            assert response.status_code == 200
            call_kwargs = mock_sync.call_args[1]
            assert call_kwargs["problem_set"] == "nvidia"

    def test_update_problems_with_force(self, test_client, mock_backend):
        """POST /admin/update-problems with force=True."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.created = []
        mock_result.updated = ["updated-problem"]
        mock_result.skipped = []
        mock_result.errors = []

        with patch('kernelbot.api.main.sync_problems', return_value=mock_result) as mock_sync:
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={"force": True}
            )
            assert response.status_code == 200
            call_kwargs = mock_sync.call_args[1]
            assert call_kwargs["force"] is True

    def test_update_problems_with_custom_repo_and_branch(self, test_client, mock_backend):
        """POST /admin/update-problems with custom repository and branch."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.created = []
        mock_result.updated = []
        mock_result.skipped = []
        mock_result.errors = []

        with patch('kernelbot.api.main.sync_problems', return_value=mock_result) as mock_sync:
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={
                    "repository": "other-org/other-repo",
                    "branch": "develop"
                }
            )
            assert response.status_code == 200
            call_kwargs = mock_sync.call_args[1]
            assert call_kwargs["repository"] == "other-org/other-repo"
            assert call_kwargs["branch"] == "develop"

    def test_update_problems_value_error(self, test_client, mock_backend):
        """POST /admin/update-problems returns 400 on ValueError."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        with patch('kernelbot.api.main.sync_problems', side_effect=ValueError("Invalid branch name")):
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={"branch": "invalid/branch"}
            )
            assert response.status_code == 400
            assert "Invalid branch name" in response.json()["detail"]

    def test_update_problems_with_errors(self, test_client, mock_backend):
        """POST /admin/update-problems includes errors in response."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.created = []
        mock_result.updated = []
        mock_result.skipped = []
        mock_result.errors = [{"name": "bad-problem", "error": "create failed: DB error"}]

        with patch('kernelbot.api.main.sync_problems', return_value=mock_result):
            response = test_client.post(
                "/admin/update-problems",
                headers={"Authorization": "Bearer test_token"},
                json={}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert len(data["errors"]) == 1
            assert data["errors"][0]["name"] == "bad-problem"


class TestAdminLeaderboardInvites:
    """Test admin leaderboard invite endpoints."""

    def _setup_db_mock(self, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

    def test_generate_invites(self, test_client, mock_backend):
        """POST /admin/invites generates codes for multiple leaderboards."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.generate_invite_codes = MagicMock(return_value=["code1", "code2"])

        response = test_client.post(
            "/admin/invites",
            headers={"Authorization": "Bearer test_token"},
            json={"leaderboards": ["lb-1", "lb-2"], "count": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["codes"] == ["code1", "code2"]
        assert data["leaderboards"] == ["lb-1", "lb-2"]
        mock_backend.db.generate_invite_codes.assert_called_once_with(["lb-1", "lb-2"], 2)

    def test_generate_invites_single_shorthand(self, test_client, mock_backend):
        """POST /admin/invites accepts single leaderboard shorthand."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.generate_invite_codes = MagicMock(return_value=["code1"])

        response = test_client.post(
            "/admin/invites",
            headers={"Authorization": "Bearer test_token"},
            json={"leaderboard": "test-lb", "count": 1},
        )
        assert response.status_code == 200
        mock_backend.db.generate_invite_codes.assert_called_once_with(["test-lb"], 1)

    def test_generate_invites_invalid_count(self, test_client, mock_backend):
        """POST /admin/invites rejects invalid count."""
        response = test_client.post(
            "/admin/invites",
            headers={"Authorization": "Bearer test_token"},
            json={"leaderboards": ["lb-1"], "count": 0},
        )
        assert response.status_code == 400

    def test_generate_invites_missing_leaderboards(self, test_client, mock_backend):
        """POST /admin/invites rejects missing leaderboards."""
        response = test_client.post(
            "/admin/invites",
            headers={"Authorization": "Bearer test_token"},
            json={"count": 5},
        )
        assert response.status_code == 400

    def test_generate_invites_requires_auth(self, test_client):
        """POST /admin/invites requires admin auth."""
        response = test_client.post(
            "/admin/invites",
            json={"leaderboards": ["lb-1"], "count": 5},
        )
        assert response.status_code == 401

    def test_list_invites(self, test_client, mock_backend):
        """GET /admin/leaderboards/{lb}/invites lists codes."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_invite_codes = MagicMock(return_value=[
            {"code": "abc", "user_id": "1", "user_name": "alice",
             "claimed_at": "2026-01-01T00:00:00Z", "created_at": "2026-01-01T00:00:00Z"},
            {"code": "def", "user_id": None, "user_name": None,
             "claimed_at": None, "created_at": "2026-01-01T00:00:00Z"},
        ])

        response = test_client.get(
            "/admin/leaderboards/test-lb/invites",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["invites"]) == 2
        assert data["invites"][0]["user_id"] == "1"
        assert data["invites"][1]["user_id"] is None

    def test_set_visibility(self, test_client, mock_backend):
        """POST /admin/leaderboards/{lb}/visibility changes visibility."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.set_leaderboard_visibility = MagicMock()

        response = test_client.post(
            "/admin/leaderboards/test-lb/visibility",
            headers={"Authorization": "Bearer test_token"},
            json={"visibility": "closed"},
        )
        assert response.status_code == 200
        mock_backend.db.set_leaderboard_visibility.assert_called_once_with("test-lb", "closed")

    def test_set_visibility_invalid(self, test_client, mock_backend):
        """POST /admin/leaderboards/{lb}/visibility rejects invalid values."""
        response = test_client.post(
            "/admin/leaderboards/test-lb/visibility",
            headers={"Authorization": "Bearer test_token"},
            json={"visibility": "private"},
        )
        assert response.status_code == 400

    def test_revoke_invite(self, test_client, mock_backend):
        """DELETE /admin/invites/{code} revokes a code."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.revoke_invite_code = MagicMock(return_value={"code": "abc123", "was_claimed": False})

        response = test_client.delete(
            "/admin/invites/abc123",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["was_claimed"] is False
        mock_backend.db.revoke_invite_code.assert_called_once_with("abc123")

    def test_revoke_invite_not_found(self, test_client, mock_backend):
        """DELETE /admin/invites/{code} returns 404 for invalid code."""
        from libkernelbot.utils import KernelBotError

        self._setup_db_mock(mock_backend)
        err = KernelBotError("Invalid invite code", code=404)
        mock_backend.db.revoke_invite_code = MagicMock(side_effect=err)

        response = test_client.delete(
            "/admin/invites/bad-code",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 404

    def test_revoke_invite_requires_auth(self, test_client):
        """DELETE /admin/invites/{code} requires admin auth."""
        response = test_client.delete("/admin/invites/abc123")
        assert response.status_code == 401


class TestUserJoin:
    """Test user invite claim endpoint."""

    def _setup_db_mock(self, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

    def test_join_success(self, test_client, mock_backend):
        """POST /user/join claims an invite code."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.validate_cli_id = MagicMock(
            return_value={"user_id": "42", "user_name": "testuser"}
        )
        mock_backend.db.claim_invite_code = MagicMock(
            return_value={"leaderboards": ["closed-lb-1", "closed-lb-2"]}
        )

        response = test_client.post(
            "/user/join",
            headers={"X-Popcorn-Cli-Id": "valid-cli-id"},
            json={"code": "invite-code-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["leaderboards"] == ["closed-lb-1", "closed-lb-2"]
        mock_backend.db.claim_invite_code.assert_called_once_with("invite-code-123", "42")

    def test_join_missing_code(self, test_client, mock_backend):
        """POST /user/join requires code field."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.validate_cli_id = MagicMock(
            return_value={"user_id": "42", "user_name": "testuser"}
        )

        response = test_client.post(
            "/user/join",
            headers={"X-Popcorn-Cli-Id": "valid-cli-id"},
            json={},
        )
        assert response.status_code == 400

    def test_join_requires_cli_auth(self, test_client):
        """POST /user/join requires CLI authentication."""
        response = test_client.post(
            "/user/join",
            json={"code": "invite-code-123"},
        )
        assert response.status_code == 400  # missing header


class TestClosedLeaderboardAccess:
    """Test that closed leaderboards gate access correctly."""

    def _setup_db_mock(self, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

    def test_closed_leaderboard_submissions_no_auth(self, test_client, mock_backend):
        """GET /submissions on closed leaderboard without auth returns 401."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_leaderboard = MagicMock(return_value={"visibility": "closed"})

        response = test_client.get("/submissions/closed-lb/A100")
        assert response.status_code == 401

    def test_closed_leaderboard_submissions_no_access(self, test_client, mock_backend):
        """GET /submissions on closed leaderboard without invite returns 403."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_leaderboard = MagicMock(return_value={"visibility": "closed"})
        mock_backend.db.check_leaderboard_access = MagicMock(return_value=False)
        mock_backend.db.validate_identity = MagicMock(
            return_value={"user_id": "1", "user_name": "test", "id_type": "cli"}
        )

        response = test_client.get(
            "/submissions/closed-lb/A100",
            headers={"X-Popcorn-Cli-Id": "valid-cli-id"},
        )
        assert response.status_code == 403

    def test_public_leaderboard_submissions_no_auth(self, test_client, mock_backend):
        """GET /submissions on public leaderboard without auth works fine."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_leaderboard = MagicMock(return_value={"visibility": "public"})
        mock_backend.db.get_leaderboard_submissions = MagicMock(return_value=[])

        response = test_client.get("/submissions/public-lb/A100")
        assert response.status_code == 200
class TestAdminExportHF:
    """Test admin HF export endpoint."""

    def test_export_hf_rejects_non_int_leaderboard_ids(self, test_client):
        """POST /admin/export-hf returns 400 for non-integer leaderboard IDs."""
        from kernelbot.api import main as api_main

        with patch.object(api_main.env, "HF_TOKEN", "hf-token"):
            response = test_client.post(
                "/admin/export-hf",
                headers={"Authorization": "Bearer test_token"},
                json={
                    "leaderboard_ids": ["1"],
                    "filename": "active-comp.parquet",
                    "private": True,
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "leaderboard_ids must be a non-empty list of integers"

    def test_export_hf_rejects_non_string_filename(self, test_client):
        """POST /admin/export-hf returns 400 for non-string filenames."""
        response = test_client.post(
            "/admin/export-hf",
            headers={"Authorization": "Bearer test_token"},
            json={
                "leaderboard_ids": [1],
                "filename": 123,
                "private": True,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "filename must end with .parquet"

    def test_export_hf_rejects_active_public_export(self, test_client, mock_backend):
        """POST /admin/export-hf returns 400 for active public exports."""
        from kernelbot.api import main as api_main

        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

        with patch.object(api_main.env, "HF_TOKEN", "hf-token"):
            with patch(
                "libkernelbot.hf_export.export_to_hf",
                side_effect=ValueError(
                    "Cannot export active leaderboards to the public dataset: active-comp"
                ),
            ):
                response = test_client.post(
                    "/admin/export-hf",
                    headers={"Authorization": "Bearer test_token"},
                    json={
                        "leaderboard_ids": [1],
                        "filename": "active-comp.parquet",
                        "private": False,
                    },
                )

        assert response.status_code == 400
        assert "Cannot export active leaderboards" in response.json()["detail"]


class TestAdminRateLimits:
    """Test admin rate limit endpoints."""

    def test_set_rate_limit(self, test_client, mock_backend):
        """PUT /admin/leaderboards/{name}/rate-limits creates a rate limit."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.set_rate_limit = MagicMock(return_value={
            "id": 1,
            "leaderboard_id": 1,
            "leaderboard_name": "test-lb",
            "mode_category": "test",
            "max_submissions_per_hour": 5,
        })

        response = test_client.put(
            "/admin/leaderboards/test-lb/rate-limits",
            headers={"Authorization": "Bearer test_token"},
            json={"mode_category": "test", "max_submissions_per_hour": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["rate_limit"]["max_submissions_per_hour"] == 5
        mock_backend.db.set_rate_limit.assert_called_once_with("test-lb", "test", 5)

    def test_set_rate_limit_invalid_category(self, test_client):
        """PUT /admin/leaderboards/{name}/rate-limits rejects invalid category."""
        response = test_client.put(
            "/admin/leaderboards/test-lb/rate-limits",
            headers={"Authorization": "Bearer test_token"},
            json={"mode_category": "invalid", "max_submissions_per_hour": 5},
        )
        assert response.status_code == 400

    def test_set_rate_limit_invalid_count(self, test_client):
        """PUT /admin/leaderboards/{name}/rate-limits rejects negative count."""
        response = test_client.put(
            "/admin/leaderboards/test-lb/rate-limits",
            headers={"Authorization": "Bearer test_token"},
            json={"mode_category": "test", "max_submissions_per_hour": -1},
        )
        assert response.status_code == 400

    def test_set_rate_limit_requires_auth(self, test_client):
        """PUT /admin/leaderboards/{name}/rate-limits requires auth."""
        response = test_client.put(
            "/admin/leaderboards/test-lb/rate-limits",
            json={"mode_category": "test", "max_submissions_per_hour": 5},
        )
        assert response.status_code == 401

    def test_get_rate_limits(self, test_client, mock_backend):
        """GET /admin/leaderboards/{name}/rate-limits returns rate limits."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.get_rate_limits = MagicMock(return_value=[
            {
                "id": 1,
                "leaderboard_id": 1,
                "leaderboard_name": "test-lb",
                "mode_category": "test",
                "max_submissions_per_hour": 5,
            }
        ])

        response = test_client.get(
            "/admin/leaderboards/test-lb/rate-limits",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["rate_limits"]) == 1

    def test_delete_rate_limit(self, test_client, mock_backend):
        """DELETE /admin/leaderboards/{name}/rate-limits/{category} removes a rate limit."""
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)
        mock_backend.db.delete_rate_limit = MagicMock(return_value=None)

        response = test_client.delete(
            "/admin/leaderboards/test-lb/rate-limits/test",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_backend.db.delete_rate_limit.assert_called_once_with("test-lb", "test")

    def test_delete_rate_limit_invalid_category(self, test_client):
        """DELETE rejects invalid mode_category."""
        response = test_client.delete(
            "/admin/leaderboards/test-lb/rate-limits/invalid",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
