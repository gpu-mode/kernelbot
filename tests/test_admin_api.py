"""Tests for admin API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock()
    backend.accepts_jobs = False
    backend.db = MagicMock()
    return backend


@pytest.fixture
def test_client(mock_backend):
    """Create a test client with mocked backend."""
    # Patch env before importing the app
    with patch.dict('os.environ', {'ADMIN_TOKEN': 'test_token'}):
        from kernelbot.api.main import app, init_api
        init_api(mock_backend)
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
        mock_backend.db.generate_stats.assert_called_once_with(True)


class TestAdminSubmissions:
    """Test admin submission endpoints."""

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


class TestAdminRateLimits:
    """Test admin rate limit endpoints."""

    def _setup_db_mock(self, mock_backend):
        mock_backend.db.__enter__ = MagicMock(return_value=mock_backend.db)
        mock_backend.db.__exit__ = MagicMock(return_value=None)

    def test_get_all_rate_limits(self, test_client, mock_backend):
        """GET /admin/rate-limits returns all rate limits."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_all_user_rate_limits = MagicMock(return_value=[
            {"user_id": "123", "max_submissions_per_hour": 10, "max_submissions_per_day": 50,
             "note": None, "created_at": None, "updated_at": None, "user_name": "testuser"},
        ])

        response = test_client.get(
            "/admin/rate-limits",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["rate_limits"]) == 1
        assert data["rate_limits"][0]["user_id"] == "123"

    def test_get_all_rate_limits_requires_auth(self, test_client):
        """GET /admin/rate-limits requires authentication."""
        response = test_client.get("/admin/rate-limits")
        assert response.status_code == 401

    def test_get_user_rate_limit(self, test_client, mock_backend):
        """GET /admin/rate-limits/{user_id} returns user's rate limit."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_user_rate_limit = MagicMock(return_value={
            "user_id": "123", "max_submissions_per_hour": 10, "max_submissions_per_day": 50,
            "note": "heavy user", "created_at": None, "updated_at": None,
        })

        response = test_client.get(
            "/admin/rate-limits/123",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["rate_limit"]["user_id"] == "123"
        assert data["rate_limit"]["note"] == "heavy user"

    def test_get_user_rate_limit_not_found(self, test_client, mock_backend):
        """GET /admin/rate-limits/{user_id} returns 404 for missing."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.get_user_rate_limit = MagicMock(return_value=None)

        response = test_client.get(
            "/admin/rate-limits/999",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404

    def test_set_user_rate_limit(self, test_client, mock_backend):
        """PUT /admin/rate-limits/{user_id} sets rate limit."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.set_user_rate_limit = MagicMock(return_value={
            "user_id": "123", "max_submissions_per_hour": 5, "max_submissions_per_day": 20,
            "note": "restricted", "created_at": None, "updated_at": None,
        })

        response = test_client.put(
            "/admin/rate-limits/123",
            headers={"Authorization": "Bearer test_token"},
            json={"max_submissions_per_hour": 5, "max_submissions_per_day": 20, "note": "restricted"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["rate_limit"]["max_submissions_per_hour"] == 5
        mock_backend.db.set_user_rate_limit.assert_called_once_with(
            user_id="123",
            max_submissions_per_hour=5,
            max_submissions_per_day=20,
            note="restricted",
        )

    def test_set_user_rate_limit_requires_at_least_one_limit(self, test_client, mock_backend):
        """PUT /admin/rate-limits/{user_id} requires at least one limit field."""
        self._setup_db_mock(mock_backend)

        response = test_client.put(
            "/admin/rate-limits/123",
            headers={"Authorization": "Bearer test_token"},
            json={"note": "just a note"},
        )
        assert response.status_code == 400
        assert "At least one of" in response.json()["detail"]

    def test_set_user_rate_limit_validates_negative(self, test_client, mock_backend):
        """PUT /admin/rate-limits/{user_id} rejects negative values."""
        self._setup_db_mock(mock_backend)

        response = test_client.put(
            "/admin/rate-limits/123",
            headers={"Authorization": "Bearer test_token"},
            json={"max_submissions_per_hour": -1},
        )
        assert response.status_code == 400
        assert "non-negative integer" in response.json()["detail"]

    def test_delete_user_rate_limit(self, test_client, mock_backend):
        """DELETE /admin/rate-limits/{user_id} deletes rate limit."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.delete_user_rate_limit = MagicMock(return_value=True)

        response = test_client.delete(
            "/admin/rate-limits/123",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["user_id"] == "123"

    def test_delete_user_rate_limit_not_found(self, test_client, mock_backend):
        """DELETE /admin/rate-limits/{user_id} returns 404 when not found."""
        self._setup_db_mock(mock_backend)
        mock_backend.db.delete_user_rate_limit = MagicMock(return_value=False)

        response = test_client.delete(
            "/admin/rate-limits/999",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 404
