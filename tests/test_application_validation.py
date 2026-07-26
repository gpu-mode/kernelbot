import asyncio
from types import SimpleNamespace

import pytest

from libkernelbot.application_validation import ApplicationValidationService
from libkernelbot.task import LeaderboardTask


def _task() -> LeaderboardTask:
    return LeaderboardTask.from_dict(
        {
            "lang": "py",
            "files": {"submission.py": "@SUBMISSION@"},
            "config": {"main": "submission.py"},
            "validation": {
                "version": "v1",
                "source": "print('validator')",
            },
        }
    )


class FakeDB:
    def __init__(self):
        self.task = _task()
        self.leaderboard = {
            "id": 7,
            "name": "cholesky",
            "task": self.task,
            "gpu_types": ["B200"],
        }
        self.submissions = [
            {
                "submission_id": submission_id,
                "submission_name": f"submission-{submission_id}.py",
            }
            for submission_id in range(1, 13)
        ]
        self.claims = set()
        self.saved = []
        self.sweeps = []
        self.requested_limits = []
        self.validated_ids = set()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_leaderboard(self, name):
        assert name == "cholesky"
        return self.leaderboard

    def claim_validation_sweep(
        self,
        *,
        leaderboard_id,
        gpu_type,
        contract_version,
        scheduled_for,
    ):
        key = (leaderboard_id, gpu_type, contract_version, scheduled_for)
        if scheduled_for is not None and key in self.claims:
            return None
        self.claims.add(key)
        return len(self.claims)

    def get_leaderboard_submissions(self, _name, _gpu, limit):
        self.requested_limits.append(limit)
        return self.submissions[:limit]

    def get_submission_code_for_validation(self, submission_id):
        return f"# submission {submission_id}"

    def get_submission_validation_ids(
        self,
        submission_ids,
        *,
        gpu_type,
        contract_version,
    ):
        assert gpu_type == "B200"
        assert contract_version == "v1"
        return set(submission_ids) & self.validated_ids

    def upsert_submission_validation(self, **values):
        self.saved.append(values)

    def complete_validation_sweep(self, sweep_id, *, status, error=None):
        self.sweeps.append((sweep_id, status, error))


class FakeLauncher:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def run_validation(self, config, _gpu):
        assert "@SUBMISSION@" not in config["sources"]["submission.py"]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return {
            "status": "completed",
            "result": {
                "passed_shapes": 2,
                "total_shapes": 2,
                "fully_validated": True,
                "geomean_sync_wall_speedup": 1.25,
                "results": [
                    {
                        "shape": {"n": 1},
                        "passed": True,
                        "unexpected_private_field": "drop me",
                    }
                ],
            },
        }


@pytest.mark.asyncio
async def test_sweep_validates_top_ten_with_bounded_concurrency():
    database = FakeDB()
    launcher = FakeLauncher()
    backend = SimpleNamespace(db=database, launcher_map={"B200": launcher})
    service = ApplicationValidationService(backend)

    summary = await service.run_sweep(
        "cholesky",
        "B200",
    )

    assert summary["status"] == "completed"
    assert len(summary["results"]) == 10
    assert len(database.saved) == 10
    assert launcher.max_active == 2
    assert database.sweeps == [(1, "completed", None)]
    assert database.saved[0]["fully_validated"] is True
    assert database.saved[0]["result"]["results"][0]["unexpected_private_field"] == "drop me"
    assert database.requested_limits == [10]


@pytest.mark.asyncio
async def test_manual_sweep_can_validate_every_ranked_user():
    database = FakeDB()
    launcher = FakeLauncher()
    backend = SimpleNamespace(db=database, launcher_map={"B200": launcher})
    service = ApplicationValidationService(backend)

    summary = await service.run_sweep(
        "cholesky",
        "B200",
        all_users=True,
    )

    assert summary["status"] == "completed"
    assert summary["all_users"] is True
    assert len(summary["results"]) == 12
    assert len(database.saved) == 12
    assert database.requested_limits == [None]
    assert launcher.max_active == 2


@pytest.mark.asyncio
async def test_manual_sweep_can_validate_only_missing_users():
    database = FakeDB()
    database.validated_ids = {1, 2, 4, 8}
    launcher = FakeLauncher()
    backend = SimpleNamespace(db=database, launcher_map={"B200": launcher})
    service = ApplicationValidationService(backend)

    summary = await service.run_sweep(
        "cholesky",
        "B200",
        all_users=True,
        only_missing=True,
    )

    assert summary["status"] == "completed"
    assert summary["only_missing"] is True
    assert [result["submission_id"] for result in summary["results"]] == [
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
    ]
    assert database.requested_limits == [None]
    assert launcher.max_active == 2
