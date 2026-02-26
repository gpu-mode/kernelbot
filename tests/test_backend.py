import datetime
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from test_leaderboard_db import _submit_leaderboard
from test_report import create_eval_result, sample_system_info

from libkernelbot import backend, consts, report
from libkernelbot.run_eval import FullResult


class MockProgressReporter(report.RunProgressReporter):
    def __init__(self, title):
        super().__init__(title)
        self._update_message = AsyncMock()
        self.display_report = AsyncMock()


class MockMultReporter(report.MultiProgressReporter):
    def __init__(self):
        super().__init__()
        self.show = AsyncMock()
        self.make_message = MagicMock()
        self.reporter_list: list[MockProgressReporter] = []

    def add_run(self, title: str) -> "report.RunProgressReporter":
        self.reporter_list.append(MockProgressReporter(title))
        return self.reporter_list[-1]


def _mock_launcher(bot: backend.KernelBackend, runs: dict, name="launcher"):
    mock_launcher = MagicMock(spec=backend.Launcher)
    mock_launcher.name = name
    mock_launcher.gpus = [consts.ModalGPU.A100]
    mock_launcher.run_submission = AsyncMock(
        return_value=FullResult(success=True, error="", system=sample_system_info(), runs=runs)
    )
    bot.register_launcher(mock_launcher)
    return mock_launcher


@pytest.mark.asyncio
async def test_handle_submission(bot: backend.KernelBackend, task_directory):
    _submit_leaderboard(bot.db, task_directory)
    with bot.db as db:
        task = db.get_leaderboard("submit-leaderboard")["task"]
    mock_launcher = _mock_launcher(bot, {"test": create_eval_result()})

    reporter = MockProgressReporter("report")

    await bot.handle_submission(
        consts.ModalGPU.A100,
        reporter,
        "pass",
        "submit.py",
        task,
        consts.SubmissionMode.LEADERBOARD,
        -1,
    )

    assert reporter.title == "report ✅ success"
    assert reporter.lines == [
        "> ✅ Compilation successful",
        "> ✅ Testing successful",
        "> ❌ Benchmarks missing",
        "> ❌ Leaderboard missing",
    ]

    call_args = reporter.display_report.call_args[0]
    assert call_args[0] == "❌ submit.py on A100 (launcher)"
    from libkernelbot.report import Log, Text

    assert call_args[1].data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Device count: `1`\n"
            "* Runtime: `CUDA`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
            "* Hostname: `test-hostname`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Log(header="Program stdout", content="log stdout"),
    ]

    assert mock_launcher.run_submission.call_count == 1
    assert mock_launcher.run_submission.call_args[0][0] == {
        "arch": "80",
        "benchmark_timeout": 180,
        "benchmarks": [{"dtype": "float32", "input_size": 10000}],
        "lang": "py",
        "main": "kernel.py",
        "mode": "leaderboard",
        "multi_gpu": False,
        "ranked_timeout": 180,
        "ranking_by": "geom",
        "score_ascending": True,
        "seed": None,
        "sources": {"kernel.py": "def kernel(): pass", "submission.py": "pass"},
        "test_timeout": 120,
        "tests": [{"dtype": "float32", "input_size": 1000}],
    }

    with bot.db as db:
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.runs")
        assert db.cursor.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_submit_leaderboard(bot: backend.KernelBackend, task_directory):
    _submit_leaderboard(bot.db, task_directory)
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)
    with bot.db as db:
        task = db.get_leaderboard("submit-leaderboard")["task"]
        s_id = db.create_submission(
            "submit-leaderboard",
            "submit.py",
            34,
            "pass",
            submit_time,
        )
    eval_result = create_eval_result("benchmark")
    mock_launcher = _mock_launcher(bot, {"leaderboard": eval_result})

    reporter = MockProgressReporter("report")

    await bot.submit_leaderboard(
        s_id,
        "pass",
        "submit.py",
        consts.ModalGPU.A100,
        reporter,
        task,
        consts.SubmissionMode.LEADERBOARD,
        seed=1337,
    )

    # make sure we're not messing up the original seed
    assert task.seed is None

    assert mock_launcher.run_submission.call_count == 1
    assert mock_launcher.run_submission.call_args[0][0] == {
        "arch": "80",
        "benchmark_timeout": 180,
        "benchmarks": [{"dtype": "float32", "input_size": 10000}],
        "lang": "py",
        "main": "kernel.py",
        "mode": "leaderboard",
        "multi_gpu": False,
        "ranked_timeout": 180,
        "ranking_by": "geom",
        "score_ascending": True,
        "seed": 1337,
        "sources": {"kernel.py": "def kernel(): pass", "submission.py": "pass"},
        "test_timeout": 120,
        "tests": [{"dtype": "float32", "input_size": 1000}],
    }

    with bot.db as db:
        sub = db.get_submission_by_id(s_id)
        assert sub == {
            "code": "pass",
            "done": False,
            "file_name": "submit.py",
            "leaderboard_id": s_id,
            "leaderboard_name": "submit-leaderboard",
            "runs": [
                {
                    "compilation": {
                        "command": "nvcc -o test test.cu",
                        "exit_code": 0,
                        "nvcc_found": True,
                        "nvcc_version": "11.8",
                        "stderr": "",
                        "stdout": "",
                        "success": True,
                    },
                    "end_time": eval_result.end.replace(tzinfo=datetime.timezone.utc),
                    "meta": {
                        "command": "./test",
                        "duration": 1.5,
                        "exit_code": 0,
                        "stderr": "",
                        "stdout": "log stdout",
                        "success": True,
                    },
                    "mode": "leaderboard",
                    "passed": True,
                    "result": {
                        "benchmark-count": "1",
                        "benchmark.0.best": "1.3",
                        "benchmark.0.err": "0.1",
                        "benchmark.0.mean": "1.5",
                        "benchmark.0.spec": "Matrix multiplication",
                        "benchmark.0.status": "pass",
                        "benchmark.0.worst": "1.8",
                    },
                    "runner": "A100",
                    "score": Decimal("1.5e-9"),
                    "secret": False,
                    "start_time": eval_result.start.replace(tzinfo=datetime.timezone.utc),
                    "system": {
                        "cpu": "Intel i9-12900K",
                        "device_count": 1,
                        "gpu": "NVIDIA RTX 4090",
                        "platform": "Linux-5.15.0",
                        "runtime": "CUDA",
                        "torch": "2.0.1+cu118",
                        "hostname": "test-hostname",
                    },
                }
            ],
            "submission_id": 1,
            "submission_time": submit_time,
            "user_id": "34",
        }


@pytest.mark.asyncio
async def test_submit_full(bot: backend.KernelBackend, task_directory):
    _submit_leaderboard(bot.db, task_directory)
    with bot.db as db:
        task = db.get_leaderboard("submit-leaderboard")["task"]

    eval_result = create_eval_result("benchmark")
    mock_launcher = _mock_launcher(bot, {"leaderboard": eval_result})

    from libkernelbot.submission import ProcessedSubmissionRequest

    req = ProcessedSubmissionRequest(
        code="pass",
        file_name="submission.py",
        user_id=5,
        user_name="user",
        gpus=["A100"],
        leaderboard="submit-leaderboard",
        task=task,
        secret_seed=42,
        task_gpus=["A100", "H100"],
    )
    reporter = MockMultReporter()
    s_id, results = await bot.submit_full(
        req, mode=consts.SubmissionMode.LEADERBOARD, reporter=reporter
    )

    expected_result = mock_launcher.run_submission.return_value
    assert len(results) == 2
    assert results == [expected_result, expected_result]

    r1, r2 = reporter.reporter_list
    assert r1.lines == [
        "> ✅ Compilation successful",
        "> ❌ Tests missing",
        "> ❌ Benchmarks missing",
        "> ✅ Leaderboard run successful",
    ]
    assert r2.lines == [
        "> ✅ Compilation successful",
        "> ❌ Tests missing",
        "> ❌ Benchmarks missing",
        "> ✅ Leaderboard run successful",
    ]
    assert r1.title == "A100 on Modal ✅ success"
    assert r2.title == "A100 on Modal (secret) ✅ success"

    with bot.db as db:
        db_results = db.get_submission_by_id(s_id)
        assert db_results == {
            "code": "pass",
            "done": True,
            "file_name": "submission.py",
            "leaderboard_id": 1,
            "leaderboard_name": "submit-leaderboard",
            "runs": [
                {
                    "compilation": {
                        "command": "nvcc -o test test.cu",
                        "exit_code": 0,
                        "nvcc_found": True,
                        "nvcc_version": "11.8",
                        "stderr": "",
                        "stdout": "",
                        "success": True,
                    },
                    "end_time": ANY,
                    "meta": {
                        "command": "./test",
                        "duration": 1.5,
                        "exit_code": 0,
                        "stderr": "",
                        "stdout": "log stdout",
                        "success": True,
                    },
                    "mode": "leaderboard",
                    "passed": True,
                    "result": {
                        "benchmark-count": "1",
                        "benchmark.0.best": "1.3",
                        "benchmark.0.err": "0.1",
                        "benchmark.0.mean": "1.5",
                        "benchmark.0.spec": "Matrix multiplication",
                        "benchmark.0.status": "pass",
                        "benchmark.0.worst": "1.8",
                    },
                    "runner": "A100",
                    "score": Decimal("1.5E-9"),
                    "secret": False,
                    "start_time": ANY,
                    "system": {
                        "cpu": "Intel i9-12900K",
                        "device_count": 1,
                        "gpu": "NVIDIA RTX 4090",
                        "platform": "Linux-5.15.0",
                        "runtime": "CUDA",
                        "torch": "2.0.1+cu118",
                        "hostname": "test-hostname",
                    },
                },
                {
                    "compilation": {
                        "command": "nvcc -o test test.cu",
                        "exit_code": 0,
                        "nvcc_found": True,
                        "nvcc_version": "11.8",
                        "stderr": "",
                        "stdout": "",
                        "success": True,
                    },
                    "end_time": ANY,
                    "meta": {
                        "command": "./test",
                        "duration": 1.5,
                        "exit_code": 0,
                        "stderr": "",
                        "stdout": "log stdout",
                        "success": True,
                    },
                    "mode": "leaderboard",
                    "passed": True,
                    "result": {
                        "benchmark-count": "1",
                        "benchmark.0.best": "1.3",
                        "benchmark.0.err": "0.1",
                        "benchmark.0.mean": "1.5",
                        "benchmark.0.spec": "Matrix multiplication",
                        "benchmark.0.status": "pass",
                        "benchmark.0.worst": "1.8",
                    },
                    "runner": "A100",
                    "score": Decimal("1.5E-9"),
                    "secret": True,
                    "start_time": ANY,
                    "system": {
                        "cpu": "Intel i9-12900K",
                        "device_count": 1,
                        "gpu": "NVIDIA RTX 4090",
                        "platform": "Linux-5.15.0",
                        "runtime": "CUDA",
                        "torch": "2.0.1+cu118",
                        "hostname": "test-hostname",
                    },
                },
            ],
            "submission_id": 1,
            "submission_time": ANY,
            "user_id": "5",
        }
