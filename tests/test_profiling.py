"""Hosted profiling contract tests; no cloud account or GPU is required."""

import base64
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from libkernelbot import run_eval
from libkernelbot.consts import SubmissionMode
from libkernelbot.profiling import ProfileOptions
from libkernelbot.submission import SubmissionRequest


@pytest.mark.parametrize(
    "values",
    [
        {"benchmark_index": -1},
        {"benchmark_index": True},
        {"ncu_launch_count": 0},
        {"ncu_kernel_name_base": "invalid"},
        {"ncu_kernel_name": ""},
    ],
)
def test_invalid_capture_options(values):
    with pytest.raises(ValueError):
        ProfileOptions(**values)


def test_reject_out_of_range_and_multi_gpu():
    with pytest.raises(ValueError):
        ProfileOptions(benchmark_index=2).validate_task([{"n": 32}], False)
    with pytest.raises(ValueError):
        ProfileOptions().validate_task([{"n": 32}], True)


def test_runner_selects_original_benchmark_index_without_mutating_config():
    config = {
        "lang": "py",
        "sources": {"eval.py": "pass"},
        "main": "eval.py",
        "mode": "profile",
        "benchmarks": [{"n": 32}, {"n": 512}],
        "profile_options": {"benchmark_index": 1, "ncu_kernel_name": "regex:solver"},
        "cpu_compile": {"artifacts": b"not-json"},
    }
    with (
        patch.object(
            run_eval, "make_system_info", return_value=run_eval.SystemInfo(runtime="CUDA")
        ),
        patch.object(run_eval, "run_pytorch_script", return_value="captured") as runner,
    ):
        result = run_eval.run_config(config)
    assert result.runs == {"profile.1": "captured"}
    assert runner.call_args.kwargs["benchmarks"] == "n: 512"
    assert runner.call_args.kwargs["profile_options"]["ncu_kernel_name"] == "regex:solver"
    assert len(config["benchmarks"]) == 2
    assert result.profile_metadata["benchmark_specs"] == {"1": {"n": 512}}
    assert len(result.profile_metadata["config_sha256"]) == 64


def test_ncu_captures_children_and_exports_reports(tmp_path):
    result = SimpleNamespace(success=True, stderr="", result={})

    def capture(command, **kwargs):
        assert command[command.index("--clock-control") + 1] == "none"
        assert command[command.index("--target-processes") + 1] == "all"
        assert command[command.index("--kernel-name") + 1] == "regex:solver|a b"
        assert command[command.index("--launch-count") + 1] == "2"
        assert kwargs["extra_env"] == {"POPCORN_NCU": "1"}
        (tmp_path / "profile.ncu-rep").write_bytes(b"report")
        return result

    with (
        patch.object(run_eval, "run_program", side_effect=capture),
        patch.object(
            run_eval.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="counter data", stderr=""),
        ),
    ):
        run, profile = run_eval.profile_program_ncu(
            ["python3", "eval.py"],
            None,
            30,
            False,
            tmp_path,
            {"ncu_kernel_name": "regex:solver|a b", "ncu_launch_count": 2},
        )
    assert run.success
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(profile.trace))) as archive:
        names = {name.split("/")[-1] for name in archive.namelist()}
        assert {"profile.ncu-rep", "ncu-details.txt", "ncu-details.csv"} <= names


def test_empty_ncu_capture_is_a_failure(tmp_path):
    result = SimpleNamespace(success=True, stderr="", stdout="No kernels profiled", result={})
    with patch.object(run_eval, "run_program", return_value=result):
        run, profile = run_eval.profile_program_ncu(["eval"], None, 30, False, tmp_path)
    assert not run.success
    assert profile is None
    assert "did not produce a report" in run.stderr
    assert run.stdout == "No kernels profiled"


def test_profile_endpoint_requires_normal_cli_auth():
    from kernelbot.api.main import app, get_db

    app.dependency_overrides[get_db] = lambda: MagicMock()
    try:
        response = TestClient(app).post(
            "/profile/qr_v2/B200", files={"file": ("submission.py", b"pass")}
        )
        assert response.status_code == 400
        assert "Missing X-Popcorn-Cli-Id" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_profile_endpoint_preserves_options_and_validates_input():
    from kernelbot.api import main

    captured = []

    async def stream(request, mode, backend):
        captured.append((request, mode))
        yield 'event: result\ndata: {"results": []}\n\n'

    request = SubmissionRequest("pass", "submission.py", 1, "test", ["B200"], "qr_v2")
    main.app.dependency_overrides[main.validate_cli_header] = lambda: {
        "user_id": 1,
        "user_name": "test",
    }
    main.app.dependency_overrides[main.get_db] = lambda: MagicMock()
    try:
        with (
            patch.object(
                main, "to_submit_info", AsyncMock(return_value=(request, SubmissionMode.PROFILE))
            ),
            patch.object(main, "_stream_submission_response", stream),
            patch.object(main, "simple_rate_limit", AsyncMock()),
        ):
            client = TestClient(main.app)
            response = client.post(
                "/profile/qr_v2/B200",
                files={"file": ("submission.py", b"pass")},
                data={
                    "benchmark_index": "1",
                    "ncu_kernel_name": "regex:solver",
                    "ncu_launch_count": "2",
                },
            )
            assert response.status_code == 200
            assert captured[0][0].profile_options == {
                "benchmark_index": 1,
                "ncu_kernel_name": "regex:solver",
                "ncu_launch_count": 2,
            }
            assert captured[0][1] == SubmissionMode.PROFILE
            for data, status in [
                ({"benchmark_index": "-1"}, 422),
                ({"ncu_launch_count": "0"}, 422),
                ({"ncu_kernel_name_base": "invalid"}, 400),
            ]:
                response = client.post(
                    "/profile/qr_v2/B200", files={"file": ("submission.py", b"pass")}, data=data
                )
                assert response.status_code == status
            assert len(captured) == 1
    finally:
        main.app.dependency_overrides.clear()
