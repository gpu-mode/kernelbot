from unittest.mock import Mock, patch

import pytest

from kernelbot import admin_cli


def test_validate_top10_enqueues_current_top_ten(capsys):
    response = Mock()
    response.json.return_value = {
        "status": "accepted",
        "leaderboard": "cholesky",
        "gpu_type": "B200",
    }

    with (
        patch.dict(
            "os.environ",
            {
                "DISCORD_CLUSTER_MANAGER_API_BASE_URL": "https://kernelbot.test/",
                "ADMIN_TOKEN": "secret",
            },
        ),
        patch("kernelbot.admin_cli.requests.post", return_value=response) as post,
        patch("sys.argv", ["kernelbot-admin", "validate-top10", "cholesky", "B200"]),
    ):
        assert admin_cli.main() == 0

    post.assert_called_once_with(
        "https://kernelbot.test/admin/application-validations/cholesky/B200",
        headers={"Authorization": "Bearer secret"},
        params={"wait": "false"},
        timeout=30,
    )
    response.raise_for_status.assert_called_once_with()
    assert '"status": "accepted"' in capsys.readouterr().out


def test_validate_top10_can_wait_and_url_encodes_names():
    response = Mock()
    response.json.return_value = {"status": "completed", "results": []}

    with (
        patch.dict("os.environ", {"ADMIN_TOKEN": "secret"}, clear=True),
        patch("kernelbot.admin_cli.requests.post", return_value=response) as post,
        patch(
            "sys.argv",
            [
                "kernelbot-admin",
                "--api-url",
                "https://kernelbot.test",
                "validate-top10",
                "batched cholesky",
                "B200",
                "--wait",
            ],
        ),
    ):
        assert admin_cli.main() == 0

    assert post.call_args.kwargs["params"] == {"wait": "true"}
    assert post.call_args.kwargs["timeout"] is None
    assert post.call_args.args[0].endswith(
        "/application-validations/batched%20cholesky/B200"
    )


def test_validate_top10_requires_admin_token():
    with (
        patch.dict(
            "os.environ",
            {"DISCORD_CLUSTER_MANAGER_API_BASE_URL": "https://kernelbot.test"},
            clear=True,
        ),
        patch("sys.argv", ["kernelbot-admin", "validate-top10", "cholesky", "B200"]),
        pytest.raises(SystemExit, match="Set ADMIN_TOKEN"),
    ):
        admin_cli.main()
