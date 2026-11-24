import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient


async def submit(project_root, code: bytes):
    # Importing at collection time would load kernelbot.env before test_admin_api patches it.
    from kernelbot.api.api_utils import to_submit_info

    db = MagicMock()
    db.__enter__.return_value.get_leaderboard.return_value = {"gpu_types": ["H100"]}
    return await to_submit_info(
        {"user_name": "user", "user_id": "1", "id_type": "cli"},
        "leaderboard",
        UploadFile(io.BytesIO(code), filename="submission_mojo.py"),
        "vectoradd_py-dev",
        "H100",
        db,
    )


@pytest.mark.asyncio
async def test_mojo_example_passes_submission_filters(project_root):
    code = (project_root / "examples" / "vectoradd_py" / "submission_mojo.py").read_bytes()

    request, _ = await submit(project_root, code)

    assert request.code == code.decode()


@pytest.mark.asyncio
async def test_submissions_mentioning_streams_are_rejected(project_root):
    with pytest.raises(HTTPException, match="another stream"):
        await submit(project_root, b"s = torch.cuda.current_stream()\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("global_name", "expected"),
    [("Alice Öztürk", "Alice Öztürk"), (None, "alice_dev"), ("", "alice_dev")],
)
async def test_discord_oauth_prefers_global_name_with_username_fallback(monkeypatch, global_name, expected):
    from kernelbot.api import api_utils

    monkeypatch.setattr(api_utils.env, "CLI_DISCORD_CLIENT_ID", "discord-client-id")
    monkeypatch.setattr(api_utils.env, "CLI_DISCORD_CLIENT_SECRET", "discord-client-secret")
    monkeypatch.setattr(api_utils.env, "CLI_TOKEN_URL", "https://discord.example/token")
    monkeypatch.setattr(api_utils.requests, "post", lambda *a, **k: MagicMock(json=lambda: {"access_token": "tok"}))
    user = {"id": "123456789", "username": "alice_dev", "global_name": global_name}
    monkeypatch.setattr(api_utils.requests, "get", lambda *a, **k: MagicMock(json=lambda: user))

    assert await api_utils._handle_discord_oauth("code", "https://example.com/cb") == ("123456789", expected)


def test_cli_auth_reset_passes_refreshed_user_name(monkeypatch):
    from kernelbot.api import main

    mock_db = MagicMock()
    monkeypatch.setenv("POPCORN_API_URL", "https://example.com")
    monkeypatch.setattr(main, "_handle_discord_oauth", AsyncMock(return_value=("123456789", "Alice Öztürk")))
    state = base64.urlsafe_b64encode(b'{"cli_id":"cli-2","is_reset":true}').decode()
    main.app.dependency_overrides[main.get_db] = lambda: mock_db
    try:
        resp = TestClient(main.app).get(f"/auth/cli/discord?code=abc&state={state}")
        assert resp.status_code == 200
        assert resp.json()["user_name"] == "Alice Öztürk"
        mock_db.__enter__.return_value.reset_user_from_cli.assert_called_once_with(
            "123456789", "cli-2", "discord", "Alice Öztürk"
        )
    finally:
        main.app.dependency_overrides.clear()
