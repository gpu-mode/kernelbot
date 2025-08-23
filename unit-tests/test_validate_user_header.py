from typing import Optional

import pytest
from fastapi import HTTPException

from kernelbot.api.main import validate_user_header
from libkernelbot.db_types import IdentityType


class DummyDBCtx:
    def __init__(self, to_return=None, to_raise: Optional[Exception] = None):
        self.to_return = to_return
        self.to_raise = to_raise
        self.seen = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def validate_identity(self, token, id_type):
        self.seen["token"] = token
        self.seen["id_type"] = id_type
        if self.to_raise:
            raise self.to_raise
        return self.to_return

@pytest.mark.asyncio
async def test_cli_header_success():
    test_db = DummyDBCtx(to_return={"user_id": "u2", "user_name": "bob"})
    res = await validate_user_header(
        x_web_auth_id=None,
        x_popcorn_cli_id="clitok",
        db_context=test_db,
    )
    assert res["user_id"] == "u2"
    assert test_db.seen["id_type"] == IdentityType.CLI

@pytest.mark.asyncio
async def test_both_headers_prefers_web():
    test_db = DummyDBCtx(to_return={"user_id": "u3", "user_name": "c"})
    _ = await validate_user_header(
        x_web_auth_id="webtok",
        x_popcorn_cli_id="clitok",
        db_context=test_db,
    )
    assert test_db.seen["token"] == "webtok"
    assert test_db.seen["id_type"] == IdentityType.WEB

@pytest.mark.asyncio
async def test_missing_header_400():
    test_db = DummyDBCtx()
    with pytest.raises(HTTPException) as ei:
        await validate_user_header(None, None, test_db)
    assert ei.value.status_code == 400

@pytest.mark.asyncio
async def test_db_error_500():
    test_db = DummyDBCtx(to_raise=RuntimeError("boom"))
    with pytest.raises(HTTPException) as ei:
        await validate_user_header("webtok", None, test_db)
    assert ei.value.status_code == 500

@pytest.mark.asyncio
async def test_unauthorized_401():
    test_db = DummyDBCtx(to_return=None)
    with pytest.raises(HTTPException) as ei:
        await validate_user_header("webtok", None, test_db)
    assert ei.value.status_code == 401
