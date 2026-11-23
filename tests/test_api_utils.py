import io
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile


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
