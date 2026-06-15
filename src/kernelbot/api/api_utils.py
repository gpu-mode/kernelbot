import asyncio
from dataclasses import asdict
from datetime import datetime
from re import sub
import time
from typing import Any, Optional, Tuple
import requests
from fastapi import HTTPException, UploadFile
from fastapi import BackgroundTasks
import json
from kernelbot.env import env
from libkernelbot.backend import KernelBackend
from libkernelbot.consts import SubmissionMode
from libkernelbot.report import (
    Log,
    MultiProgressReporter,
    RunProgressReporter,
    RunResultReport,
    Text,
)
from libkernelbot.submission import SubmissionRequest, prepare_submission
from src.kernelbot.api.main import simple_rate_limit
from src.libkernelbot.leaderboard_db import LeaderboardDB


async def _handle_discord_oauth(
    code: str, redirect_uri: str
) -> tuple[str, str]:
    """Handles the Discord OAuth code exchange and user info retrieval."""
    client_id = env.CLI_DISCORD_CLIENT_ID
    client_secret = env.CLI_DISCORD_CLIENT_SECRET
    token_url = env.CLI_TOKEN_URL
    user_api_url = "https://discord.com/api/users/@me"

    if not client_id:
        raise HTTPException(
            status_code=500, detail="Discord client ID not configured."
        )
    if not client_secret:
        raise HTTPException(
            status_code=500, detail="Discord client secret not configured."
        )
    if not token_url:
        raise HTTPException(
            status_code=500, detail="Discord token URL not configured."
        )

    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    try:
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with Discord token endpoint: {e}",
        ) from e

    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to get access token from Discord: {token_response.text}",
        )

    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        user_response = requests.get(user_api_url, headers=headers)
        user_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with Discord user endpoint: {e}",
        ) from e

    user_json = user_response.json()
    user_id = user_json.get("id")
    user_name = user_json.get("username")

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user ID or username from Discord.",
        )

    return user_id, user_name


async def _handle_github_oauth(
    code: str, redirect_uri: str
) -> tuple[str, str]:
    """Handles the GitHub OAuth code exchange and user info retrieval."""
    client_id = env.CLI_GITHUB_CLIENT_ID
    client_secret = env.CLI_GITHUB_CLIENT_SECRET

    token_url = "https://github.com/login/oauth/access_token"
    user_api_url = "https://api.github.com/user"

    if not client_id:
        raise HTTPException(
            status_code=500, detail="GitHub client ID not configured."
        )
    if not client_secret:
        raise HTTPException(
            status_code=500, detail="GitHub client secret not configured."
        )

    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    print(token_data)
    headers = {"Accept": "application/json"}  # Request JSON response for token

    try:
        token_response = requests.post(
            token_url, data=token_data, headers=headers
        )
        token_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with GitHub token endpoint: {e}",
        ) from e

    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to get access token from GitHub: {token_response.text}",
        )

    auth_headers = {"Authorization": f"Bearer {access_token}"}
    try:
        user_response = requests.get(user_api_url, headers=auth_headers)
        user_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with GitHub user endpoint: {e}",
        ) from e

    user_json = user_response.json()
    user_id = str(
        user_json.get("id")
    )  # GitHub ID is integer, convert to string for consistency
    user_name = user_json.get("login")  # GitHub uses 'login' for username

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user ID or username from GitHub.",
        )

    return user_id, user_name


class MultiProgressReporterAPI(MultiProgressReporter):
    def __init__(self):
        self.runs = []

    async def show(self, title: str):
        return

    def add_run(self, title: str) -> "RunProgressReporterAPI":
        rep = RunProgressReporterAPI(title)
        self.runs.append(rep)
        return rep

    def make_message(self):
        return


class RunProgressReporterAPI(RunProgressReporter):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.long_report = ""

    async def _update_message(self):
        pass

    async def display_report(self, title: str, report: RunResultReport):
        for part in report.data:
            if isinstance(part, Text):
                self.long_report += part.text
            elif isinstance(part, Log):
                self.long_report += f"\n\n## {part.header}:\n"
                self.long_report += f"```\n{part.content}```"


async def to_submission_info(
    user_info: Any,
    submission_mode: str,
    file: UploadFile,
    leaderboard_name: str,
    gpu_type: str,
    db_context: LeaderboardDB,
) -> tuple[SubmissionRequest, SubmissionMode]:
    user_name = user_info["user_name"]
    user_id = user_info["user_id"]

    try:
        submission_mode_enum: SubmissionMode = SubmissionMode(
            submission_mode.lower()
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid submission mode value: '{submission_mode}'",
        ) from None

    if submission_mode_enum in [SubmissionMode.PROFILE]:
        raise HTTPException(
            status_code=400,
            detail="Profile submissions are not currently supported via API",
        )

    allowed_modes = [
        SubmissionMode.TEST,
        SubmissionMode.BENCHMARK,
        SubmissionMode.LEADERBOARD,
    ]
    if submission_mode_enum not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Submission mode '{submission_mode}' is not supported for this endpoint",
        )

    try:
        with db_context as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            gpus = leaderboard_item.get("gpu_types", [])
            if gpu_type not in gpus:
                supported_gpus = ", ".join(gpus) if gpus else "None"
                raise HTTPException(
                    status_code=400,
                    detail=f"GPU type '{gpu_type}' is not supported for "
                    f"leaderboard '{leaderboard_name}'. Supported GPUs: {supported_gpus}",
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while validating leaderboard/GPU: {e}",
        ) from e

    try:
        submission_content = await file.read()
        if not submission_content:
            raise HTTPException(
                status_code=400,
                detail="Empty file submitted. Please provide a file with code.",
            )
        if len(submission_content) > 1_000_000:
            raise HTTPException(
                status_code=413,
                detail="Submission file is too large (limit: 1MB).",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading submission file: {e}"
        ) from e

    try:
        submission_code = submission_content.decode("utf-8")
        submission_request = SubmissionRequest(
            code=submission_code,
            file_name=file.filename or "submission.py",
            user_id=user_id,
            user_name=user_name,
            gpus=[gpu_type],
            leaderboard=leaderboard_name,
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode submission file content as UTF-8.",
        ) from None
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error creating submission request: {e}",
        ) from e

    return submission_request, submission_mode_enum


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def _run_submission(
    submission: SubmissionRequest,
    mode: SubmissionMode,
    backend: KernelBackend,
    submission_id: Optional[int] = None,
):
    try:
        req = prepare_submission(submission, backend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if len(req.gpus) != 1:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    reporter = MultiProgressReporterAPI()
    sub_id, results = await backend.submit_full(
        req, mode, reporter, submission_id
    )
    return (
        results,
        [rep.get_message() + "\n" + rep.long_report for rep in reporter.runs],
        sub_id,
    )


def start_detached_run(
    submission_request: SubmissionRequest,
    submission_mode_enum: SubmissionMode,
    backend: KernelBackend,
    db: "LeaderboardDB",
    background_tasks: BackgroundTasks,
) -> int:
    """Starts a submission in the background and returns the submission id immediately."""

    # create submission id, so that it can be return to client before task is started
    with db:
        req = submission_request
        sub_id = db.create_submission(
            leaderboard=req.leaderboard,
            file_name=req.file_name,
            code=req.code,
            user_id=req.user_id,
            time=datetime.now(),
            user_name=req.user_name,
        )

    # makes the task run in the background
    background_tasks.add_task(
        _run_submission,
        submission_request,
        submission_mode_enum,
        backend,
        db,
        sub_id,
    )
    return sub_id


async def sse_stream_submission(
    submission_request: SubmissionRequest,
    submission_mode_enum: SubmissionMode,
    backend: KernelBackend,
):
    start_time = time.time()
    task: asyncio.Task | None = None
    try:
        task = asyncio.create_task(
            _run_submission(submission_request, submission_mode_enum, backend)
        )

        while not task.done():
            elapsed_time = time.time() - start_time
            yield (
                "event: status\n"
                f"data: {json.dumps({'status': 'processing','elapsed_time': round(elapsed_time, 2)}, default=json_serializer)}\n\n"
            )
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'status': 'error','detail': 'Submission cancelled'}, default=json_serializer)}\n\n"
                )
                return

        result, reports = await task
        result_data = {
            "status": "success",
            "results": [asdict(r) for r in result],
            "reports": reports,
        }
        yield "event: result\n" f"data: {json.dumps(result_data, default=json_serializer)}\n\n"

    except HTTPException as http_exc:
        error_data = {
            "status": "error",
            "detail": http_exc.detail,
            "status_code": http_exc.status_code,
        }
        yield "event: error\n" f"data: {json.dumps(error_data, default=json_serializer)}\n\n"
    except Exception as e:
        error_type = type(e).__name__
        error_data = {
            "status": "error",
            "detail": f"An unexpected error occurred: {error_type}",
            "raw_error": str(e),
        }
        yield "event: error\n" f"data: {json.dumps(error_data, default=json_serializer)}\n\n"
    finally:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
