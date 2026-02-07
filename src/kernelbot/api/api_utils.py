from typing import Any

import requests
from fastapi import HTTPException, UploadFile

from kernelbot.env import env
from libkernelbot.backend import KernelBackend
from libkernelbot.consts import SubmissionMode
from libkernelbot.leaderboard_db import LeaderboardDB
from libkernelbot.report import (
    Log,
    MultiProgressReporter,
    RunProgressReporter,
    RunResultReport,
    Text,
)
from libkernelbot.submission import (
    SubmissionRequest,
    prepare_submission,
)


async def _handle_discord_oauth(code: str, redirect_uri: str) -> tuple[str, str]:
    """Handles the Discord OAuth code exchange and user info retrieval."""
    client_id = env.CLI_DISCORD_CLIENT_ID
    client_secret = env.CLI_DISCORD_CLIENT_SECRET
    token_url = env.CLI_TOKEN_URL
    user_api_url = "https://discord.com/api/users/@me"

    if not client_id:
        raise HTTPException(status_code=500, detail="Discord client ID not configured.")
    if not client_secret:
        raise HTTPException(status_code=500, detail="Discord client secret not configured.")
    if not token_url:
        raise HTTPException(status_code=500, detail="Discord token URL not configured.")

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
            status_code=500, detail="Failed to retrieve user ID or username from Discord."
        )

    return user_id, user_name


async def _handle_github_oauth(code: str, redirect_uri: str) -> tuple[str, str]:
    """Handles the GitHub OAuth code exchange and user info retrieval."""
    client_id = env.CLI_GITHUB_CLIENT_ID
    client_secret = env.CLI_GITHUB_CLIENT_SECRET

    token_url = "https://github.com/login/oauth/access_token"
    user_api_url = "https://api.github.com/user"

    if not client_id:
        raise HTTPException(status_code=500, detail="GitHub client ID not configured.")
    if not client_secret:
        raise HTTPException(status_code=500, detail="GitHub client secret not configured.")

    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    print(token_data)
    headers = {"Accept": "application/json"}  # Request JSON response for token

    try:
        token_response = requests.post(token_url, data=token_data, headers=headers)
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
    user_id = str(user_json.get("id"))  # GitHub ID is integer, convert to string for consistency
    user_name = user_json.get("login")  # GitHub uses 'login' for username

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve user ID or username from GitHub."
        )

    return user_id, user_name


async def _run_submission(
    submission: SubmissionRequest, mode: SubmissionMode, backend: KernelBackend
):
    try:
        req = prepare_submission(submission, backend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if len(req.gpus) != 1:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    reporter = MultiProgressReporterAPI()
    sub_id, results = await backend.submit_full(req, mode, reporter)
    return results, [rep.get_message() + "\n" + rep.long_report for rep in reporter.runs]


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


# ruff: noqa: C901
async def to_submit_info(
    user_info: Any,
    submission_mode: str,
    file: UploadFile,
    leaderboard_name: str,
    gpu_type: str,
    db_context: LeaderboardDB,
) -> tuple[SubmissionRequest, SubmissionMode]:  # noqa: C901
    user_name = user_info["user_name"]
    user_id = user_info["user_id"]

    try:
        submission_mode_enum: SubmissionMode = SubmissionMode(submission_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid submission mode value: '{submission_mode}'",
        ) from None

    allowed_modes = [
        SubmissionMode.TEST,
        SubmissionMode.BENCHMARK,
        SubmissionMode.PROFILE,
        SubmissionMode.LEADERBOARD,
    ]
    if submission_mode_enum not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Submission mode '{submission_mode}' is not supported for this endpoint",
        )

    try:
        with db_context as db:
            # Per-user rate limit: max 1 submission per hour on Modal B200 for leaderboard 730
            if gpu_type == "B200":
                lb_id = db.get_leaderboard_id(leaderboard_name)
                if lb_id == 730:
                    last_submission_time = db.check_user_rate_limit(user_id)
                    if last_submission_time:
                        raise HTTPException(
                            status_code=429,
                            detail=(
                                f"Rate limit exceeded. You can submit once per hour. "
                                f"Last submission: {last_submission_time.isoformat()}. "
                                f"Consider using the NVIDIA runner instead of Modal for faster iteration."
                            ),
                        )

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
            detail=f"Internal server error while validating submission: {e}",
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
        raise HTTPException(status_code=400, detail=f"Error reading submission file: {e}") from e

    try:
        submission_code = submission_content.decode("utf-8")
        if "stream" in submission_code.lower():
            raise HTTPException(
                status_code=500,
                detail="Your code contains work on another stream. This is not allowed and may result in your disqualification. If you think this is a mistake, please contact us.",  # noqa: E501
            )
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error creating submission request: {e}",
        ) from e

    return submission_request, submission_mode_enum
