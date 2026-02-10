import asyncio
import base64
import dataclasses
import datetime
import io
import json
import math
import os
import pprint
import threading
import uuid
import zipfile
import zlib
from typing import Awaitable, Callable, Optional

import github
import requests
from github import Github, UnknownObjectException
from github.GithubObject import NotSet, Opt
from github.Workflow import Workflow
from github.WorkflowRun import WorkflowRun

from libkernelbot.consts import (
    AMD_REQUIREMENTS,
    DEFAULT_GITHUB_TIMEOUT_MINUTES,
    GPU,
    NVIDIA_REQUIREMENTS,
    TIMEOUT_BUFFER_MINUTES,
    GitHubGPU,
    SubmissionMode,
)
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import (
    CompileResult,
    EvalResult,
    FullResult,
    ProfileResult,
    RunResult,
    SystemInfo,
)
from libkernelbot.utils import KernelBotError, setup_logging

from .launcher import Launcher

logger = setup_logging()


def get_timeout(config: dict) -> int:
    # Model submissions compute timeout from their own config
    if config.get("lang") == "model":
        mc = config.get("model_config", {})
        total_seconds = (
            mc.get("install_timeout", 600)
            + mc.get("server_startup_timeout", 300)
            + mc.get("benchmark_timeout", 1200)
        )
        return math.ceil(total_seconds / 60)

    mode = config.get("mode")
    sec_map = {
        SubmissionMode.TEST.value: config.get("test_timeout"),
        SubmissionMode.BENCHMARK.value: config.get("benchmark_timeout"),
        SubmissionMode.LEADERBOARD.value: config.get("ranked_timeout"),
    }
    seconds = sec_map.get(mode) or DEFAULT_GITHUB_TIMEOUT_MINUTES * 60
    return math.ceil(seconds / 60)


class GitHubLauncher(Launcher):
    def __init__(self, repo: str, token: str, branch: str):
        super().__init__(name="GitHub", gpus=GitHubGPU)
        self.repo = repo
        self.tokens = self._load_github_tokens(token)
        self._token_lock = threading.Lock()
        self._token_idx = 0
        self.branch = branch

    @staticmethod
    def _load_github_tokens(fallback_token: str) -> list[str]:
        primary = (os.getenv("GITHUB_TOKEN") or fallback_token).strip()
        backup = (os.getenv("GITHUB_TOKEN_BACKUP") or "").strip()

        tokens: list[str] = []
        for t in (primary, backup):
            if t and t not in tokens:
                tokens.append(t)

        if not tokens:
            raise KernelBotError(
                "No GitHub tokens configured. Set GITHUB_TOKEN "
                "(and optionally GITHUB_TOKEN_BACKUP)."
            )

        return tokens

    def _next_token(self) -> str:
        with self._token_lock:
            token = self.tokens[self._token_idx]
            self._token_idx = (self._token_idx + 1) % len(self.tokens)
            return token

    async def run_submission(  # noqa: C901
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        gpu_vendor = None
        if gpu_type.value in ["MI300", "MI250", "MI300x8"]:
            selected_workflow = "amd_workflow.yml"
            runner_name = {
                "MI300": "amdgpu-mi300-x86-64",
                "MI250": "amdgpu-mi250-x86-64",
                "MI300x8": "amdgpu-mi300-8-x86-64",
            }[gpu_type.value]
            gpu_vendor = "AMD"
            requirements = AMD_REQUIREMENTS
        elif gpu_type.value == "NVIDIA":
            selected_workflow = "nvidia_workflow.yml"
            gpu_vendor = "NVIDIA"
            requirements = NVIDIA_REQUIREMENTS
        else:
            raise ValueError(f"Invalid GPU type: {gpu_type.value}")

        lang = config["lang"]
        if lang == "cu" and gpu_vendor == "AMD":
            # TODO implement HIP
            raise NotImplementedError("Cannot use CUDA runs with AMD GPUs")

        if lang == "model" and gpu_vendor == "AMD":
            raise NotImplementedError("Model competitions are not supported on AMD GPUs")

        # Override workflow for model submissions
        if lang == "model":
            selected_workflow = "nvidia_model_workflow.yml"

        lang_name = {"py": "Python", "cu": "CUDA", "model": "Model"}[lang]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {selected_workflow}")
        run = GitHubRun(self.repo, self._next_token(), self.branch, selected_workflow)
        logger.info(f"Successfully created GitHub run: {run.run_id}")

        # For model submissions, the archive is too large for workflow dispatch inputs.
        # Upload it as a Git blob and pass the SHA reference instead.
        archive_blob_sha = None
        if lang == "model" and "submission_archive" in config:
            archive_b64 = config.pop("submission_archive")
            blob = await asyncio.to_thread(
                run.repo.create_git_blob, archive_b64, "base64"
            )
            archive_blob_sha = blob.sha  # noqa: F841
            config["archive_blob_sha"] = blob.sha
            logger.info(f"Uploaded submission archive as blob {blob.sha}")

        payload = base64.b64encode(zlib.compress(json.dumps(config).encode("utf-8"))).decode(
            "utf-8"
        )

        inputs = {"payload": payload}
        if lang == "py":
            inputs["requirements"] = requirements
            if gpu_vendor == "AMD":
                inputs["runner"] = runner_name

        if not await run.trigger(inputs):
            raise RuntimeError("Failed to trigger GitHub Action. Please check the configuration.")

        await status.push("⏳ Waiting for workflow to start...")
        logger.info("Waiting for workflow to start...")

        timeout = get_timeout(config) + TIMEOUT_BUFFER_MINUTES

        logger.info(f"Waiting for workflow to complete... (timeout: {timeout} minutes)")
        await run.wait_for_completion(
            lambda x: self.wait_callback(x, status), timeout_minutes=timeout
        )
        await status.update(f"Workflow [{run.run_id}](<{run.html_url}>) completed")
        logger.info(f"Workflow [{run.run_id}]({run.html_url}) completed")
        await status.push("Downloading artifacts...")
        logger.info("Downloading artifacts...")

        index = run.get_artifact_index()

        if "run-result" not in index:
            logger.error("Could not find `run-result` among artifacts: %s", index.keys())
            await status.push("Downloading artifacts...  failed")
            return FullResult(
                success=False, error="Could not download artifacts", runs={}, system=SystemInfo()
            )

        artifact = await run.download_artifact(index["run-result"])
        logs = artifact["result.json"].decode("utf-8")

        await status.update("Downloading artifacts... done")
        logger.info("Downloading artifacts... done")

        data = json.loads(logs)
        runs = {}
        # convert json back to EvalResult structures, which requires
        # special handling for datetime and our dataclasses.

        for k, v in data["runs"].items():
            comp_res = None if v.get("compilation") is None else CompileResult(**v["compilation"])
            run_res = None if v.get("run") is None else RunResult(**v["run"])
            profile_res = None if v.get("profile") is None else ProfileResult(**v["profile"])

            # Update profile artifact to the actual download URL.
            # For the GitHub launcher the profile_artifact currently just contains
            # the name of the artifact.
            if profile_res is not None and "profile-data" in index:
                profile_res.download_url = index["profile-data"].public_download_url

            res = EvalResult(
                start=datetime.datetime.fromisoformat(v["start"]),
                end=datetime.datetime.fromisoformat(v["end"]),
                compilation=comp_res,
                run=run_res,
                profile=profile_res,
            )
            runs[k] = res

        system = SystemInfo(**data.get("system", {}))
        return FullResult(success=True, error="", runs=runs, system=system)

    async def wait_callback(self, run: "GitHubRun", status: RunProgressReporter):
        await status.update(
            f"⏳ Workflow [{run.run_id}](<{run.html_url}>): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )


@dataclasses.dataclass
class GitHubArtifact:
    name: str
    archive_download_url: str
    public_download_url: str


_WORKFLOW_FILE_CACHE: dict[tuple[str, str], Workflow] = {}


def patched_create_dispatch(
    workflow: Workflow,
    ref: github.Branch.Branch | github.Tag.Tag | github.Commit.Commit | str,
    inputs: Opt[dict] = NotSet,
) -> bool:
    """
    :calls: `POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches <https://docs.github.com/en/rest/reference/actions#create-a-workflow-dispatch-event>`_
    """
    assert (
        isinstance(ref, github.Branch.Branch)
        or isinstance(ref, github.Tag.Tag)
        or isinstance(ref, github.Commit.Commit)
        or isinstance(ref, str)
    ), ref
    assert inputs is NotSet or isinstance(inputs, dict), inputs
    if isinstance(ref, github.Branch.Branch):
        ref = ref.name
    elif isinstance(ref, github.Commit.Commit):
        ref = ref.sha
    elif isinstance(ref, github.Tag.Tag):
        ref = ref.name
    if inputs is NotSet:
        inputs = {}
    status, _, _ = workflow._requester.requestJson(
        "POST", f"{workflow.url}/dispatches", input={"ref": ref, "inputs": inputs}
    )

    return status == 200 or status == 204



class GitHubRun:
    def __init__(self, repo: str, token: str, branch: str, workflow_file: str):
        gh = Github(token)
        try:
            self.repo = gh.get_repo(repo)
        except UnknownObjectException as e:
            raise KernelBotError(f"Could not find GitHub repository {repo}: 404") from e
        self.token = token
        self.branch = branch
        self.workflow_file = workflow_file
        self.run: Optional[WorkflowRun.WorkflowRun] = None
        self.start_time = None

    @property
    def run_id(self):
        if self.run is None:
            return None
        return self.run.id

    @property
    def html_url(self):
        if self.run is None:
            return None
        return self.run.html_url

    @property
    def status(self):
        if self.run is None:
            return None
        return self.run.status

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return None
        return datetime.datetime.now(datetime.timezone.utc) - self.start_time

    async def get_workflow(self) -> Workflow:
        cache_key = (self.workflow_file, self.token)
        if cache_key in _WORKFLOW_FILE_CACHE:
            logger.info(f"Returning cached workflow {self.workflow_file}")
            return _WORKFLOW_FILE_CACHE[cache_key]
        logger.info(f"Fetching workflow {self.workflow_file} from GitHub")
        workflow = self.repo.get_workflow(self.workflow_file)
        _WORKFLOW_FILE_CACHE[cache_key] = workflow
        return workflow

    async def trigger(self, inputs: dict) -> bool:  # noqa: C901
        """
        Trigger this run with the provided inputs.
        Sets `self.run` to the new WorkflowRun on success.

        Returns: Whether the run was successfully triggered,
        """
        run_id = str(uuid.uuid4())

        inputs_with_run_id = {**inputs, "run_id": run_id}

        if self.workflow_file == "amd_workflow.yml":
            expected_run_name = f"AMD Job - {run_id}"
        elif self.workflow_file == "nvidia_workflow.yml":
            expected_run_name = f"NVIDIA Job - {run_id}"
        elif self.workflow_file == "nvidia_model_workflow.yml":
            expected_run_name = f"Model Job - {run_id}"
        else:
            raise ValueError(f"Unknown workflow file: {self.workflow_file}")

        trigger_time = datetime.datetime.now(datetime.timezone.utc)
        try:
            workflow = await self.get_workflow()
        except UnknownObjectException as e:
            logger.error(f"Could not find workflow {self.workflow_file}", exc_info=e)
            raise ValueError(f"Could not find workflow {self.workflow_file}") from e

        logger.info(
            "Dispatching workflow %s on branch %s with run_id %s",
            self.workflow_file,
            self.branch,
            run_id,
        )
        logger.debug(
            "Dispatching workflow %s on branch %s with inputs %s",
            self.workflow_file,
            self.branch,
            pprint.pformat(inputs_with_run_id),
        )
        success = await asyncio.to_thread(
            patched_create_dispatch, workflow, self.branch, inputs=inputs_with_run_id
        )  # noqa: E501

        if success:
            wait_seconds = 10
            logger.info(
                f"Workflow dispatch successful. Waiting {wait_seconds}s for the run to appear..."
            )
            await asyncio.sleep(wait_seconds)
            recent_runs_paginated = await asyncio.to_thread(
                workflow.get_runs, event="workflow_dispatch"
            )

            logger.info(f"Looking for workflow run with name: '{expected_run_name}'")
            found_run = None
            runs_checked = 0
            try:
                run_iterator = recent_runs_paginated.__iter__()
                while runs_checked < 100:
                    try:
                        run = next(run_iterator)
                        runs_checked += 1
                        logger.debug(
                            f"Checking run {run.id} with name '{run.name}'"
                            f" created at {run.created_at.isoformat()}"
                        )
                        if run.name == expected_run_name and run.created_at.replace(
                            tzinfo=datetime.timezone.utc
                        ) > trigger_time - datetime.timedelta(seconds=30):
                            found_run = run
                            logger.info(
                                f"Found matching workflow run: ID {found_run.id} "
                                f"with name '{found_run.name}'"
                            )
                            break
                    except StopIteration:
                        logger.debug("Reached end of recent runs list.")
                        break
            except Exception as e:
                logger.error(f"Error iterating through recent runs: {e}", exc_info=True)
                return False

            if found_run:
                self.run = found_run
                return True
            else:
                logger.warning(
                    f"Could not find a workflow run with name '{expected_run_name}' "
                    f"created after {trigger_time.isoformat()}."
                )
                return False
        else:
            logger.error(
                f"Failed to dispatch workflow {self.workflow_file} on branch {self.branch}."
            )
            return False

    async def wait_for_completion(
        self, callback: Callable[["GitHubRun"], Awaitable[None]], timeout_minutes: int = 10
    ):
        if self.run is None:
            raise ValueError("Run needs to be triggered before a status check!")

        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        timeout = datetime.timedelta(minutes=timeout_minutes)

        while True:
            try:
                run_update = await asyncio.to_thread(self.repo.get_workflow_run, self.run_id)
                self.run = run = run_update

                if self.elapsed_time > timeout:
                    try:
                        self.run.cancel()
                        # Wait briefly to ensure cancellation is processed
                        # And Verify the run was actually cancelled
                        await asyncio.sleep(5)
                        run = self.repo.get_workflow_run(self.run_id)
                        if run.status != "completed":
                            logger.warning(f"Failed to cancel workflow run {self.run_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling workflow: {str(e)}", exc_info=e)
                        raise

                    logger.warning(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )
                    raise TimeoutError(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )

                if run.status == "completed":
                    return

                await callback(self)
                await asyncio.sleep(30)  # Yield control while waiting
            except TimeoutError:
                raise  # Re-raise the specific TimeoutError from the timeout block
            except Exception as e:
                logger.error(f"Error waiting for GitHub run {self.run_id}: {e}", exc_info=e)
                raise  # Re-raise other exceptions

    def get_artifact_index(self) -> dict[str, GitHubArtifact]:
        logger.info("Creating artifact index for run %s", self.run_id)
        artifacts = self.run.get_artifacts()

        extracted = {}

        for artifact in artifacts:
            extracted[artifact.name] = GitHubArtifact(
                name=artifact.name,
                archive_download_url=artifact.archive_download_url,
                # Non-machine users cannot download from the archive_download_url and
                # the GitHub API does not give us access to the public download url.
                public_download_url=f"{self.repo.html_url}/actions/runs/{self.run_id}/artifacts/{artifact.id}",
            )

        return extracted

    async def download_artifact(self, artifact: GitHubArtifact) -> dict:
        logger.info("Attempting to download artifact '%s' for run %s", artifact.name, self.run_id)

        url = artifact.archive_download_url
        headers = {"Authorization": f"token {self.token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            artifact_dict = {}
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file in z.namelist():
                    with z.open(file) as f:
                        artifact_dict[file] = f.read()

            logger.info("Downloaded artifact '%s' for run %s", artifact.name, self.run_id)
            return artifact_dict
        else:
            raise RuntimeError(
                f"Failed to download artifact {artifact.name}. Status code: {response.status_code}"
            )
