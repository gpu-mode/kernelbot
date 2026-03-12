import asyncio
import base64
import datetime
import json
import math
import os
import uuid
import zlib
from typing import Optional

import requests

from libkernelbot.consts import (
    DEFAULT_GITHUB_TIMEOUT_MINUTES,
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

# Northflank-specific timeout buffer to allow for container image pulling
NORTHFLANK_TIMEOUT_BUFFER_MINUTES = 30


def get_timeout(config: dict) -> int:
    """Calculate timeout in minutes based on submission mode."""
    mode = config.get("mode")
    sec_map = {
        SubmissionMode.TEST.value: config.get("test_timeout"),
        SubmissionMode.BENCHMARK.value: config.get("benchmark_timeout"),
        SubmissionMode.LEADERBOARD.value: config.get("ranked_timeout"),
    }
    seconds = sec_map.get(mode) or DEFAULT_GITHUB_TIMEOUT_MINUTES * 60
    return math.ceil(seconds / 60)


class NorthflankLauncher(Launcher):
    """
    Launcher that executes kernel benchmarks using Northflank jobs.

    Northflank is a managed container platform that can run jobs with GPU support.
    """

    def __init__(
        self,
        api_token: str,
        project_id: str,
        job_id: str,
        repo_url: Optional[str] = None,
        repo_branch: Optional[str] = None,
    ):
        """
        Initialize the Northflank launcher.

        Args:
            api_token: Northflank API token for authentication
            project_id: Northflank project ID where jobs are defined
            job_id: Job ID for GPU workloads
            repo_url: Optional Git repository URL (defaults to gpu-mode/kernelbot)
            repo_branch: Optional Git branch to clone (defaults to main)
        """
        super().__init__(name="Northflank", gpus=GitHubGPU)
        self.token = api_token or os.getenv("NORTHFLANK_API_TOKEN")
        if not self.token:
            raise KernelBotError("Northflank API token required. Set NORTHFLANK_API_TOKEN or pass api_token parameter.")
        self.project_id = project_id
        self.job_id = job_id
        self.repo_url = repo_url or os.getenv(
            "NORTHFLANK_REPO_URL", "https://github.com/gpu-mode/kernelbot.git"
        )
        self.repo_branch = repo_branch or os.getenv("NORTHFLANK_REPO_BRANCH", "main")

    async def run_submission(
        self, config: dict, gpu_type, status: RunProgressReporter
    ) -> FullResult:
        """
        Execute a kernel submission on Northflank.

        Args:
            config: Submission configuration (mode, lang, problem, etc.)
            gpu_type: GPU type (unused, kept for API compatibility)
            status: Progress reporter for status updates

        Returns:
            FullResult containing benchmark results and system info
        """
        lang = config["lang"]
        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        logger.info(f"Attempting to trigger Northflank job {self.job_id} for {lang_name}")

        # Generate run ID for this submission
        run_id = str(uuid.uuid4())

        run = NorthflankRun(
            project_id=self.project_id,
            job_id=self.job_id,
            token=self.token,
            repo_url=self.repo_url,
            repo_branch=self.repo_branch,
        )

        # Store run_id for later download
        run.internal_run_id = run_id

        # Encode config as compressed base64 payload
        payload = base64.b64encode(zlib.compress(json.dumps(config).encode("utf-8"))).decode("utf-8")

        # Prepare environment variables for the job
        env_vars = {
            "PAYLOAD": payload,
            "RUN_ID": run_id,
        }

        # Generate presigned URL for result upload
        try:
            from minio import Minio
            from datetime import timedelta

            endpoint = os.getenv("MINIO_ENDPOINT")
            access_key = os.getenv("MINIO_ACCESS_KEY")
            secret_key = os.getenv("MINIO_SECRET_KEY")
            bucket = os.getenv("MINIO_BUCKET")
            use_ssl = os.getenv("MINIO_USE_SSL", "true").lower() == "true"

            if all([endpoint, access_key, secret_key, bucket]):
                client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=use_ssl,
                )

                # Generate presigned PUT URL (valid for 2 hours)
                object_key = f"results/{run_id}/result.json"
                upload_url = client.presigned_put_object(
                    bucket,
                    object_key,
                    expires=timedelta(hours=2)
                )

                env_vars["RESULT_UPLOAD_URL"] = upload_url
                logger.info(f"Generated presigned upload URL for {object_key}")
            else:
                logger.warning("MinIO configuration incomplete, presigned URL not generated")

        except ImportError:
            logger.warning("minio package not installed, presigned URL not generated")
        except Exception as e:
            logger.warning(f"Failed to generate presigned upload URL: {e}")
            # Continue without upload URL - job will fail to upload but won't crash

        logger.info(f"Triggering Northflank job with run_id: {env_vars['RUN_ID']}")

        if not await run.trigger(env_vars):
            raise RuntimeError("Failed to trigger Northflank job. Please check the configuration.")

        await status.push("⏳ Waiting for job to complete...")
        logger.info("Waiting for job to complete...")

        timeout = get_timeout(config) + NORTHFLANK_TIMEOUT_BUFFER_MINUTES

        logger.info(f"Waiting for job to complete... (timeout: {timeout} minutes)")
        await run.wait_for_completion(lambda x: self.wait_callback(x, status), timeout_minutes=timeout)

        await status.update(f"Job [{run.run_id}](<{run.job_url}>) completed")
        logger.info(f"Job [{run.run_id}]({run.job_url}) completed")
        await status.push("Downloading results from MinIO...")
        logger.info("Downloading results from MinIO...")

        # Download result from MinIO
        try:
            result_data = await run.download_from_minio()
        except Exception as e:
            logger.error(f"Could not download result from MinIO: {e}", exc_info=True)
            await status.push("Downloading results from MinIO... failed")
            return FullResult(
                success=False, error=f"Could not download results from MinIO: {str(e)}", runs={}, system=SystemInfo()
            )

        await status.update("Downloading results from MinIO... done")
        logger.info("Downloading results from MinIO... done")

        data = json.loads(result_data)
        runs = {}

        # Convert JSON back to EvalResult structures
        for k, v in data["runs"].items():
            comp_res = None if v.get("compilation") is None else CompileResult(**v["compilation"])
            run_res = None if v.get("run") is None else RunResult(**v["run"])
            profile_res = None if v.get("profile") is None else ProfileResult(**v["profile"])

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

    async def wait_callback(self, run: "NorthflankRun", status: RunProgressReporter):
        """Callback for status updates during job execution."""
        await status.update(
            f"⏳ Job [{run.run_id}](<{run.job_url}>): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )


class NorthflankRun:
    """
    Represents a single Northflank job run.

    This class handles:
    - Triggering a job with environment variables
    - Polling job status until completion
    - Downloading results from logs
    """

    API_BASE = "https://api.northflank.com/v1"

    def __init__(
        self,
        project_id: str,
        job_id: str,
        token: str,
        repo_url: str = "https://github.com/gpu-mode/kernelbot.git",
        repo_branch: str = "main",
    ):
        """
        Initialize a Northflank job run.

        Args:
            project_id: Northflank project ID
            job_id: Northflank job ID to execute
            token: API token for authentication
            repo_url: Git repository URL to clone
            repo_branch: Git branch to checkout
        """
        self.project_id = project_id
        self.job_id = job_id
        self.token = token
        self.repo_url = repo_url
        self.repo_branch = repo_branch
        self.run_id: Optional[str] = None
        self.run_name: Optional[str] = None
        self.internal_run_id: Optional[str] = None  # RUN_ID passed to job for MinIO
        self.start_time: Optional[datetime.datetime] = None
        self._status: Optional[str] = None
        self._job_url: Optional[str] = None

    @property
    def job_url(self) -> str:
        """Get the Northflank UI URL for this job run."""
        if self._job_url:
            return self._job_url
        return f"https://app.northflank.com/projects/{self.project_id}/jobs/{self.job_id}"

    @property
    def status(self) -> Optional[str]:
        """Get the current job status."""
        return self._status

    @property
    def elapsed_time(self) -> Optional[datetime.timedelta]:
        """Get elapsed time since job started."""
        if self.start_time is None:
            return None
        return datetime.datetime.now(datetime.timezone.utc) - self.start_time

    def _make_request(
        self, method: str, endpoint: str, json_data: Optional[dict] = None
    ) -> requests.Response:
        """Make an authenticated request to the Northflank API."""
        url = f"{self.API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            timeout=30,
        )

        return response

    async def wait_for_completion(
        self,
        callback,
        timeout_minutes: int = 10,
    ):
        """
        Wait for the job to complete, polling status periodically.

        Args:
            callback: Async function to call on each status update
            timeout_minutes: Maximum time to wait before cancelling

        Raises:
            TimeoutError: If the job exceeds the timeout
        """
        if self.run_id is None:
            raise ValueError("Job needs to be triggered before status check!")

        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        timeout = datetime.timedelta(minutes=timeout_minutes)

        # Endpoint to get specific run details
        endpoint = f"/projects/{self.project_id}/jobs/{self.job_id}/runs/{self.run_id}"

        while True:
            try:
                if self.elapsed_time > timeout:
                    # Abort the job run on timeout
                    try:
                        logger.info(f"Aborting job {self.run_id} due to timeout")
                        abort_endpoint = f"/projects/{self.project_id}/jobs/{self.job_id}/runs/{self.run_id}"
                        await asyncio.to_thread(self._make_request, "DELETE", abort_endpoint)
                        logger.info(f"Job {self.run_id} aborted successfully")
                        # Wait briefly for abort to process
                        await asyncio.sleep(2)
                    except Exception as e:
                        logger.warning(f"Failed to abort job {self.run_id}: {e}")
                        # Continue with timeout error even if abort fails

                    logger.warning(f"Job {self.run_id} exceeded {timeout_minutes} minute timeout")
                    raise TimeoutError(f"Job {self.run_id} exceeded {timeout_minutes} minute timeout")

                # Get job run details
                response = await asyncio.to_thread(self._make_request, "GET", endpoint)

                if response.status_code == 200:
                    data = response.json()
                    run_data = data.get("data", {})

                    # Update status from Northflank API
                    self._status = run_data.get("status", "UNKNOWN")
                    concluded = run_data.get("concluded", False)

                    logger.debug(f"Job {self.run_id} status: {self._status}, concluded: {concluded}")

                    # Check if job is complete
                    if concluded or self._status in ("SUCCESS", "FAILED"):
                        logger.info(f"Job {self.run_id} completed with status: {self._status}")
                        return
                else:
                    logger.warning(
                        f"Failed to get job status. Status code: {response.status_code}, "
                        f"Response: {response.text}"
                    )
                    self._status = "UNKNOWN"

                await callback(self)
                await asyncio.sleep(30)  # Poll every 30 seconds

            except TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error waiting for job {self.run_id}: {e}", exc_info=e)
                raise

    async def download_from_minio(self) -> str:
        """
        Download the result from MinIO.

        Expects MinIO configuration via environment variables:
        - MINIO_ENDPOINT: MinIO server endpoint
        - MINIO_ACCESS_KEY: Access key for authentication
        - MINIO_SECRET_KEY: Secret key for authentication
        - MINIO_BUCKET: Bucket name
        - MINIO_USE_SSL: Whether to use SSL (default: true)

        Returns:
            String content of the result JSON

        Raises:
            RuntimeError: If download fails
        """
        try:
            from minio import Minio
        except ImportError:
            raise RuntimeError("minio package not installed. Install with: pip install minio")

        logger.info(f"Downloading result from storage for run {self.run_id}")
        logger.info(f"Using internal_run_id: {self.internal_run_id} for object key")

        # Get storage configuration from environment
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = os.getenv("MINIO_BUCKET")
        use_ssl = os.getenv("MINIO_USE_SSL", "true").lower() == "true"

        if not all([endpoint, access_key, secret_key, bucket]):
            raise RuntimeError(
                "MinIO configuration missing. Required env vars: "
                "MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET"
            )

        # Construct object key
        # Use the internal_run_id that was passed to the job
        object_key = f"results/{self.internal_run_id}/result.json"

        try:
            # Create MinIO client
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=use_ssl,
            )

            # Download object
            response = await asyncio.to_thread(
                client.get_object,
                bucket,
                object_key
            )

            # Read content
            result_data = response.read().decode("utf-8")
            response.close()
            response.release_conn()

            logger.info(f"Successfully downloaded result from MinIO: {object_key}")
            return result_data

        except Exception as e:
            raise RuntimeError(f"Failed to download result from MinIO: {str(e)}") from e


    async def trigger(self, env_vars: dict) -> bool:
        """
        Trigger the Northflank job with the full benchmark command.

        The runner will upload results to MinIO after completion.

        Args:
            env_vars: Dictionary of environment variables to pass to the job

        Returns:
            True if the job was successfully triggered, False otherwise
        """
        internal_run_id = env_vars.get("RUN_ID", str(uuid.uuid4()))
        self.internal_run_id = internal_run_id  # Store for later MinIO download

        logger.info(f"Triggering job with internal_run_id (for MinIO): {internal_run_id}")

        # Northflank job run API endpoint
        endpoint = f"/projects/{self.project_id}/jobs/{self.job_id}/runs"

        # Extract repo name from URL for cd command
        repo_name = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        # Build command to run inside container
        command_parts = [
            # Clone the repository
            f"git clone -b {self.repo_branch} {self.repo_url}",
            f"cd {repo_name}",
            # Install kernelbot and minio
            "pip install --break-system-packages -e .",
            "pip install --break-system-packages minio",
            # Run the northflank-runner.py script (will upload to MinIO)
            "python3 src/runners/northflank-runner.py",
        ]

        full_command = " && ".join(command_parts)

        logger.debug(f"Generated command: {full_command}")

        # Prepare the request payload
        payload = {
            "runtimeEnvironment": env_vars,
            "deployment": {
                "docker": {
                    "configType": "customEntrypointCustomCommand",
                    "customEntrypoint": "/bin/bash",
                    "customCommand": f"-c {json.dumps(full_command)}",
                }
            },
        }

        logger.info(f"Triggering Northflank job {self.job_id} with internal run_id {internal_run_id}")

        try:
            response = await asyncio.to_thread(self._make_request, "POST", endpoint, payload)

            if response.status_code in (200, 201, 202):
                result = response.json()
                logger.info(f"Job triggered successfully: {result}")

                # Store the Northflank-assigned run ID and name
                if "data" in result:
                    self.run_id = result["data"].get("id")
                    self.run_name = result["data"].get("runName")

                    if self.run_id:
                        self._job_url = f"https://app.northflank.com/projects/{self.project_id}/jobs/{self.job_id}/runs/{self.run_id}"
                        logger.info(f"Northflank assigned run ID: {self.run_id}, run name: {self.run_name}")
                    else:
                        logger.warning("Northflank did not return a run ID")
                        return False

                self.start_time = datetime.datetime.now(datetime.timezone.utc)
                return True
            else:
                logger.error(
                    f"Failed to trigger job. Status: {response.status_code}, " f"Response: {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error triggering Northflank job: {e}", exc_info=True)
            return False
