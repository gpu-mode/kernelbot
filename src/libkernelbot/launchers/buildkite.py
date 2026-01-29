import asyncio
import base64
import datetime
import json
import math
import zlib
from typing import Awaitable, Callable

import requests

from libkernelbot.consts import (
    DEFAULT_GITHUB_TIMEOUT_MINUTES,
    GPU,
    TIMEOUT_BUFFER_MINUTES,
    BuildkiteGPU,
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
from libkernelbot.utils import setup_logging

from .launcher import Launcher

logger = setup_logging()

# Buildkite API base URL
BUILDKITE_API_BASE = "https://api.buildkite.com/v2"


def get_timeout(config: dict) -> int:
    """Get timeout in minutes from config, matching GitHub launcher pattern."""
    mode = config.get("mode")
    sec_map = {
        SubmissionMode.TEST.value: config.get("test_timeout"),
        SubmissionMode.BENCHMARK.value: config.get("benchmark_timeout"),
        SubmissionMode.LEADERBOARD.value: config.get("ranked_timeout"),
    }
    seconds = sec_map.get(mode) or DEFAULT_GITHUB_TIMEOUT_MINUTES * 60
    return math.ceil(seconds / 60)


class BuildkiteLauncher(Launcher):
    """
    Launcher for Buildkite-based GPU runners.

    Buildkite agents are configured per-GPU with isolated resources:
    - Each agent bound to single GPU via CUDA_VISIBLE_DEVICES
    - CPU/RAM limits enforced via systemd cgroups
    - Queue tags route jobs to specific GPU types (e.g., queue=nvidia-h100-0)
    """

    def __init__(self, org: str, pipeline: str, token: str):
        """
        Initialize Buildkite launcher.

        Args:
            org: Buildkite organization slug (e.g., "gpu-mode")
            pipeline: Pipeline slug (e.g., "kernelbot-runner")
            token: Buildkite API token with build creation permissions
        """
        super().__init__(name="Buildkite", gpus=BuildkiteGPU)
        self.org = org
        self.pipeline = pipeline
        self.token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        """
        Run a submission on a Buildkite agent.

        Args:
            config: Submission configuration dict
            gpu_type: GPU type to run on (determines queue routing)
            status: Progress reporter for user feedback

        Returns:
            FullResult with compilation and run results
        """
        # Compress config (same as GitHub launcher)
        payload = base64.b64encode(zlib.compress(json.dumps(config).encode("utf-8"))).decode(
            "utf-8"
        )

        # Create build via Buildkite API
        build_url = f"{BUILDKITE_API_BASE}/organizations/{self.org}/pipelines/{self.pipeline}/builds"

        # Queue name from GPU type value (e.g., "nvidia-h100")
        # Buildkite will route to any agent with matching queue tag
        queue_name = gpu_type.value

        build_data = {
            "commit": "HEAD",
            "branch": "main",
            "message": f"Kernel submission on {gpu_type.name}",
            "env": {
                "SUBMISSION_PAYLOAD": payload,
                "GPU_QUEUE": queue_name,
            },
        }

        logger.info(f"Creating Buildkite build for {gpu_type.name} on queue {queue_name}")

        try:
            response = await asyncio.to_thread(
                requests.post, build_url, headers=self._headers, json=build_data
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to create Buildkite build: {e}")
            return FullResult(
                success=False,
                error=f"Failed to create Buildkite build: {str(e)}",
                runs={},
                system=SystemInfo(),
            )

        build = response.json()
        build_number = build["number"]
        build_url_html = build["web_url"]

        logger.info(f"Created Buildkite build #{build_number}: {build_url_html}")
        await status.push(f"⏳ Buildkite build [#{build_number}](<{build_url_html}>) started...")

        # Poll for completion
        timeout = get_timeout(config) + TIMEOUT_BUFFER_MINUTES
        build_api_url = f"{BUILDKITE_API_BASE}/organizations/{self.org}/pipelines/{self.pipeline}/builds/{build_number}"

        try:
            await self._wait_for_completion(
                build_api_url,
                build_number,
                build_url_html,
                timeout,
                lambda state, elapsed: self._status_callback(
                    status, build_number, build_url_html, state, elapsed
                ),
            )
        except TimeoutError as e:
            logger.error(f"Buildkite build #{build_number} timed out")
            return FullResult(
                success=False,
                error=str(e),
                runs={},
                system=SystemInfo(),
            )
        except Exception as e:
            logger.error(f"Error waiting for Buildkite build: {e}")
            return FullResult(
                success=False,
                error=f"Build error: {str(e)}",
                runs={},
                system=SystemInfo(),
            )

        await status.update(f"✅ Build [#{build_number}](<{build_url_html}>) completed")

        # Download artifacts
        await status.push("Downloading artifacts...")
        logger.info(f"Downloading artifacts for build #{build_number}")

        try:
            result = await self._download_and_parse_result(build_api_url)
            await status.update("Downloading artifacts... done")
            return result
        except Exception as e:
            logger.error(f"Failed to download artifacts: {e}")
            await status.update("Downloading artifacts... failed")
            return FullResult(
                success=False,
                error=f"Failed to download artifacts: {str(e)}",
                runs={},
                system=SystemInfo(),
            )

    async def _wait_for_completion(
        self,
        build_api_url: str,
        build_number: int,
        build_url_html: str,
        timeout_minutes: int,
        callback: Callable[[str, float], Awaitable[None]],
    ):
        """Poll Buildkite API until build completes or times out."""
        start_time = datetime.datetime.now(datetime.timezone.utc)
        timeout = datetime.timedelta(minutes=timeout_minutes)

        while True:
            try:
                response = await asyncio.to_thread(
                    requests.get, build_api_url, headers=self._headers
                )
                response.raise_for_status()
                build = response.json()

                elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

                if elapsed > timeout.total_seconds():
                    # Try to cancel the build
                    cancel_url = f"{build_api_url}/cancel"
                    await asyncio.to_thread(
                        requests.put, cancel_url, headers=self._headers
                    )
                    raise TimeoutError(
                        f"Build #{build_number} cancelled - exceeded {timeout_minutes} minute timeout"
                    )

                state = build.get("state", "unknown")

                if state in ("passed", "failed", "canceled", "blocked"):
                    if state != "passed":
                        logger.warning(f"Build #{build_number} finished with state: {state}")
                    return

                await callback(state, elapsed)
                await asyncio.sleep(10)  # Poll every 10 seconds

            except TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error polling build status: {e}")
                raise

    async def _status_callback(
        self,
        status: RunProgressReporter,
        build_number: int,
        build_url_html: str,
        state: str,
        elapsed: float,
    ):
        """Update status with current build state."""
        await status.update(
            f"⏳ Build [#{build_number}](<{build_url_html}>): {state} ({elapsed:.1f}s)"
        )

    async def _download_and_parse_result(self, build_api_url: str) -> FullResult:
        """Download artifacts and parse result.json."""
        # Get artifacts list
        artifacts_url = f"{build_api_url}/artifacts"
        response = await asyncio.to_thread(
            requests.get, artifacts_url, headers=self._headers
        )
        response.raise_for_status()
        artifacts = response.json()

        # Find result.json artifact
        result_artifact = None
        profile_artifact = None
        for artifact in artifacts:
            if artifact.get("filename") == "result.json":
                result_artifact = artifact
            elif artifact.get("path", "").startswith("profile_data/"):
                profile_artifact = artifact

        if not result_artifact:
            raise RuntimeError("Could not find result.json artifact")

        # Download result.json
        download_url = result_artifact.get("download_url")
        response = await asyncio.to_thread(
            requests.get, download_url, headers=self._headers
        )
        response.raise_for_status()

        # Parse result
        data = response.json()
        runs = {}

        for k, v in data.get("runs", {}).items():
            comp_res = None if v.get("compilation") is None else CompileResult(**v["compilation"])
            run_res = None if v.get("run") is None else RunResult(**v["run"])
            profile_res = None if v.get("profile") is None else ProfileResult(**v["profile"])

            # Add profile download URL if available
            if profile_res is not None and profile_artifact:
                profile_res.download_url = profile_artifact.get("download_url")

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
