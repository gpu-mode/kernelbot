"""Buildkite launcher for kernel evaluation jobs.

Uses single-queue model where all agents on a node share the same queue.
Buildkite automatically routes jobs to idle agents.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Any

import httpx

from libkernelbot.consts import GPU, BuildkiteGPU
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

logger = setup_logging(__name__)

BUILDKITE_API = "https://api.buildkite.com/v2"


@dataclass
class BuildkiteConfig:
    """Buildkite launcher configuration."""

    org_slug: str = "mark-saroufim"
    pipeline_slug: str = "kernelbot"
    api_token: str = field(default_factory=lambda: os.environ.get("BUILDKITE_API_TOKEN", ""))

    # Docker image for jobs
    image: str = "ghcr.io/gpu-mode/kernelbot:latest"

    # Timeouts
    poll_interval_seconds: int = 10
    max_wait_seconds: int = 900  # 15 minutes

    # Resource defaults
    cpus: int = 8
    memory: str = "64g"


@dataclass
class BuildkiteResult:
    """Result from a Buildkite job."""

    success: bool
    error: str | None
    result: dict[str, Any] | None
    build_url: str | None = None
    build_number: int | None = None


class BuildkiteLauncher(Launcher):
    """Launcher that submits jobs to Buildkite."""

    def __init__(self, config: BuildkiteConfig | None = None):
        super().__init__(name="Buildkite", gpus=BuildkiteGPU)
        self.config = config or BuildkiteConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    def _encode_payload(self, config: dict[str, Any]) -> str:
        """Compress and base64-encode config."""
        json_bytes = json.dumps(config).encode("utf-8")
        compressed = zlib.compress(json_bytes)
        return base64.b64encode(compressed).decode("ascii")

    def _get_queue_for_gpu(self, gpu_type: GPU) -> str:
        """Map GPU type to Buildkite queue name."""
        queue_map = {
            "B200_BK": "b200",
            "H100_BK": "h100",
            "MI300_BK": "mi300",
            "L40S_BK": "test",  # Test infrastructure
        }
        return queue_map.get(gpu_type.name, gpu_type.name.lower().replace("_bk", ""))

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        """
        Launch a kernel evaluation job on Buildkite.

        Args:
            config: Evaluation configuration dict
            gpu_type: Which GPU to run on
            status: Progress reporter for status updates

        Returns:
            FullResult with success status and results
        """
        queue = self._get_queue_for_gpu(gpu_type)
        run_id = f"sub-{config.get('submission_id', 'unknown')}-{gpu_type.name}"

        await status.push(f"Submitting to Buildkite queue: {queue}")
        logger.info(f"Submitting job {run_id} to Buildkite queue {queue}")

        result = await self._launch(
            run_id=run_id,
            config=config,
            queue=queue,
            status=status,
        )

        if not result.success:
            return FullResult(
                success=False,
                error=result.error or "Buildkite job failed",
                runs={},
                system=SystemInfo(),
            )

        if result.result is None:
            return FullResult(
                success=False,
                error="No result returned from Buildkite job",
                runs={},
                system=SystemInfo(),
            )

        # Parse the result
        return self._parse_result(result.result)

    async def _launch(
        self,
        run_id: str,
        config: dict[str, Any],
        queue: str,
        status: RunProgressReporter,
        inline_steps: list[dict[str, Any]] | None = None,
    ) -> BuildkiteResult:
        """
        Launch a kernel evaluation job.

        Args:
            run_id: Unique identifier for this run
            config: Evaluation configuration dict
            queue: GPU queue name (e.g., "b200", "mi300")
            status: Progress reporter
            inline_steps: Optional inline pipeline steps (for testing without pipeline config)

        Returns:
            BuildkiteResult with success status and results
        """
        client = await self._get_client()
        payload = self._encode_payload(config)

        # Create build
        url = (
            f"{BUILDKITE_API}/organizations/{self.config.org_slug}"
            f"/pipelines/{self.config.pipeline_slug}/builds"
        )

        build_data = {
            "commit": "HEAD",
            "branch": "buildkite-infrastructure",
            "message": f"Kernel eval: {run_id}",
            "env": {
                "KERNELBOT_RUN_ID": run_id,
                "KERNELBOT_PAYLOAD": payload,
                "KERNELBOT_QUEUE": queue,
                "KERNELBOT_IMAGE": self.config.image,
                "KERNELBOT_CPUS": str(self.config.cpus),
                "KERNELBOT_MEMORY": self.config.memory,
            },
            "meta_data": {
                "run_id": run_id,
                "queue": queue,
            },
        }

        # If inline steps provided, use them instead of pipeline from repo
        if inline_steps:
            build_data["steps"] = inline_steps

        try:
            response = await client.post(url, json=build_data)
            response.raise_for_status()
            build = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to create build: {e}")
            return BuildkiteResult(
                success=False,
                error=f"Failed to create build: {e}",
                result=None,
            )

        build_url = build.get("web_url")
        build_number = build.get("number")
        logger.info(f"Build created: {build_url}")
        await status.update(f"Build created: [{build_number}](<{build_url}>)")

        # Wait for completion
        return await self._wait_for_build(build, run_id, status)

    async def _wait_for_build(
        self, build: dict, run_id: str, status: RunProgressReporter
    ) -> BuildkiteResult:
        """Poll until build completes and download artifacts."""
        client = await self._get_client()
        build_url = build.get("url")
        web_url = build.get("web_url")
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < self.config.max_wait_seconds:
            try:
                response = await client.get(build_url)
                response.raise_for_status()
                build = response.json()
            except httpx.HTTPError as e:
                logger.warning(f"Error polling build: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)
                continue

            state = build.get("state")
            elapsed = asyncio.get_event_loop().time() - start

            if state == "passed":
                await status.update(f"Build completed: [{build.get('number')}](<{web_url}>)")
                result = await self._download_result(build)
                return BuildkiteResult(
                    success=True,
                    error=None,
                    result=result,
                    build_url=web_url,
                    build_number=build.get("number"),
                )

            if state in ("failed", "canceled", "blocked"):
                return BuildkiteResult(
                    success=False,
                    error=f"Build {state}",
                    result=None,
                    build_url=web_url,
                    build_number=build.get("number"),
                )

            await status.update(
                f"⏳ Build [{build.get('number')}](<{web_url}>): {state} ({elapsed:.1f}s)"
            )
            await asyncio.sleep(self.config.poll_interval_seconds)

        return BuildkiteResult(
            success=False,
            error="Build timed out",
            result=None,
            build_url=web_url,
            build_number=build.get("number"),
        )

    async def _download_result(self, build: dict) -> dict[str, Any] | None:
        """Download result.json artifact."""
        client = await self._get_client()

        # Get artifacts from first job
        jobs = build.get("jobs", [])
        if not jobs:
            return None

        job = jobs[0]
        artifacts_url = job.get("artifacts_url")
        if not artifacts_url:
            return None

        try:
            response = await client.get(artifacts_url)
            response.raise_for_status()
            artifacts = response.json()

            for artifact in artifacts:
                if artifact.get("filename") == "result.json":
                    download_url = artifact.get("download_url")
                    # Buildkite returns a 302 redirect to S3
                    # We need to follow it without the auth header
                    result_resp = await client.get(download_url, follow_redirects=False)
                    if result_resp.status_code == 302:
                        # Get the redirect URL and fetch without auth
                        s3_url = result_resp.headers.get("location")
                        async with httpx.AsyncClient(timeout=30.0) as s3_client:
                            result_resp = await s3_client.get(s3_url)
                            result_resp.raise_for_status()
                            return result_resp.json()
                    else:
                        result_resp.raise_for_status()
                        return result_resp.json()
        except Exception as e:
            logger.error(f"Failed to download artifacts: {e}")

        return None

    def _parse_result(self, data: dict[str, Any]) -> FullResult:
        """Parse result.json into FullResult."""
        runs = {}

        for k, v in data.get("runs", {}).items():
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

    async def get_queue_status(self, queue: str) -> dict[str, Any]:
        """Get status of agents in a queue."""
        client = await self._get_client()
        url = f"{BUILDKITE_API}/organizations/{self.config.org_slug}/agents"

        try:
            response = await client.get(url)
            response.raise_for_status()
            agents = response.json()
        except httpx.HTTPError as e:
            return {"error": str(e), "agents": []}

        queue_agents = []
        for agent in agents:
            agent_queue = None
            for meta in agent.get("metadata", []):
                if meta.startswith("queue="):
                    agent_queue = meta.split("=", 1)[1]
                    break

            if agent_queue == queue:
                queue_agents.append({
                    "name": agent.get("name"),
                    "state": agent.get("connection_state"),
                    "busy": agent.get("job") is not None,
                    "gpu_index": next(
                        (m.split("=")[1] for m in agent.get("metadata", [])
                         if m.startswith("gpu-index=")),
                        None
                    ),
                })

        return {
            "queue": queue,
            "total": len(queue_agents),
            "idle": sum(1 for a in queue_agents if not a["busy"]),
            "agents": queue_agents,
        }

    def create_artifact_test_steps(self, queue: str) -> list[dict[str, Any]]:
        """Create inline steps for artifact upload/download testing."""
        # Python script that decodes payload and writes result.json
        script = '''
import base64
import json
import os
import zlib
from datetime import datetime

run_id = os.environ.get("KERNELBOT_RUN_ID", "unknown")
payload_b64 = os.environ.get("KERNELBOT_PAYLOAD", "")

print("=== Artifact Test ===")
print(f"Run ID: {run_id}")
print(f"GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")

# Decode payload if present
config = {}
if payload_b64:
    try:
        compressed = base64.b64decode(payload_b64)
        config_json = zlib.decompress(compressed).decode("utf-8")
        config = json.loads(config_json)
        print(f"Decoded config keys: {list(config.keys())}")
    except Exception as e:
        print(f"Could not decode payload: {e}")

# Create result matching FullResult structure
result = {
    "success": True,
    "error": "",
    "runs": {},
    "system": {
        "gpu_name": os.environ.get("NVIDIA_VISIBLE_DEVICES", "unknown"),
        "cuda_version": "test",
        "python_version": "3.11",
    },
}

# Write result.json
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\\n=== Result ===")
print(json.dumps(result, indent=2))
print("\\nResult written to result.json")
'''
        return [
            {
                "label": ":test_tube: Artifact Test",
                "agents": {"queue": queue},
                "plugins": [
                    {
                        "docker#v5.11.0": {
                            "image": "python:3.11-slim",
                            "propagate-environment": True,
                            "environment": [
                                "KERNELBOT_PAYLOAD",
                                "KERNELBOT_RUN_ID",
                                "NVIDIA_VISIBLE_DEVICES",
                            ],
                        }
                    }
                ],
                "command": f"python3 -c {json.dumps(script)}",
                "artifact_paths": ["result.json"],
                "timeout_in_minutes": 5,
            }
        ]
