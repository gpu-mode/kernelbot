"""Admin-triggered application-level validation for ranked kernel submissions."""

from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any

from libkernelbot.consts import get_gpu_by_name
from libkernelbot.task import LeaderboardTask, build_validation_config
from libkernelbot.utils import KernelBotError, setup_logging

logger = setup_logging(__name__)

TOP_K = 10
MAX_CONCURRENCY = 2


class ApplicationValidationService:
    def __init__(self, backend):
        self.backend = backend
        self._manual_tasks: set[asyncio.Task] = set()

    async def stop(self) -> None:
        tasks = list(self._manual_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._manual_tasks.clear()

    def enqueue_manual_sweep(
        self,
        leaderboard_name: str,
        gpu_type: str,
        *,
        all_users: bool = False,
        only_missing: bool = False,
    ) -> None:
        task = asyncio.create_task(
            self.run_sweep(
                leaderboard_name,
                gpu_type,
                all_users=all_users,
                only_missing=only_missing,
            ),
            name=f"manual-application-validation-{leaderboard_name}-{gpu_type}",
        )
        self._manual_tasks.add(task)
        task.add_done_callback(self._manual_task_done)

    def _manual_task_done(self, task: asyncio.Task) -> None:
        self._manual_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Manual application validation sweep failed",
                exc_info=task.exception(),
            )

    async def _validate_submission(
        self,
        entry: dict,
        *,
        task: LeaderboardTask,
        gpu_type: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        submission_id = entry["submission_id"]
        contract_version = task.validation.version
        async with semaphore:
            try:
                with self.backend.db as db:
                    source = db.get_submission_code_for_validation(submission_id)
                config = build_validation_config(task, source)

                gpu = get_gpu_by_name(gpu_type)
                if gpu is None:
                    raise KernelBotError(f"Unknown GPU type {gpu_type!r}")
                launcher = self.backend.launcher_map.get(gpu.value)
                if launcher is None:
                    raise KernelBotError(
                        f"No validation launcher is registered for {gpu_type!r}"
                    )
                response = await launcher.run_validation(config, gpu)
                status = (
                    "completed"
                    if response.get("status") == "completed"
                    else "failed"
                )
                result = response.get("result", {})
                if not isinstance(result, dict):
                    raise ValueError("validator result must be an object")

                total_shapes = int(result.get("total_shapes", 0))
                passed_shapes = int(result.get("passed_shapes", 0))
                if status == "completed" and (
                    total_shapes <= 0
                    or not 0 <= passed_shapes <= total_shapes
                ):
                    raise ValueError("validator returned invalid shape counts")

                speedup = result.get("geomean_sync_wall_speedup")
                if speedup is not None:
                    speedup = float(speedup)
                    if not math.isfinite(speedup) or speedup <= 0:
                        raise ValueError("validator returned an invalid speedup")
                fully_validated = bool(
                    status == "completed"
                    and passed_shapes == total_shapes
                    and result.get("fully_validated")
                )

                with self.backend.db as db:
                    db.upsert_submission_validation(
                        submission_id=submission_id,
                        gpu_type=gpu_type,
                        contract_version=contract_version,
                        status=status,
                        passed_shapes=passed_shapes,
                        total_shapes=total_shapes,
                        fully_validated=fully_validated,
                        geomean_sync_wall_speedup=speedup,
                        result=result,
                        error=response.get("error"),
                    )
                return {
                    "submission_id": submission_id,
                    "status": status,
                    "passed_shapes": passed_shapes,
                    "total_shapes": total_shapes,
                    "fully_validated": fully_validated,
                    "geomean_sync_wall_speedup": speedup,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:1000]
                logger.warning(
                    "Application validation failed for submission=%s: %s",
                    submission_id,
                    error,
                )
                with self.backend.db as db:
                    db.upsert_submission_validation(
                        submission_id=submission_id,
                        gpu_type=gpu_type,
                        contract_version=contract_version,
                        status="failed",
                        passed_shapes=0,
                        total_shapes=0,
                        fully_validated=False,
                        geomean_sync_wall_speedup=None,
                        result={},
                        error=error,
                    )
                return {
                    "submission_id": submission_id,
                    "status": "failed",
                    "passed_shapes": 0,
                    "total_shapes": 0,
                    "fully_validated": False,
                }

    async def run_sweep(
        self,
        leaderboard_name: str,
        gpu_type: str,
        *,
        all_users: bool = False,
        only_missing: bool = False,
    ) -> dict[str, Any]:
        with self.backend.db as db:
            leaderboard = db.get_leaderboard(leaderboard_name)
            validation = leaderboard["task"].validation
            if validation is None:
                raise KernelBotError(
                    f"Leaderboard {leaderboard_name!r} has no application validation",
                    code=400,
                )
            if gpu_type not in leaderboard["gpu_types"]:
                raise KernelBotError(
                    f"GPU {gpu_type!r} is not configured for {leaderboard_name!r}",
                    code=400,
                )
            sweep_id = db.claim_validation_sweep(
                leaderboard_id=leaderboard["id"],
                gpu_type=gpu_type,
                contract_version=validation.version,
                scheduled_for=None,
            )
            submissions = db.get_leaderboard_submissions(
                leaderboard_name,
                gpu_type,
                limit=None if all_users else TOP_K,
            )
            if only_missing:
                validated_ids = db.get_submission_validation_ids(
                    [entry["submission_id"] for entry in submissions],
                    gpu_type=gpu_type,
                    contract_version=validation.version,
                )
                submissions = [
                    entry
                    for entry in submissions
                    if entry["submission_id"] not in validated_ids
                ]

        logger.info(
            "Running application validation sweep %s for leaderboard=%s gpu=%s submissions=%s",
            sweep_id,
            leaderboard_name,
            gpu_type,
            len(submissions),
        )
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        try:
            results = await asyncio.gather(
                *(
                    self._validate_submission(
                        entry,
                        task=leaderboard["task"],
                        gpu_type=gpu_type,
                        semaphore=semaphore,
                    )
                    for entry in submissions
                )
            )
            with self.backend.db as db:
                db.complete_validation_sweep(sweep_id, status="completed")
            return {
                "status": "completed",
                "sweep_id": sweep_id,
                "leaderboard": leaderboard_name,
                "gpu_type": gpu_type,
                "contract_version": validation.version,
                "all_users": all_users,
                "only_missing": only_missing,
                "results": results,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1000]
            with self.backend.db as db:
                db.complete_validation_sweep(
                    sweep_id,
                    status="failed",
                    error=error,
                )
            raise
