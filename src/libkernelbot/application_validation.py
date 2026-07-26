"""Nightly application-level validation for ranked kernel submissions."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import math
from typing import Any
from zoneinfo import ZoneInfo

from libkernelbot.consts import SubmissionMode, get_gpu_by_name
from libkernelbot.kernelguard import (
    enforce_submission_precheck,
    should_precheck_submission,
)
from libkernelbot.task import (
    ApplicationValidation,
    LeaderboardTask,
    build_validation_config,
)
from libkernelbot.utils import KernelBotError, setup_logging

logger = setup_logging(__name__)


def _kernelguard_precheck(source: str, file_name: str) -> None:
    if not should_precheck_submission(SubmissionMode.LEADERBOARD):
        raise KernelBotError(
            "KernelGuard must be enabled before application validation can run"
        )
    enforce_submission_precheck(source, file_name)


def _due_date(
    validation: ApplicationValidation,
    now: datetime.datetime,
) -> datetime.date | None:
    timezone = ZoneInfo(validation.schedule.timezone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    local_now = now.astimezone(timezone)
    scheduled_time = datetime.time(
        validation.schedule.hour,
        validation.schedule.minute,
        tzinfo=timezone,
    )
    if local_now < datetime.datetime.combine(
        local_now.date(),
        scheduled_time,
    ):
        return None
    return local_now.date()


def _safe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Whitelist aggregate and per-shape metrics safe to keep in the database."""
    public: dict[str, Any] = {}
    for key in (
        "passed_shapes",
        "total_shapes",
        "fully_validated",
        "geomean_sync_wall_speedup",
    ):
        if key in result:
            public[key] = result[key]

    public_shapes = []
    for shape in result.get("results", []):
        if not isinstance(shape, dict):
            continue
        public_shapes.append(
            {
                key: shape[key]
                for key in (
                    "batch",
                    "n",
                    "steps",
                    "passed",
                    "numerically_stable",
                    "converged",
                    "route",
                    "torch_fallback_calls",
                    "sync_wall_speedup",
                    "factor_residual",
                    "baseline_final_loss",
                    "candidate_final_loss",
                )
                if key in shape
            }
        )
    if public_shapes:
        public["results"] = public_shapes
    return public


class ApplicationValidationService:
    def __init__(
        self,
        backend,
        *,
        enabled: bool = False,
        poll_seconds: int = 60,
        precheck=_kernelguard_precheck,
    ):
        self.backend = backend
        self.enabled = enabled
        self.poll_seconds = poll_seconds
        self.precheck = precheck
        self._scheduler_task: asyncio.Task | None = None
        self._manual_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self.enabled or self._scheduler_task is not None:
            return
        logger.info("Starting application validation scheduler")
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(),
            name="application-validation-scheduler",
        )

    async def stop(self) -> None:
        tasks = list(self._manual_tasks)
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            tasks.append(self._scheduler_task)
            self._scheduler_task = None
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._manual_tasks.clear()

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self.run_due_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Application validation scheduler iteration failed")
            await asyncio.sleep(self.poll_seconds)

    async def run_due_once(
        self,
        now: datetime.datetime | None = None,
    ) -> list[dict[str, Any]]:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        with self.backend.db as db:
            leaderboards = db.get_leaderboards()

        summaries = []
        for leaderboard in leaderboards:
            validation = leaderboard["task"].validation
            if validation is None:
                continue
            scheduled_for = _due_date(validation, now)
            if scheduled_for is None:
                continue
            for gpu_type in leaderboard["gpu_types"]:
                summaries.append(
                    await self.run_sweep(
                        leaderboard["name"],
                        gpu_type,
                        scheduled_for=scheduled_for,
                    )
                )
        return summaries

    def enqueue_manual_sweep(
        self,
        leaderboard_name: str,
        gpu_type: str,
    ) -> None:
        task = asyncio.create_task(
            self.run_sweep(leaderboard_name, gpu_type, scheduled_for=None),
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
        validation: ApplicationValidation,
        gpu_type: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        submission_id = entry["submission_id"]
        async with semaphore:
            try:
                with self.backend.db as db:
                    source = db.get_submission_code_for_validation(submission_id)
                self.precheck(source, entry["submission_name"])
                config = build_validation_config(task, source)
                del source

                gpu = get_gpu_by_name(gpu_type)
                if gpu is None:
                    raise KernelBotError(f"Unknown GPU type {gpu_type!r}")
                launcher = self.backend.launcher_map.get(gpu.value)
                if launcher is None:
                    raise KernelBotError(
                        f"No validation launcher is registered for {gpu_type!r}"
                    )
                response = await launcher.run_validation(config, gpu)
                del config

                safe_result = _safe_result(response.get("result", {}))
                status = (
                    "completed"
                    if response.get("status") == "completed"
                    else "failed"
                )
                expected_total = len(validation.shapes)
                passed_shapes = int(safe_result.get("passed_shapes", 0))
                reported_total = int(
                    safe_result.get("total_shapes", expected_total)
                )
                if status == "completed" and reported_total != expected_total:
                    raise ValueError(
                        "validator result shape count does not match contract"
                    )
                if not 0 <= passed_shapes <= expected_total:
                    raise ValueError(
                        "validator result has an invalid passed shape count"
                    )
                fully_validated = bool(
                    status == "completed"
                    and passed_shapes == expected_total
                    and safe_result.get("fully_validated", False)
                )
                speedup = safe_result.get("geomean_sync_wall_speedup")
                if speedup is not None:
                    speedup = float(speedup)
                    if not math.isfinite(speedup) or speedup <= 0:
                        raise ValueError("validator result has an invalid speedup")
                error = response.get("error")
                with self.backend.db as db:
                    db.upsert_submission_validation(
                        submission_id=submission_id,
                        gpu_type=gpu_type,
                        contract_name=validation.name,
                        contract_version=validation.version,
                        status=status,
                        passed_shapes=passed_shapes,
                        total_shapes=expected_total,
                        fully_validated=fully_validated,
                        geomean_sync_wall_speedup=speedup,
                        result=safe_result,
                        error=str(error)[:1000] if error else None,
                    )
                return {
                    "submission_id": submission_id,
                    "status": status,
                    "passed_shapes": passed_shapes,
                    "total_shapes": expected_total,
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
                        contract_name=validation.name,
                        contract_version=validation.version,
                        status="failed",
                        passed_shapes=0,
                        total_shapes=len(validation.shapes),
                        fully_validated=False,
                        geomean_sync_wall_speedup=None,
                        result={},
                        error=error,
                    )
                return {
                    "submission_id": submission_id,
                    "status": "failed",
                    "passed_shapes": 0,
                    "total_shapes": len(validation.shapes),
                    "fully_validated": False,
                }

    async def run_sweep(
        self,
        leaderboard_name: str,
        gpu_type: str,
        *,
        scheduled_for: datetime.date | None,
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
                scheduled_for=scheduled_for,
                top_k=validation.top_k,
            )
            if sweep_id is None:
                return {
                    "status": "already_claimed",
                    "leaderboard": leaderboard_name,
                    "gpu_type": gpu_type,
                    "scheduled_for": scheduled_for,
                }
            submissions = db.get_leaderboard_submissions(
                leaderboard_name,
                gpu_type,
                limit=validation.top_k,
            )

        logger.info(
            "Running application validation sweep %s for leaderboard=%s gpu=%s submissions=%s",
            sweep_id,
            leaderboard_name,
            gpu_type,
            len(submissions),
        )
        semaphore = asyncio.Semaphore(validation.max_concurrency)

        try:
            results = await asyncio.gather(
                *(
                    self._validate_submission(
                        entry,
                        task=leaderboard["task"],
                        validation=validation,
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
                "scheduled_for": scheduled_for,
                "contract_version": validation.version,
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
