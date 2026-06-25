from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Type

from libkernelbot.consts import GPU
from libkernelbot.report import RunProgressReporter


@dataclass
class RunnerQueueStatus:
    runner: str
    gpu: str
    queued_jobs: int | None
    running_jobs: int | None = None
    available_runners: int | None = None
    status: str = "available"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Launcher:
    def __init__(self, name: str, gpus: Type[Enum]):
        self.name = name
        self.gpus = gpus

    async def run_submission(self, config: dict, gpu_type: GPU, status: RunProgressReporter):
        raise NotImplementedError()

    async def get_queue_status(
        self, gpu_type: GPU, config: dict | None = None
    ) -> RunnerQueueStatus:
        return RunnerQueueStatus(
            runner=self.name,
            gpu=gpu_type.name,
            queued_jobs=None,
            status="unavailable",
            error="queue status is not supported for this runner",
        )
