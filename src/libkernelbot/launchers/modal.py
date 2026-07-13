import asyncio

import modal

from libkernelbot.consts import GPU, ModalGPU
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult
from libkernelbot.utils import setup_logging

from .launcher import Launcher, RunnerQueueStatus

logger = setup_logging(__name__)


class ModalLauncher(Launcher):
    def __init__(self, add_include_dirs: list):
        super().__init__("Modal", gpus=ModalGPU)
        self.additional_include_dirs = add_include_dirs

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        if config["lang"] == "cu":
            config["include_dirs"] = config.get("include_dirs", []) + self.additional_include_dirs
        func_name = self._function_name(config, gpu_type)

        logger.info(f"Starting Modal run using {func_name}")

        await status.push("⏳ Waiting for Modal run to finish...")

        function = modal.Function.from_name("discord-bot-runner", func_name)
        result = await function.remote.aio(config=config)

        await status.update("✅ Waiting for modal run to finish... Done")

        return result

    def _function_name(self, config: dict, gpu_type: GPU) -> str:
        func_type = "pytorch" if config["lang"] == "py" else "cuda"
        return f"run_{func_type}_script_{gpu_type.value.lower()}"

    async def get_queue_status(
        self, gpu_type: GPU, config: dict | None = None
    ) -> RunnerQueueStatus:
        func_name = self._function_name(config or {"lang": "cu"}, gpu_type)
        loop = asyncio.get_event_loop()

        try:
            stats = await loop.run_in_executor(
                None,
                lambda: modal.Function.from_name(
                    "discord-bot-runner", func_name
                ).get_current_stats(),
            )
        except Exception as e:
            logger.warning("Could not get Modal queue stats for %s", func_name, exc_info=e)
            return RunnerQueueStatus(
                runner=self.name,
                gpu=gpu_type.name,
                queued_jobs=None,
                status="unavailable",
                error=str(e),
            )

        return RunnerQueueStatus(
            runner=self.name,
            gpu=gpu_type.name,
            queued_jobs=getattr(stats, "backlog", None),
            available_runners=getattr(stats, "num_total_runners", None),
        )
