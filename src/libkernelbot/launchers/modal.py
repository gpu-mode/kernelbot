import asyncio
import os
import time

import modal

from libkernelbot.consts import GPU, ModalGPU, Timeout
from libkernelbot.python_precompile import should_precompile
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult
from libkernelbot.utils import setup_logging

from .launcher import Launcher, RunnerQueueStatus

logger = setup_logging(__name__)


class ModalLauncher(Launcher):
    def __init__(self, add_include_dirs: list, *, app_name: str | None = None):
        super().__init__("Modal", gpus=ModalGPU)
        self.additional_include_dirs = add_include_dirs
        self.app_name = app_name or os.environ.get("KERNELBOT_MODAL_APP", "discord-bot-runner")

    def _get_function(self, name: str):
        return modal.Function.from_name(self.app_name, name)

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        if config["lang"] == "cu":
            config["include_dirs"] = config.get("include_dirs", []) + self.additional_include_dirs
        func_name = self._function_name(config, gpu_type)

        logger.info(f"Starting Modal run using {func_name}")

        await status.push("⏳ Waiting for Modal run to finish...")

        if os.environ.get("KERNELBOT_CPU_COMPILE", "1") != "0" and should_precompile(config):
            started = time.perf_counter()
            try:
                compiler = self._get_function("compile_python_submission")
                packet = await asyncio.wait_for(
                    compiler.remote.aio(config=config),
                    timeout=Timeout.COMPILE + 60,
                )
            except Exception as exc:
                # Old deployments and unavailable CPU workers retain the existing GPU path.
                packet = {
                    "artifacts": b"",
                    "info": {
                        "status": "fallback",
                        "duration": time.perf_counter() - started,
                        "reason": str(exc)[-1000:],
                    },
                }
            config = {**config, "cpu_compile": packet}
        function = self._get_function(func_name)
        result = await function.remote.aio(config=config)
        if getattr(result, "cpu_compile", None) is not None:
            logger.info("Modal CPU compilation outcome: %s", result.cpu_compile)

        await status.update("✅ Waiting for modal run to finish... Done")

        return result

    async def run_validation(self, config: dict, gpu_type: GPU) -> dict:
        func_name = f"run_validation_script_{gpu_type.value.lower()}"
        logger.info(
            "Starting Modal application validation using %s for contract %s",
            func_name,
            config.get("version"),
        )
        function = self._get_function(func_name)
        return await function.remote.aio(config=config)

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
                lambda: self._get_function(func_name).get_current_stats(),
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
