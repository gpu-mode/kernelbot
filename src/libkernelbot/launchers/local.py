import asyncio

from libkernelbot.consts import GPU, MetalGPU
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult, run_config
from libkernelbot.utils import setup_logging

from .launcher import Launcher

logger = setup_logging(__name__)


class LocalLauncher(Launcher):
    def __init__(self):
        super().__init__("Local", gpus=MetalGPU)

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        if config["lang"] == "cu":
            raise NotImplementedError("CUDA is not supported on Metal GPUs")

        logger.info(f"Starting local run for {gpu_type.name}")
        await status.push(f"⏳ Running locally on {gpu_type.name}...")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: run_config(config))

        await status.update(f"✅ Local run on {gpu_type.name} complete")
        return result
