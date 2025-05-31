# Generic launcher POSTs to a specific URL
import asyncio
import datetime
import json

import requests

from consts import GPU, OtherGPU
from report import RunProgressReporter
from run_eval import FullResult, CompileResult, RunResult, EvalResult, SystemInfo
from utils import setup_logging, KernelBotError

from .launcher import Launcher

logger = setup_logging(__name__)


class GenericLauncher(Launcher):
    def __init__(self, url: str, token: str):
        super().__init__("Generic", gpus=OtherGPU)
        self.url = url
        self.token = token

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        loop = asyncio.get_event_loop()
        logger.info(f"Calling {self.url}")

        await status.push("⏳ Waiting for run to finish...")
        result = await loop.run_in_executor(
            None,
            lambda: requests.post(self.url, json={"config": config, "token": self.token})
        )

        print(result.text)

        await status.update("✅ Waiting for run to finish... Done")
        if result.status_code != 200:
            logger.error("Error running submission. Status code %d, Message: %s", result.status_code, result.text)
            raise KernelBotError(f"Error running submission. Status code {result.status_code}")

        # TODO: this code is duplicated :(
        data = result.json()
        runs = {}
        # convert json back to EvalResult structures, which requires
        # special handling for datetime and our dataclasses.
        for k, v in data["runs"].items():
            if "compilation" in v and v["compilation"] is not None:
                comp = CompileResult(**v["compilation"])
            else:
                comp = None
            run = RunResult(**v["run"])
            res = EvalResult(
                start=datetime.datetime.fromisoformat(v["start"]),
                end=datetime.datetime.fromisoformat(v["end"]),
                compilation=comp,
                run=run,
            )
            runs[k] = res

        system = SystemInfo(**data.get("system", {}))
        return FullResult(success=True, error="", runs=runs, system=system)
