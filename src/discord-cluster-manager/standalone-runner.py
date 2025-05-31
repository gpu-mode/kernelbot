import asyncio
import os
from dataclasses import asdict
from pydantic import BaseModel

import uvicorn

from run_eval import run_config

from fastapi import FastAPI, HTTPException

app = FastAPI()


_serial_run = asyncio.Semaphore(1)
_runner_token = None


class RunRequest(BaseModel):
    config: dict
    token: str


@app.post("/run")
async def run(request: RunRequest) -> dict:
    # only one submission can run at any given time
    if request.token != _runner_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    async with _serial_run:
        return asdict(run_config(request.config))


async def run_server(port):
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        limit_concurrency=2,
    )
    server = uvicorn.Server(config)

    # we need this as discord and fastapi both run on the same event loop
    await server.serve()


def main():
    with asyncio.Runner() as runner:
        runner.run(run_server(port=int(os.environ.get("PORT") or 8000)))


if __name__ == "__main__":
    main()
