"""Exercise the native Python launcher against an ephemeral, real Modal app.

PYTHONPATH=src:src/runners uv run python scripts/debug_python_precompile.py
"""

import argparse
import asyncio
import dataclasses
import json
import os
import textwrap
from pathlib import Path

import modal

os.environ.setdefault("KERNELBOT_MODAL_APP", "kernelbot-cpu-compile-debug")

from modal_runner import compile_python_submission  # noqa: E402
from modal_runner_archs import app, pytorch_functions  # noqa: E402

from libkernelbot.consts import GPU_TO_SM, SubmissionMode, get_gpu_by_name  # noqa: E402
from libkernelbot.launchers.modal import ModalLauncher  # noqa: E402
from libkernelbot.report import RunProgressReporter  # noqa: E402
from libkernelbot.task import build_task_config, make_task_definition  # noqa: E402


class DebugLauncher(ModalLauncher):
    def _get_function(self, name):
        # Use the same registered functions without deploying over the live app.
        if name == "compile_python_submission":
            return compile_python_submission
        return pytorch_functions[name.removeprefix("run_pytorch_script_")]


class ConsoleReporter(RunProgressReporter):
    async def push(self, message):
        print(message, flush=True)

    async def update(self, message):
        print(message, flush=True)


def submissions(example: Path) -> dict[str, tuple[str, str | None]]:
    inline = (example / "submission_cuda_inline.py").read_text()
    begin = inline.index("add_module = load_inline(")
    end = inline.index("\n\n\ndef add(", begin)
    lazy = (
        inline[:begin]
        + "def build_module():\n"
        + textwrap.indent(inline[begin:end].replace("add_module =", "return", 1), "    ")
        + inline[end:]
    ).replace("return add_module.add_cuda(A, B)", "return build_module().add_cuda(A, B)")
    return {
        "inline": (inline, "reused"),
        "gpu-import": ("import torch\ntorch.empty(1, device='cuda')\n" + inline, "fallback"),
        "lazy": (lazy, "skipped"),
        "triton": ((example / "submission_triton.py").read_text(), None),
        "torch": ("def custom_kernel(data):\n    return data[0] + data[1]\n", None),
        "changed-source": (
            inline.replace(
                "add_module = load_inline(",
                "add_cuda_source += '// GPU' if torch.cuda.is_available() else '// CPU'\n"
                "add_module = load_inline(",
            ),
            "fallback",
        ),
        "changed-header": (
            "import torch\nfrom pathlib import Path\n"
            "Path('/tmp/kernelbot-debug-header.h').write_text(\n"
            "    '#define OFFSET ' + ('0' if torch.cuda.is_available() else '1'))\n"
            + inline.replace(
                'add_cuda_source = """',
                'add_cuda_source = """\n#include "/tmp/kernelbot-debug-header.h"',
            ).replace("C[idx] = A[idx] + B[idx];", "C[idx] = A[idx] + B[idx] + OFFSET;"),
            "fallback",
        ),
    }


async def run(args):
    example = Path(__file__).resolve().parents[1] / "examples" / "vectoradd_py"
    task = make_task_definition(example).task
    gpu = get_gpu_by_name(args.gpu)
    launcher = DebugLauncher([])
    cases = submissions(example)
    results = {}
    async with app.run.aio():
        for name in args.cases.split(","):
            source, expected = cases[name]
            config = build_task_config(
                task=task,
                submission_content=source,
                arch=GPU_TO_SM[gpu.name],
                mode=SubmissionMode.LEADERBOARD if name == "inline" else SubmissionMode.TEST,
            )
            config["benchmarks"] = [{"size": 256, "seed": 54352}]
            result = await launcher.run_submission(config, gpu, ConsoleReporter(name))
            info = result.cpu_compile
            print(name, json.dumps(dataclasses.asdict(info) if info else None), flush=True)
            assert result.success, result.error
            assert all(item.run and item.run.passed for item in result.runs.values()), result
            assert (info.status if info else None) == expected, result
            results[name] = dataclasses.asdict(result)
            if args.output:
                Path(args.output).write_text(json.dumps(results, default=str, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument(
        "--cases", default="inline,gpu-import,lazy,triton,torch,changed-source,changed-header"
    )
    parser.add_argument("--output", default="/tmp/kernelbot-native-cpu-compile.json")
    with modal.enable_output():
        asyncio.run(run(parser.parse_args()))
