"""Instrument the real run_config boundary for trusted example submissions."""

import dataclasses
import json
import sys
import time
from pathlib import Path

from libkernelbot import run_eval


def main():
    """Execute one submission with fresh processes or a persistent RPC worker."""
    config = json.loads(Path(sys.argv[1]).read_text())
    mode = sys.argv[2]
    system = run_eval.SystemInfo(**json.loads(Path("/experiment/system.json").read_text()))
    run_eval.make_system_info = lambda: system
    original = run_eval.run_program
    if mode == "persistent":
        from persistent import run_program

        original = run_program
    phases = []

    def measured_program(args, seed, timeout, multi_gpu=False, extra_env=None):
        started = time.monotonic()
        result = original(args, seed, timeout, multi_gpu, extra_env)
        phases.append(
            {
                "phase": args[2] if len(args) > 2 else "compile-import",
                "wall_s": time.monotonic() - started,
                "exit_code": int(result.exit_code),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result

    run_eval.run_program = measured_program
    try:
        result = run_eval.run_config(config)
        row = {
            "passed": result.success and all(r.run and r.run.passed for r in result.runs.values()),
            "result": dataclasses.asdict(result),
        }
    except Exception:
        import traceback

        row = {"passed": False, "error": traceback.format_exc()}
    row["phases"] = phases
    Path("result.json").write_text(json.dumps(row, default=str))
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
