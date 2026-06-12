import math
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

from reference import check_implementation, generate_input
from submission import custom_kernel

WARMUP_ITERS = 10
BENCH_ITERS = 100


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def log(self, key, value):
        print(f"{key}: {value}", file=self.file, flush=True)


def get_test_cases(file_name):
    content = Path(file_name).read_text()
    tests = []
    pattern = r"\s*([a-zA-Z_]+):\s*([a-zA-Z_]+|[+-]?[0-9]+)\s*"
    for line in content.splitlines():
        if not line.strip():
            continue
        case = {}
        for part in line.split(";"):
            m = re.fullmatch(pattern, part)
            if not m:
                print(f"invalid test case: '{line}'", file=sys.stderr)
                sys.exit(113)
            key, val = m[1], m[2]
            try:
                val = int(val)
            except ValueError:
                pass
            case[key] = val
        tests.append(case)
    return tests


def run_testing(logger, tests):
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test)
        data = generate_input(**test)
        output = custom_kernel(data)
        mx.eval(output)
        error = check_implementation(data, output)
        if error:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", error)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_benchmarking(logger, tests):
    # warmup
    data = generate_input(**tests[0])
    for _ in range(WARMUP_ITERS):
        mx.eval(custom_kernel(data))

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test)
        data = generate_input(**test)
        mx.eval(data)

        output = custom_kernel(data)
        mx.eval(output)
        error = check_implementation(data, output)
        if error:
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", error)
            passed = False
            continue

        durations = []
        for i in range(BENCH_ITERS):
            start = time.perf_counter_ns()
            mx.eval(custom_kernel(data))
            durations.append(time.perf_counter_ns() - start)
            if i > 1:
                avg = sum(durations) / len(durations)
                std = math.sqrt(sum((d - avg) ** 2 for d in durations) / (len(durations) - 1))
                if std / math.sqrt(len(durations)) / avg < 0.01:
                    break

        avg = sum(durations) / len(durations)
        logger.log(f"benchmark.{idx}.runs", len(durations))
        logger.log(f"benchmark.{idx}.mean", avg)

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111
    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    tests = get_test_cases(sys.argv[2])

    with PopcornOutput(int(fd)) as logger:
        if mode == "test":
            return run_testing(logger, tests)
        if mode in ("benchmark", "leaderboard"):
            return run_benchmarking(logger, tests)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
