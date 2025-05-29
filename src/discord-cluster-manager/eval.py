import argparse
import math
import os
import sys
import time

import torch
from reference import check_implementation, generate_input, ref_kernel
from consts import REFERENCE_TIMING_ARG


class PopcornLogger:
    def __init__(self, fd):
        self.channel = open(fd, "w")

    def log(self, key: str, value):
        print(f"{key}: {value}\n", file=self.channel)


def correctness(rng: torch.Generator) -> bool:
    from submission import custom_kernel
    for _ in range(10):  # check multiple times
        inputs = generate_input(torch.randint(0, int(2**31), (), generator=rng).item())
        custom_output = custom_kernel(inputs)
        ref_output = ref_kernel(inputs)

        if not check_implementation(custom_output, ref_output):
            return False

    print("custom implementation matches the reference implementation.")
    return True


def metric(logger: PopcornLogger, rng: torch.Generator, time_reference_impl: bool = False):
    print("timing kernel")
    warmup_runs = 10
    timed_runs = 100
    if time_reference_impl:
        logger.log("Timing Reference Implementation")
    else:
        # in the case of a reference run we don't have a submission
        logger.log("Timing Submitted Custom Implementation")
        from submission import custom_kernel

    # Warmup Code
    print("warming up...")
    for _ in range(warmup_runs):
        inputs = generate_input(torch.randint(0, int(2**31), (), generator=rng).item())
        if time_reference_impl:
            _ = ref_kernel(inputs)
        else:
            _ = custom_kernel(inputs)
    torch.cuda.synchronize()

    # Timing Code
    times = []

    for _ in range(timed_runs):
        inputs = generate_input(torch.randint(0, int(2**31), (), generator=rng).item())

        start_time = time.time()
        if time_reference_impl:
            ref_output = ref_kernel(inputs)
        else:
            custom_output = custom_kernel(inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

        if not time_reference_impl:
            ref_output = ref_kernel(inputs)
            torch.cuda.synchronize()
            if not check_implementation(custom_output, ref_output):
                logger.log("check", "fail")
                exit(112)

    total_time = sum(times)
    average_duration = total_time / timed_runs
    variance = sum(map(lambda x: (x - average_duration) ** 2, times))  # noqa
    standard_deviation = math.sqrt(variance / (timed_runs - 1))
    standard_error = standard_deviation / math.sqrt(timed_runs)

    logger.log("check", "pass")
    logger.log("duration.mean", average_duration * 1e9)
    logger.log("duration.std", standard_deviation * 1e9)
    logger.log("duration.err", standard_error * 1e9)
    logger.log("duration.best", min(times) * 1e9)
    logger.log("duration.worst", max(times) * 1e9)

    kernel_name = "Reference" if time_reference_impl else "Submitted"
    print(f"{kernel_name} kernel runtime: {average_duration:.4f} ± {standard_error:.4} seconds")


def main():
    parser = argparse.ArgumentParser(description='Evaluate kernel implementation.')
    parser.add_argument(
        REFERENCE_TIMING_ARG, action='store_true', help='Time ref kernel.'
    )
    args = parser.parse_args()
    print(f"starting script")
    try:
        logger = PopcornLogger(int(os.environ["POPCORN_FD"]))
    except Exception as e:
        print(e, file=sys.stderr)
        exit(111)

    seed = int(os.environ.get("POPCORN_FD", 42))
    rng = torch.Generator()
    rng.manual_seed(seed)
    print(f"seed: {seed}")
    print(f"time ref: {args.time_ref}")
    print(f"correctness: {not args.time_ref}")
    if not args.time_ref:
        if not correctness(rng):
            logger.log("check", "fail")
            exit(112)
    
    metric(logger, rng, time_reference_impl=args.time_ref)


if __name__ == "__main__":
    main()
