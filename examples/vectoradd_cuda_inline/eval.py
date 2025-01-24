import logging
import time
import os
import sys
import math
from logging import getLogger, StreamHandler

from utils import set_seed, check_implementation
from submission import custom_kernel
from reference import ref_kernel, generate_input

WARMUP_RUNS = 10
TIMED_RUNS = 100

logger = getLogger("PopcornOutput")
logger.setLevel(logging.INFO)

def setup_logger(fd):
    file_obj = os.fdopen(int(fd), 'w')
    handler = StreamHandler(file_obj)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def log(key, value):
    logger.info(f"{key}: {value}")

def measure_runtime():
    print("Warming up...")

    warmup_data = generate_input()
    for _ in range(WARMUP_RUNS):
        custom_kernel(warmup_data)
    
    durations = []

    for _ in range(TIMED_RUNS):
        data = generate_input()
        start = time.time()
        submission_output = custom_kernel(data)
        end = time.time()
        durations.append((end - start) * 1e9)

        reference_output = ref_kernel(data)
        if not check_implementation(submission_output, reference_output):
            log("check", "fail")
            sys.exit(112)
    
    total_duration = sum(durations)
    best = min(durations)
    worst = max(durations)
    average_duration = total_duration / TIMED_RUNS

    variance = sum([(d - average_duration) ** 2 for d in durations])
    standard_deviation = math.sqrt(variance / (TIMED_RUNS - 1))
    standard_error = standard_deviation / math.sqrt(TIMED_RUNS)

    log("check", "pass")
    log("duration.mean", average_duration)
    log("duration.std", standard_deviation)
    log("duration.err", standard_error)
    log("duration.best", best)
    log("duration.worst", worst)

    print(f"Average kernel runtime: {average_duration / 1e6} ± {standard_error / 1e6} µs")

def main():
    fd = os.getenv("POPCORN_FD")
    if fd:
        setup_logger(fd)
    else:
        return 111

    seed = os.getenv("POPCORN_SEED")
    seed = int(seed) if seed else 42

    set_seed(seed)
    data = generate_input()
    reference_output = ref_kernel(data)
    submission_output = custom_kernel(data)

    if not check_implementation(submission_output, reference_output):
        log("check", "fail")
        return 112

    measure_runtime()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
