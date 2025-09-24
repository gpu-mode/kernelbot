import os
import signal
import sys
from multiprocessing import Pool

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def timeout_handler(signum, frame):
    print("✗ TIMEOUT: Process hung")
    sys.exit(1)


def test_worker(args):
    rank, world_size, master_port = args
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        print(f"Rank {rank}: Init NCCL...")
        dist.init_process_group(
            "nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}"),
        )
        signal.alarm(0)

        device = torch.device(f"cuda:{rank}")
        tensor = torch.ones(100, device=device) * rank

        signal.alarm(15)
        dist.all_reduce(tensor)
        signal.alarm(0)

        print(f"✓ Rank {rank}: sum = {tensor[0].item()}")
        dist.destroy_process_group()
        return True

    except Exception as e:
        signal.alarm(0)
        print(f"✗ Rank {rank}: {e}")
        return False


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Testing {num_gpus} GPUs - 4 rounds")

    for round_num in range(4):
        print(f"=== ROUND {round_num + 1} ===")
        master_port = 29500 + round_num

        mp.set_start_method("spawn", force=True)

        # Prepare worker arguments
        worker_args = [(rank, num_gpus, master_port) for rank in range(num_gpus)]

        with Pool(processes=num_gpus) as pool:
            try:
                # Use map_async with timeout
                result = pool.map_async(test_worker, worker_args)
                results = result.get(timeout=60)

                # Check if all workers succeeded
                if not all(results):
                    print(f"✗ ROUND {round_num + 1} FAILED")
                    sys.exit(1)

            except mp.TimeoutError:
                print(f"✗ ROUND {round_num + 1} HUNG")
                pool.terminate()
                pool.join()
                sys.exit(1)
            except Exception as e:
                print(f"✗ ROUND {round_num + 1} ERROR: {e}")
                sys.exit(1)

        print(f"✓ ROUND {round_num + 1} PASSED")

    print("✓ ALL ROUNDS PASSED")


if __name__ == "__main__":
    main()
