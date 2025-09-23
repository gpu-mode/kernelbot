import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import signal
import sys

def timeout_handler(signum, frame):
    print('✗ TIMEOUT: Process hung')
    sys.exit(1)

def test_worker(rank, world_size, master_port):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        print(f'Rank {rank}: Init NCCL...')
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        signal.alarm(0)
        
        device = torch.device(f'cuda:{rank}')
        tensor = torch.ones(100, device=device) * rank
        
        signal.alarm(15)
        dist.all_reduce(tensor)
        signal.alarm(0)
        
        print(f'✓ Rank {rank}: sum = {tensor[0].item()}')
        dist.destroy_process_group()
        
    except Exception as e:
        signal.alarm(0)
        print(f'✗ Rank {rank}: {e}')
        sys.exit(1)

def main():
    num_gpus = torch.cuda.device_count()
    print(f'Testing {num_gpus} GPUs - 4 rounds')
    
    for round_num in range(4):
        print(f'=== ROUND {round_num + 1} ===')
        master_port = 29500 + round_num
        
        mp.set_start_method('spawn', force=True)
        processes = []
        
        for rank in range(num_gpus):
            p = mp.Process(target=test_worker, args=(rank, num_gpus, master_port))
            p.start()
            processes.append(p)
        
        for i, p in enumerate(processes):
            p.join(timeout=60)
            if p.exitcode != 0:
                print(f'✗ ROUND {round_num + 1} FAILED')
                for rp in processes:
                    if rp.is_alive():
                        rp.terminate()
                sys.exit(1)
            elif p.is_alive():
                print(f'✗ ROUND {round_num + 1} HUNG')
                p.terminate()
                sys.exit(1)
        
        print(f'✓ ROUND {round_num + 1} PASSED')
    
    print('✓ ALL ROUNDS PASSED')

if __name__ == '__main__':
    main()