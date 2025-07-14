#!/usr/bin/env python3
import modal
import sys
import os

# Change to the correct directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

# Run pytest via Modal on GPU
func = modal.Function.from_name('discord-bot-runner', 'run_cuda_script_t4')
with open('scripts/ci_test_cuda.py', 'r') as f:
    code = f.read()

result = func.remote(config={
    'code': code,
    'language': 'py',
    'timeout': 300
})

print('Modal execution result:', result)
if result.get('success') != True:
    print('CUDA test failed on Modal')
    sys.exit(1) 