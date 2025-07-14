#!/usr/bin/env python3
import modal
import sys
import os

# Change to the correct directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

# Add the src directory to Python path for Modal deserialization
sys.path.append('src/discord-cluster-manager')

# Import required modules so they're available for deserialization
from run_eval import FullResult, EvalResult, CompileResult, RunResult, SystemInfo
from consts import ExitCode, SubmissionMode

# Run simple Python test via Modal on GPU
func = modal.Function.from_name('discord-bot-runner', 'run_pytorch_script_t4')
with open('scripts/ci_test_python_simple.py', 'r') as f:
    code = f.read()

# Pass config in the correct format that run_config expects
config = {
    'lang': 'py',
    'sources': {'ci_test_python_simple.py': code},
    'main': 'ci_test_python_simple.py',
    'mode': SubmissionMode.TEST.value,
    'tests': [],
    'benchmarks': [],
    'seed': None,
    'ranking_by': 'last',
    'test_timeout': 300,
    'benchmark_timeout': 300,
    'ranked_timeout': 300,
}

result = func.remote(config=config)

print('Modal execution result:', result)
if not result.success:
    print('PyTorch test failed on Modal')
    print('Error:', result.error)
    sys.exit(1)

# Check if any test runs failed
for run_name, run_result in result.runs.items():
    if run_result.run and not run_result.run.success:
        print(f'Test run {run_name} failed')
        sys.exit(1)
        
print('PyTorch test passed on Modal') 