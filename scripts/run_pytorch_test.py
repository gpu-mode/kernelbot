#!/usr/bin/env python3
import modal
import sys
import os
from pathlib import Path

# Change to the correct directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

# Add the src directory to Python path for Modal deserialization
sys.path.append('src/discord-cluster-manager')

# Import required modules so they're available for deserialization
from run_eval import FullResult, EvalResult, CompileResult, RunResult, SystemInfo
from consts import ExitCode, SubmissionMode
from task import make_task_definition, build_task_config

# Run Python test via Modal on GPU using the same framework as Discord bot
func = modal.Function.from_name('discord-bot-runner', 'run_pytorch_script_t4')

# Load the task definition exactly like the Discord bot does
task = make_task_definition("examples/identity_py")

# Read the submission file
submission = Path("examples/identity_py/submission.py").read_text()

# Build config using the same function as the Discord bot
config = build_task_config(
    task=task.task,
    submission_content=submission,
    arch=None,
    mode=SubmissionMode.TEST,
)

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