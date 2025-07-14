#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from consts import ExitCode, SubmissionMode
from run_eval import run_cuda_script

def main():
    print("Running simple CUDA test...")
    
    # Load test files
    try:
        ref = Path("examples/identity_cuda/reference.cuh").read_text()
        task_h = Path("examples/identity_cuda/task.h").read_text()
        utils_h = Path("examples/utils.h").read_text()
        eval_cu = Path("examples/eval.cu").read_text()
        submission_cu = Path("examples/identity_cuda/submission.cu").read_text()
        
        header_files = {"reference.cuh": ref, "task.h": task_h, "utils.h": utils_h}
        source_files = {"eval.cu": eval_cu, "submission.cu": submission_cu}
        
        print("✓ Test files loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load test files: {e}")
        return 1
    
    # Run the CUDA test
    try:
        eval_result = run_cuda_script(
            source_files,
            header_files,
            arch=None,
            mode=SubmissionMode.TEST.value,
            tests="size: 256; seed: 42\n",
        )
        
        print("✓ CUDA script execution completed")
        
        # Check compilation
        if not eval_result.compilation.success:
            print(f"✗ Compilation failed: {eval_result.compilation.stderr}")
            return 1
        
        print("✓ Compilation successful")
        
        # Check execution
        if not eval_result.run.success:
            print(f"✗ Execution failed: {eval_result.run.stderr}")
            return 1
            
        print("✓ Execution successful")
        
        # Check test results
        if eval_result.run.result.get("check") != "pass":
            print(f"✗ Test validation failed: {eval_result.run.result}")
            return 1
            
        print("✓ Test validation passed")
        print("🎉 All CUDA tests passed!")
        return 0
        
    except Exception as e:
        print(f"✗ CUDA test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 