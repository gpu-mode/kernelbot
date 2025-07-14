#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from consts import ExitCode, SubmissionMode
from run_eval import run_pytorch_script

def main():
    print("Running simple Python test...")
    
    # Load test files
    try:
        ref = Path("examples/identity_py/reference.py").read_text()
        task = Path("examples/identity_py/task.py").read_text()
        py_eval = Path("examples/eval.py").read_text()
        utils = Path("examples/utils.py").read_text()
        submission = Path("examples/identity_py/submission.py").read_text()
        
        files = {
            "eval.py": py_eval, 
            "reference.py": ref, 
            "utils.py": utils, 
            "task.py": task,
            "submission.py": submission
        }
        
        print("✓ Test files loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load test files: {e}")
        return 1
    
    # Run the Python test
    try:
        eval_result = run_pytorch_script(
            files,
            "eval.py",
            mode=SubmissionMode.TEST.value,
            tests="size: 256; seed: 42\n",
        )
        
        print("✓ Python script execution completed")
        
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
        print("🎉 All Python tests passed!")
        return 0
        
    except Exception as e:
        print(f"✗ Python test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 