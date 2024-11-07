import os
import subprocess
import tempfile
import shutil
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_packages(package_names, pip_path, use_uv=False, python_path=None):
    if use_uv:
        subprocess.run([pip_path, "install", "uv"])
        subprocess.run([python_path, "-m", "uv", "pip", "install", "--no-deps"] + package_names)
        subprocess.run([python_path, "-m", "uv", "pip", "install"] + package_names)
        return subprocess.CompletedProcess(args=[], returncode=0)
    else:
        return subprocess.run([pip_path, "install"] + package_names)

def simulate_workflow(script_content, clear_cache=False, use_uv=False):
    start_time = time.time()
    
    pip_cache_dir = os.path.expanduser("~/.cache/pip")
    uv_cache_dir = os.path.expanduser("~/.cache/uv")
    
    if clear_cache:
        logger.info("Clearing package caches...")
        if os.path.exists(pip_cache_dir):
            shutil.rmtree(pip_cache_dir)
        if os.path.exists(uv_cache_dir):
            shutil.rmtree(uv_cache_dir)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")
        
        logger.info("Creating virtual environment...")
        subprocess.run(["python3.10", "-m", "venv", f"{temp_dir}/venv"])
        
        pip_path = f"{temp_dir}/venv/bin/pip" if os.name != 'nt' else f"{temp_dir}/venv/Scripts/pip"
        python_path = f"{temp_dir}/venv/bin/python" if os.name != 'nt' else f"{temp_dir}/venv/Scripts/python"
        logger.info(f"Using pip at: {pip_path}")
        logger.info(f"Using python at: {python_path}")
        
        logger.info("Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"])
        
        logger.info("Installing numpy and torch...")
        install_start = time.time()
        if use_uv:
            install_packages(["numpy", "torch"], pip_path, use_uv=True, python_path=python_path)
        else:
            cmd = [pip_path, "install"]
            if clear_cache:
                cmd.append("--no-cache-dir")
            cmd.extend(["numpy", "torch"])
            subprocess.run(cmd)
        install_duration = time.time() - install_start
        logger.info(f"Package installation took {install_duration:.2f} seconds")
        
        train_path = os.path.join(temp_dir, "train.py")
        logger.info(f"Writing training script to {train_path}")
        with open(train_path, "w") as f:
            f.write(script_content)
        
        log_path = os.path.join(temp_dir, "training.log")
        logger.info(f"Running training script, logging to {log_path}")
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [python_path, "train.py"],
                cwd=temp_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        
        logger.info("Reading training logs...")
        with open(log_path, "r") as f:
            logs = f.read()
            
        logger.info(f"Training completed with return code: {result.returncode}")
        
    total_duration = time.time() - start_time
    logger.info(f"Total workflow duration: {total_duration:.2f} seconds")
    return result.returncode, logs, total_duration

if __name__ == "__main__":
    try:
        logger.info("Checking Python 3.10 installation...")
        python_version = subprocess.run(["python", "-c", "import sys; print(sys.version.split()[0])"], 
                                      capture_output=True, 
                                      text=True, 
                                      check=True).stdout.strip()
        if not python_version.startswith("3.10"):
            logger.error(f"Python 3.10 is required but found {python_version}")
            sys.exit(1)
        logger.info(f"Found Python version: {python_version}")
        
        numpy_test_script = """
import numpy as np

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.matmul(matrix1, matrix2)
print("Matrix multiplication result:\\n", result)
"""

        pytorch_test_script = """
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
tensor2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
result = torch.matmul(tensor1, tensor2)
print("Tensor multiplication result:\\n", result)
"""
        
        logger.info("\nRunning test with UV package installer...")
        return_code, logs, uv_time = simulate_workflow(numpy_test_script, clear_cache=False, use_uv=True)
        
        logger.info("\nRunning test without cache...")
        return_code, logs, uncached_time = simulate_workflow(numpy_test_script, clear_cache=True)

        logger.info("\nRunning test with cached packages...")
        return_code, logs, cached_time = simulate_workflow(numpy_test_script, clear_cache=False)
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS".center(50))
        print("="*50)
        print(f"{'Test Type':<30}{'Duration':<20}")
        print("-"*50)
        print(f"{'With Cache':<30}{f'{cached_time:.2f}s':<20}")
        print(f"{'Without Cache':<30}{f'{uncached_time:.2f}s':<20}")
        print(f"{'With UV':<30}{f'{uv_time:.2f}s':<20}")
        print(f"{'Cache Speedup':<30}{f'{uncached_time - cached_time:.2f}s':<20}")
        print(f"{'UV vs Cache Speedup':<30}{f'{cached_time - uv_time:.2f}s':<20}")
        print("="*50)
        
        print("\nTest Output:")
        print("-"*50)
        print(logs)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Failed to determine Python version")
        sys.exit(1)