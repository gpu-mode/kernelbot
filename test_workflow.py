import os
import subprocess
import tempfile
import shutil
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_workflow(script_content):
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
        subprocess.run([pip_path, "install", "numpy", "torch"])
        
        logger.info("Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"])
        
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
        return result.returncode, logs

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
        
        test_script = """
import numpy as np
import torch
print("Test script running...")
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
"""
        
        return_code, logs = simulate_workflow(test_script)
        print("\nTest output:")
        print(logs)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Failed to determine Python version")
        sys.exit(1)