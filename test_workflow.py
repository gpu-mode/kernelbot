import subprocess
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_pytorch_install():
    cache_dir = os.path.expanduser("~/.cache/pip")
    
    if os.path.exists(cache_dir):
        subprocess.run(["rm", "-rf", cache_dir])
    
    start_time = time.time()
    result = subprocess.run(
        ["pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"pip install failed with output:\n{result.stdout}\n{result.stderr}")
        raise AssertionError("First pip install failed")
        
    first_run_time = time.time() - start_time
    logger.info(f"First run (no cache) took: {first_run_time:.2f} seconds")
    
    start_time = time.time()
    result = subprocess.run(
        ["pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Second pip install failed with output:\n{result.stdout}\n{result.stderr}")
        raise AssertionError("Second pip install failed")
        
    second_run_time = time.time() - start_time
    logger.info(f"Second run (with cache) took: {second_run_time:.2f} seconds")
    
    improvement = (first_run_time - second_run_time) / first_run_time * 100
    logger.info(f"Cache improved installation time by {improvement:.1f}%")
    
    assert improvement > 0, "Cache did not improve installation time"
    assert improvement > 10, f"Cache improvement ({improvement:.1f}%) was less than 10%"

if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__])
    except ImportError:
        test_pytorch_install()