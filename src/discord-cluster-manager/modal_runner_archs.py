# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
import modal
from modal_runner import app, cuda_image, modal_run_config

gpus = ["T4", "L4", "A100-80GB", "H100!"]
for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!")
    app.function(gpu=gpu, image=cuda_image, name=f"run_cuda_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )
    app.function(gpu=gpu, image=cuda_image, name=f"run_pytorch_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )


@app.function(
    image=modal.Image.debian_slim().pip_install("requests"), max_containers=1, timeout=600
)
def run_pytorch_script_b200(config: dict, timeout: int):
    """Send a config and timeout to the server and return the response."""
    import requests

    ip_addr = "34.59.196.5"
    port = "33001"

    payload = {"config": config, "timeout": timeout}

    try:
        response = requests.post(f"http://{ip_addr}:{port}", json=payload, timeout=timeout + 5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


@app.local_entrypoint()
def test_b200(timeout: int = 300):
    config = {}
    run_pytorch_script_b200.remote(config, timeout)
