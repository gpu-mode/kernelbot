# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
from modal_runner import app, cuda_image, modal_run_config, model_image, model_weights, sccache_vol

gpus = ["T4", "L4", "L4:4", "A100-80GB", "H100!", "B200"]
for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(gpu=gpu, image=cuda_image, name=f"run_cuda_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )
    app.function(gpu=gpu, image=cuda_image, name=f"run_pytorch_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )

# Model competition functions — vLLM fork benchmarking
model_gpus = ["H100!", "B200"]
for gpu in model_gpus:
    gpu_slug = gpu.lower().strip("!")
    app.function(
        gpu=gpu,
        image=model_image,
        volumes={"/models": model_weights, "/sccache": sccache_vol},
        name=f"run_model_benchmark_{gpu_slug}",
        serialized=True,
        timeout=3600,
    )(modal_run_config)
