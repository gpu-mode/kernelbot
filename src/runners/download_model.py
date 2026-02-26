"""Download model weights to the Modal volume.

Usage:
    modal run src/runners/download_model.py --model meta-llama/Llama-3.1-8B
"""

from pathlib import Path

import modal

app = modal.App("model-weight-downloader")
volume = modal.Volume.from_name("model-weights", create_if_missing=True)
MODEL_DIR = Path("/models")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=image,
    volumes={MODEL_DIR.as_posix(): volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_model(model: str, revision: str = "main"):
    from huggingface_hub import snapshot_download

    dest = MODEL_DIR / model
    print(f"Downloading {model} (revision={revision}) to {dest} ...")
    snapshot_download(repo_id=model, local_dir=dest, revision=revision)
    volume.commit()
    print(f"Done. Model saved to {dest}")


@app.local_entrypoint()
def main(model: str, revision: str = "main"):
    download_model.remote(model=model, revision=revision)
