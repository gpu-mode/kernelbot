"""Download model weights to the Modal volume.

Usage:
    modal run src/runners/download_model.py --model meta-llama/Llama-3.1-8B
"""

import modal

app = modal.App("model-weight-downloader")
volume = modal.Volume.from_name("model-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("huggingface_hub", "transformers", "torch")
)


@app.function(image=image, volumes={"/models": volume}, timeout=3600)
def download_model(model: str, revision: str = "main"):
    from huggingface_hub import snapshot_download

    print(f"Downloading {model} (revision={revision}) to /models/...")
    snapshot_download(
        repo_id=model,
        revision=revision,
        local_dir=f"/models/models--{model.replace('/', '--')}",
    )
    volume.commit()
    print(f"Done. Model saved to /models/models--{model.replace('/', '--')}")


@app.local_entrypoint()
def main(model: str, revision: str = "main"):
    download_model.remote(model=model, revision=revision)
