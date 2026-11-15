"""Upload the checksum-pinned NVIDIA drop to KernelBot's private Modal volume.

PYTHONPATH=src/runners python scripts/upload_nvcc_toolchain.py --source /path/to/binaries
This does not deploy a runner or publish the binaries outside the Modal workspace.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import modal
from nvcc_toolchain import BINARY_SHA256, RELEASE


def upload(source: Path, volume_name: str):
    for name, expected in BINARY_SHA256.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Compiler checksum mismatch: {name}")
    manifest = json.dumps({"release": RELEASE, "sha256": BINARY_SHA256}, sort_keys=True) + "\n"
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    manifest_path = f"/{RELEASE}/manifest.json"
    try:
        existing = b"".join(volume.read_file(manifest_path)).decode()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != manifest:
            raise ValueError("Existing compiler release has a different manifest")
        print(f"{RELEASE} is already present in {volume_name}")
        return
    with tempfile.TemporaryDirectory() as directory:
        metadata = Path(directory) / "manifest.json"
        metadata.write_text(manifest)
        with volume.batch_upload(force=False) as batch:
            for name in BINARY_SHA256:
                batch.put_file(source / name, f"/{RELEASE}/{name}")
            batch.put_file(metadata, manifest_path)
    print(f"Uploaded verified {RELEASE} to private volume {volume_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--volume",
        default=os.environ.get("KERNELBOT_NVCC_TOOLCHAIN_VOLUME", "kernelbot-nvcc-toolchains"),
    )
    args = parser.parse_args()
    upload(args.source, args.volume)
