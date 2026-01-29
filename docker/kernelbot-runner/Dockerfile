# Kernelbot Runner Docker Image
#
# This image is used by Buildkite agents to run kernel submissions.
# It matches the Modal runner configuration for consistent behavior.
#
# Build:
#   docker build -t ghcr.io/gpu-mode/kernelbot-runner:latest -f docker/kernelbot-runner/Dockerfile .
#
# Run locally (for testing):
#   docker run --gpus '"device=0"' -e SUBMISSION_PAYLOAD="..." kernelbot-runner:latest

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

LABEL org.opencontainers.image.source="https://github.com/gpu-mode/kernelbot"
LABEL org.opencontainers.image.description="Kernelbot GPU runner for kernel competitions"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3-pip \
    git \
    gcc-13 \
    g++-13 \
    clang-18 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.13 /usr/bin/python

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (matching modal_runner.py)
COPY docker/kernelbot-runner/requirements-runner.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-runner.txt

# Install PyTorch with CUDA 13.0 support
RUN pip install \
    torch==2.9.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Install additional frameworks
RUN pip install \
    tinygrad~=0.10

# Install NVIDIA CUDA packages
RUN pip install \
    nvidia-cupynumeric~=25.3 \
    nvidia-cutlass-dsl==4.3.5 \
    "cuda-core[cu13]" \
    "cuda-python[all]==13.0"

# Copy kernelbot library and runner
WORKDIR /opt/kernelbot
COPY src/libkernelbot /opt/kernelbot/libkernelbot
COPY src/runners/buildkite-runner.py /opt/kernelbot/

# Set PYTHONPATH so libkernelbot is importable
ENV PYTHONPATH="/opt/kernelbot:$PYTHONPATH"

# Default command
CMD ["python", "/opt/kernelbot/buildkite-runner.py"]
