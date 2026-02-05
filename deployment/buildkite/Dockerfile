# Kernelbot evaluation image
# Pre-built with all dependencies for fast cold starts
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone and install kernelbot
WORKDIR /opt/kernelbot
RUN git clone --depth 1 --branch buildkite-infrastructure https://github.com/gpu-mode/kernelbot.git .

# Install dependencies with uv
RUN uv sync

# Install PyTorch and GPU packages
RUN uv pip install torch triton numpy --index-url https://download.pytorch.org/whl/cu124

# Ensure venv is activated for any commands
ENV VIRTUAL_ENV=/opt/kernelbot/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    python -c "import triton; print(f'Triton installed')" && \
    python -c "from libkernelbot.run_eval import run_config; print('kernelbot installed')"

# Default command
CMD ["python", "/opt/kernelbot/src/runners/buildkite-runner.py"]
