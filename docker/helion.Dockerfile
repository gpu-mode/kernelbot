FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Install Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    git \
    gcc-13 \
    g++-13 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python

# Install pip and uv
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir uv

# Core build tools
RUN uv pip install --system \
    ninja~=1.11 \
    wheel~=0.45 \
    packaging~=25.0 \
    numpy~=2.3

# PyTorch (CUDA 13.0 wheels)
RUN uv pip install --system \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Helion
RUN uv pip install --system helion

# # tinygrad
# RUN uv pip install --system tinygrad~=0.10

# # NVIDIA CUDA packages
# RUN uv pip install --system \
#     nvidia-cupynumeric~=25.3 \
#     nvidia-cutlass-dsl==4.3.5 \
#     "cuda-core[cu13]" \
#     "cuda-python[all]==13.0"

# # CUTLASS C++ headers
# RUN git clone --depth 1 --branch v4.3.5 https://github.com/NVIDIA/cutlass.git /opt/cutlass
# ENV CUTLASS_PATH=/opt/cutlass
# ENV CPLUS_INCLUDE_PATH=/opt/cutlass/include:/opt/cutlass/tools/util/include
