FROM ghcr.io/actions/actions-runner:latest

# Install CUDA 13.1 toolkit
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    gnupg \
    && sudo rm -rf /var/lib/apt/lists/* \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
    cuda-toolkit-13-1 \
    && sudo rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-13.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}"

# Install Python 3.13
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    software-properties-common \
    && sudo add-apt-repository ppa:deadsnakes/ppa \
    && sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    git \
    gcc-13 \
    g++-13 \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
    && sudo ln -sf /usr/bin/python3.13 /usr/local/bin/python

# Install pip and uv
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir uv

# Core build tools
RUN sudo uv pip install --system \
    ninja~=1.11 \
    wheel~=0.45 \
    packaging~=25.0 \
    numpy~=2.3

# PyTorch (CUDA 13.0 wheels)
RUN sudo uv pip install --system \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Helion
RUN sudo uv pip install --system helion

# # tinygrad
# RUN sudo uv pip install --system tinygrad~=0.10

# # NVIDIA CUDA packages
# RUN sudo uv pip install --system \
#     nvidia-cupynumeric~=25.3 \
#     nvidia-cutlass-dsl==4.3.5 \
#     "cuda-core[cu13]" \
#     "cuda-python[all]==13.0"

# # CUTLASS C++ headers
# RUN git clone --depth 1 --branch v4.3.5 https://github.com/NVIDIA/cutlass.git /opt/cutlass
# ENV CUTLASS_PATH=/opt/cutlass
# ENV CPLUS_INCLUDE_PATH=/opt/cutlass/include:/opt/cutlass/tools/util/include
