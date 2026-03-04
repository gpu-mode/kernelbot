FROM ghcr.io/actions/actions-runner:latest

ENV CXX=clang++

RUN sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    jq \
    sudo \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    python3-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-dev \
    && sudo rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

RUN sudo apt update -y \
    && sudo usermod -a -G render,video runner \
    && wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb \
    && sudo apt install -y ./amdgpu-install_7.1.70100-1_all.deb \
    && sudo apt update -y \
    && sudo apt install -y rocm

ENV ROCM_PATH=/opt/rocm

RUN sudo pip install --break-system-packages --no-cache-dir torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1

RUN git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter \
    && git checkout f3be04a12a0cfd6b5e2c7a94edc774f1bc24460d \
    && sudo pip install --break-system-packages -r requirements.txt \
    && sudo python3 setup.py develop

RUN sudo mkdir -p /home/runner/aiter/aiter/jit/build \
    && sudo chown -R runner:runner /home/runner/aiter/aiter/jit/build

RUN sudo pip install --break-system-packages \
    ninja \
    numpy \
    packaging \
    wheel \
    tinygrad

RUN sudo pip install --break-system-packages git+https://github.com/ROCm/iris.git

ENV LD_LIBRARY_PATH="/opt/rocm/lib"
