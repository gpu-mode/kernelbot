FROM rocm/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update -y \
    && sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository -y ppa:git-core/ppa \
    && sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    curl \
    tar \
    sudo \
    python3-pip \
    git \
    build-essential \
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
    python3.10-venv \
    && sudo rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install ninja tinygrad

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

# Setup GitHub Actions runner
WORKDIR /actions-runner
RUN curl -O -L https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.323.0.tar.gz

RUN pip install \
    ninja \
    numpy \
    packaging \
    wheel \
    triton \
    tinygrad

CMD ["/bin/bash"]