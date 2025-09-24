FROM ghcr.io/actions/actions-runner:latest

ENV CXX=clang++
ENV UCX_CXX=g++
ENV UCX_CC=gcc

RUN sudo apt-get update -y \
    && sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository -y ppa:git-core/ppa \
    && sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
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
    python3.10-venv \
    && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update && sudo apt-get install -y python3.10 python3-pip python-is-python3 python3-setuptools python3-wheel libpython3.10

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

RUN sudo apt update -y \
    && sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" \
    && sudo usermod -a -G render,video runner \
    && wget https://repo.radeon.com/amdgpu-install/6.3.1/ubuntu/jammy/amdgpu-install_6.3.60301-1_all.deb \
    && sudo apt install -y ./amdgpu-install_6.3.60301-1_all.deb \
    && sudo apt update -y \
    && sudo apt install -y rocm

RUN sudo pip install --upgrade pip

RUN sudo pip install --no-cache-dir torch==2.10.0.dev20250916+rocm6.3 pytorch-triton-rocm --index-url https://download.pytorch.org/whl/nightly/rocm6.3

RUN git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter \
    && git checkout 1d88633958236e942cba3c283864282f7af3ebc5 \
    && sudo pip install -r requirements.txt \
    && sudo python3 setup.py develop

RUN sudo mkdir -p /home/runner/aiter/aiter/jit/build \
    && sudo chown -R runner:runner /home/runner/aiter/aiter/jit/build

RUN sudo pip install \
    ninja \
    numpy \
    packaging \
    wheel \
    tinygrad

RUN sudo pip install git+https://github.com/ROCm/iris.git

RUN sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    libtool \
    pkg-config \
    build-essential \
    gfortran \
    flex \
    bison \
    libomp-dev \
    libhwloc-dev \
    libnuma-dev \
    && sudo rm -rf /var/lib/apt/lists/*

ENV UCX_INSTALL_DIR=/opt/ucx
ENV OMPI_INSTALL_DIR=/opt/openmpi
ENV ROCSHMEM_INSTALL_DIR=/opt/rocshmem
ENV ROCM_PATH=/opt/rocm

RUN cd /tmp \
    && git clone https://github.com/openucx/ucx.git -b v1.17.x \
    && cd ucx \
    && ./autogen.sh \
    && CC=gcc CXX=g++ ./configure --prefix=${UCX_INSTALL_DIR} --with-rocm=${ROCM_PATH} --enable-mt --disable-optimizations \
    && make -j$(nproc) \
    && sudo make install \
    && cd / \
    && sudo rm -rf /tmp/ucx

RUN cd /tmp \
    && git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x \
    && cd ompi \
    && ./autogen.pl \
    && ./configure --prefix=${OMPI_INSTALL_DIR} --with-rocm=${ROCM_PATH} --with-ucx=${UCX_INSTALL_DIR} \
    && make -j$(nproc) \
    && sudo make install \
    && cd / \
    && sudo rm -rf /tmp/ompi

ENV PATH="${OMPI_INSTALL_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib:/opt/rocm/lib"


RUN cd /tmp \
    && git clone https://github.com/ROCm/rocSHMEM.git \
    && cd rocSHMEM \
    && mkdir build \
    && cd build \
    && MPI_ROOT=${OMPI_INSTALL_DIR} UCX_ROOT=${UCX_INSTALL_DIR} CMAKE_PREFIX_PATH="${ROCM_PATH}:$CMAKE_PREFIX_PATH" \
       sudo ../scripts/build_configs/ipc_single -DCMAKE_INSTALL_PREFIX=/opt/rocshmem \
    && cd / \
    && sudo rm -rf /tmp/rocSHMEM


ENV ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR}
ENV LD_LIBRARY_PATH="${ROCSHMEM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"