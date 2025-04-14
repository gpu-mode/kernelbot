FROM rocm/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    tar \
    sudo \
    python3-pip \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip && \
    pip3 install ninja tinygrad

WORKDIR /actions-runner
RUN curl -O -L https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.323.0.tar.gz
