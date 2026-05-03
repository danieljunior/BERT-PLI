# FROM python:3.6-slim
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG PYTORCH_GIT_REF=main
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG MAX_JOBS=4
FROM ${CUDA_IMAGE}

ENV DFA_URL="http://dfanalyzer:22000/"
# Set working directory
WORKDIR /app
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV USE_NCCL=1
ENV BUILD_TEST=0
ENV MAX_JOBS=${MAX_JOBS}
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgomp1 \
    libomp-dev \
    libopenblas-dev \
    libssl-dev \
    libffi-dev \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if you have them
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir numpy==1.26.4
RUN git clone --depth 1 --branch ${PYTORCH_GIT_REF} https://github.com/pytorch/pytorch.git /opt/pytorch && \
    cd /opt/pytorch && \
    git submodule sync && git submodule update --init --recursive && \
    python3 -m pip install --no-cache-dir -r /opt/pytorch/requirements.txt && \
    python3 setup.py install && \
    cd /app && \
    rm -rf /opt/pytorch
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
# COPY . .
COPY provenance/dfa-lib-python provenance/dfa-lib-python
WORKDIR /app/provenance/dfa-lib-python
RUN python3 setup.py install
WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
# https://pytorch.org/get-started/previous-versions/
# https://download.pytorch.org/whl/cu100/torch_stable.html
# https://hub.docker.com/r/pure/python/tags
# https://github.com/ThuYShao/BERT-PLI-IJCAI2020/forks?include=active&page=1&period=&sort_by=stargazer_counts
