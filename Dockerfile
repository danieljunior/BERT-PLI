# FROM python:3.6-slim
ARG PYTORCH_IMAGE=pytorch/pytorch:nightly-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DFA_URL="http://dfanalyzer:22000/"
# Set working directory
WORKDIR /app
RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements if you have them
COPY requirements.txt .
RUN pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# COPY . .
COPY provenance/dfa-lib-python provenance/dfa-lib-python
WORKDIR /app/provenance/dfa-lib-python
RUN python setup.py install
WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
# https://pytorch.org/get-started/previous-versions/
# https://download.pytorch.org/whl/cu100/torch_stable.html
# https://hub.docker.com/r/pure/python/tags
# https://github.com/ThuYShao/BERT-PLI-IJCAI2020/forks?include=active&page=1&period=&sort_by=stargazer_counts
