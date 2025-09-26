# FROM python:3.6-slim
FROM pure/python:3.6-cuda10.0-cudnn7-runtime

# Set working directory
WORKDIR /app
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \ 
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y curl libgomp1

# Copy requirements if you have them
COPY requirements.txt .
# RUN curl -O https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl && \
RUN curl -O https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl && \
    curl -O https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
    pip install torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
    rm torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# COPY . .

CMD ["tail", "-f", "/dev/null"]
# https://pytorch.org/get-started/previous-versions/
# https://download.pytorch.org/whl/cu100/torch_stable.html
# https://hub.docker.com/r/pure/python/tags
# https://github.com/ThuYShao/BERT-PLI-IJCAI2020/forks?include=active&page=1&period=&sort_by=stargazer_counts