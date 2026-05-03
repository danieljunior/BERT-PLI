#!/bin/bash

# Function to run a command and check for failure
run_command() {
    local cmd=$1
    local name=$2
    echo "Running $name..."
    if ! eval "$cmd"; then
        echo "$name failed"
        exit 1
    fi
}
run_command "docker run -itd --name dfanalyzer -p 22000:22000 -p 50000:50000 dfanalyzer" "dfanalyzer container"
run_command "docker build --no-cache --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 --build-arg PYTORCH_GIT_REF=main --build-arg TORCH_CUDA_ARCH_LIST=12.0 --build-arg MAX_JOBS=4 --tag bert-pli:cuda124-src ." "bert-pli docker build"
run_command "docker run -itd --shm-size 5gb --name bert-pli --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=1 -e DFA_URL=http://dfanalyzer:22000/ -v ${PWD}:/app -v /home/danieljunior/workspace/datasets/jurídicos/COLIEE\ dataset:/app/data --link dfanalyzer:dfanalyzer bert-pli:cuda124-src tail -f /dev/null" "bert-pli container"
