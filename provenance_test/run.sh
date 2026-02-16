#!/bin/bash

run_command() {
    local cmd=$1
    local name=$2
    echo "Running $name..."
    if ! eval "$cmd"; then
        echo "$name failed"
        exit 1
    fi
}

run_command "python create_dataflow.py"
run_command "python parse_coliee_dataset.py"
run_command "python finetune_bert.py"
run_command "python poolout.py"