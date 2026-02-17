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

run_command "python /app/provenance/prospective_service.py"
run_command "python /app/provenance/parse_coliee_dataset.py"
run_command "python /app/provenance/finetune_bert.py"
run_command "python /app/provenance/poolout.py"