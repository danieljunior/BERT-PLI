#!/bin/bash
DATAFLOW_TAG="${1:-}"
if [ -z "$DATAFLOW_TAG" ]; then
    echo "Error: DATAFLOW_TAG argument is required"
    exit 1
fi

# Export once at the beginning
export DATAFLOW_TAG

run_command() {
    local cmd=$1
    local name=$2
    echo "Running $name..."
    if ! eval "$cmd"; then
        echo "$name failed"
        exit 1
    fi
}

run_command "python /app/provenance/parse_coliee_dataset.py"
run_command "python /app/provenance/finetune_bert.py"
run_command "python /app/provenance/poolout.py"