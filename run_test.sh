#!/bin/bash

# Script to run three Python commands in sequence, stopping on failure and indicating which one failed.
# Usage: ./run_test.sh <config> <gpu> <checkpoint> <result1> <result2> <labels> <result3>

# Assign positional arguments to variables
CONFIG=$1
GPU=$2
CHECKPOINT=$3
RESULT1=$4
RESULT2=$5
LABELS=$6
RESULT3=$7

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

# Run the first command
run_command "python3 test.py -c $CONFIG -g $GPU --checkpoint $CHECKPOINT --result $RESULT1" "test.py"

# Run the second command
run_command "python parse_results.py parse $RESULT1 $RESULT2" "parse_results.py parse"

# Run the third command
run_command "python parse_results.py evaluate $LABELS $RESULT2 $RESULT3" "parse_results.py evaluate"

echo "All commands succeeded"