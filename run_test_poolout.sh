#!/bin/bash

# Script to run three Python commands in sequence, stopping on failure and indicating which one failed.
# Usage: ./run_test_poolout.sh <config1> <gpu> <checkpoint> <result1> <in2> <result2> <config2> <result3> <in4> <result4>

# Assign positional arguments to variables
CONFIG1=$1
GPU=$2
CHECKPOINT=$3
RESULT1=$4
IN2=$5
RESULT2=$6
CONFIG2=$7
RESULT3=$8
IN4=$9
RESULT4=${10}

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
run_command "python3 poolout.py -c $CONFIG1 -g $GPU --checkpoint $CHECKPOINT --result $RESULT1" "poolout.py"

# Run the second command
run_command "python3 poolout_to_train.py -in $IN2 -out $RESULT1 --result $RESULT2" "poolout_to_train.py"

run_command "python3 poolout.py -c $CONFIG2 -g $GPU --checkpoint $CHECKPOINT --result $RESULT3" "poolout.py"

run_command "python3 poolout_to_train.py -in $IN4 -out $RESULT3 --result $RESULT4" "poolout_to_train.py"

echo "All commands succeeded"
