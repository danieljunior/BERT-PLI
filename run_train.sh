#!/bin/bash

# Script to run three Python commands in sequence, stopping on failure and indicating which one failed.
# Usage: ./run_pipeline.sh <config1> <gpu1> <checkpoint1> <result1> <in2> <out2> <result2> <config3> <gpu3> <config4> <gpu4>

# Assign positional arguments to variables
CONFIG1=$1
GPU1=$2
CHECKPOINT1=$3
RESULT1=$4
IN2=$5
OUT2=$6
RESULT2=$7
CONFIG3=$8
GPU3=$9
CONFIG4=$10
GPU4=$11

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
run_command "python3 poolout.py -c $CONFIG1 -g $GPU1 --checkpoint $CHECKPOINT1 --result $RESULT1" "poolout.py"

# Run the second command
run_command "python3 poolout_to_train.py -in $IN2 -out $OUT2 --result $RESULT2" "poolout_to_train.py"

# Run the third command 
run_command "python3 train.py -c $CONFIG3 -g $GPU3" "train.py"

# Run the fourth command 
run_command "python3 train.py -c $CONFIG4 -g $GPU4" "train.py"

echo "All commands succeeded"