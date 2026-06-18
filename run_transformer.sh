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


# run_command "python3 test.py -c /app/config/nlp/AttenTransformer.config -g 0,1 --checkpoint /app/output/checkpoints/vanilla/attentransformer/11.pkl --result /app/output/results/vanilla/transformer_results.json" "test.py (transformer)"

# run_command "python3 parse_results.py parse /app/output/results/vanilla/transformer_results.json /app/output/results/vanilla/transformer_parsed_results.json" "parse_results.py (lstm)"

# run_command "python3 parse_results.py evaluate /app/data/COLIEE/task1_test_labels_2024.json /app/output/results/vanilla/transformer_parsed_results.json /app/output/results/vanilla/transformer_metrics.json" "parse_results.py (evaluate lstm)"

run_command "python3 train.py -c /app/config/nlp/AttenTransformerSummarized.config -g 0,1" "train.py (transformer summarized)"
run_command "python3 train.py -c /app/config/nlp/AttenTransformer.config -g 0,1" "train.py (transformer)"

# run_command "python3 test.py -c $LSTM_CONFIG -g $LSTM_GPU --checkpoint $LSTM_CHECKPOINT --result $LSTM_RESULTS" "test.py (lstm)"

# run_command "python parse_results.py parse $LSTM_RESULTS $PARSED_LSTM_RESULTS" "parse_results.py (lstm)"

# run_command "python parse_results.py evaluate $TEST_LABELS $PARSED_LSTM_RESULTS $LSTM_METRICS" "parse_results.py (evaluate lstm)"

echo "All commands succeeded"
