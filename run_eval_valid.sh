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

V2_VANILLA_GRU_CONFIG="config/nlp/v2/vanilla/AttenGRU.config"
V2_VANILLA_CHECKPOINT_DIR="output/checkpoints/vanilla/v2_attengru"
V2_VANILLA_GRU_RESULTS="output/results/vanilla/v2_attengru_valid_metrics.json"

V2_VANILLA_LSTM_CONFIG="config/nlp/v2/vanilla/AttenLSTM.config"
V2_VANILLA_LSTM_CHECKPOINT_DIR="output/checkpoints/vanilla/v2_attenlstm"
V2_VANILLA_LSTM_RESULTS="output/results/vanilla/v2_attenlstm_valid_metrics.json"

V2_SUMMARIZED_GRU_CONFIG="config/nlp/v2/summarized/AttenGRU.config"
V2_SUMMARIZED_CHECKPOINT_DIR="output/checkpoints/summarized/v2_attengru"
V2_SUMMARIZED_GRU_RESULTS="output/results/summarized/v2_attengru_valid_metrics.json"

V2_SUMMARIZED_LSTM_CONFIG="config/nlp/v2/summarized/AttenLSTM.config"
V2_SUMMARIZED_LSTM_CHECKPOINT_DIR="output/checkpoints/summarized/v2_attenlstm"
V2_SUMMARIZED_LSTM_RESULTS="output/results/summarized/v2_attenlstm_valid_metrics.json"

V3_VANILLA_GRU_CONFIG="config/nlp/v3/vanilla/AttenGRU.config"
V3_VANILLA_CHECKPOINT_DIR="output/checkpoints/vanilla/v3_attengru"
V3_VANILLA_GRU_RESULTS="output/results/vanilla/v3_attengru_valid_metrics.json"

V3_VANILLA_LSTM_CONFIG="config/nlp/v3/vanilla/AttenLSTM.config"
V3_VANILLA_LSTM_CHECKPOINT_DIR="output/checkpoints/vanilla/v3_attenlstm"
V3_VANILLA_LSTM_RESULTS="output/results/vanilla/v3_attenlstm_valid_metrics.json"

V3_SUMMARIZED_GRU_CONFIG="config/nlp/v3/summarized/AttenGRU.config"
V3_SUMMARIZED_GRU_CHECKPOINT_DIR="output/checkpoints/summarized/v3_attengru"
V3_SUMMARIZED_GRU_RESULTS="output/results/summarized/v3_attengru_valid_metrics.json"

V3_SUMMARIZED_LSTM_CONFIG="config/nlp/v3/summarized/AttenLSTM.config"
V3_SUMMARIZED_LSTM_CHECKPOINT_DIR="output/checkpoints/summarized/v3_attenlstm"
V3_SUMMARIZED_LSTM_RESULTS="output/results/summarized/v3_attenlstm_valid_metrics.json"

run_command "python3 eval_valid.py -c $V2_VANILLA_GRU_CONFIG -g 0 --checkpoint-dir $V2_VANILLA_CHECKPOINT_DIR --result $V2_VANILLA_GRU_RESULTS" "eval_valid.py (v2_vanilla_gru)"
run_command "python3 eval_valid.py -c $V2_VANILLA_LSTM_CONFIG -g 0 --checkpoint-dir $V2_VANILLA_LSTM_CHECKPOINT_DIR --result $V2_VANILLA_LSTM_RESULTS" "eval_valid.py (v2_vanilla_lstm)"

run_command "python3 eval_valid.py -c $V2_SUMMARIZED_GRU_CONFIG -g 0 --checkpoint-dir $V2_SUMMARIZED_CHECKPOINT_DIR --result $V2_SUMMARIZED_GRU_RESULTS" "eval_valid.py (v2_summarized_gru)"
run_command "python3 eval_valid.py -c $V2_SUMMARIZED_LSTM_CONFIG -g 0 --checkpoint-dir $V2_SUMMARIZED_LSTM_CHECKPOINT_DIR --result $V2_SUMMARIZED_LSTM_RESULTS" "eval_valid.py (v2_summarized_lstm)"

run_command "python3 eval_valid.py -c $V3_VANILLA_GRU_CONFIG -g 0 --checkpoint-dir $V3_VANILLA_CHECKPOINT_DIR --result $V3_VANILLA_GRU_RESULTS" "eval_valid.py (v3_vanilla_gru)"
run_command "python3 eval_valid.py -c $V3_VANILLA_LSTM_CONFIG -g 0 --checkpoint-dir $V3_VANILLA_LSTM_CHECKPOINT_DIR --result $V3_VANILLA_LSTM_RESULTS" "eval_valid.py (v3_vanilla_lstm)"

run_command "python3 eval_valid.py -c $V3_SUMMARIZED_GRU_CONFIG -g 0 --checkpoint-dir $V3_SUMMARIZED_GRU_CHECKPOINT_DIR --result $V3_SUMMARIZED_GRU_RESULTS" "eval_valid.py (v3_summarized_gru)"
run_command "python3 eval_valid.py -c $V3_SUMMARIZED_LSTM_CONFIG -g 0 --checkpoint-dir $V3_SUMMARIZED_LSTM_CHECKPOINT_DIR --result $V3_SUMMARIZED_LSTM_RESULTS" "eval_valid.py (v3_summarized_lstm)"

echo "All commands succeeded"