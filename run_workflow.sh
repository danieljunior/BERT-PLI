#!/bin/bash
set -euo pipefail

DATAFLOW_TAG="${1:-}"
if [ -z "$DATAFLOW_TAG" ]; then
    echo "Error: DATAFLOW_TAG argument is required"
    exit 1
fi

# Export once at the beginning
export DATAFLOW_TAG

# Load .env (same folder as script)
ENV_FILE="$(dirname "$0")/.run_env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo ".env file not found: $ENV_FILE"
  exit 1
fi

# Optional: validate required vars
: "${FINETUNE_CONFIG:?Missing FINETUNE_CONFIG in .env}"
: "${POOLOUT_CONFIG:?Missing POOLOUT_CONFIG in .env}"
: "${POOLOUT_GPU:?Missing POOLOUT_GPU in .env}"
: "${BERT_CHECKPOINT:?Missing BERT_CHECKPOINT in .env}"
: "${POOLOUT_RESULT:?Missing POOLOUT_RESULT in .env}"
: "${PAIRS_DATA:?Missing PAIRS_DATA in .env}"
: "${TRAIN_DATA:?Missing TRAIN_DATA in .env}"
: "${LSTM_CONFIG:?Missing LSTM_CONFIG in .env}"
: "${LSTM_GPU:?Missing LSTM_GPU in .env}"
: "${LSTM_CHECKPOINT:?Missing LSTM_CHECKPOINT in .env}"
: "${LSTM_RESULTS:?Missing LSTM_RESULTS in .env}"
: "${GRU_CONFIG:?Missing GRU_CONFIG in .env}"
: "${GRU_GPU:?Missing GRU_GPU in .env}"
: "${GRU_CHECKPOINT:?Missing GRU_CHECKPOINT in .env}"
: "${GRU_RESULTS:?Missing GRU_RESULTS in .env}"
: "${PARSED_GRU_RESULTS:?Missing PARSED_GRU_RESULTS in .env}"
: "${TRUE_LABELS:?Missing TRUE_LABELS in .env}"
: "${GRU_METRICS:?Missing GRU_METRICS in .env}"
: "${PARSED_LSTM_RESULTS:?Missing PARSED_LSTM_RESULTS in .env}"
: "${LSTM_METRICS:?Missing LSTM_METRICS in .env}"

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

run_command "python3 coliee_to_sts_parser.py" "coliee_to_sts_parser.py"

run_command "python3 train.py -c $FINETUNE_CONFIG -g 0" "train.py (finetune)"

run_command "python3 poolout.py -c $POOLOUT_CONFIG -g $POOLOUT_GPU --checkpoint $BERT_CHECKPOINT --result $POOLOUT_RESULT" "poolout.py"

run_command "python3 poolout_to_train.py -in $PAIRS_DATA -out $POOLOUT_RESULT --result $TRAIN_DATA" "poolout_to_train.py"

run_command "python3 train.py -c $LSTM_CONFIG -g $LSTM_GPU" "train.py (lstm)"

run_command "python3 test.py -c $LSTM_CONFIG -g $LSTM_GPU --checkpoint $LSTM_CHECKPOINT --result $LSTM_RESULTS" "test.py (lstm)"

run_command "python parse_results.py parse $LSTM_RESULTS $PARSED_LSTM_RESULTS" "parse_results.py (lstm)"

run_command "python parse_results.py evaluate $TRUE_LABELS $PARSED_LSTM_RESULTS $LSTM_METRICS" "parse_results.py (evaluate lstm)"

run_command "python3 train.py -c $GRU_CONFIG -g $GRU_GPU" "train.py (gru)"

run_command "python3 test.py -c $GRU_CONFIG -g $GRU_GPU --checkpoint $GRU_CHECKPOINT --result $GRU_RESULTS" "test.py (gru)"

run_command "python parse_results.py parse $GRU_RESULTS $PARSED_GRU_RESULTS" "parse_results.py (gru)"

run_command "python parse_results.py evaluate $TRUE_LABELS $PARSED_GRU_RESULTS $GRU_METRICS" "parse_results.py (evaluate gru)"

echo "All commands succeeded"