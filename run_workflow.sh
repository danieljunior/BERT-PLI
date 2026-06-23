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
: "${POOLOUT_TEST_CONFIG:?Missing POOLOUT_TEST_CONFIG in .env}"
: "${BERT_CHECKPOINT:?Missing BERT_CHECKPOINT in .env}"
: "${POOLOUT_RESULT:?Missing POOLOUT_RESULT in .env}"
: "${POOLOUT_TEST_RESULT:?Missing POOLOUT_TEST_RESULT in .env}"
: "${TRAIN_PAIRS:?Missing TRAIN_PAIRS in .env}"
: "${TRAIN_LABELS:?Missing TRAIN_LABELS in .env}"
: "${TRAIN_DATA:?Missing TRAIN_DATA in .env}"
: "${TEST_DATA:?Missing TEST_DATA in .env}"
: "${LSTM_CONFIG:?Missing LSTM_CONFIG in .env}"
: "${LSTM_GPU:?Missing LSTM_GPU in .env}"
: "${LSTM_CHECKPOINT:?Missing LSTM_CHECKPOINT in .env}"
: "${LSTM_RESULTS:?Missing LSTM_RESULTS in .env}"
: "${GRU_CONFIG:?Missing GRU_CONFIG in .env}"
: "${GRU_GPU:?Missing GRU_GPU in .env}"
: "${GRU_CHECKPOINT:?Missing GRU_CHECKPOINT in .env}"
: "${GRU_RESULTS:?Missing GRU_RESULTS in .env}"
: "${PARSED_GRU_RESULTS:?Missing PARSED_GRU_RESULTS in .env}"
: "${TEST_PAIRS:?Missing TEST_PAIRS in .env}"
: "${TEST_LABELS:?Missing TEST_LABELS in .env}"
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

# run_command "python3 coliee_to_sts_parser.py -p ./data/COLIEE/task1_train_files_2024/ -l ./data/COLIEE/task1_TRAIN_PAIRS_2024.json -vo ./data/COLIEE/train_vanilla_sentences.json -so ./data/COLIEE/train_summarized_sentences.json" "coliee_to_sts_parser.py"

# run_command "python3 coliee_to_sts_parser.py -p ./data/COLIEE/task1_test_files_2024/ -l ./data/COLIEE/task1_TEST_PAIRS_2024.json -vo ./data/COLIEE/test_vanilla_sentences.json -so ./data/COLIEE/test_summarized_sentences.json --test" "coliee_to_sts_parser.py (test)"

# run_command "python3 train.py -c $FINETUNE_CONFIG -g 0" "train.py (finetune)"

# run_command "sleep 1" "sleep (wait for finetune to finish)"

# run_command "python3 poolout.py -c $POOLOUT_CONFIG -g $POOLOUT_GPU --checkpoint $BERT_CHECKPOINT --result $POOLOUT_RESULT" "poolout.py"

# run_command "sleep 1" "sleep (wait for poolout to finish)"

# run_command "python3 poolout_to_train.py -in $TRAIN_PAIRS -out $POOLOUT_RESULT --result $TRAIN_DATA" "poolout_to_train.py"

# run_command "sleep 1" "sleep (wait for conversion to finish)"

# run_command "python3 poolout.py -c $POOLOUT_TEST_CONFIG -g $POOLOUT_GPU --checkpoint $BERT_CHECKPOINT --result $POOLOUT_TEST_RESULT --test" "poolout.py"

# run_command "sleep 1" "sleep (wait for poolout to finish)"

# run_command "python3 poolout_to_train.py -in $TEST_PAIRS -out $POOLOUT_TEST_RESULT --result $TEST_DATA --test" "poolout_to_train.py"

# run_command "sleep 1" "sleep (wait for conversion to finish)"

#### LSTM
run_command "python3 train.py -c $LSTM_CONFIG -g $LSTM_GPU" "train.py (lstm)"

run_command "sleep 5" "sleep (wait for training to finish)"

LSTM_CHECKPOINT_DIR=$(dirname "$LSTM_CHECKPOINT")
rm -f /tmp/lstm_eval_valid.json
run_command "python3 eval_valid.py -c $LSTM_CONFIG -g $LSTM_GPU --checkpoint-dir $LSTM_CHECKPOINT_DIR --result /tmp/lstm_eval_valid.json" "eval_valid.py (lstm)"
BEST_LSTM_CHECKPOINT=$(python3 -c "import json; res=json.load(open('/tmp/lstm_eval_valid.json'))['results']; print(max(res, key=lambda x: x['metrics']['f1'])['checkpoint'])")
echo "Best LSTM checkpoint: $BEST_LSTM_CHECKPOINT"

run_command "python3 test.py -c $LSTM_CONFIG -g $LSTM_GPU --checkpoint $BEST_LSTM_CHECKPOINT --result $LSTM_RESULTS" "test.py (lstm)"

run_command "sleep 5" "sleep (wait for testing to finish)"

run_command "python parse_results.py parse $LSTM_RESULTS $PARSED_LSTM_RESULTS" "parse_results.py (lstm)"

run_command "python parse_results.py evaluate $TEST_LABELS $PARSED_LSTM_RESULTS $LSTM_METRICS" "parse_results.py (evaluate lstm)"

### GRU
run_command "python3 train.py -c $GRU_CONFIG -g $GRU_GPU" "train.py (gru)"

run_command "sleep 5" "sleep (wait for training to finish)"

GRU_CHECKPOINT_DIR=$(dirname "$GRU_CHECKPOINT")
rm -f /tmp/gru_eval_valid.json
run_command "python3 eval_valid.py -c $GRU_CONFIG -g $GRU_GPU --checkpoint-dir $GRU_CHECKPOINT_DIR --result /tmp/gru_eval_valid.json" "eval_valid.py (gru)"
BEST_GRU_CHECKPOINT=$(python3 -c "import json; res=json.load(open('/tmp/gru_eval_valid.json'))['results']; print(max(res, key=lambda x: x['metrics']['f1'])['checkpoint'])")
echo "Best GRU checkpoint: $BEST_GRU_CHECKPOINT"

run_command "python3 test.py -c $GRU_CONFIG -g $GRU_GPU --checkpoint $BEST_GRU_CHECKPOINT --result $GRU_RESULTS" "test.py (gru)"

run_command "sleep 5" "sleep (wait for conversion to finish)"
  
run_command "python parse_results.py parse $GRU_RESULTS $PARSED_GRU_RESULTS" "parse_results.py (gru)"

run_command "python parse_results.py evaluate $TEST_LABELS $PARSED_GRU_RESULTS $GRU_METRICS" "parse_results.py (evaluate gru)"

#### TRANSFORMER

run_command "python3 train.py -c $TRANSFORMER_CONFIG -g $TRANSFORMER_GPU" "train.py (transformer)"

run_command "sleep 5" "sleep (wait for training to finish)"

TRANSFORMER_CHECKPOINT_DIR=$(dirname "$TRANSFORMER_CHECKPOINT")
rm -f /tmp/transformer_eval_valid.json
run_command "python3 eval_valid.py -c $TRANSFORMER_CONFIG -g $TRANSFORMER_GPU --checkpoint-dir $TRANSFORMER_CHECKPOINT_DIR --result /tmp/transformer_eval_valid.json" "eval_valid.py (transformer)"
BEST_TRANSFORMER_CHECKPOINT=$(python3 -c "import json; res=json.load(open('/tmp/transformer_eval_valid.json'))['results']; print(max(res, key=lambda x: x['metrics']['f1'])['checkpoint'])")
echo "Best Transformer checkpoint: $BEST_TRANSFORMER_CHECKPOINT"

run_command "python3 test.py -c $TRANSFORMER_CONFIG -g $TRANSFORMER_GPU --checkpoint $BEST_TRANSFORMER_CHECKPOINT --result $TRANSFORMER_RESULTS" "test.py (transformer)"

run_command "sleep 5" "sleep (wait for conversion to finish)"
  
run_command "python parse_results.py parse $TRANSFORMER_RESULTS $PARSED_TRANSFORMER_RESULTS" "parse_results.py (transformer)"

run_command "python parse_results.py evaluate $TEST_LABELS $PARSED_TRANSFORMER_RESULTS $TRANSFORMER_METRICS" "parse_results.py (evaluate transformer)"

echo "All commands succeeded"