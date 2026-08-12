#!/bin/bash
set -euo pipefail

echo "Running vanilla v2 experiments"
./run_workflow.sh "bert-pli"

echo "Running vanilla v3 workflow"
./run_workflow_v3_vanilla.sh "bert-pli"

echo "Running summarized v3 workflow"
./run_workflow_v3_summarized.sh "bert-pli"

echo "All experiments succeeded"