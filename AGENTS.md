# PROJECT KNOWLEDGE BASE

**Generated:** 2026-09-25

## OVERVIEW
BERT-PLI (BERT-based Paragraph-Level Interaction) research and experimentation pipeline for Legal Case Retrieval (COLIEE benchmark).
The repository supports:
- Training and evaluating BERT-PLI ranking models with multiple aggregation architectures (`AttenGRU`, `AttenLSTM`, `AttenTransformer`, and `BertPoolOutMax`).
- Comparing text representations across multiple dataset versions (`v1`, `v2`, `v3`) and granularity variants (`vanilla` sentence-level, `summarized` via TextRank/Sumy, and `paragraph`-level).
- Checkpoint validation selection (`eval_valid.py`, `run_best_model.py`, `run_eval_valid.sh`).
- Quantitative attention summarization (entropy, Gini coefficient, top-k concentration) and qualitative heatmaps for divergent model predictions (`attention_summary.py`, `attention_divergent_predictions.py`).
- Statistical significance analysis using McNemar tests (`mcnemar_test.py`).
- Scientific dataflow provenance tracking with DfAnalyzer (MonetDB + REST API).
- Embedding and vector retrieval toolchain using Qdrant in `support/`.

## STRUCTURE
```text
.
├── config/              # Model, parser, and experiment configs (v1/v2/v3, divergent, nlp)
├── data/                # COLIEE dataset, parsed JSONs, and label mappings
├── dataset/             # PyTorch dataset definitions (BertDataset, AttenDataset, etc.)
├── DfAnalyzer-Docker/   # MonetDB + Java REST service for provenance tracking
├── formatter/           # Data formatters for model inputs and outputs
├── knowledge_base/      # Research notes, McNemar analysis reports, hypothesis validation, and guides
├── model/               # PyTorch neural architectures (BERT-PLI, AttenRNN, AttenTransformer)
├── output/              # Model checkpoints and experimental results (JSON, CSV, heatmaps)
│   ├── checkpoints/     # Saved model weights
│   └── results/         # Parsed predictions, McNemar CSVs, and divergent attention stats
├── provenance/          # DfAnalyzer Python library (dfa-lib-python) and tracking wrappers
├── support/             # Modern Python (>=3.9) Qdrant embedding and clustering toolchain
├── tests/               # Automated unit and integration tests (test_attention_summary.py)
├── tools/               # Auxiliary scripts and initializers
├── eval_valid.py        # Validation checkpoint evaluator (F1, Precision, Recall)
├── attention_summary.py # Quantitative attention distribution summarizer
├── mcnemar_test.py      # Pairwise statistical significance tester
└── train.py             # Main model training entry point
```

## WHERE TO LOOK
| Task / Subsystem | Location | Notes |
|---|---|---|
| **Training & Inference** | `train.py`, `bert_pli_train.py`, `test.py`, `bert_pli_test.py` | Core model training and test loops |
| **Model Architectures** | `model/nlp/` | `AttenGRU`, `AttenLSTM`, `AttenTransformer`, `BertPLI` |
| **Data Parsers & Slicing** | `coliee_to_sts_parser.py`, `split_train_labels.py`, `poolout.py`, `poolout_to_train.py` | Extracts sentence/paragraph/summary representations |
| **Validation & Best Checkpoints** | `eval_valid.py`, `run_best_model.py`, `run_eval_valid.sh` | Evaluates validation checkpoints to identify peak F1/Recall |
| **Attention Divergence Analysis** | `attention_summary.py`, `attention_metrics.py`, `divergent_subset_loader.py` | Calculates entropy, Gini, and top-k mass on divergent predictions |
| **Attention Heatmaps** | `attention_divergent_predictions.py`, `extract_attention.py` | Visualizes paragraph attention distribution for divergent pairs |
| **Statistical Testing** | `mcnemar_test.py`, `requirements_mcnemar.txt` | Pairwise McNemar $\chi^2$ and exact binomial tests |
| **Workflows & Pipelines** | `run_workflow.sh`, `run_workflow_v3.sh`, `run_workflow_v3_vanilla.sh`, `run_workflow_v3_summarized.sh` | End-to-end automation scripts |
| **Provenance Tracking** | `provenance/`, `provenance_workflow.py`, `DfAnalyzer-Docker/` | DfAnalyzer integration for dataflow monitoring |
| **Knowledge Base & Reports** | `knowledge_base/` | Research findings, hypothesis validation, and user guides |
| **Vector DB / Modern Toolchain** | `support/` | Isolated Python >=3.9 stack with Qdrant and HDBSCAN |
| **Unit Tests** | `tests/` | Unit tests for attention metrics and divergent loaders |
| **Docker Infrastructure** | `Dockerfile`, `compose.yaml`, `docker_up.sh` | Containerized runtime (Python 3.10 + PyTorch CUDA) |

## ENVIRONMENTS & CONVENTIONS

### 1. Multi-Environment Setup
- **Root / Container Environment**: Configured via `Dockerfile` (`python:3.10-slim`) and `requirements.txt`. Used for training, inference, and BERT representations.
- **Statistical / Analysis Environment (`.venv_mcnemar`)**: Dedicated venv using `requirements_mcnemar.txt` (`statsmodels`, `scipy`, `tabulate`, `pandas`). Used for `mcnemar_test.py` and running test suites.
- **Support Environment (`support/.venv`)**: Isolated Python >=3.9 environment managed via `uv` (`pyproject.toml`, `uv.lock`) for Qdrant vector operations.

### 2. Experiment & Config Conventions
- **Experiment Versions**: `v1`, `v2`, `v3` correspond to different pipeline revisions and feature representations.
- **Granularity Variants**:
  - `vanilla`: Sentence-level BERT interaction representation.
  - `summarized`: TextRank-summarized sentence representation (via Sumy).
  - `paragraph`: Paragraph-level segmentation representation.
- **Attention Divergence Modes**:
  - `inter`: Comparison across model architectures (e.g., GRU vs. LSTM).
  - `intra`: Comparison across representation variants (e.g., vanilla vs. summarized).
- **Identifier Conventions**: Pair IDs are formatted as `{query_doc_id}_{candidate_doc_id}` (without `.txt` suffix).
- **Prediction Files**: Standard COLIEE JSON format `{"<query_doc>.txt": ["<candidate_doc>.txt", ...]}` stored as `*_parsed_results.json`.

## ANTI-PATTERNS (THIS PROJECT)
- **Do NOT mix environments**: Never install `.venv_mcnemar` packages or `support/` dependencies directly into the root environment, and vice versa.
- **Do NOT run host Python for statistical/unit tests without venv**: Run unit tests and `mcnemar_test.py` with `.venv_mcnemar/bin/python` to avoid missing `statsmodels`/`numpy`.
- **Do NOT hardcode local dataset paths**: Use Docker-mounted paths (`/app/data` or `/app/data/COLIEE`) when executing inside containers.
- **Do NOT commit large binaries**: Model checkpoints (`output/checkpoints/`), raw prediction caches, `.venv_*`, and `knowledge_base/` are gitignored.
- **Do NOT modify provenance schemas without DfAnalyzer dataflow alignment**: Ensure `DATAFLOW_TAG` matches the dataflow specification registered in MonetDB.

## COMMON COMMANDS

### 1. Tests & Statistical Analysis
```bash
# Run unit tests
.venv_mcnemar/bin/python3 -m unittest discover tests

# Run McNemar statistical test across all model groups
.venv_mcnemar/bin/python3 mcnemar_test.py --results_dir output/results --groups paragraph summarized vanilla

# Run McNemar test on specific model pairs
.venv_mcnemar/bin/python3 mcnemar_test.py --model_a output/results/vanilla/v1_gru_parsed_results.json --model_b output/results/vanilla/v1_lstm_parsed_results.json
```

### 2. Attention Metrics & Divergence Analysis
```bash
# Quantitative attention metrics for intra-model divergence (vanilla vs summarized)
python3 attention_summary.py --experiment-version v1 --type intra --plots

# Quantitative attention metrics for inter-model divergence (GRU vs LSTM)
python3 attention_summary.py --experiment-version v1 --type inter --plots

# Attention heatmap visualization for divergent predictions
python3 attention_divergent_predictions.py -ev v1 -t inter
```

### 3. Checkpoint Evaluation & Validation
```bash
# Evaluate validation checkpoints across all v2 and v3 variants
./run_eval_valid.sh

# Evaluate specific model checkpoint
python3 eval_valid.py -c config/nlp/v2/vanilla/AttenGRU.config -g 0 --checkpoint-dir output/checkpoints/vanilla/v2_attengru --result output/results/vanilla/v2_attengru_valid_metrics.json
```

### 4. Workflows & Training
```bash
# Run multi-version full workflow (v2 + v3)
./run_workflow_v3.sh

# Train specific model configuration
python3 train.py -c config/nlp/v2/vanilla/AttenGRU.config -g 0
```

### 5. Docker & DfAnalyzer Services
```bash
# Build and run DfAnalyzer provenance service
cd DfAnalyzer-Docker && docker build -t dfanalyzer .
docker run -itd --name dfanalyzer -p 22000:22000 -p 50000:50000 dfanalyzer

# Spin up containers via helper script
./docker_up.sh
```
