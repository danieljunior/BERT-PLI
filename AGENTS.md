# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-08

## OVERVIEW
BERT-PLI training pipeline for Legal Case Retrieval and DfAnalyzer Java/MonetDB service for provenance tracking.

## STRUCTURE
```text
.
├── config/          # Model and parser configurations
├── data/            # COLIEE dataset and intermediate files
├── dataset/         # PyTorch dataset definitions
├── DfAnalyzer-Docker/ # External MonetDB/Java service
├── formatter/       # Data formatters
├── model/           # PyTorch models (BERT, RNNs, Attention)
├── output/          # Training checkpoints and metrics
├── provenance/      # DfAnalyzer python integration and storage
├── support/         # Qdrant + embeddings toolchain (Python >=3.9)
└── train.py         # Main training entry point
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training scripts | `train.py`, `bert_pli_train.py` | Main entry points |
| Model defs | `model/nlp/` | PyTorch architectures |
| Data loading | `dataset/` | Custom PyTorch Dataset classes |
| Workflows | `run_workflow.sh` | End-to-end pipeline |
| Docker setup | `compose.yaml`, `Dockerfile` | Relies on CUDA 10/13+ |

## CONVENTIONS
- Python 3.6 + legacy torch wheels for root environment.
- Docker `DATAFLOW_TAG` required for end-to-end runs.
- Qdrant logic isolated in `support/` with modern Python.

## ANTI-PATTERNS (THIS PROJECT)
- Do NOT mix `support/` (Py3.9+) with root `requirements.txt` (Py3.6).
- Do NOT edit dataset paths without updating Docker mounts.

## COMMANDS
```bash
# Build & Run DfAnalyzer
cd DfAnalyzer-Docker && docker build -t dfanalyzer .
docker run -itd --name dfanalyzer -p 22000:22000 -p 50000:50000 dfanalyzer

# Train
python3 train.py -c config/nlp/BertPoint.config -g 0
```
