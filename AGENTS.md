# AGENTS

## Repo layout
- Root contains the BERT-PLI training code (Python) plus Docker setup; DfAnalyzer is a separate Java/MonetDB service under `DfAnalyzer-Docker/`.
- `support/` is a separate toolchain (Qdrant + embeddings) with its own Python >=3.9 deps; do not mix with the main Py3.6 environment.

## Primary workflows
- `run_workflow.sh` is the end-to-end pipeline; it requires a `DATAFLOW_TAG` argument and loads `.run_env` from the repo root. It expects `/app/...` paths and exits early if any env var is missing.
- `run_test.sh` and `run_test_poolout.sh` chain test -> parse -> evaluate as used in README examples.

## Data and mounts
- Training scripts assume COLIEE data under `/app/data/COLIEE/` and outputs under `/app/output/`; `compose.yaml` mounts the dataset into `/app/data/COLIEE` inside the container.
- The dataset path in Docker commands includes spaces and non-ASCII characters; always quote/escape or adjust it when editing `compose.yaml` or `docker_up.sh`.

## Docker and services
- The root `Dockerfile` targets Python 3.6 + CUDA 10 and installs legacy torch wheels; it does NOT `COPY . .` (commented), so you must mount the repo as a volume to run code.
- `DFA_URL` must point at the DfAnalyzer service; `docker_up.sh` and README run the containers with `--link dfanalyzer:dfanalyzer` and `-e DFA_URL=http://dfanalyzer:22000/`.
- `DfAnalyzer-Docker/README.txt` notes that `docker compose down` resets the MonetDB database and that the build expects extra assets downloaded from the linked Google Drive.

## Support/Qdrant
- `support/README.md` is the source of truth for Qdrant workflows; it assumes a running Qdrant container and uses `uv run` commands from `support/`.
