`docker run -itd --privileged --shm-size 5gb --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant/storage:/qdrant/storage qdrant/qdrant:latest` # RUN qdrant


`DOCKER_BUILDKIT=1 docker build . -t bert-pli-support`

`NV_GPU=5 nvidia-docker run -itd --rm --privileged --shm-size 5gb --name bert-pli-support --link qdrant:qdrant -v ${PWD}:/src -v /home/dsilva/workspace/BERT-PLI/data:/src/data -v /home/dsilva/workspace/BERT-PLI/output:/src/output bert-pli-support:latest tail -f /dev/null`

`uv run qdrant_coliee_loader.py data/train_paragraphs_processed_data.json --collection-name coliee-train`

`uv run qdrant_density_calculator.py --collection coliee-train`

`uv run qdrant_relevance_calculator.py --collection coliee-train`

`uv run qdrant_extract_relevant_segments.py   --collection coliee-test  --labels-file data/task1_test_labels_2024.json --negative-samples-file data/test_paragraphs_processed_data.json --output-file data/test_relevant_paragraphs_processed_data.json`