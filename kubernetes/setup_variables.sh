#!/bin/bash
#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTFjMTY5Zi03ZGUwLTQ4ODYtYWI0Zi1kZDEzNjlkMGI5ZjQifQ=="
export NEPTUNE_PROJECT=MachineLearningBrewery/capit-v5
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'

export WANDB_API_KEY="821661c6ee1657a2717093701ab76574ae1a9be0"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=capit-v5

export EXPERIMENT_NAME=capit-v5
export HF_USERNAME="Antreas"
export HF_TOKEN=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

export TOKENIZERS_PARALLELISM=False

export CODE_DIR=/app/
export PROJECT_DIR=/app/
export EXPERIMENT_NAME_PREFIX="capit-v5"
export EXPERIMENTS_DIR=/data/experiments/
export EXPERIMENT_DIR=/data/experiments/

export DATASET_DIR=/data/
export MODEL_DIR=/data/models/

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-c
export CLUSTER_PROJECT=tali-multi-modal

export DOCKER_IMAGE_PATH="ghcr.io/antreasantoniou/capit:latest"
