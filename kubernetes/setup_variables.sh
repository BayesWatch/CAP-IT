#!/bin/bash
export WANDB_API_KEY="821661c6ee1657a2717093701ab76574ae1a9be0"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=capit-debug-multi-gpu

export HF_USERNAME="Antreas"
export HF_TOKEN=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

export TOKENIZERS_PARALLELISM=true

export PROJECT_DIR=/app/
export EXPERIMENTS_DIR=/run/experiments
export EXPERIMENT_DIR=/run/experiments

export DATASET_DIR=/data/
export MODEL_DIR=/run/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export EXPERIMENT_NAME_PREFIX="capit-v3"
export DOCKER_IMAGE_PATH="ghcr.io/antreasantoniou/capit:latest"
