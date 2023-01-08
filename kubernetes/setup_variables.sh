#!/bin/bash
export WANDB_API_KEY="03f97dc1ae484a463ef95ba2e03fb12f7eeddd85"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=capit-v2

export HF_USERNAME="evolvingfungus"
export HF_TOKEN=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

export TOKENIZERS_PARALLELISM=false

export PROJECT_DIR=/app/
export EXPERIMENTS_DIR=/run/experiments
export EXPERIMENT_DIR=/run/experiments

export DATASET_DIR=/data/
export MODEL_DIR=/run/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export EXPERIMENT_NAME_PREFIX="capit-v2"
export DOCKER_IMAGE_PATH="ghcr.io/bayeswatch/capit:latest"
