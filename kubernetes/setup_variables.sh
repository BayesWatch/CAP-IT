#!/bin/bash
export WANDB_API_KEY="myapikey"
export WANDB_ENTITY="myentity"
export WANDB_PROJECT="myproject"

export HF_USERNAME="myusername"
export HF_TOKEN="mytoken"

export TOKENIZERS_PARALLELISM=false

export EXPERIMENTS_DIR=/run/experiments
export EXPERIMENT_DIR=/run/experiments

export DATASET_DIR=/data/
export MODEL_DIR=/run/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export EXPERIMENT_NAME_PREFIX="capit-v2-debug"
export DOCKER_IMAGE_PATH="ghcr.io/bayeswatch/capit:latest"

mkdir -p "$HOME/.huggingface"
touch "$HOME/.huggingface/token"

echo $HF_TOKEN >"$HOME/.huggingface/token"
