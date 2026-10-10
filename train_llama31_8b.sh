#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=llama31_8b
METHOD=medusa_reflex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Model-specific defaults; caller environment always takes precedence.
# LAUNCHER_USE_ENV=1 remains accepted but is no longer required.
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
export MODEL="${MODEL:-/workspace/storage-shared/models/Llama-3.1-8B-Instruct}"
export MODEL_TYPE="${MODEL_TYPE:-llama}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
export DATA_ROOT="${DATA_ROOT:-/workspace/storage-shared/nlp/minhpn19/data}"
export DATASET="${DATASET:-simplelr}"
export PRETRAIN_DATASET="${PRETRAIN_DATASET:-sharegpt}"
export TARGET_ADAPTER="${TARGET_ADAPTER:-$OUTPUT_ROOT/initial_target/${MODEL_KEY}_seed42}"
export PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
export MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-$PRETRAIN_MODEL_ROOT}"
export DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$PRETRAIN_MODEL_ROOT/latest_checkpoint}"
export TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-$OUTPUT_ROOT/train/$MODEL_KEY}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUTPUT_ROOT/triton_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
export ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
export DRAFT_ACCUMULATION_STEPS="${DRAFT_ACCUMULATION_STEPS:-1}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
export PRETRAIN_ACCUMULATION_STEPS="${PRETRAIN_ACCUMULATION_STEPS:-8}"
export TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
export GENERATION_LENGTH_POLICY="${GENERATION_LENGTH_POLICY:-specnaacl_compatible}"
export GRPO_BENCHMARK="${GRPO_BENCHMARK:-1}"
export COMMON_ENV="${COMMON_ENV:-$SCRIPT_DIR/configs/_shared/b200_common.env}"
export MODEL_ENV="${MODEL_ENV:-$SCRIPT_DIR/configs/$MODEL_KEY/b200.env}"
export LAUNCHER_ENV="${LAUNCHER_ENV:-$SCRIPT_DIR/configs/$MODEL_KEY/launcher.env}"
# Keep RESUME/RUN_DIR/RUN_NAME and explicit training budgets supplied by caller.
source "$SCRIPT_DIR/scripts/launch/train_model.sh" "$@"
