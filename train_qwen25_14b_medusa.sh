#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen25_14b
METHOD=medusa
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Edit this block for this model. Reset stale exports when switching models.
# LAUNCHER_USE_ENV=1 explicitly selects environment overrides instead.
if [[ "${LAUNCHER_USE_ENV:-0}" != 1 ]]; then
  export PYTHON_BIN="$(command -v python || command -v python3)"
  export MODEL="/workspace/storage-shared/models/Qwen2.5-14B-Instruct"
  export MODEL_TYPE="qwen2"
  export OUTPUT_ROOT="$SCRIPT_DIR/outputs"
  export DATA_ROOT="/workspace/storage-shared/nlp/minhpn19/data"
  export DATASET="simplelr"
  export DATASET_PATH="$DATA_ROOT/simplelr_abel_level3to5/train.parquet"
  export PRETRAIN_DATASET="sharegpt"
  export PRETRAIN_DATASET_PATH="$DATA_ROOT/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json"
  export TARGET_ADAPTER="$OUTPUT_ROOT/initial_target/${MODEL_KEY}_seed42"
  export PRETRAIN_MODEL_ROOT="$OUTPUT_ROOT/pretrain/$MODEL_KEY"
  export MODEL_OUTPUT_ROOT="$PRETRAIN_MODEL_ROOT"
  export DRAFT_CHECKPOINT="$PRETRAIN_MODEL_ROOT/latest_checkpoint"
  export TRAIN_MODEL_ROOT="$OUTPUT_ROOT/train/$MODEL_KEY"
  export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"
  export CUDA_VISIBLE_DEVICES="0"
  export NPROC_PER_NODE="1"
  export MODEL_DTYPE="bf16"
  export ATTENTION_IMPLEMENTATION="sdpa"
  export BATCH_SIZE="8"
  export ACCUMULATION_STEPS="4"
  export DRAFT_ACCUMULATION_STEPS="1"
  export PRETRAIN_BATCH_SIZE="2"
  export PRETRAIN_ACCUMULATION_STEPS="16"
  export TRAIN_SUBSET_SEED="42"
  export GENERATION_LENGTH_POLICY="specnaacl_compatible"
  export GRPO_BENCHMARK="1"
  export COMMON_ENV="$SCRIPT_DIR/configs/_shared/b200_common.env"
  export MODEL_ENV="$SCRIPT_DIR/configs/$MODEL_KEY/b200.env"
  export LAUNCHER_ENV="$SCRIPT_DIR/configs/$MODEL_KEY/launcher.env"
fi
# Keep RESUME/RUN_DIR/RUN_NAME and explicit training budgets supplied by caller.
source "$SCRIPT_DIR/scripts/launch/train_model.sh" "$@"
