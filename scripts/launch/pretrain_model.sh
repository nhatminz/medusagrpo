#!/usr/bin/env bash
# Medusa ShareGPT pretraining launcher.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
requested_pretrain_batch="${PRETRAIN_BATCH_SIZE:-}"
requested_pretrain_accum="${PRETRAIN_ACCUMULATION_STEPS:-}"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
PRETRAIN_BATCH_SIZE="${requested_pretrain_batch:-${DRAFT_PRETRAIN_BATCH_SIZE:-4}}"
PRETRAIN_ACCUMULATION_STEPS="${requested_pretrain_accum:-${DRAFT_PRETRAIN_ACCUMULATION_STEPS:-8}}"
: "${MODEL:?MODEL is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"

case "${PRETRAIN_DATASET,,}" in
  sharegpt)
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
    ;;
  gsm8k)
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}"
    ;;
  simplelr|simplelr_abel|simplelr_abel_level3to5)
    PRETRAIN_DATASET="simplelr"
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}"
    ;;
  dapo|dapo-math|dapo_math)
    PRETRAIN_DATASET="dapo"
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}"
    ;;
  *)
    [[ -n "$PRETRAIN_DATASET_PATH" ]] || { echo "ERROR: unknown PRETRAIN_DATASET; set PRETRAIN_DATASET_PATH" >&2; exit 2; }
    ;;
esac

MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
timestamp="$(date -u +%Y%m%dT%H%M%S)"
short_uuid="$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${PRETRAIN_DATASET}__medusa-pretrain__seed${TRAIN_SUBSET_SEED}__${timestamp}__${short_uuid}}"
RUN_DIR="${RUN_DIR:-$MODEL_OUTPUT_ROOT/$RUN_NAME}"

PRETRAIN_RESUME="${RESUME:-}"
if [[ "${RESUME:-}" == "auto" && -e "$MODEL_OUTPUT_ROOT/active_run" ]]; then
  active="$(readlink -f "$MODEL_OUTPUT_ROOT/active_run")"
  if [[ ! -f "$active/checkpoints/pretrain_complete.json" ]]; then
    RUN_DIR="$active"
    RUN_NAME="$(basename "$RUN_DIR")"
  fi
elif [[ -n "${RESUME:-}" && "${RESUME:-}" != "auto" ]]; then
  resume_path="$(readlink -f "$RESUME")"
  if [[ -d "$resume_path/checkpoints" ]]; then
    RUN_DIR="$resume_path"
    RUN_NAME="$(basename "$RUN_DIR")"
    PRETRAIN_RESUME="$RUN_DIR/checkpoints/$RUN_NAME-latest/training_state.pt"
  else
    [[ -d "$resume_path" ]] && resume_path="$resume_path/training_state.pt"
    PRETRAIN_RESUME="$resume_path"
    RUN_DIR="$(dirname "$(dirname "$(dirname "$resume_path")")")"
    RUN_NAME="$(basename "$RUN_DIR")"
  fi
fi

TRAIN_DATA_PATH="$PRETRAIN_DATASET_PATH"
if [[ "${PRETRAIN_DATASET_PATH##*.}" != "json" || "$PRETRAIN_DATASET" != "sharegpt" ]]; then
  TRAIN_DATA_PATH="$RUN_DIR/data/sharegpt.json"
fi
cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE"
  "$PROJECT_DIR/train_draft.py" --model_dir "$MODEL" --dataset_dir "$TRAIN_DATA_PATH"
  --model_type "$MODEL_TYPE" --version_name "$RUN_NAME" --num_epochs "$PRETRAIN_EPOCHS"
  --batch_size "$PRETRAIN_BATCH_SIZE" --accumulation_steps "$PRETRAIN_ACCUMULATION_STEPS"
  --lr "$PRETRAIN_LR" --max_length "$PRETRAIN_MAX_LENGTH" --max_samples "$PRETRAIN_MAX_SAMPLES"
  --num_workers "$PRETRAIN_DATALOADER_WORKERS" --save_interval "$PRETRAIN_SAVE_INTERVAL"
  --seed "$TRAIN_SUBSET_SEED" --dtype "$MODEL_DTYPE" --attn_implementation "$ATTENTION_IMPLEMENTATION"
  --saved_model_dir "$RUN_DIR/checkpoints" --log_dir "$RUN_DIR/logs"
  --model_output_root "$MODEL_OUTPUT_ROOT" --resume "$PRETRAIN_RESUME" --load_lora_path "$TARGET_ADAPTER")
if (($#)); then cmd+=("$@"); fi
printf 'Run dir: %s\nModel: %s\nDataset: %s\n' "$RUN_DIR" "$MODEL" "$PRETRAIN_DATASET_PATH"
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-false}" == true ]];then return 0 2>/dev/null || exit 0;fi
[[ -f "$MODEL/config.json" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -f "$PRETRAIN_DATASET_PATH" ]] || { echo "Missing dataset: $PRETRAIN_DATASET_PATH" >&2; exit 2; }
if [[ -n "$TARGET_ADAPTER" ]];then
  [[ -f "$TARGET_ADAPTER/adapter_config.json" ]] || { echo "Missing initial target LoRA: $TARGET_ADAPTER" >&2; exit 2; }
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --requirements "$PROJECT_DIR/requirements.txt" --require-cuda
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$MODEL_OUTPUT_ROOT"
if [[ "$TRAIN_DATA_PATH" != "$PRETRAIN_DATASET_PATH" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_local_pretrain_data.py" \
    --input "$PRETRAIN_DATASET_PATH" --output "$TRAIN_DATA_PATH" --max-samples "$PRETRAIN_MAX_SAMPLES"
fi
ln -sfn "$RUN_DIR" "$MODEL_OUTPUT_ROOT/active_run"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/write_run_metadata.py" --run-dir "$RUN_DIR" --kind pretrain \
  --item "model=$MODEL" --item "dataset=$PRETRAIN_DATASET_PATH" --item "epochs=$PRETRAIN_EPOCHS" \
  --item "draft_architecture=MedusaParallel3" --item "run_name=$RUN_NAME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}" 2>&1 | tee -a "$RUN_DIR/logs/console.log"
ln -sfn "$RUN_DIR" "$MODEL_OUTPUT_ROOT/latest_run"
