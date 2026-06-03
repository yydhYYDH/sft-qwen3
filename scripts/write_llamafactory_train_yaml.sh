#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-qwen2_vl}"
DATASET_NAME="${DATASET_NAME:-image_sft}"
TRAIN_YAML="${TRAIN_YAML:-$PROJECT_DIR/configs/qwen35_vl_freeze_sft.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/qwen35-vl-small/freeze/image_sft}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
CUTOFF_LEN="${CUTOFF_LEN:-4096}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
BF16="${BF16:-true}"

FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
FREEZE_MULTI_MODAL_PROJECTOR="${FREEZE_MULTI_MODAL_PROJECTOR:-false}"
FREEZE_TRAINABLE_LAYERS="${FREEZE_TRAINABLE_LAYERS:-8}"
FREEZE_TRAINABLE_MODULES="${FREEZE_TRAINABLE_MODULES:-all}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/qwen-vl-small bash scripts/write_llamafactory_train_yaml.sh

This writes a LLaMA-Factory freeze-SFT YAML only. It does not install anything.

Freeze behavior:
  freeze_vision_tower=true
  freeze_trainable_layers=8

In LLaMA-Factory, positive freeze_trainable_layers means the last N LLM layers
are trainable, so earlier LLM layers are frozen.
EOF
  exit 0
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

mkdir -p "$(dirname "$TRAIN_YAML")"

cat > "$TRAIN_YAML" <<EOF
stage: sft
do_train: true
finetuning_type: freeze

model_name_or_path: $MODEL_PATH
template: $MODEL_TEMPLATE
trust_remote_code: true
freeze_vision_tower: $FREEZE_VISION_TOWER
freeze_multi_modal_projector: $FREEZE_MULTI_MODAL_PROJECTOR
freeze_trainable_layers: $FREEZE_TRAINABLE_LAYERS
freeze_trainable_modules: $FREEZE_TRAINABLE_MODULES

dataset_dir: data
dataset: $DATASET_NAME
cutoff_len: $CUTOFF_LEN
max_samples: null
overwrite_cache: true
preprocessing_num_workers: $PREPROCESSING_NUM_WORKERS

output_dir: $OUTPUT_DIR
logging_steps: $LOGGING_STEPS
save_steps: $SAVE_STEPS
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
learning_rate: $LEARNING_RATE
num_train_epochs: $NUM_TRAIN_EPOCHS
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: $BF16
EOF

echo "$TRAIN_YAML"
