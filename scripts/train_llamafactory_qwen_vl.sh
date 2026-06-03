#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$PROJECT_DIR/LLaMA-Factory}"
SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-$PROJECT_DIR/data/screenshot_summary}"
DATASET_FILE="${DATASET_FILE:-$SFT_PROJECT_DIR/processed/llamafactory_sft.json}"
RAW_DIR="${RAW_DIR:-$SFT_PROJECT_DIR/raw}"
TRAIN_YAML="${TRAIN_YAML:-$PROJECT_DIR/configs/qwen35_vl_freeze_sft.yaml}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  SFT_PROJECT_DIR=/path/to/project \
  MODEL_PATH=/path/to/qwen-vl-small LLAMAFACTORY_DIR=/path/to/LLaMA-Factory \
    bash scripts/train_llamafactory_qwen_vl.sh

This script does not install dependencies. Prepare conda/PyTorch/LLaMA-Factory
before running it.
EOF
  exit 0
fi

if [[ ! -d "$LLAMAFACTORY_DIR" ]]; then
  echo "LLaMA-Factory directory does not exist: $LLAMAFACTORY_DIR" >&2
  exit 1
fi

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "Dataset file does not exist: $DATASET_FILE" >&2
  echo "Run scripts/generate_image_sft.py first." >&2
  exit 1
fi

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Raw image directory does not exist: $RAW_DIR" >&2
  exit 1
fi

bash "$PROJECT_DIR/scripts/write_llamafactory_train_yaml.sh" >/dev/null

cd "$LLAMAFACTORY_DIR"
mkdir -p data/raw data/processed
cp -r "$RAW_DIR"/. data/raw/
cp "$DATASET_FILE" data/processed/llamafactory_sft.json

python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/dataset_info.json")
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
data["image_sft"] = {
    "file_name": "processed/llamafactory_sft.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "images": "images",
    },
}
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "Training config: $TRAIN_YAML"
llamafactory-cli train "$TRAIN_YAML"
