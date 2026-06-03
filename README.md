# Image SFT Data Generation

This repo contains helper scripts that send images to an OpenAI-compatible
vision chat API and produce LLaMA-Factory SFT datasets.

## Project Layout

Each data project is self-contained:

```text
data/
  screenshot_summary/
    raw/
    processed/
    failed/
    prompt.txt
  chat_summary/
    raw/
    resized/
    processed/
    failed/
    prompt.txt
```

Current projects:

```text
data/screenshot_summary  # uses data/screenshot_summary/prompt.txt
data/chat_summary        # uses data/chat_summary/prompt.txt
```

## Usage

```bash
export OPENAI_BASE_URL="http://123.60.91.241:9003/v1"
export OPENAI_MODEL="Qwen3.5-35B-A3B"

python3 scripts/generate_image_sft.py --workers 4
```

`OPENAI_API_KEY` is optional. If it is empty, the request is sent without an
`Authorization` header.

By default, the script uses:

```text
--project-dir data/screenshot_summary
```

That means it reads images from `data/screenshot_summary/raw`, writes per-image
JSON files to `data/screenshot_summary/processed`, and reads the prompt from
`data/screenshot_summary/prompt.txt`.

Prompt priority is:

```text
--prompt > IMAGE_SFT_PROMPT > --prompt-file
```

Run the chat project:

```bash
python3 scripts/generate_image_sft.py \
  --project-dir data/chat_summary \
  --workers 8
```

Useful options:

```bash
python3 scripts/generate_image_sft.py \
  --project-dir data/screenshot_summary \
  --workers 8 \
  --aggregate-every 20 \
  --model Qwen3.5-35B-A3B \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --no-enable-thinking
```

Those sampling defaults are already built into the script.

Existing per-image JSON files are skipped by default. Use `--force` to regenerate
them.

Aggregate dataset files are rewritten every 20 completed images by default. Use
`--aggregate-every 0` to write aggregates only at the end, or a smaller value
such as `--aggregate-every 1` for the most conservative checkpointing.

Before writing SFT records, the script removes any text up to and including the
last `</think>` marker in the model response. The remaining answer must be valid
JSON. If JSON parsing fails, that image is marked as failed and excluded from the
aggregate LLaMA-Factory datasets. Failure markers are written under `data/failed`
inside each project by default.

## Outputs

For each image, the script writes:

```text
data/<project>/processed/<image_stem>.json
```

It also writes two aggregate datasets:

```text
data/<project>/processed/llamafactory_sft.json
data/<project>/processed/llamafactory_openai_sft.json
```

`llamafactory_sft.json` uses LLaMA-Factory's Alpaca multimodal format:

```json
{
  "instruction": "<image>\n请仔细观察这张图片...",
  "input": "",
  "output": "模型回答",
  "images": ["raw/example.jpg"]
}
```

If LLaMA-Factory uses its default `dataset_dir=./data`, copy or keep this repo's
`data/raw` and `data/processed` under that dataset directory, then register the
aggregate file in LLaMA-Factory's `data/dataset_info.json` like this:

```json
{
  "image_sft": {
    "file_name": "processed/llamafactory_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "images": "images"
    }
  }
}
```

The script also writes `processed/llamafactory_openai_sft.json`. To use that
OpenAI-message-style file, register it as sharegpt with OpenAI tags:

```json
{
  "image_sft_openai": {
    "file_name": "processed/llamafactory_openai_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

LLaMA-Factory requires the number of `<image>` tags in the prompt text to match
the number of paths in `images`.

## LLaMA-Factory Training

Prepare conda, PyTorch, and LLaMA-Factory separately. These scripts do not
install dependencies.

Write the training YAML only:

```bash
MODEL_PATH=/path/to/your/qwen-vl-small \
bash scripts/write_llamafactory_train_yaml.sh
```

The default YAML path is:

```text
configs/qwen35_vl_freeze_sft.yaml
```

Start training:

```bash
MODEL_PATH=/path/to/your/qwen-vl-small \
LLAMAFACTORY_DIR=/path/to/LLaMA-Factory \
bash scripts/train_llamafactory_qwen_vl.sh
```

The train script copies this repo's raw images and
`data/processed/llamafactory_sft.json` into LLaMA-Factory, updates
`data/dataset_info.json`, writes the freeze SFT yaml, and starts training.

The generated YAML uses LLaMA-Factory freeze fine-tuning:

```yaml
stage: sft
do_train: true
finetuning_type: freeze

model_name_or_path: /path/to/your/qwen-vl-small
template: qwen2_vl
trust_remote_code: true
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_trainable_layers: 8
freeze_trainable_modules: all

dataset_dir: data
dataset: image_sft
cutoff_len: 4096
max_samples: null
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: saves/qwen35-vl-small/freeze/image_sft
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
```

Freeze controls can be changed with environment variables:

```bash
FREEZE_TRAINABLE_LAYERS=12 \
FREEZE_VISION_TOWER=true \
FREEZE_MULTI_MODAL_PROJECTOR=false \
MODEL_PATH=/path/to/your/qwen-vl-small \
bash scripts/write_llamafactory_train_yaml.sh
```

In LLaMA-Factory, positive `freeze_trainable_layers` means the last N LLM layers
are trainable. Earlier LLM layers are frozen. `freeze_vision_tower: true` freezes
the vision module.

If the target is an official Qwen2.5-VL small model, use a real checkpoint such
as `Qwen/Qwen2.5-VL-3B-Instruct` and keep `template: qwen2_vl`. If your target
is an internal Qwen3.5 vision-language model, keep its local checkpoint path and
set `template` to the closest compatible template used by that checkpoint. A
pure text-only Qwen model cannot train on this image dataset without converting
the data into text-only samples first.
