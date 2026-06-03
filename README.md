# Image SFT Data Generation

This repo contains a helper script that sends every image under `data/raw` to an
OpenAI-compatible vision chat API and writes one JSON file per image under
`data/processed`.

## Usage

```bash
export OPENAI_BASE_URL="http://123.60.91.241:9003/v1"
export OPENAI_MODEL="Qwen3.5-35B-A3B"

python3 scripts/generate_image_sft.py --workers 4
```

`OPENAI_API_KEY` is optional. If it is empty, the request is sent without an
`Authorization` header.

The default prompt is read from:

```text
prompts/summary_prompt.txt
```

Prompt priority is:

```text
--prompt > IMAGE_SFT_PROMPT > --prompt-file
```

Useful options:

```bash
python3 scripts/generate_image_sft.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --dataset-dir data \
  --prompt-file prompts/summary_prompt.txt \
  --workers 8 \
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

Before writing SFT records, the script removes any text up to and including the
last `</think>` marker in the model response. The remaining answer must be valid
JSON. If JSON parsing fails, that image is marked as failed and excluded from the
aggregate LLaMA-Factory datasets. Failure markers are written under `data/failed`
by default.

## Outputs

For each image, the script writes:

```text
data/processed/<image_stem>.json
```

It also writes two aggregate datasets:

```text
data/processed/llamafactory_sft.json
data/processed/llamafactory_openai_sft.json
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

Create a fresh conda environment:

```bash
conda create -n llamafactory-qwen-sft python=3.10 -y
conda activate llamafactory-qwen-sft
```

Install PyTorch first. Pick the command that matches your CUDA version:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install LLaMA-Factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Put this repo's generated data under LLaMA-Factory's default dataset directory:

```bash
mkdir -p data/raw data/processed
cp -r /mnt/e/WAIC/sft/data/raw/. data/raw/
cp /mnt/e/WAIC/sft/data/processed/llamafactory_sft.json data/processed/
```

Add the dataset entry to `data/dataset_info.json`:

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

Create `examples/train_lora/qwen35_vl_lora_sft.yaml`:

```yaml
stage: sft
do_train: true
finetuning_type: lora

model_name_or_path: /path/to/your/qwen3.5-vl-small
template: qwen2_vl
trust_remote_code: true

dataset_dir: data
dataset: image_sft
cutoff_len: 4096
max_samples: null
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: saves/qwen35-vl-small/lora/image_sft
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

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
```

Start training:

```bash
llamafactory-cli train examples/train_lora/qwen35_vl_lora_sft.yaml
```

If the target is an official Qwen2.5-VL small model, use a real checkpoint such
as `Qwen/Qwen2.5-VL-3B-Instruct` and keep `template: qwen2_vl`. If your target
is an internal Qwen3.5 vision-language model, keep its local checkpoint path and
set `template` to the closest compatible template used by that checkpoint. A
pure text-only Qwen model cannot train on this image dataset without converting
the data into text-only samples first.
