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

Install LLaMA-Factory in a Python 3.12 conda environment:

```bash
conda create -n llamafactory-qwen-sft python=3.12 -y
conda activate llamafactory-qwen-sft

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

This repo provides an example YAML:

```text
configs/qwen35_vl_freeze_sft.yaml
```

Before training, copy the dataset project you want into LLaMA-Factory's
`data/` directory. For the screenshot project:

```bash
cd /path/to/LLaMA-Factory
mkdir -p data/screenshot_summary

cd /path/to/this-repo
cp -r data/screenshot_summary/raw /path/to/LLaMA-Factory/data/screenshot_summary/
cp -r data/screenshot_summary/processed /path/to/LLaMA-Factory/data/screenshot_summary/
```

Register the dataset in LLaMA-Factory's `data/dataset_info.json`:

```json
{
  "image_sft": {
    "file_name": "screenshot_summary/processed/llamafactory_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "images": "images"
    }
  }
}
```

Then edit `configs/qwen35_vl_freeze_sft.yaml`.

Fields you usually need to change:

- `model_name_or_path`: your local or Hugging Face model checkpoint.
- `template`: keep `qwen2_vl` for Qwen2.5-VL-style models; change only if your model requires another LLaMA-Factory template.
- `dataset_dir`: set to `data` if the dataset lives under LLaMA-Factory's `data/`.
- `dataset`: must match the key you added in `dataset_info.json`, for example `image_sft`.
- `output_dir`: where checkpoints and logs should be written.
- `per_device_train_batch_size` and `gradient_accumulation_steps`: adjust for GPU memory.
- `learning_rate`, `num_train_epochs`, `cutoff_len`: tune for your run.
- `bf16`: set `false` if your GPU does not support bf16.
- `freeze_trainable_layers`: positive N means only the last N LLM layers are trainable; earlier LLM layers are frozen.
- `freeze_vision_tower`: `true` freezes the vision module.
- `freeze_multi_modal_projector`: `false` keeps the projector trainable.

Run training directly with LLaMA-Factory:

```bash
cd /path/to/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
llamafactory-cli train /path/to/this-repo/configs/qwen35_vl_freeze_sft.yaml
```

Change `CUDA_VISIBLE_DEVICES` if you only want to use part of the GPUs, for
example `export CUDA_VISIBLE_DEVICES=0,1`.

If you use the merged dataset, register it as:

```json
{
  "merged_image_sft": {
    "file_name": "merged/processed/llamafactory_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "images": "images"
    }
  }
}
```

Then set this in the YAML:

```yaml
dataset: merged_image_sft
```

If the target is an official Qwen2.5-VL small model, use a real checkpoint such
as `Qwen/Qwen2.5-VL-3B-Instruct` and keep `template: qwen2_vl`. If your target
is an internal Qwen3.5 vision-language model, keep its local checkpoint path and
set `template` to the closest compatible template used by that checkpoint. A
pure text-only Qwen model cannot train on this image dataset without converting
the data into text-only samples first.
