# 图片 SFT 数据生成与 LLaMA-Factory 训练

这个仓库用于把图片发送给 OpenAI 兼容的视觉模型接口，生成可用于
LLaMA-Factory 微调的数据集。

## 项目结构

每个数据项目都是独立目录，包含自己的图片、输出和 prompt：

```text
data/
  screenshot_summary/
    raw/          # 原始图片
    processed/    # 单图 JSON 和汇总数据
    failed/       # 失败记录
    prompt.txt    # 该项目使用的 prompt
  chat_summary/
    raw/
    resized/
    processed/
    failed/
    prompt.txt
```

当前有两个项目：

```text
data/screenshot_summary  # 截图总结
data/chat_summary        # 聊天总结
```

`data/` 已加入 `.gitignore`，不会提交到 Git。

## 生成 SFT 数据

默认处理 `data/screenshot_summary`：

```bash
export OPENAI_BASE_URL="http://123.60.91.241:9003/v1"
export OPENAI_MODEL="Qwen3.5-35B-A3B"

python3 scripts/generate_image_sft.py --workers 4
```

`OPENAI_API_KEY` 可以为空。为空时，请求不会带 `Authorization` 头。

处理聊天项目：

```bash
python3 scripts/generate_image_sft.py \
  --project-dir data/chat_summary \
  --workers 8
```

常用参数：

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

prompt 优先级：

```text
--prompt > IMAGE_SFT_PROMPT > --prompt-file > data/<project>/prompt.txt
```

脚本会跳过已经存在的单图 JSON。需要重新生成时加：

```bash
--force
```

汇总文件默认每完成 20 张图片重写一次：

```bash
--aggregate-every 20
```

如果很怕中断，可以用：

```bash
--aggregate-every 1
```

如果只想最后写汇总：

```bash
--aggregate-every 0
```

## 输出文件

每张图片会生成一个单独 JSON：

```text
data/<project>/processed/<image_stem>.json
```

同时生成两个汇总文件：

```text
data/<project>/processed/llamafactory_sft.json
data/<project>/processed/llamafactory_openai_sft.json
```

`llamafactory_sft.json` 是 LLaMA-Factory Alpaca 多模态格式：

```json
{
  "instruction": "<image>\n你的 prompt",
  "input": "",
  "output": "模型回答",
  "images": ["raw/example.jpg"]
}
```

脚本会先截断模型输出中的思考过程：保留最后一个 `</think>` 后面的内容。
截断后的内容必须能被 `json.loads` 解析，否则该图片会标记失败，并从汇总数据中排除。
失败记录写入：

```text
data/<project>/failed/
```

## 缩小聊天图片

把 `data/chat_summary/raw` 下的图片按宽高各缩小到原来的 1/4：

```bash
python3 scripts/resize_images.py
```

默认输出到：

```text
data/chat_summary/resized
```

覆盖已有缩略图：

```bash
python3 scripts/resize_images.py --force
```

## 合并两个数据集

两个项目都处理完成后，合并：

```bash
python3 scripts/merge_llamafactory_sft.py --strict
```

如果某个项目还没处理完，可以跳过缺失项目：

```bash
python3 scripts/merge_llamafactory_sft.py --strict --skip-missing
```

默认合并：

```text
data/screenshot_summary/processed/llamafactory_sft.json
data/chat_summary/processed/llamafactory_sft.json
```

输出：

```text
data/merged/processed/llamafactory_sft.json
```

合并脚本会把图片路径从：

```json
"images": ["raw/example.jpg"]
```

改成：

```json
"images": ["screenshot_summary/raw/example.jpg"]
```

或者：

```json
"images": ["chat_summary/raw/example.jpg"]
```

这样训练时 `dataset_dir: data` 就能正确找到图片。

## 测试集与小模型预测

测试项目目录已经预留：

```text
data/test_summary/
  raw/          # 你把测试图片放这里
  processed/    # 大模型生成的测试集回答
  predictions/  # 小模型预测结果
  failed/
  prompt.txt
```

先把测试图片放进：

```text
data/test_summary/raw
```

用大模型生成测试图片对应的参考回答：

```bash
python3 scripts/generate_image_sft.py \
  --project-dir data/test_summary \
  --workers 4
```

这会生成：

```text
data/test_summary/processed/llamafactory_sft.json
```

用本地小模型生成预测回答：

```bash
python3 scripts/run_local_vlm_predictions.py \
  --project-dir data/test_summary \
  --model-path /path/to/your/small-vlm \
  --max-new-tokens 1024
```

预测结果会保存到：

```text
data/test_summary/predictions
```

如果要重新生成预测，加：

```bash
--force
```

## LLaMA-Factory 数据注册

如果只训练截图项目，把数据复制到 LLaMA-Factory：

```bash
cd /path/to/LLaMA-Factory
mkdir -p data/screenshot_summary

cd /path/to/this-repo
cp -r data/screenshot_summary/raw /path/to/LLaMA-Factory/data/screenshot_summary/
cp -r data/screenshot_summary/processed /path/to/LLaMA-Factory/data/screenshot_summary/
```

在 LLaMA-Factory 的 `data/dataset_info.json` 中注册：

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

如果使用合并数据集，注册：

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

LLaMA-Factory 要求文本中的 `<image>` 数量和 `images` 路径数量一致。

## 训练 YAML

先创建 Python 3.12 conda 环境，并安装 CUDA 12.1 版 PyTorch 和 LLaMA-Factory：

```bash
conda create -n llamafactory-qwen-sft python=3.12 -y
conda activate llamafactory-qwen-sft

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

示例文件：

```text
configs/qwen35_vl_freeze_sft.yaml
```

通常需要改这些字段：

- `model_name_or_path`：你的本地模型路径或 Hugging Face 模型名。
- `template`：Qwen2.5-VL 风格模型通常用 `qwen2_vl`。
- `dataset_dir`：如果数据在 LLaMA-Factory 的 `data/` 下，就写 `data`。
- `dataset`：必须和 `dataset_info.json` 里的 key 一致，比如 `image_sft` 或 `merged_image_sft`。
- `output_dir`：训练输出目录。
- `per_device_train_batch_size`：单卡 batch size，根据显存调。
- `gradient_accumulation_steps`：梯度累积步数。
- `learning_rate`：学习率。
- `num_train_epochs`：训练轮数。
- `cutoff_len`：上下文长度。
- `bf16`：显卡不支持 bf16 时改成 `false`。
- `freeze_trainable_layers`：正数 N 表示只训练最后 N 层 LLM，前面的 LLM 层冻结。
- `freeze_vision_tower`：`true` 表示冻结视觉模块。
- `freeze_multi_modal_projector`：`false` 表示多模态 projector 参与训练。

启动训练：

```bash
cd /path/to/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
llamafactory-cli train /path/to/this-repo/configs/qwen35_vl_freeze_sft.yaml
```

如果只想使用部分 GPU，修改 `CUDA_VISIBLE_DEVICES` 即可，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

注意：目标模型必须是视觉语言模型。纯文本 Qwen 模型不能直接训练包含 `images` 的数据集。
