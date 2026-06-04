#!/usr/bin/env python3
"""Run a local vision-language model on project images and save predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with a local VLM.")
    parser.add_argument(
        "--project-dir",
        default="data/test_summary",
        help="Project directory containing raw, prompt.txt, and predictions.",
    )
    parser.add_argument("--raw-dir", default=None, help="Directory containing images.")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction JSON files.")
    parser.add_argument("--prompt-file", default=None, help="Prompt file. Defaults to <project>/prompt.txt.")
    parser.add_argument("--model-path", required=True, help="Local or Hugging Face model path.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing prediction JSON files.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir)
    if args.raw_dir is None:
        args.raw_dir = str(project_dir / "raw")
    if args.output_dir is None:
        args.output_dir = str(project_dir / "predictions")
    if args.prompt_file is None:
        args.prompt_file = str(project_dir / "prompt.txt")


def find_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_path_for(image_path: Path, raw_dir: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(raw_dir)
    stem = "__".join(relative.with_suffix("").parts)
    return output_dir / f"{stem}.json"


def dtype_from_name(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def build_messages(prompt: str, image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_inputs(processor: Any, prompt: str, image_path: Path, model_device: torch.device) -> dict[str, Any]:
    messages = build_messages(prompt, image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    return {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_one(model: Any, processor: Any, prompt: str, image_path: Path, args: argparse.Namespace) -> str:
    model_device = next(model.parameters()).device
    inputs = build_inputs(processor, prompt, image_path, model_device)

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[:, input_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def write_prediction(
    output_path: Path,
    image_path: Path,
    prompt: str,
    answer: str,
    args: argparse.Namespace,
) -> None:
    payload = {
        "source_image": image_path.as_posix(),
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "model_path": args.model_path,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)


def main() -> int:
    args = parse_args()
    resolve_paths(args)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    prompt_file = Path(args.prompt_file)

    if not raw_dir.exists():
        raise SystemExit(f"Raw directory does not exist: {raw_dir}")
    if not prompt_file.exists():
        raise SystemExit(f"Prompt file does not exist: {prompt_file}")

    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit(f"Prompt file is empty: {prompt_file}")

    images = find_images(raw_dir)
    if not images:
        raise SystemExit(f"No images found in {raw_dir}")

    model, processor = load_model_and_processor(args)

    created = 0
    skipped = 0
    for index, image_path in enumerate(images, start=1):
        output_path = output_path_for(image_path, raw_dir, output_dir)
        if output_path.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{len(images)} skipped] {image_path} -> {output_path}")
            continue

        answer = generate_one(model, processor, prompt, image_path, args)
        write_prediction(output_path, image_path, prompt, answer, args)
        created += 1
        print(f"[{index}/{len(images)} created] {image_path} -> {output_path}")

    print(f"done total={len(images)} created={created} skipped={skipped} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
