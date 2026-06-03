#!/usr/bin/env python3
"""Generate image SFT records with an OpenAI-compatible vision model."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DEFAULT_PROMPT = (
    "请仔细观察这张图片，生成适合用于训练多模态小模型的高质量中文回答。"
    "请包含：1. 图片整体内容；2. 关键文字或界面元素；3. 用户可能正在进行的操作或场景；"
    "4. 对图片有帮助的结构化细节。不要编造看不见的信息。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible vision API for every image in data/raw."
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("PROJECT_DIR", "data/screenshot_summary"),
        help="Project directory containing raw, processed, failed, and prompt.txt.",
    )
    parser.add_argument("--raw-dir", default=None, help="Directory containing images.")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory for per-image JSON outputs.",
    )
    parser.add_argument(
        "--failed-dir",
        default=None,
        help="Directory for per-image failure marker JSON outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Aggregated LLaMA-Factory Alpaca-format JSON output.",
    )
    parser.add_argument(
        "--openai-output",
        default=None,
        help="Aggregated OpenAI-message-format JSON output.",
    )
    parser.add_argument(
        "--aggregate-every",
        type=int,
        default=int(os.environ.get("AGGREGATE_EVERY", "20")),
        help="Rewrite aggregate dataset files every N completed images. Use 0 to only write at the end.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR"),
        help="Dataset root used by LLaMA-Factory. Image paths are stored relative to it.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. Takes precedence over IMAGE_SFT_PROMPT and --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        default=os.environ.get("IMAGE_SFT_PROMPT_FILE"),
        help="File containing the prompt text. Used when --prompt and IMAGE_SFT_PROMPT are unset.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen3.5-35B-A3B"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://123.60.91.241:9003/v1"),
        help="OpenAI-compatible base URL, for example https://api.openai.com/v1.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("TIMEOUT", "120")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.7")),
    )
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.8")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "20")))
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("MIN_P", "0.0")))
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=float(os.environ.get("PRESENCE_PENALTY", "1.5")),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.environ.get("REPETITION_PENALTY", "1.0")),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ENABLE_THINKING", "false").lower() in {"1", "true", "yes"},
        help="Pass chat_template_kwargs.enable_thinking to compatible servers.",
    )
    parser.add_argument("--retries", type=int, default=int(os.environ.get("RETRIES", "3")))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-image JSON even when it already exists.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir)
    if args.raw_dir is None:
        args.raw_dir = str(project_dir / "raw")
    if args.processed_dir is None:
        args.processed_dir = str(project_dir / "processed")
    if args.failed_dir is None:
        args.failed_dir = str(project_dir / "failed")
    if args.output is None:
        args.output = str(Path(args.processed_dir) / "llamafactory_sft.json")
    if args.openai_output is None:
        args.openai_output = str(Path(args.processed_dir) / "llamafactory_openai_sft.json")
    if args.dataset_dir is None:
        args.dataset_dir = str(project_dir)
    if args.prompt_file is None:
        args.prompt_file = str(project_dir / "prompt.txt")


def resolve_prompt(args: argparse.Namespace) -> str:
    prompt = args.prompt if args.prompt is not None else os.environ.get("IMAGE_SFT_PROMPT")
    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            raise SystemExit("Prompt is empty. Provide --prompt or a non-empty IMAGE_SFT_PROMPT.")
        return prompt

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise SystemExit(
            f"Prompt file does not exist: {prompt_file}. "
            "Provide --prompt or create the prompt file."
        )

    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit(f"Prompt file is empty: {prompt_file}")
    return prompt


def find_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_path_for(image_path: Path, raw_dir: Path, processed_dir: Path) -> Path:
    relative = image_path.relative_to(raw_dir)
    stem = "__".join(relative.with_suffix("").parts)
    return processed_dir / f"{stem}.json"


def failure_path_for(image_path: Path, raw_dir: Path, failed_dir: Path) -> Path:
    relative = image_path.relative_to(raw_dir)
    stem = "__".join(relative.with_suffix("").parts)
    return failed_dir / f"{stem}.json"


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(args: argparse.Namespace, image_path: Path) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        },
    }


def post_chat_completion(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def extract_answer(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response shape: {response}") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part.strip() for part in parts if part).strip()
    return str(content).strip()


def strip_thinking_text(answer: str) -> str:
    think_end = answer.rfind("</think>")
    if think_end == -1:
        return answer.strip()
    return answer[think_end + len("</think>") :].strip()


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    first_line_end = stripped.find("\n")
    if first_line_end == -1:
        return stripped

    opening = stripped[:first_line_end].strip()
    if not opening.startswith("```"):
        return stripped

    body = stripped[first_line_end + 1 :].strip()
    if not body.endswith("```"):
        return stripped

    return body[:-3].strip()


def normalize_json_answer(answer: str) -> str:
    stripped = strip_markdown_code_fence(strip_thinking_text(answer))
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        preview = stripped[:200].replace("\n", "\\n")
        raise RuntimeError(
            "Answer is not valid JSON after </think> and Markdown fence stripping: "
            f"{preview}"
        ) from exc
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def call_with_retries(args: argparse.Namespace, image_path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            payload = build_payload(args, image_path)
            response = post_chat_completion(args, payload)
            return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            sleep_seconds = min(30.0, (2**attempt) + random.random())
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed after {args.retries + 1} attempts: {last_error}")


def build_record(
    *,
    args: argparse.Namespace,
    image_path: Path,
    raw_dir: Path,
    dataset_dir: Path,
    answer: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    try:
        image_rel_path = image_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        image_rel_path = image_path.as_posix()
    return {
        "source_image": image_rel_path,
        "prompt": args.prompt,
        "answer": answer,
        "llamafactory_alpaca": {
            "instruction": "<image>\n" + args.prompt,
            "input": "",
            "output": answer,
            "images": [image_rel_path],
        },
        "llamafactory_openai": {
            "messages": [
                {"role": "user", "content": "<image>\n" + args.prompt},
                {"role": "assistant", "content": answer},
            ],
            "images": [image_rel_path],
        },
        "metadata": {
            "model": args.model,
            "base_url": args.base_url.rstrip("/"),
            "raw_dir": raw_dir.as_posix(),
            "dataset_dir": dataset_dir.as_posix(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "raw_response": response,
    }


def refresh_record_answers(record: dict[str, Any]) -> bool:
    original_answer = str(record.get("answer", ""))
    normalized_answer = normalize_json_answer(original_answer)
    changed = normalized_answer != original_answer

    record["answer"] = normalized_answer
    if "llamafactory_alpaca" in record:
        old_output = record["llamafactory_alpaca"].get("output")
        record["llamafactory_alpaca"]["output"] = normalized_answer
        changed = changed or old_output != normalized_answer
    if "llamafactory_openai" in record:
        for message in record["llamafactory_openai"].get("messages", []):
            if message.get("role") == "assistant":
                old_content = message.get("content")
                message["content"] = normalized_answer
                changed = changed or old_content != normalized_answer
                break
    return changed


def load_record_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = refresh_record_answers(data)
    if changed:
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    return data


def process_one(
    args: argparse.Namespace,
    raw_dir: Path,
    dataset_dir: Path,
    processed_dir: Path,
    image_path: Path,
) -> tuple[str, Path]:
    json_path = output_path_for(image_path, raw_dir, processed_dir)
    if json_path.exists() and not args.force:
        load_record_file(json_path)
        return "skipped", json_path

    response = call_with_retries(args, image_path)
    answer = normalize_json_answer(extract_answer(response))
    record = build_record(
        args=args,
        image_path=image_path,
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
        answer=answer,
        response=response,
    )

    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(json_path)
    return "created", json_path


def load_records(processed_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(processed_dir.glob("*.json")):
        if path.name in {"llamafactory_sft.json", "llamafactory_openai_sft.json"}:
            continue
        try:
            data = load_record_file(path)
        except (json.JSONDecodeError, RuntimeError) as exc:
            print(f"[invalid] {path}: {exc}")
            continue
        if "llamafactory_alpaca" in data:
            records.append(data)
    return records


def write_aggregates(records: list[dict[str, Any]], output: Path, openai_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    alpaca_rows = [record["llamafactory_alpaca"] for record in records]
    openai_rows = [record["llamafactory_openai"] for record in records]
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    openai_output_tmp = openai_output.with_suffix(openai_output.suffix + ".tmp")
    output_tmp.write_text(
        json.dumps(alpaca_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    openai_output_tmp.write_text(
        json.dumps(openai_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_tmp.replace(output)
    openai_output_tmp.replace(openai_output)


def refresh_aggregates(processed_dir: Path, output: Path, openai_output: Path) -> int:
    records = load_records(processed_dir)
    if records:
        write_aggregates(records, output, openai_output)
    return len(records)


def write_failure_marker(image_path: Path, raw_dir: Path, failed_dir: Path, error: Exception) -> Path:
    failed_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_path_for(image_path, raw_dir, failed_dir)
    payload = {
        "source_image": image_path.as_posix(),
        "error": str(error),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp_path = failure_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(failure_path)
    return failure_path


def print_progress(
    *,
    done: int,
    total: int,
    created: int,
    skipped: int,
    failed: int,
    current: Path | None = None,
    final: bool = False,
) -> None:
    width = shutil.get_terminal_size((100, 20)).columns
    percent = (done / total * 100) if total else 100.0
    bar_width = 28
    filled = int(bar_width * done / total) if total else bar_width
    bar = "#" * filled + "-" * (bar_width - filled)
    suffix = f" current={current.name}" if current else ""
    line = (
        f"[{bar}] {done}/{total} {percent:5.1f}% "
        f"created={created} skipped={skipped} failed={failed}{suffix}"
    )
    if len(line) > width:
        line = line[: max(0, width - 3)] + "..."
    print("\r" + line.ljust(width), end="\n" if final else "", flush=True)


def maybe_refresh_aggregates(
    *,
    done: int,
    last_aggregate_done: int,
    aggregate_every: int,
    processed_dir: Path,
    output: Path,
    openai_output: Path,
) -> tuple[int, int | None]:
    if aggregate_every <= 0 or done - last_aggregate_done < aggregate_every:
        return last_aggregate_done, None
    valid_records = refresh_aggregates(processed_dir, output, openai_output)
    return done, valid_records


def main() -> int:
    args = parse_args()
    resolve_paths(args)
    args.prompt = resolve_prompt(args)
    raw_dir = Path(args.raw_dir)
    dataset_dir = Path(args.dataset_dir)
    processed_dir = Path(args.processed_dir)
    failed_dir = Path(args.failed_dir)
    output = Path(args.output)
    openai_output = Path(args.openai_output)

    if not raw_dir.exists():
        raise SystemExit(f"Raw directory does not exist: {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    images = find_images(raw_dir)
    if not images:
        raise SystemExit(f"No images found in {raw_dir}")

    print(
        f"found={len(images)} workers={args.workers} model={args.model} "
        f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"enable_thinking={args.enable_thinking}"
    )
    created = 0
    skipped = 0
    failed = 0
    done = 0
    last_aggregate_done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, args, raw_dir, dataset_dir, processed_dir, image_path): image_path
            for image_path in images
        }
        print_progress(
            done=done,
            total=len(images),
            created=created,
            skipped=skipped,
            failed=failed,
        )
        for future in concurrent.futures.as_completed(futures):
            image_path = futures[future]
            try:
                status, json_path = future.result()
            except Exception as exc:
                failed += 1
                done += 1
                failure_path = write_failure_marker(image_path, raw_dir, failed_dir, exc)
                print()
                print(f"[failed] {image_path}: {exc} -> {failure_path}")
                print_progress(
                    done=done,
                    total=len(images),
                    created=created,
                    skipped=skipped,
                    failed=failed,
                    current=image_path,
                )
                last_aggregate_done, valid_records = maybe_refresh_aggregates(
                    done=done,
                    last_aggregate_done=last_aggregate_done,
                    aggregate_every=args.aggregate_every,
                    processed_dir=processed_dir,
                    output=output,
                    openai_output=openai_output,
                )
                if valid_records is not None:
                    print()
                    print(f"[aggregate] valid_records={valid_records} at_done={done}")
                    print_progress(
                        done=done,
                        total=len(images),
                        created=created,
                        skipped=skipped,
                        failed=failed,
                        current=image_path,
                    )
                continue

            if status == "created":
                created += 1
            elif status == "skipped":
                skipped += 1
            done += 1
            print_progress(
                done=done,
                total=len(images),
                created=created,
                skipped=skipped,
                failed=failed,
                current=image_path,
            )
            last_aggregate_done, valid_records = maybe_refresh_aggregates(
                done=done,
                last_aggregate_done=last_aggregate_done,
                aggregate_every=args.aggregate_every,
                processed_dir=processed_dir,
                output=output,
                openai_output=openai_output,
            )
            if valid_records is not None:
                print()
                print(f"[aggregate] valid_records={valid_records} at_done={done}")
                print_progress(
                    done=done,
                    total=len(images),
                    created=created,
                    skipped=skipped,
                    failed=failed,
                    current=image_path,
                )

    print_progress(
        done=done,
        total=len(images),
        created=created,
        skipped=skipped,
        failed=failed,
        final=True,
    )

    valid_records = refresh_aggregates(processed_dir, output, openai_output)
    if valid_records:
        aggregate_message = f"aggregate={output} openai_aggregate={openai_output}"
    else:
        aggregate_message = "aggregate=not_written_no_records"
    print(f"done created={created} skipped={skipped} failed={failed} {aggregate_message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
