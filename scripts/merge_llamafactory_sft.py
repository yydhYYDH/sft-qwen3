#!/usr/bin/env python3
"""Merge project-level LLaMA-Factory SFT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PROJECTS = ["data/screenshot_summary", "data/chat_summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LLaMA-Factory SFT datasets.")
    parser.add_argument(
        "--projects",
        nargs="+",
        default=DEFAULT_PROJECTS,
        help="Project directories containing processed/llamafactory_sft.json.",
    )
    parser.add_argument(
        "--input-name",
        default="processed/llamafactory_sft.json",
        help="Dataset file path relative to each project directory.",
    )
    parser.add_argument(
        "--output",
        default="data/merged/processed/llamafactory_sft.json",
        help="Merged dataset output path.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data",
        help="Dataset root used by LLaMA-Factory. Image paths are written relative to it.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image referenced by a row does not exist under dataset root.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip projects whose input dataset file does not exist.",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Dataset must be a JSON array: {path}")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"Row {index} is not an object in {path}")
        rows.append(item)
    return rows


def rewrite_image_path(image_path: str, project_dir: Path, dataset_root: Path) -> str:
    absolute_or_project_path = Path(image_path)
    if absolute_or_project_path.is_absolute():
        full_path = absolute_or_project_path
    else:
        full_path = project_dir / absolute_or_project_path

    try:
        return full_path.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Image path is outside dataset root: image={image_path} project={project_dir}"
        ) from exc


def merge_project(
    project_dir: Path,
    input_name: str,
    dataset_root: Path,
    strict: bool,
    skip_missing: bool,
) -> list[dict[str, Any]]:
    dataset_path = project_dir / input_name
    if not dataset_path.exists():
        if skip_missing:
            print(f"[missing] {dataset_path}")
            return []
        raise RuntimeError(f"Dataset file does not exist: {dataset_path}")

    rows = load_json_array(dataset_path)
    merged_rows = []
    for index, row in enumerate(rows):
        if "images" not in row:
            raise RuntimeError(f"Row {index} missing images: {dataset_path}")

        merged_row = dict(row)
        rewritten_images = [
            rewrite_image_path(str(image_path), project_dir, dataset_root)
            for image_path in row.get("images", [])
        ]
        if strict:
            for rewritten in rewritten_images:
                image_file = dataset_root / rewritten
                if not image_file.exists():
                    raise RuntimeError(f"Missing image: {image_file}")

        merged_row["images"] = rewritten_images
        merged_row["source_project"] = project_dir.name
        merged_rows.append(merged_row)
    return merged_rows


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output = Path(args.output)

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    merged_rows: list[dict[str, Any]] = []
    for project in args.projects:
        project_dir = Path(project)
        rows = merge_project(
            project_dir,
            args.input_name,
            dataset_root,
            args.strict,
            args.skip_missing,
        )
        merged_rows.extend(rows)
        print(f"[merged] {project_dir}: {len(rows)} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output)
    print(f"done total={len(merged_rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
