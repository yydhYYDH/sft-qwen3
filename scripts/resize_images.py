#!/usr/bin/env python3
"""Resize images to one quarter of their original width and height."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize images under a directory.")
    parser.add_argument(
        "--input-dir",
        default="data/chat_summary/raw",
        help="Directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/chat_summary/resized",
        help="Directory for resized images.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Scale factor for both width and height.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_one(image_path: Path, input_dir: Path, output_dir: Path, scale: float, force: bool) -> str:
    relative = image_path.relative_to(input_dir)
    output_path = output_dir / relative
    if output_path.exists() and not force:
        return f"[skipped] {image_path} -> {output_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        save_kwargs = {}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0, "optimize": True})
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
        resized.save(output_path, **save_kwargs)

    return f"[created] {image_path} {width}x{height} -> {output_path} {new_size[0]}x{new_size[1]}"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.scale <= 0:
        raise SystemExit("--scale must be greater than 0")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    created = 0
    skipped = 0
    for image_path in images:
        message = resize_one(image_path, input_dir, output_dir, args.scale, args.force)
        if message.startswith("[created]"):
            created += 1
        elif message.startswith("[skipped]"):
            skipped += 1
        print(message)

    print(f"done total={len(images)} created={created} skipped={skipped} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
