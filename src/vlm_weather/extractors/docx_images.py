"""Extract images from DOCX files via document relationships (rels).

This method handles both embedded and external (HTTP) images,
and provides a utility to fix historical ``*-图0.*`` naming.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from docx import Document
from PIL import Image

from vlm_weather.common import SUPPORTED_IMAGE_EXTENSIONS, ensure_dir, file_stem, find_files

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30
DEFAULT_MIN_HEIGHT = 500


def save_images_from_docx(
    docx_path: str | Path,
    output_dir: str | Path,
    min_height: int = DEFAULT_MIN_HEIGHT,
    request_timeout: int = REQUEST_TIMEOUT,
) -> int:
    """Save all images from a DOCX file via rels.

    Returns:
        Number of images saved.
    """
    ensure_dir(output_dir)
    doc = Document(str(docx_path))
    stem = file_stem(docx_path)

    embedded_digits: list[int] = []
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype or rel.is_external:
            continue
        partname = rel.target_part.partname.split("/")[-1].split(".")[0]
        digits = "".join(ch for ch in partname if ch.isdigit())
        if digits:
            embedded_digits.append(int(digits))
    min_embedded = min(embedded_digits) if embedded_digits else 0

    saved = 0
    external_counter = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue

        if rel.is_external:
            external_counter += 1
            image_name = f"{stem}-图{external_counter}"
            image_path = Path(output_dir) / image_name
            try:
                response = requests.get(rel.target, timeout=request_timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error("Failed to download external image %s: %s", rel.target, exc)
                continue
            image_path.write_bytes(response.content)
            saved += 1
            continue

        blob = rel.target_part.blob
        try:
            image = Image.open(BytesIO(blob))
        except Exception as exc:
            logger.error("Cannot decode image: %s", exc)
            continue
        if image.height < min_height:
            continue

        partname = rel.target_part.partname.split("/")[-1]
        digits = "".join(ch for ch in partname if ch.isdigit())
        if not digits:
            continue
        new_index = int(digits) - min_embedded + 1
        new_basename = partname.replace(digits, str(new_index)).replace("image", "图")
        image_name = f"{stem}-{new_basename}"
        image_path = Path(output_dir) / image_name
        image_path.write_bytes(blob)
        saved += 1
    return saved


def rename_zero_suffix_images(folder: str | Path, dry_run: bool = False) -> int:
    """Rename ``xxx-图0.ext`` to ``xxx-图1.ext``.

    Returns:
        Number of files renamed.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    targets: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        targets.extend(folder.glob(f"*.{ext}"))
        targets.extend(folder.glob(f"*.{ext.upper()}"))

    if not targets:
        logger.info("No images found in %s", folder)
        return 0

    renamed = 0
    for old_path in targets:
        stem = old_path.stem
        if not stem.endswith("-图0"):
            continue
        new_stem = stem[:-3] + "-图1"
        new_path = old_path.with_name(new_stem + old_path.suffix)

        if new_path.exists():
            logger.warning("Skipping: target already exists %s", new_path)
            continue

        if dry_run:
            logger.info("[dry-run] %s -> %s", old_path.name, new_path.name)
        else:
            old_path.rename(new_path)
            logger.info("Renamed: %s -> %s", old_path.name, new_path.name)
        renamed += 1
    return renamed


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> None:
    """Batch process all DOCX files in *input_dir* and save images to *output_dir*."""
    images_dir = Path(output_dir) if output_dir else Path(input_dir) / "images"
    total = 0
    for docx_path in find_files(input_dir, [".docx"]):
        logger.info("--- %s", docx_path)
        total += save_images_from_docx(docx_path, images_dir, min_height=min_height)
    logger.info("Total images saved: %d to %s", total, images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DOCX images via relationships")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Batch extract images from DOCX files")
    extract.add_argument("--input_dir", required=True, help="Root directory with .docx files")
    extract.add_argument("--output_dir", default=None, help="Image output directory")
    extract.add_argument("--min_height", type=int, default=DEFAULT_MIN_HEIGHT,
                         help="Skip images shorter than this (default 500)")

    rename = sub.add_parser("rename-zero", help="Rename *-图0.ext to *-图1.ext")
    rename.add_argument("--folder", required=True, help="Image folder")
    rename.add_argument("--dry_run", action="store_true", help="Print without renaming")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "extract":
        process_directory(args.input_dir, args.output_dir, args.min_height)
    elif args.command == "rename-zero":
        rename_zero_suffix_images(args.folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
