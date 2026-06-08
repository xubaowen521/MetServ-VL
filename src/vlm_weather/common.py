"""Shared utilities for file discovery, format conversion, and path handling."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tif",
    "tiff",
)


def find_files(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Recursively collect files matching the given extensions.

    Args:
        directory: Root directory to search.
        extensions: File extensions to match (case-insensitive).
            Examples: ``[".doc", ".docx"]`` or ``["doc", "docx"]``.

    Returns:
        List of matched file paths as :class:`pathlib.Path` objects.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    normalized = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    )
    matched: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in normalized:
            matched.append(path)
    return matched


def rename_jpeg_to_jpg(folder: str | Path) -> int:
    """Rename all ``*.jpeg`` files in *folder* to ``*.jpg``.

    Returns:
        Number of files renamed.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    renamed = 0
    for old_path in folder.iterdir():
        if not old_path.is_file() or old_path.suffix.lower() != ".jpeg":
            continue
        new_path = old_path.with_suffix(".jpg")
        if new_path.exists():
            logger.warning("Skipping rename: target already exists: %s", new_path)
            continue
        old_path.rename(new_path)
        renamed += 1
        logger.info("Renamed: %s -> %s", old_path, new_path)
    return renamed


def file_stem(path: str | Path) -> str:
    """Return the file name without its final suffix.

    This keeps identifiers such as ``000-图1`` stable when the original file is
    ``000-图1.docx`` or ``000-图1.jpg``.
    """
    return Path(path).stem


def convert_doc_to_docx(doc_path: str | Path, docx_path: str | Path) -> bool:
    """Convert a ``.doc`` file to ``.docx`` using Microsoft Word COM (Windows only).

    Args:
        doc_path: Source ``.doc`` file path.
        docx_path: Destination ``.docx`` file path.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    try:
        import comtypes.client  # noqa: PLC0415
    except ImportError:
        logger.error("comtypes is not installed; cannot convert .doc -> .docx")
        return False

    doc_path = Path(doc_path)
    docx_path = Path(docx_path)

    if not doc_path.is_file():
        logger.warning("Source file does not exist: %s", doc_path)
        return False

    word = None
    try:
        word = comtypes.client.CreateObject("Word.Application")
        doc = word.Documents.Open(str(doc_path.resolve()))
        # FileFormat=16 corresponds to .docx
        doc.SaveAs(str(docx_path.resolve()), FileFormat=16)
        doc.Close()
        logger.info("Converted: %s -> %s", doc_path, docx_path)
        return True
    except Exception as exc:
        logger.error("Conversion failed for %s: %s", doc_path, exc)
        return False
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if it does not exist.

    Returns:
        The resolved :class:`pathlib.Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
