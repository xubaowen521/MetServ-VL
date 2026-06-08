"""SentenceTransformer-based embedding similarity evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-4B")


def calculate_similarity(
    data: list[dict[str, Any]],
    model: SentenceTransformer,
) -> dict[str, Any]:
    """Compute cosine similarity between ``human_answer`` and ``assistant_answer``.

    Args:
        data: List of evaluation entries.
        model: Loaded SentenceTransformer model.

    Returns:
        Dictionary with ``items`` (per-entry results) and ``average_similarity``.
    """
    results: list[dict[str, Any]] = []
    similarities: list[float] = []

    for item in data:
        human = item.get("human_answer", "")
        assistant = item.get("assistant_answer", "")
        idx = item.get("index")

        emb_human = model.encode([human])
        emb_assistant = model.encode([assistant])
        sim = float(model.similarity(emb_human, emb_assistant).item())

        similarities.append(sim)
        results.append(
            {
                "index": idx,
                "human_answer": human,
                "assistant_answer": assistant,
                "similarity": round(sim, 4),
            }
        )
        logger.info("Item %s similarity: %.4f", idx, sim)

    avg = sum(similarities) / len(similarities) if similarities else 0.0
    logger.info("Average similarity: %.4f", avg)
    return {"items": results, "average_similarity": round(avg, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute embedding similarity between human and model answers."
    )
    parser.add_argument("--file", type=str, required=True, help="JSON data file")
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="SentenceTransformer model path or name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading model: %s", args.model_path)
    model = SentenceTransformer(args.model_path)

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    results = calculate_similarity(data, model)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
