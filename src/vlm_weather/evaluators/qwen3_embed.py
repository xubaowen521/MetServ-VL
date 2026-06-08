"""Qwen3-Embedding-based similarity evaluation using ModelScope."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from modelscope import AutoModel, AutoTokenizer
from torch import Tensor

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.environ.get("QWEN3_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

_SAMPLE_DATA: list[dict[str, Any]] = [
    {
        "index": 0,
        "image_path": "200000694-图1.jpg",
        "human_answer": "未来10天（18-27日），江南西南部、华南北部...",
        "assistant_answer": "未来10天，新疆北部、西北地区东南部...",
        "question": "请基于提供的预报图给出具体预报",
    }
]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool hidden states using the last non-padding token."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def calculate_similarity(
    text1: str,
    text2: str,
    tokenizer: Any,
    model: Any,
    max_length: int = 8192,
) -> float:
    """Compute cosine similarity between two texts."""
    batch_dict = tokenizer(
        [text1, text2],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return float((embeddings[0] @ embeddings[1]).item())


def evaluate_file(
    file_path: str | Path | None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Evaluate a JSON file or demo data using Qwen3 embedding similarity.

    Args:
        file_path: Path to JSON file, or ``None`` for demo data.
        model_name: Qwen3 embedding model identifier.

    Returns:
        Evaluation result dictionary.
    """
    if file_path:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if not data:
            logger.warning("No data found; using demo data")
            data = _SAMPLE_DATA
    else:
        data = _SAMPLE_DATA

    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    results: list[dict[str, Any]] = []
    similarities: list[float] = []

    for item in data:
        try:
            idx = item["index"]
            human = item.get("human_answer", "")
            assistant = item.get("assistant_answer", "")
            sim = calculate_similarity(human, assistant, tokenizer, model)
            similarities.append(sim)
            results.append(
                {
                    "index": idx,
                    "question": item.get("question", ""),
                    "similarity": round(sim, 4),
                    "human_answer": human,
                    "assistant_answer": assistant,
                    "image_path": item.get("image_path", ""),
                }
            )
            logger.info("Processed item %d: similarity=%.4f", idx, sim)
        except Exception as exc:
            logger.error("Failed to process item %s: %s", item.get("index", "?"), exc)

    avg = sum(similarities) / len(similarities) if similarities else 0.0
    return {
        "total_cases": len(results),
        "average_similarity": round(avg, 4),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate forecast texts with Qwen3 embedding similarity."
    )
    parser.add_argument("--file", type=str, default=None, help="JSON data file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Qwen3 embedding model name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = evaluate_file(args.file, model_name=args.model)

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_summary.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_file)
    logger.info("Total cases: %d | Avg similarity: %.4f", result["total_cases"], result["average_similarity"])


if __name__ == "__main__":
    main()
